"""Fast batched recovery utilities."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from daae.data.cell_io import parse_cell_file
from daae.data.features import extract_features_from_arrays
from daae.models.decode import decode_insertion_mask, prototype_directions, support_keep_ratio


@dataclass
class DefendedSample:
    cls_id: int
    inst_id: int
    def_path: Path
    features: np.ndarray  # (C, max_len)
    valid_len: int
    timestamps: np.ndarray
    directions: np.ndarray
    raw_directions: Optional[np.ndarray] = None
    orig_dirs: Optional[np.ndarray] = None  # unused at unknown-class recover
    support_cls_id: Optional[int] = None  # inferred train class; not the filename label


@dataclass
class ClassPrototypeCache:
    target_ratio: float
    proto_dirs: List[np.ndarray]
    proto_masks: List[np.ndarray]


def _load_defended_sample(args: Tuple) -> DefendedSample:
    """args: (def_path, cls_id, inst_id, max_len, trace_format[, orig_path])."""
    path, cls_id, inst_id, max_len, trace_format = args[:5]
    orig_path = args[5] if len(args) > 5 else None
    trace = parse_cell_file(path)
    ts = trace.timestamps[:max_len]
    raw = np.asarray(
        (trace.raw_directions if trace.raw_directions is not None else trace.directions)[:max_len],
        dtype=np.float32,
    )
    dirs = np.where(raw > 0.0, 1.0, -1.0).astype(np.float32)
    fmt = trace_format if trace_format in ("direction", "paired_dir") else "direction"
    features, valid_len = extract_features_from_arrays(ts, raw, max_len, fmt)
    orig_dirs = None
    if orig_path is not None:
        orig = parse_cell_file(orig_path)
        orig_dirs = np.asarray(orig.directions[:max_len], dtype=np.float32)
    return DefendedSample(
        cls_id=cls_id,
        inst_id=inst_id,
        def_path=Path(path),
        features=features,
        valid_len=valid_len,
        timestamps=ts,
        directions=dirs,
        raw_directions=raw,
        orig_dirs=orig_dirs,
    )


def preload_defended_samples(
    items: Sequence,
    max_len: int,
    trace_format: str,
    workers: int = 8,
    progress: bool = True,
) -> List[DefendedSample]:
    """Parallel load of defended traces (no alignment — recovery only).

    ``items`` may be:
      - paired tuples ``(orig_path, def_path, cls_id, inst_id)``, or
      - defended-only tuples ``(def_path, cls_id, inst_id)``.
    """
    tasks = []
    for item in items:
        if len(item) == 4:
            orig_path, def_path, cls_id, inst_id = item
            tasks.append(
                (str(def_path), int(cls_id), int(inst_id), max_len, trace_format, str(orig_path))
            )
        elif len(item) == 3:
            def_path, cls_id, inst_id = item
            tasks.append((str(def_path), int(cls_id), int(inst_id), max_len, trace_format))
        else:
            raise ValueError(f"Expected 3- or 4-tuple, got {item!r}")

    if workers <= 1 or len(tasks) <= 1:
        return [_load_defended_sample(t) for t in tasks]

    out: List[Optional[DefendedSample]] = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_load_defended_sample, t): i for i, t in enumerate(tasks)}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            out[i] = fut.result()
            done += 1
            if progress and (done % 500 == 0 or done == len(tasks)):
                print(f"  preloaded {done}/{len(tasks)} defended traces", flush=True)
    return [s for s in out if s is not None]


def build_prototype_cache(
    support_dirs: np.ndarray,
    support_masks: np.ndarray,
    valid_lens: np.ndarray,
    max_candidates: int = 1,
) -> ClassPrototypeCache:
    """Precompute direction prototypes for one class (K support traces)."""
    k = support_masks.shape[0]
    # Prefer the longest support keep sequences so decode does not systematically
    # under-keep relative to typical original lengths in the class.
    scored = []
    ratios = []
    for ki in range(k):
        sl = max(int(valid_lens[ki]), 1)
        kept = float(support_masks[ki, :sl].sum())
        ratios.append(kept / float(sl))
        scored.append((-kept, ki))
    scored.sort(key=lambda x: (x[0], x[1]))
    ids = [ki for _, ki in scored[:max_candidates]]
    # Floor keep-ratio at the longest support (not the median) to avoid over-stripping.
    target = max(ratios) if ratios else support_keep_ratio(support_masks)
    protos = []
    pmasks = []
    for ki in ids:
        sl = int(valid_lens[ki])
        protos.append(np.ascontiguousarray(
            prototype_directions(support_dirs[ki], support_masks[ki], sl)
        ))
        pmasks.append(np.ascontiguousarray(support_masks[ki, :sl].copy()))
    return ClassPrototypeCache(target_ratio=float(target), proto_dirs=protos, proto_masks=pmasks)


def _decode_one(args: Tuple) -> Tuple[int, np.ndarray]:
    bi, dirs, logits, proto_dirs, proto_masks, target_ratio, calibrate_steps, max_candidates = args
    decoded = decode_insertion_mask(
        dirs,
        logits,
        [],
        proto_masks,
        target_ratio=target_ratio,
        calibrate=True,
        calibrate_steps=calibrate_steps,
        max_candidates=max_candidates,
        precomputed_protos=proto_dirs,
    )
    return bi, decoded


def batch_insertion_decode(
    dirs_batch: np.ndarray,
    logits_batch: np.ndarray,
    lens_batch: np.ndarray,
    proto_cache: ClassPrototypeCache,
    calibrate_steps: int = 3,
    max_candidates: int = 1,
    workers: int = 0,
) -> np.ndarray:
    """CPU insertion decode for a batch sharing the same class prototypes.

    ``workers>1`` runs traces in parallel (Numba kernels release the GIL).
    """
    b, seq_len = logits_batch.shape
    keep = np.zeros((b, seq_len), dtype=np.float32)
    protos = proto_cache.proto_dirs[:max_candidates]
    masks = proto_cache.proto_masks
    target = proto_cache.target_ratio
    if workers <= 0:
        workers = min(b, os.cpu_count() or 4)

    tasks = []
    for bi in range(b):
        valid = int(lens_batch[bi])
        tasks.append((
            bi,
            np.ascontiguousarray(dirs_batch[bi, :valid], dtype=np.int8),
            np.ascontiguousarray(logits_batch[bi, :valid], dtype=np.float64),
            protos,
            masks,
            target,
            calibrate_steps,
            max_candidates,
        ))

    if workers <= 1 or b <= 1:
        for task in tasks:
            bi, decoded = _decode_one(task)
            keep[bi, : len(decoded)] = decoded
        return keep

    with ThreadPoolExecutor(max_workers=min(workers, b)) as pool:
        for bi, decoded in pool.map(_decode_one, tasks):
            keep[bi, : len(decoded)] = decoded
    return keep


def group_samples_by_class(samples: List[DefendedSample]) -> Dict[int, List[DefendedSample]]:
    grouped: Dict[int, List[DefendedSample]] = {}
    for s in samples:
        grouped.setdefault(s.cls_id, []).append(s)
    return grouped
