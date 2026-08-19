"""Fast parallel preload and disk cache for DAAE datasets."""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from daae.data.alignment import ALIGNMENT_VERSION, align_traces
from daae.data.cell_io import parse_cell_file
from daae.data.features import extract_features_from_arrays


def _pad_1d(arr: np.ndarray, max_len: int, pad_value: float = 0.0) -> np.ndarray:
    out = np.full(max_len, pad_value, dtype=np.float32)
    n = min(len(arr), max_len)
    out[:n] = arr[:n]
    return out


def _preload_sample_tuple(
    orig_path: str,
    def_path: str,
    cls_id: int,
    max_len: int,
    trace_format: str,
    alignment_method: str = "hamming",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Worker-friendly preload; returns numpy arrays (no torch)."""
    orig = parse_cell_file(orig_path)
    defended = parse_cell_file(def_path)

    # Truncate before alignment: LCS/hamming DP is O(n*m) and allocates (n+1)*(m+1).
    # Aligning full FRUGAL/Palette traces (often >> max_len) can thrash for days.
    orig_dirs = orig.directions.astype(np.float32)[:max_len]
    def_dirs = defended.directions.astype(np.float32)[:max_len]
    def_ts = defended.timestamps[:max_len]
    def_raw_dirs = defended.directions[:max_len]

    alignment = align_traces(orig_dirs, def_dirs, method=alignment_method)
    recon_target = np.zeros(len(def_dirs), dtype=np.float32)
    valid_idx = alignment.orig_indices >= 0
    recon_target[valid_idx] = orig_dirs[alignment.orig_indices[valid_idx]]

    def_features, def_len = extract_features_from_arrays(
        def_ts, def_raw_dirs, max_len, trace_format
    )

    return (
        def_features,
        _pad_1d(recon_target, max_len),
        _pad_1d(alignment.keep_mask, max_len),
        def_len,
        min(len(orig_dirs), max_len),
        cls_id,
    )


def _cache_tag(
    split_name: str,
    pairs: List[Tuple[Path, Path, int, int]],
    max_len: int,
    trace_format: str,
    alignment_method: str = "hamming",
) -> str:
    h = hashlib.sha1()
    h.update(split_name.encode())
    h.update(str(max_len).encode())
    h.update(trace_format.encode())
    h.update(ALIGNMENT_VERSION.encode())
    h.update(alignment_method.encode())
    for orig, def_path, cls_id, inst_id in pairs[:5]:
        h.update(f"{orig}|{def_path}|{cls_id}|{inst_id}".encode())
    h.update(str(len(pairs)).encode())
    return h.hexdigest()[:16]


def _cache_path(cache_dir: Path, split_name: str, tag: str) -> Path:
    return cache_dir / f"{split_name}_{tag}.npz"


def preload_pairs_parallel(
    pairs: List[Tuple[Path, Path, int, int]],
    max_len: int,
    trace_format: str,
    workers: int,
    split_name: str = "train",
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
    alignment_method: str = "hamming",
) -> Dict[str, np.ndarray]:
    """Preload all pairs in parallel; optionally read/write .npz cache."""
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tag = _cache_tag(split_name, pairs, max_len, trace_format, alignment_method)
        cache_file = _cache_path(cache_dir, split_name, tag)
        if use_cache and cache_file.is_file():
            print(f"Loading {split_name} cache: {cache_file}", flush=True)
            return dict(np.load(cache_file))

    workers = max(1, min(workers, os.cpu_count() or 1, len(pairs)))
    tasks = [
        (str(o), str(d), c, max_len, trace_format, alignment_method)
        for o, d, c, _ in pairs
    ]

    results: List[Optional[Tuple]] = [None] * len(tasks)
    print(
        f"Preloading {split_name} split ({len(tasks)} traces, {workers} workers, "
        f"alignment={alignment_method}, max_len={max_len})...",
        flush=True,
    )

    # Progress every ~1% (min 25) so long LCS jobs are not silent for hours.
    step = max(25, len(tasks) // 100) if len(tasks) >= 50 else max(1, len(tasks))

    if workers == 1:
        for i, task in enumerate(tasks):
            results[i] = _preload_sample_tuple(*task)
            if (i + 1) % step == 0 or (i + 1) == len(tasks):
                print(f"  {split_name}: {i + 1}/{len(tasks)}", flush=True)
    else:
        # Submit in chunks to avoid queuing tens of thousands of futures at once.
        chunk = max(workers * 4, 64)
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(tasks), chunk):
                batch = tasks[start : start + chunk]
                futures = {
                    pool.submit(_preload_sample_tuple, *task): start + offset
                    for offset, task in enumerate(batch)
                }
                for fut in as_completed(futures):
                    idx = futures[fut]
                    results[idx] = fut.result()
                    done += 1
                    if done % step == 0 or done == len(tasks):
                        print(f"  {split_name}: {done}/{len(tasks)}", flush=True)

    defended = np.stack([r[0] for r in results], axis=0)
    recon_target = np.stack([r[1] for r in results], axis=0)
    dummy_mask = np.stack([r[2] for r in results], axis=0)
    defended_len = np.array([r[3] for r in results], dtype=np.int64)
    original_len = np.array([r[4] for r in results], dtype=np.int64)
    labels = np.array([r[5] for r in results], dtype=np.int64)

    arrays = {
        "defended": defended,
        "recon_target": recon_target,
        "dummy_mask": dummy_mask,
        "defended_len": defended_len,
        "original_len": original_len,
        "labels": labels,
    }

    if cache_dir is not None and use_cache:
        print(f"Writing {split_name} cache (compressed)…", flush=True)
        np.savez_compressed(cache_file, **arrays)
        print(f"Saved {split_name} cache: {cache_file}", flush=True)

    return arrays


class StackedPairedDataset(Dataset):
    """Memory-efficient dataset backed by pre-stacked tensors (optional GPU residence)."""

    def __init__(
        self,
        arrays: Dict[str, np.ndarray],
        device: Optional[torch.device] = None,
    ):
        dtype = torch.float32
        long_dtype = torch.long

        def _tensor(arr: np.ndarray, floating: bool = True) -> torch.Tensor:
            t = torch.as_tensor(arr, dtype=dtype if floating else long_dtype)
            if device is not None:
                t = t.to(device, non_blocking=True)
            return t

        self.defended = _tensor(arrays["defended"])
        self.recon_target = _tensor(arrays["recon_target"])
        self.dummy_mask = _tensor(arrays["dummy_mask"])
        self.defended_len = _tensor(arrays["defended_len"], floating=False)
        self.original_len = _tensor(arrays["original_len"], floating=False)
        self.labels = _tensor(arrays["labels"], floating=False)
        self.device = device

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "defended": self.defended[idx],
            "recon_target": self.recon_target[idx],
            "dummy_mask": self.dummy_mask[idx],
            "defended_len": self.defended_len[idx],
            "original_len": self.original_len[idx],
            "label": self.labels[idx],
        }
