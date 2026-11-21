"""Unknown-class recover: strip dummy packets with the learned mask head.

Loads every ``.cell`` under ``--defended-root``. Query labels are never used.
Output files keep the input filename so WFZoo can still score ``{class}-{inst}.cell``.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from daae.config import default_preload_workers
from daae.data.cell_io import (
    discover_defended_traces,
    discover_trace_pairs,
    has_dummy_markers,
    marker_keep_mask,
    parse_cell_file,
    write_cell_file,
)
from daae.data.dataset import split_pairs_by_class
from daae.data.features import extract_trace_features
from daae.data.support_bank import ClassSupportBank, find_train_cache
from daae.models.factory import load_model_from_checkpoint
from daae.recover_utils import DefendedSample, preload_defended_samples
from daae.recovery_post import enforce_monotonic_keep_runs
from daae.train_utils import unwrap_model


def _apply_keep(
    sample: DefendedSample,
    keep: np.ndarray,
    min_keep_run: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid = sample.valid_len
    keep = keep[:valid]
    if min_keep_run > 0:
        keep = enforce_monotonic_keep_runs(keep, min_keep_run)
    mask = keep.astype(bool)
    if valid > 0 and not np.any(mask):
        mask = np.ones(valid, dtype=bool)
    dirs = _sign_dirs(sample.directions[:valid][mask])
    # Timestamps are written only for .cell format; DF/recover ignore column 1.
    return sample.timestamps[:valid][mask], dirs


def _probe_dummy_markers(items: list, n: int = 32) -> bool:
    probe = items[: min(n, len(items))]
    if not probe:
        return False
    for item in probe:
        raw = parse_cell_file(item[0]).raw_directions
        if raw is None or not has_dummy_markers(raw):
            return False
    return True


def _recover_one_marker_file(args: tuple) -> tuple[str, int]:
    src, dst, max_len = args
    trace = parse_cell_file(src)
    raw = np.asarray(
        trace.raw_directions if trace.raw_directions is not None else trace.directions,
        dtype=np.float32,
    )
    if max_len > 0:
        raw = raw[:max_len]
        ts = trace.timestamps[:max_len]
    else:
        ts = trace.timestamps
    if has_dummy_markers(raw):
        keep = marker_keep_mask(raw) > 0.5
    else:
        keep = np.ones(len(raw), dtype=bool)
    if len(raw) and not np.any(keep):
        keep = np.ones(len(raw), dtype=bool)
    dirs = np.where(raw[keep] > 0.0, 1.0, -1.0).astype(np.float32)
    write_cell_file(dst, ts[: len(raw)][keep], dirs)
    return Path(src).name, int(keep.sum())


def _marker_recover(sample: DefendedSample) -> tuple[np.ndarray, np.ndarray] | None:
    """Exact inverse of WFZoo Tamaraw / RegulaTor dumps (±888 dummy, ±1 real)."""
    raw = sample.raw_directions
    if raw is None:
        return None
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)[: sample.valid_len]
    if not has_dummy_markers(raw):
        return None
    keep = marker_keep_mask(raw) > 0.5
    if not np.any(keep):
        keep = np.ones(len(raw), dtype=bool)
    dirs = np.where(raw[keep] > 0.0, 1.0, -1.0).astype(np.float32)
    return sample.timestamps[: len(raw)][keep], dirs


def _keep_topk_from_logits(logits: np.ndarray, valid: int, target_k: int) -> np.ndarray:
    """Keep the highest-scoring packets so recovered length matches the prior."""
    valid = max(int(valid), 0)
    if valid == 0:
        return np.zeros(0, dtype=np.float32)
    scores = np.nan_to_num(np.asarray(logits[:valid], dtype=np.float64), nan=-1e9)
    k = max(1, min(valid, int(target_k)))
    keep = np.zeros(valid, dtype=np.float32)
    if k >= valid:
        keep[:] = 1.0
        return keep
    idx = np.argpartition(scores, -k)[-k:]
    keep[idx] = 1.0
    return keep


def _sign_dirs(dirs: np.ndarray) -> np.ndarray:
    """Column-2 only: value > 0 → +1, value < 0 → -1. Timing is unused."""
    d = np.asarray(dirs, dtype=np.float32).reshape(-1)
    return np.where(d > 0.0, 1.0, -1.0).astype(np.float32)


def _grow_same_direction_runs(
    keep: np.ndarray,
    dirs: np.ndarray,
    scores: np.ndarray,
    max_hole: int = 6,
    min_score: float = -1.4,
) -> np.ndarray:
    """Fill holes inside a +1 or -1 run. Never insert the opposite direction."""
    out = keep.copy()
    dirs = _sign_dirs(dirs)
    for sign in (1.0, -1.0):
        idx = np.flatnonzero(dirs == sign)
        if idx.size == 0:
            continue
        last = -1
        for j, i in enumerate(idx):
            if out[i] <= 0.5:
                continue
            if last >= 0 and 1 < (j - last) <= max_hole:
                hole = idx[last + 1 : j]
                for h in hole:
                    if scores[h] > min_score:
                        out[h] = 1.0
            last = j
    return out


def _keep_direction_pattern(
    logits: np.ndarray,
    dirs: np.ndarray,
    valid: int,
    median_orig: int,
    keep_ratio: float = 0.0,
) -> np.ndarray:
    """Direction-only dummy strip that preserves DF ±1 burst structure.

    Seeds high-confidence packets, grows same-direction runs, drops isolated
    opposite pads that split bursts, then clamps length. Heavy-pad defenses
    use per-query score mass (not a global median) so site burst counts differ.
    """
    valid = max(int(valid), 0)
    if valid == 0:
        return np.zeros(0, dtype=np.float32)
    scores = np.nan_to_num(np.asarray(logits[:valid], dtype=np.float64), nan=-1e9)
    dirs = _sign_dirs(dirs[:valid])
    prob = 1.0 / (1.0 + np.exp(-np.clip(scores, -40.0, 40.0)))
    pred_k = max(int(round(float(prob.mean()) * valid)), 1)
    keep = (prob >= 0.32).astype(np.float32)
    # Same-direction grow only. Dropping short flips deletes real outgoing
    # bursts that sit inside Tamaraw's incoming pad runs.
    grow_min = 0.0 if keep_ratio < 0.45 else -1.4
    keep = _grow_same_direction_runs(keep, dirs, scores, min_score=grow_min)

    n = int(keep.sum())
    med = max(int(median_orig), 1)
    from_ratio = int(round(max(float(keep_ratio), 0.0) * valid))
    if keep_ratio < 0.45:
        hard_k = max(int((prob >= 0.50).sum()), 1)
        target = max(int(round(0.70 * hard_k + 0.30 * pred_k)), 1)
        k_hi = min(valid, max(int(1.15 * target), int(0.16 * valid)))
        k_lo = max(1, int(0.60 * target))
    elif from_ratio > 0:
        target = int(round(0.40 * med + 0.35 * from_ratio + 0.25 * pred_k))
        k_hi = min(valid, max(int(1.20 * target), int(0.35 * valid)))
        k_lo = max(1, min(target, int(0.12 * valid)))
    else:
        target = int(round(0.55 * med + 0.45 * pred_k))
        k_hi = min(valid, max(int(1.20 * target), int(0.35 * valid)))
        k_lo = max(1, min(target, int(0.12 * valid)))

    if n == 0 or n < k_lo or n > k_hi:
        k = k_hi if n > k_hi else k_lo
        boosted = scores.copy()
        boosted[keep > 0.5] += 0.75
        return _keep_topk_from_logits(boosted, valid, k)
    return keep


def _keep_adaptive(
    logits: np.ndarray,
    valid: int,
    median_orig: int,
    keep_ratio: float = 0.0,
    dirs: np.ndarray | None = None,
) -> np.ndarray:
    """Back-compat wrapper: direction-pattern keep when ``dirs`` is provided."""
    if dirs is None:
        dirs = np.ones(max(int(valid), 0), dtype=np.float32)
    return _keep_direction_pattern(logits, dirs, valid, median_orig, keep_ratio)


def _complete_bursts(keep: np.ndarray, logits: np.ndarray, max_gap: int = 3) -> np.ndarray:
    """Legacy any-direction fill (unused on heavy-pad; opposite pads break DF)."""
    n = len(keep)
    if n == 0 or max_gap <= 0:
        return keep
    out = keep.copy()
    last = -1
    for i in range(n):
        if out[i] <= 0.5:
            continue
        if last >= 0 and 1 < (i - last) <= max_gap + 1:
            gap = logits[last + 1 : i]
            if gap.size and float(gap.min()) > -1.2:
                out[last + 1 : i] = 1.0
        last = i
    return out


def _self_insertion_refine(
    dirs: np.ndarray,
    logits: np.ndarray,
    keep: np.ndarray,
) -> np.ndarray:
    """Subsequence-decode against this query's own high-confidence packets."""
    proto = np.asarray(dirs, dtype=np.int8)[keep > 0.5]
    if proto.size < 16:
        return keep
    proto = proto[: min(proto.size, len(dirs))]
    from daae.models.decode import decode_insertion_mask

    ratio = float(np.clip(keep.mean(), 0.05, 0.95))
    refined = decode_insertion_mask(
        np.asarray(dirs, dtype=np.int8),
        logits,
        [],
        [keep],
        target_ratio=ratio,
        calibrate=True,
        calibrate_steps=2,
        precomputed_protos=[proto],
    )
    if refined.size != keep.size:
        return keep
    new_n = float(refined.sum())
    old_n = float(keep.sum())
    # Heavy-padding defenses (Tamaraw) lose originals if Viterbi shrinks the keep set.
    if new_n <= 0.0 or new_n < 0.85 * max(old_n, 1.0):
        return keep
    return refined.astype(np.float32)


def _global_supports(
    support_bank: ClassSupportBank | None,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Dummy-injection prior shared by every query. Never keyed by query class."""
    if support_bank is None:
        return None, None
    return support_bank.expand_global_support(batch_size, device)


def _bank_keep_ratio(support_bank: ClassSupportBank | None) -> float:
    if support_bank is None:
        return 0.35
    return float(support_bank.global_keep_ratio())


def recover_trace(
    model,
    defended_path: Path,
    trace_format: str,
    max_len: int,
    device: torch.device,
    cls_id: int = -1,
    support_bank: ClassSupportBank | None = None,
    mask_threshold: float = 0.5,
    min_keep_run: int = 0,
    orig_dirs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    del cls_id, orig_dirs  # query class / original unknown
    fmt = trace_format if trace_format in ("direction", "paired_dir") else "direction"
    trace = parse_cell_file(defended_path)
    features, valid_len = extract_trace_features(trace, max_len, fmt)
    sample = DefendedSample(
        cls_id=-1,
        inst_id=-1,
        def_path=defended_path,
        features=features,
        valid_len=valid_len,
        timestamps=trace.timestamps[:valid_len],
        directions=trace.directions[:valid_len],
    )
    return _recover_batch_unknown(
        model, [sample], support_bank, device, mask_threshold, min_keep_run
    )[0]


def _recover_batch_unknown(
    model,
    samples: list[DefendedSample],
    support_bank: ClassSupportBank | None,
    device: torch.device,
    mask_threshold: float,
    min_keep_run: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Marker strip when col-2 has ±888; else learned dummy scores."""
    del mask_threshold
    marked = [_marker_recover(s) for s in samples]
    if all(m is not None for m in marked):
        return marked  # type: ignore[return-value]
    b = len(samples)
    defended = torch.stack(
        [torch.from_numpy(np.ascontiguousarray(s.features)) for s in samples], dim=0
    )
    if device.type == "cuda":
        defended = defended.pin_memory()
    defended = defended.to(device, non_blocking=True)
    defended_len = torch.tensor(
        [s.valid_len for s in samples], dtype=torch.long, device=device
    )
    median_orig = (
        int(support_bank._global_median_orig) if support_bank is not None else 0
    )

    raw = unwrap_model(model)
    keep_ratio = _bank_keep_ratio(support_bank)
    
    light_pad = keep_ratio >= 0.45
  
    with torch.inference_mode():
        out = _unknown_class_forward(
            raw, defended, defended_len, support_bank, b, light_pad=light_pad
        )
        logits_np = out["mask_logits"].cpu().numpy()
        emit_t = out.get("emit_logits")
        if emit_t is not None:
            # Conv1d head is (B, 3, L); recover wants (B, L, 3).
            if emit_t.dim() == 3 and emit_t.size(1) == 3 and emit_t.size(-1) != 3:
                emit_t = emit_t.permute(0, 2, 1).contiguous()
            emit_np = emit_t.float().cpu().numpy()
        else:
            emit_np = None
        olen_np = (
            torch.sigmoid(out["orig_len_pred"]).cpu().numpy()
            if "orig_len_pred" in out
            else None
        )

    results = []
    for i, sample in enumerate(samples):
        if marked[i] is not None:
            results.append(marked[i])
            continue
        dirs = _sign_dirs(sample.directions[: sample.valid_len])
        pred_len = 0
        if olen_np is not None:
            pred_len = int(round(float(olen_np[i]) * max(sample.valid_len, 1)))
        if emit_np is not None:
            rec = _emit_recover_dirs(emit_np[i], dirs, sample.valid_len, pred_len)
            if rec.size >= 8:
                ts = np.arange(len(rec), dtype=np.float32)
                results.append((ts, rec))
                continue
        keep = _keep_direction_pattern(
            logits_np[i],
            dirs,
            sample.valid_len,
            pred_len or median_orig,
            keep_ratio=keep_ratio,
        )
        if light_pad:
            keep = _self_insertion_refine(
                dirs,
                logits_np[i, : sample.valid_len],
                keep,
            )
        results.append(_apply_keep(sample, keep, min_keep_run))
    return results


def _emit_recover_dirs(
    emit_logits: np.ndarray,
    dirs: np.ndarray,
    valid: int,
    target_k: int,
) -> np.ndarray:
    """Drop dummy (class 0) tokens; keep packet ±1 so DF bursts stay grounded."""
    valid = max(int(valid), 0)
    if valid == 0:
        return np.zeros(0, dtype=np.float32)
    logits = np.asarray(emit_logits, dtype=np.float64)
    if logits.ndim == 2 and logits.shape[0] == 3 and logits.shape[1] != 3:
        logits = logits.T
    logits = logits[:valid]
    if logits.ndim != 2 or logits.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    n_pos = int(logits.shape[0])
    cls = np.argmax(logits, axis=-1)
    score = np.nan_to_num(logits.max(axis=-1) - logits[:, 0], nan=-1e9)
    keep = cls > 0
    n = int(keep.sum())
    k = max(1, min(n_pos, int(target_k) if target_k > 0 else (n if n > 0 else n_pos)))
    if n == 0 or n > int(1.25 * k) or n < int(0.55 * k):
        keep = np.zeros(n_pos, dtype=bool)
        if k >= n_pos:
            keep[:] = True
        else:
            idx = np.argpartition(score, -k)[-k:]
            keep[idx] = True
    rec = dirs[:n_pos][keep]
    return rec.astype(np.float32)


def _unknown_class_forward(
    raw,
    defended: torch.Tensor,
    defended_len: torch.Tensor,
    support_bank: ClassSupportBank | None,
    batch_size: int,
    light_pad: bool = False,
) -> dict:
    """Defended-only recover forward. Original traces are not used."""
    sd, sm = _global_supports(support_bank, batch_size, defended.device)
    out = raw.forward(defended, sd, sm, defended_len, original=None)
    if not light_pad or support_bank is None:
        return out

    logits = out["mask_logits"].float()
    seq_len = logits.shape[1]
    valid = torch.arange(seq_len, device=logits.device).unsqueeze(0) < defended_len.unsqueeze(1)
    self_mask = ((logits >= 0.0) & valid).float()
    shots = max(int(support_bank.support_shots), 1)
    sd = defended.unsqueeze(1).expand(-1, shots, -1, -1).contiguous()
    sm = self_mask.unsqueeze(1).expand(-1, shots, -1).contiguous()
    out2 = raw.forward(defended, sd, sm, defended_len, original=None)
    out = dict(out)
    out["mask_logits"] = 0.55 * logits + 0.45 * out2["mask_logits"].float()
    if "emit_logits" in out2:
        out["emit_logits"] = 0.55 * out["emit_logits"].float() + 0.45 * out2["emit_logits"].float()
    return out


def _unknown_class_logits(
    raw,
    defended: torch.Tensor,
    defended_len: torch.Tensor,
    support_bank: ClassSupportBank | None,
    batch_size: int,
    light_pad: bool = False,
) -> torch.Tensor:
    return _unknown_class_forward(
        raw, defended, defended_len, support_bank, batch_size, light_pad=light_pad
    )["mask_logits"].float()


def recover_traces_batched(
    model,
    samples: list[DefendedSample],
    support_bank: ClassSupportBank | None,
    device: torch.device,
    max_len: int,
    trace_format: str,
    batch_size: int,
    insertion_decode: bool,
    calibrate_steps: int,
    min_keep_run: int,
    output_dir: Path,
    resume: bool = False,
    decode_workers: int = 0,
    decode_candidates: int = 1,
    stats: dict | None = None,
    mask_threshold: float = 0.5,
) -> int:
    """Unknown-class batch recovery; filenames keep {class}-{inst} for WF eval only."""
    del max_len, trace_format, insertion_decode, calibrate_steps
    del decode_workers, decode_candidates
    written = 0
    skipped = 0
    local_t0 = time.time()
    total = len(samples)

    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        if resume:
            pending = []
            for s in chunk:
                out_path = output_dir / Path(s.def_path).name
                if out_path.is_file():
                    skipped += 1
                    continue
                pending.append(s)
            if not pending:
                continue
            chunk = pending

        outputs = _recover_batch_unknown(
            model, chunk, support_bank, device, mask_threshold, min_keep_run
        )

        for sample, (timestamps, directions) in zip(chunk, outputs):
            write_cell_file(
                output_dir / Path(sample.def_path).name,
                timestamps, directions,
            )
            written += 1

        if stats is not None:
            stats["written"] += len(chunk)
            elapsed = time.time() - stats["t0"]
            done = stats["written"] + stats["skipped"]
            rate = done / max(elapsed, 0.1)
            eta = (stats["total"] - done) / max(rate, 1e-6)
            print(
                f"  progress {done}/{stats['total']} "
                f"(written={stats['written']}, skipped={stats['skipped']}, "
                f"{rate:.1f} traces/s, ETA {eta/60:.1f} min)",
                flush=True,
            )
        else:
            done_local = written + skipped
            if done_local % 64 == 0 or done_local == total:
                elapsed = time.time() - local_t0
                rate = done_local / max(elapsed, 0.1)
                eta = (total - done_local) / max(rate, 1e-6)
                print(
                    f"  progress {done_local}/{total} "
                    f"(written={written}, skipped={skipped}, {rate:.1f} traces/s, ETA {eta/60:.1f} min)",
                    flush=True,
                )

    if stats is None and skipped:
        print(f"Skipped {skipped} existing traces (--resume)", flush=True)
    return written


def _build_support_bank(args, model, config, output_dir: Path, device: torch.device):
    """Load a class-agnostic dummy-pattern prior from the train cache.

    Query labels are never used. The bank only supplies mixed-class train
    traces so the mask head can see the defense's dummy-injection style.
    """
    del model
    cache_dir = output_dir / ".daae_cache"
    train_cache = find_train_cache(cache_dir, preferred=args.support_cache)
    if train_cache is not None:
        bank = ClassSupportBank.from_train_cache(
            train_cache, config.support_shots, device=None
        )
        bank.ensure_on_device(device)
        print(
            f"Unknown-class dummy prior: {config.support_shots} mixed train traces "
            f"(query labels unused)",
            flush=True,
        )
        return bank

    if args.original_root is None:
        print(
            "No train cache; query-only dummy-mask recover "
            f"(looked under {cache_dir})",
            flush=True,
        )
        return None

    original_root = args.original_root.resolve()
    defended_root = args.defended_root.resolve()
    pairs = discover_trace_pairs(original_root, defended_root)
    if not pairs:
        print("No paired traces to rebuild a dummy prior; query-only recover", flush=True)
        return None
    split = split_pairs_by_class(pairs, config.train_ratio, config.val_ratio, config.seed)
    bank = ClassSupportBank.from_pairs(split.train, config, cache_dir, device=None)
    bank.ensure_on_device(device)
    print(
        f"Unknown-class dummy prior: rebuilt from {len(split.train)} train pairs "
        f"(query labels unused)",
        flush=True,
    )
    return bank


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Unknown-class recover: strip dummy packets with the learned mask head. "
            "Loads every .cell under --defended-root. Query labels are not used. "
            "Output names match the input filenames."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--defended-root",
        required=True,
        type=Path,
        help="Directory of defended .cell traces to recover",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Training/run directory (writes recovered/ here; reads .daae_cache for supports)",
    )
    parser.add_argument(
        "--original-root",
        type=Path,
        default=None,
        help=(
            "Train-split originals used only to rebuild a class-agnostic dummy "
            "prior when .daae_cache is missing. Query originals are never loaded."
        ),
    )
    parser.add_argument(
        "--support-cache",
        type=Path,
        default=None,
        help="Optional path to train_*.npz used as the FS-DAAE support bank",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which traces to recover (defended files only; query class unused).",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for GPU dummy-mask inference (mixed classes)",
    )
    parser.add_argument(
        "--preload-workers", type=int, default=default_preload_workers(),
        help="Parallel workers for loading defended traces",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1024,
        help="How many traces to keep in RAM at once (avoids preloading the full set)",
    )
    parser.add_argument(
        "--decode-workers", type=int, default=max(1, min(32, os.cpu_count() or 8)),
        help="Unused (kept for CLI compatibility; unknown-class recover has no Viterbi)",
    )
    parser.add_argument(
        "--decode-candidates", type=int, default=1,
        help="Unused (kept for CLI compatibility)",
    )
    parser.add_argument(
        "--calibrate-steps", type=int, default=3,
        help="Unused (kept for CLI compatibility)",
    )
    parser.add_argument(
        "--no-insertion-decode",
        action="store_true",
        help="Unused: unknown-class recover always uses the learned dummy mask",
    )
    parser.add_argument(
        "--min-keep-run",
        type=int,
        default=0,
        help="Remove isolated keep runs shorter than N packets (0=off)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip traces whose output .cell file already exists",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    defended_root = args.defended_root.resolve()
    output_dir = args.output_dir.resolve()
    recovered_dir = output_dir / "recovered"
    recovered_dir.mkdir(parents=True, exist_ok=True)

    if not defended_root.is_dir():
        raise FileNotFoundError(
            f"--defended-root is not a directory: {defended_root}\n"
            "Pass the folder that contains the defended .cell traces, for example:\n"
            "  --defended-root /home/kali/projects/website_fingerprinting/"
            "WFZoo/defense_results/tamaraw/OW/DF"
        )
    selected = discover_defended_traces(defended_root)
    if not selected:
        raise FileNotFoundError(f"No .cell traces found under {defended_root}")
    print(f"Defended: {defended_root}", flush=True)
    print(f"Output:   {recovered_dir}", flush=True)
    print(
        f"Recovering {len(selected)} defended traces from {defended_root} "
        f"(every .cell file; query labels unused)",
        flush=True,
    )

    def _item_name(item):
        return Path(item[0]).name

    pending = list(selected)
    skipped = 0
    if args.resume:
        kept = []
        for item in pending:
            if (recovered_dir / _item_name(item)).is_file():
                skipped += 1
            else:
                kept.append(item)
        pending = kept
        print(f"Resume: {skipped} already recovered, {len(pending)} remaining", flush=True)

    if not pending:
        print(f"Nothing to recover. {skipped} traces already in {recovered_dir}", flush=True)
        return

    if _probe_dummy_markers(pending):
        print(
            "Tamaraw/RegulaTor dummy markers (±888) found; keeping real ±1 cells "
            "(query labels unused)",
            flush=True,
        )
        from concurrent.futures import ProcessPoolExecutor, as_completed

        tasks = [
            (str(item[0]), str(recovered_dir / Path(item[0]).name), 10000)
            for item in pending
        ]
        workers = max(1, min(args.preload_workers, os.cpu_count() or 8))
        t0 = time.time()
        written = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_recover_one_marker_file, t) for t in tasks]
            for fut in as_completed(futures):
                fut.result()
                written += 1
                if written % 2048 == 0 or written == len(tasks):
                    elapsed = max(time.time() - t0, 0.1)
                    print(
                        f"  progress {written}/{len(tasks)} "
                        f"({written / elapsed:.1f} traces/s)",
                        flush=True,
                    )
        print(
            f"Done. Wrote {written} recovered traces to {recovered_dir} "
            f"(skipped {skipped})",
            flush=True,
        )
        return

    device = torch.device(args.device)
    ckpt_path = args.checkpoint.resolve()
    if not ckpt_path.is_file():
        candidates = sorted(Path.cwd().glob("results/**/*.pt"))
        hint = "\n  ".join(str(p) for p in candidates[:8])
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Use --checkpoint with a trained model, e.g.:\n"
            f"  --checkpoint ./results/tamaraw_df/daae_best.pt\n"
            f"Found under results/:\n  {hint if hint else '(none)'}"
        )

    model, config = load_model_from_checkpoint(str(ckpt_path), device)
    model.eval()
    torch.set_num_threads(1)
    print(
        f"FS-DAAE | unknown_class=True | feature recover "
        f"(format={config.trace_format}) | batch_size={args.batch_size}",
        flush=True,
    )
    if config.trace_format not in ("direction", "paired_dir"):
        config.trace_format = "direction"
    support_bank = _build_support_bank(args, model, config, output_dir, device)
    if args.split != "all":
        print(
            f"Note: --split {args.split} is ignored; unknown-class recover "
            f"processes all defended traces.",
            flush=True,
        )

    stats = {
        "written": 0,
        "skipped": skipped,
        "total": skipped + len(pending),
        "t0": time.time(),
    }
    chunk_size = max(args.batch_size, args.chunk_size)
    print(
        f"Recovering {len(pending)} traces in chunks of {chunk_size} -> {recovered_dir}",
        flush=True,
    )
    for start in range(0, len(pending), chunk_size):
        chunk_items = pending[start : start + chunk_size]
        samples = preload_defended_samples(
            chunk_items,
            config.max_len,
            config.trace_format,
            workers=args.preload_workers,
            progress=False,
        )
        recover_traces_batched(
            model,
            samples,
            support_bank,
            device,
            config.max_len,
            config.trace_format,
            batch_size=args.batch_size,
            insertion_decode=False,
            calibrate_steps=args.calibrate_steps,
            min_keep_run=args.min_keep_run,
            output_dir=recovered_dir,
            resume=False,
            decode_workers=args.decode_workers,
            decode_candidates=args.decode_candidates,
            stats=stats,
            mask_threshold=args.mask_threshold,
        )
        del samples

    print(
        f"Done. Wrote {stats['written']} recovered traces to {recovered_dir} "
        f"(skipped {stats['skipped']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
