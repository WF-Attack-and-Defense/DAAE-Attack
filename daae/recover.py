"""Apply trained DAAE / FS-DAAE to recover dummy-removed traces.

Defended-only mode (recommended for deployment):
  Provide --defended-root + --checkpoint + training --output-dir (for
  .daae_cache/train_*.npz support bank). Originals are NOT required.

Paired mode (legacy / rebuild support bank):
  Also pass --original-root so train-split pairs can be aligned on the fly.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from daae.cli import add_alignment_arguments, add_dataset_scope_arguments
from daae.config import default_preload_workers
from daae.data.cell_io import (
    discover_defended_traces,
    discover_trace_pairs,
    parse_cell_file,
    write_cell_file,
)
from daae.data.dataset import split_pairs_by_class
from daae.data.features import extract_trace_features
from daae.data.support_bank import ClassSupportBank, find_train_cache
from daae.models.decode import warmup_decode_kernels
from daae.models.factory import load_model_from_checkpoint
from daae.models.fs_daae import FS_DAAE
from daae.models.fs_daae_legacy import FS_DAAE_Legacy
from daae.recover_utils import (
    DefendedSample,
    batch_insertion_decode,
    group_samples_by_class,
    preload_defended_samples,
)
from daae.recovery_post import enforce_monotonic_keep_runs
from daae.train_utils import unwrap_model


def _is_fs_model(model) -> bool:
    return isinstance(unwrap_model(model), (FS_DAAE, FS_DAAE_Legacy))


def recover_trace(
    model,
    defended_path: Path,
    trace_format: str,
    max_len: int,
    device: torch.device,
    cls_id: int,
    support_bank: ClassSupportBank | None = None,
    mask_threshold: float = 0.5,
    insertion_decode: bool = True,
    use_crf_decode: bool = False,
    crf_keep_bias: float = 0.0,
    min_keep_run: int = 0,
    calibrate_steps: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    trace = parse_cell_file(defended_path)
    features, valid_len = extract_trace_features(trace, max_len, trace_format)

    x = torch.from_numpy(features).unsqueeze(0).to(device)
    defended_len = torch.tensor([valid_len], dtype=torch.long, device=device)

    support_defended, support_mask = None, None
    proto_caches = None
    raw = unwrap_model(model)
    if isinstance(raw, (FS_DAAE, FS_DAAE_Legacy)) and support_bank is not None:
        support_defended, support_mask = support_bank.get(cls_id, device)
        proto_caches = [support_bank.get_prototype_cache(cls_id)]

    with torch.no_grad():
        recover_kw = dict(
            mask_threshold=mask_threshold,
            prototype_caches=proto_caches,
            calibrate_steps=calibrate_steps,
        )
        if isinstance(raw, FS_DAAE_Legacy):
            out = raw.recover(
                x, defended_len, support_defended, support_mask,
                use_subsequence_refine=False,
                use_crf_decode=use_crf_decode and not insertion_decode,
                crf_keep_bias=crf_keep_bias,
                **recover_kw,
            )
        elif isinstance(raw, FS_DAAE):
            out = raw.recover(
                x, defended_len, support_defended, support_mask,
                insertion_decode=insertion_decode,
                **recover_kw,
            )
        else:
            out = raw.recover(x, defended_len, mask_threshold=mask_threshold)

    keep = out["keep_mask"][0, :valid_len].cpu().numpy()
    if min_keep_run > 0:
        keep = enforce_monotonic_keep_runs(keep, min_keep_run)

    directions = trace.directions[:valid_len][keep.astype(bool)]
    timestamps = trace.timestamps[:valid_len][keep.astype(bool)]
    return timestamps, directions


def _recover_batch_same_class(
    model,
    samples: list[DefendedSample],
    support_bank: ClassSupportBank,
    device: torch.device,
    max_len: int,
    insertion_decode: bool,
    use_crf_decode: bool,
    crf_keep_bias: float,
    calibrate_steps: int,
    min_keep_run: int,
    decode_workers: int = 0,
    decode_candidates: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    b = len(samples)
    cls_id = samples[0].cls_id
    proto = support_bank.get_prototype_cache(cls_id)

    defended = torch.stack(
        [torch.from_numpy(np.ascontiguousarray(s.features)) for s in samples], dim=0
    )
    if device.type == "cuda":
        defended = defended.pin_memory()
    defended = defended.to(device, non_blocking=True)
    defended_len = torch.tensor(
        [s.valid_len for s in samples], dtype=torch.long, device=device
    )
    support_defended, support_mask = support_bank.expand_support(cls_id, b, device)
    raw = unwrap_model(model)
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    if insertion_decode and isinstance(raw, (FS_DAAE, FS_DAAE_Legacy)):
        with torch.inference_mode(), torch.amp.autocast(
            device_type=amp_device, enabled=device.type == "cuda"
        ):
            out = raw.forward(defended, support_defended, support_mask, defended_len)
            logits_np = out["mask_logits"].float().cpu().numpy()
        dirs_np = np.stack([s.features[0] for s in samples], axis=0)
        lens = np.array([s.valid_len for s in samples], dtype=np.int32)
        keep_np = batch_insertion_decode(
            dirs_np,
            logits_np,
            lens,
            proto,
            calibrate_steps=calibrate_steps,
            max_candidates=decode_candidates,
            workers=decode_workers,
        )
    else:
        proto_caches = [proto] * b
        with torch.inference_mode():
            recover_kw = dict(
                prototype_caches=proto_caches,
                calibrate_steps=calibrate_steps,
            )
            if isinstance(raw, FS_DAAE_Legacy):
                out = raw.recover(
                    defended, defended_len, support_defended, support_mask,
                    use_subsequence_refine=False,
                    use_crf_decode=use_crf_decode and not insertion_decode,
                    crf_keep_bias=crf_keep_bias,
                    **recover_kw,
                )
            elif isinstance(raw, FS_DAAE):
                out = raw.recover(
                    defended, defended_len, support_defended, support_mask,
                    insertion_decode=insertion_decode,
                    **recover_kw,
                )
            else:
                out = raw.recover(defended, defended_len)
        keep_np = out["keep_mask"].cpu().numpy()

    results = []
    for i, sample in enumerate(samples):
        valid = sample.valid_len
        keep = keep_np[i, :valid]
        if min_keep_run > 0:
            keep = enforce_monotonic_keep_runs(keep, min_keep_run)
        mask = keep.astype(bool)
        results.append((
            sample.timestamps[:valid][mask],
            sample.directions[:valid][mask],
        ))
    return results


def recover_traces_batched(
    model,
    samples: list[DefendedSample],
    support_bank: ClassSupportBank | None,
    device: torch.device,
    max_len: int,
    trace_format: str,
    batch_size: int,
    insertion_decode: bool,
    use_crf_decode: bool,
    crf_keep_bias: float,
    calibrate_steps: int,
    min_keep_run: int,
    output_dir: Path,
    resume: bool = False,
    decode_workers: int = 0,
    decode_candidates: int = 1,
    stats: dict | None = None,
) -> int:
    """Batch recovery grouped by class; returns number of traces written."""
    grouped = group_samples_by_class(samples)
    written = 0
    skipped = 0
    local_t0 = time.time()
    total = len(samples)

    for cls_id in sorted(grouped.keys()):
        cls_samples = grouped[cls_id]
        for start in range(0, len(cls_samples), batch_size):
            chunk = cls_samples[start : start + batch_size]
            if resume:
                pending = []
                for s in chunk:
                    out_path = output_dir / f"{s.cls_id}-{s.inst_id}.cell"
                    if out_path.is_file():
                        skipped += 1
                        continue
                    pending.append(s)
                if not pending:
                    continue
                chunk = pending

            if support_bank is not None and _is_fs_model(model):
                outputs = _recover_batch_same_class(
                    model, chunk, support_bank, device, max_len,
                    insertion_decode, use_crf_decode, crf_keep_bias,
                    calibrate_steps, min_keep_run,
                    decode_workers=decode_workers,
                    decode_candidates=decode_candidates,
                )
            else:
                outputs = []
                for s in chunk:
                    ts, dirs = recover_trace(
                        model, s.def_path, trace_format, max_len, device,
                        s.cls_id, support_bank=None, min_keep_run=min_keep_run,
                    )
                    outputs.append((ts, dirs))

            for sample, (timestamps, directions) in zip(chunk, outputs):
                write_cell_file(
                    output_dir / f"{sample.cls_id}-{sample.inst_id}.cell",
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
    """Load few-shot supports from train cache, or rebuild from paired originals."""
    if not _is_fs_model(model):
        return None

    cache_dir = output_dir / ".daae_cache"
    train_cache = find_train_cache(cache_dir, preferred=args.support_cache)
    if train_cache is not None:
        bank = ClassSupportBank.from_train_cache(
            train_cache, config.support_shots, device=None
        )
        bank.ensure_on_device(device)
        print(
            f"Few-shot support: {config.support_shots} shots/class from cache, "
            f"{len(bank._prototype_cache)} classes",
            flush=True,
        )
        return bank

    if args.original_root is None:
        raise FileNotFoundError(
            "FS-DAAE recovery needs a support bank, but no train cache was found under "
            f"{cache_dir} and --original-root was not provided.\n"
            "Either:\n"
            "  1) Point --output-dir at the training run that contains .daae_cache/train_*.npz, or\n"
            "  2) Pass --support-cache /path/to/train_XXXX.npz, or\n"
            "  3) Pass --original-root so supports can be rebuilt from paired traces."
        )

    original_root = args.original_root.resolve()
    defended_root = args.defended_root.resolve()
    pairs = discover_trace_pairs(
        original_root, defended_root, config.num_classes, config.num_instances
    )
    if not pairs:
        raise FileNotFoundError(
            f"No paired traces under {original_root} and {defended_root}"
        )
    split = split_pairs_by_class(pairs, config.train_ratio, config.val_ratio, config.seed)
    bank = ClassSupportBank.from_pairs(split.train, config, cache_dir, device=None)
    bank.ensure_on_device(device)
    print(
        f"Few-shot support: rebuilt from {len(split.train)} train pairs "
        f"({config.support_shots} shots/class)",
        flush=True,
    )
    return bank


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Recover original-like traces from defended traffic using a trained DAAE. "
            "Originals are optional at inference; use the training .daae_cache for supports."
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
        help="Optional originals (only needed to rebuild support bank or use --split train/val/test)",
    )
    parser.add_argument(
        "--support-cache",
        type=Path,
        default=None,
        help="Optional path to train_*.npz used as the FS-DAAE support bank",
    )
    add_dataset_scope_arguments(parser)
    add_alignment_arguments(parser, default=None)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which traces to recover. Without --original-root only 'all' is supported.",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for GPU inference (grouped by class)",
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
        help="CPU threads for insertion Viterbi (Numba nogil)",
    )
    parser.add_argument(
        "--decode-candidates", type=int, default=1,
        help="Support prototypes per trace (1 is much faster; 2 is the old default)",
    )
    parser.add_argument(
        "--calibrate-steps", type=int, default=3,
        help="Binary-search steps for insertion decode",
    )
    parser.add_argument(
        "--no-insertion-decode",
        action="store_true",
        help="Disable insertion-only subsequence decode (not recommended)",
    )
    parser.add_argument(
        "--crf-decode",
        action="store_true",
        help="Legacy BiLSTM+CRF Viterbi decode instead of insertion decode",
    )
    parser.add_argument(
        "--crf-keep-bias",
        type=float,
        default=1.0,
        help="CRF keep-label penalty (legacy only)",
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

    device = torch.device(args.device)
    ckpt_path = args.checkpoint.resolve()
    if not ckpt_path.is_file():
        candidates = sorted(Path.cwd().glob("results/**/*.pt"))
        hint = "\n  ".join(str(p) for p in candidates[:8])
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Use --checkpoint with a trained model, e.g.:\n"
            f"  --checkpoint ./results/frugal-smoke/daae_best_mask.pt\n"
            f"Found under results/:\n  {hint if hint else '(none)'}"
        )

    model, config = load_model_from_checkpoint(str(ckpt_path), device)
    model.eval()
    torch.set_num_threads(1)
    config.num_classes = args.num_classes
    config.num_instances = args.num_instances
    if args.alignment_method is not None:
        config.alignment_method = args.alignment_method
    arch = getattr(config, "architecture", "unet")
    insertion = not args.no_insertion_decode and not args.crf_decode
    print(
        f"Model architecture: {arch} | insertion_decode={insertion} | "
        f"batch_size={args.batch_size} | calibrate_steps={args.calibrate_steps} | "
        f"decode_workers={args.decode_workers} | chunk_size={args.chunk_size}",
        flush=True,
    )
    print(f"Defended: {defended_root}", flush=True)
    print(f"Output:   {recovered_dir}", flush=True)
    print("Warming insertion-decode kernels...", flush=True)
    warmup_decode_kernels()

    support_bank = _build_support_bank(args, model, config, output_dir, device)

    # Select which defended traces to recover.
    if args.original_root is not None and args.split != "all":
        original_root = args.original_root.resolve()
        pairs = discover_trace_pairs(
            original_root, defended_root, config.num_classes, config.num_instances
        )
        if not pairs:
            raise FileNotFoundError(
                f"No paired traces under {original_root} and {defended_root}"
            )
        split = split_pairs_by_class(
            pairs, config.train_ratio, config.val_ratio, config.seed
        )
        selected = getattr(split, args.split)
        print(
            f"Recovering split={args.split} ({len(selected)} paired traces)",
            flush=True,
        )
    else:
        if args.split != "all" and args.original_root is None:
            print(
                f"Note: --split {args.split} requires --original-root; "
                f"recovering all defended traces instead.",
                flush=True,
            )
        selected = discover_defended_traces(
            defended_root, config.num_classes, config.num_instances
        )
        if not selected:
            raise FileNotFoundError(
                f"No defended traces found under {defended_root} "
                f"(classes={config.num_classes}, instances={config.num_instances})"
            )
        print(f"Recovering {len(selected)} defended traces (no originals)", flush=True)

    def _item_ids(item):
        if len(item) == 4:
            return int(item[2]), int(item[3])
        return int(item[1]), int(item[2])

    pending = list(selected)
    skipped = 0
    if args.resume:
        kept = []
        for item in pending:
            cls_id, inst_id = _item_ids(item)
            if (recovered_dir / f"{cls_id}-{inst_id}.cell").is_file():
                skipped += 1
            else:
                kept.append(item)
        pending = kept
        print(f"Resume: {skipped} already recovered, {len(pending)} remaining", flush=True)

    if not pending:
        print(f"Nothing to recover. {skipped} traces already in {recovered_dir}", flush=True)
        return

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
            insertion_decode=insertion,
            use_crf_decode=args.crf_decode,
            crf_keep_bias=args.crf_keep_bias,
            calibrate_steps=args.calibrate_steps,
            min_keep_run=args.min_keep_run,
            output_dir=recovered_dir,
            resume=False,
            decode_workers=args.decode_workers,
            decode_candidates=args.decode_candidates,
            stats=stats,
        )
        del samples

    print(
        f"Done. Wrote {stats['written']} recovered traces to {recovered_dir} "
        f"(skipped {stats['skipped']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
