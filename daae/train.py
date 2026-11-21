"""Train FS-DAAE for dummy-packet removal."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from daae.cli import add_dataset_scope_arguments, add_path_arguments, resolve_paths
from daae.config import DAAEConfig, default_batch_size, default_preload_workers
from daae.data.dataset import build_dataloaders
from daae.data.support_bank import ClassSupportBank
from daae.models.factory import create_model, forward_model, raw_state_dict
from daae.models.fs_daae import FS_DAAE
from daae.models.losses import daae_loss
from daae.train_utils import (
    amp_setup,
    make_valid_mask,
    maybe_compile,
    move_batch,
    setup_gpu_training,
    unwrap_model,
)


def _loss_kwargs(config: DAAEConfig) -> dict:
    return dict(
        trace_format=config.trace_format,
        mask_weight=config.mask_loss_weight,
        recon_weight=config.recon_loss_weight,
        focal_gamma=config.focal_gamma,
        use_focal=config.use_focal,
        dice_weight=config.dice_weight,
        front_mask_weight=config.front_mask_weight,
        length_ratio_weight=config.length_ratio_weight,
        burst_weight=getattr(config, "burst_weight", 0.4),
        rank_weight=getattr(config, "rank_weight", 0.5),
        emit_weight=1.0,
        orig_len_weight=0.5,
        distill_weight=0.4,
    )


def _is_fs_model(model) -> bool:
    return isinstance(unwrap_model(model), FS_DAAE)


def train_epoch(model, loader, optimizer, device, config, scaler, use_amp: bool, amp_dtype=None):
    model.train()
    totals = {"loss": 0.0, "mask_loss": 0.0, "recon_loss": 0.0}
    n = 0
    skipped = 0
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    autocast_kwargs = {"device_type": amp_device, "enabled": use_amp}
    if amp_dtype is not None:
        autocast_kwargs["dtype"] = amp_dtype

    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        # Train the query-only path used at unknown-class recover.
        if batch.get("support_defended") is not None and torch.rand(1).item() < 0.35:
            batch = dict(batch)
            batch["support_defended"] = None
            batch["support_mask"] = None

        # Student (defended-only / mixed supports) — this is the recover path.
        student_batch = dict(batch)
        student_batch["use_original"] = False
        teacher_logits = None
        with torch.amp.autocast(**autocast_kwargs):
            outputs = forward_model(model, student_batch)
            if batch.get("original") is not None:
                teacher_batch = dict(batch)
                teacher_batch["use_original"] = True
                teacher_out = forward_model(model, teacher_batch)
                teacher_logits = teacher_out["mask_logits"]
        # Loss outside autocast (fp32) — BCE/focal + long-seq attn are NaN-prone in fp16.
        kw = _loss_kwargs(config)
        loss, metrics = daae_loss(
            outputs, batch, teacher_logits=teacher_logits, **kw
        )
        if teacher_logits is not None:
            t_loss, t_metrics = daae_loss(teacher_out, batch, **{**kw, "distill_weight": 0.0})
            loss = loss + 0.5 * t_loss
            metrics["teacher_mask"] = t_metrics["mask_loss"]

        if not torch.isfinite(loss):
            skipped += 1
            if skipped <= 3:
                print(f"  warning: non-finite loss, skipping batch ({skipped})", flush=True)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.update()
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            if not torch.isfinite(grad_norm):
                skipped += 1
                if skipped <= 3:
                    print(f"  warning: non-finite grads, skipping step ({skipped})", flush=True)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            if not torch.isfinite(grad_norm):
                skipped += 1
                if skipped <= 3:
                    print(f"  warning: non-finite grads, skipping step ({skipped})", flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()

        for k in totals:
            totals[k] += metrics[k]
        n += 1
    if skipped:
        print(f"  skipped {skipped} unstable batches this epoch", flush=True)
    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def eval_epoch(
    model,
    loader,
    device,
    config,
    use_amp: bool,
    support_bank: Optional[ClassSupportBank] = None,
    amp_dtype=None,
):
    model.eval()
    totals = {
        "loss": 0.0,
        "mask_loss": 0.0,
        "recon_loss": 0.0,
        "mask_acc": 0.0,
        "keep_recall": 0.0,
        "keep_prec": 0.0,
        "keep_f1": 0.0,
    }
    n = 0
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    autocast_kwargs = {"device_type": amp_device, "enabled": use_amp}
    if amp_dtype is not None:
        autocast_kwargs["dtype"] = amp_dtype

    for batch in loader:
        batch = move_batch(batch, device)
        # Test loader is not few-shot wrapped; attach a class-agnostic dummy prior.
        if (
            support_bank is not None
            and _is_fs_model(model)
            and batch.get("support_defended") is None
        ):
            batch = support_bank.attach_global_to_batch(batch, device)

        with torch.amp.autocast(**autocast_kwargs):
            outputs = forward_model(model, batch)
        _, metrics = daae_loss(outputs, batch, **_loss_kwargs(config))
        if not math.isfinite(metrics["loss"]):
            continue
        pred_mask = (torch.sigmoid(outputs["mask_logits"].float()) >= 0.5).to(
            batch["dummy_mask"].dtype
        )
        valid = make_valid_mask(batch["defended_len"], pred_mask.shape[1])
        mask_acc = ((pred_mask == batch["dummy_mask"]).float() * valid).sum() / valid.sum().clamp(
            min=1
        )
        gt = batch["dummy_mask"].float() * valid
        pr = pred_mask.float() * valid
        tp = (pr * gt).sum()
        keep_n = gt.sum().clamp(min=1.0)
        pred_n = pr.sum().clamp(min=1.0)
        rec = tp / keep_n
        prec = tp / pred_n
        f1 = (2.0 * prec * rec) / (prec + rec).clamp(min=1e-6)

        metrics["mask_acc"] = float(mask_acc.item())
        metrics["keep_recall"] = float(rec.item())
        metrics["keep_prec"] = float(prec.item())
        metrics["keep_f1"] = float(f1.item())
        for k in totals:
            totals[k] += metrics[k]
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Train FS-DAAE for dummy-packet removal"
    )
    add_path_arguments(parser)
    add_dataset_scope_arguments(parser)
    parser.add_argument("--train-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--support-shots",
        type=int,
        default=5,
        help="Number of paired support traces per class (few-shot K)",
    )
    parser.add_argument(
        "--trace-format",
        choices=["direction", "paired_dir", "direction_timing", "direction_timing_gap"],
        default="paired_dir",
        help="paired_dir: ±1 + burst + CUMUL (no timing). direction: ±1 only.",
    )
    parser.add_argument("--max-len", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=default_batch_size())
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="AdamW learning rate (default 5e-4; 1e-3 is often unstable with AMP on long traces)",
    )
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument("--no-preload-val", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-gpu-data", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-focal", action="store_true")
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--length-ratio-weight", type=float, default=1.0)
    parser.add_argument("--front-mask-weight", type=float, default=2.5)
    parser.add_argument("--burst-weight", type=float, default=0.4)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Warm-start from an existing FS-DAAE checkpoint (fine-tune)",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip the held-out test sweep after training",
    )
    parser.add_argument("--preload-workers", type=int, default=default_preload_workers())
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    original_root, defended_root, output_dir = resolve_paths(args)
    print(
        f"Starting FS-DAAE train | "
        f"classes={args.num_classes} instances={args.num_instances} | "
        f"alignment=marker|lcs | max_len={args.max_len}",
        flush=True,
    )
    device = setup_gpu_training() if args.device is None else torch.device(args.device)
    if device.type == "cuda" and args.device is None:
        setup_gpu_training()

    use_amp, amp_dtype, use_scaler = amp_setup(device, enabled=not args.no_amp)
    gpu_tensors = device.type == "cuda" and not args.no_gpu_data
    tensor_device = device if gpu_tensors else None

    config = DAAEConfig(
        num_classes=args.num_classes,
        num_instances=args.num_instances,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        support_shots=args.support_shots,
        focal_gamma=args.focal_gamma,
        use_focal=not args.no_focal,
        dice_weight=args.dice_weight,
        length_ratio_weight=args.length_ratio_weight,
        front_mask_weight=args.front_mask_weight,
        burst_weight=args.burst_weight,
        rank_weight=args.rank_weight,
        trace_format=args.trace_format,
        max_len=args.max_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        seed=args.seed,
        preload=not args.no_preload,
        preload_val=not args.no_preload_val,
        preload_workers=args.preload_workers,
        use_cache=not args.no_cache,
        gpu_tensors=gpu_tensors,
        use_amp=use_amp,
        compile_model=not args.no_compile,
    )

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".daae_cache"

    train_loader, val_loader, test_loader, split, train_stacked = build_dataloaders(
        original_root, defended_root, config, cache_dir=cache_dir, tensor_device=tensor_device
    )

    support_bank = None
    if train_stacked is not None:
        support_bank = ClassSupportBank.from_stacked(train_stacked, config.support_shots)
        print(f"Few-shot support bank: {config.support_shots} shots/class from {len(split.train)} train pairs")

    amp_name = "off"
    if use_amp:
        amp_name = "bf16" if amp_dtype == torch.bfloat16 else "fp16"
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        print(
            f"GPU: {name} ({mem_gb:.0f} GB) | model={config.model_type} | "
            f"AMP={amp_name} | batch={config.batch_size} | lr={config.lr}",
            flush=True,
        )
    print(f"Original: {original_root}")
    print(f"Defended: {defended_root}")

    model = create_model(config).to(device)
    if args.init_checkpoint is not None:
        ckpt_path = args.init_checkpoint.resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"--init-checkpoint not found: {ckpt_path}")
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = blob.get("model_state", blob)
        missing, unexpected = unwrap_model(model).load_state_dict(state, strict=False)
        print(
            f"Warm-start {ckpt_path} | missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    model = maybe_compile(model, enabled=config.compile_model and device.type == "cuda")

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if device.type == "cuda":
        try:
            optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, fused=True)
        except TypeError:
            pass
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if use_scaler else None

    best_val = float("inf")
    best_mask_acc = 0.0
    history = []

    for epoch in range(1, config.epochs + 1):
        print(f"Epoch {epoch:03d}/{config.epochs} — training…", flush=True)
        train_m = train_epoch(
            model, train_loader, optimizer, device, config, scaler, use_amp, amp_dtype
        )
        print(f"Epoch {epoch:03d}/{config.epochs} — validating…", flush=True)
        val_m = eval_epoch(
            model, val_loader, device, config, use_amp, support_bank=support_bank, amp_dtype=amp_dtype
        )
        scheduler.step()
        history.append({"epoch": epoch, "train": train_m, "val": val_m})
        print(
            f"Epoch {epoch:03d} | train loss={train_m['loss']:.4f} "
            f"| val loss={val_m['loss']:.4f} mask_acc={val_m['mask_acc']:.4f} "
            f"keep_f1={val_m.get('keep_f1', 0):.4f} rec={val_m.get('keep_recall', 0):.3f} "
            f"prec={val_m.get('keep_prec', 0):.3f}",
            flush=True,
        )
        if math.isfinite(val_m["loss"]) and val_m["loss"] < best_val:
            best_val = val_m["loss"]
            ckpt = {
                "model_state": raw_state_dict(model),
                "config": config.__dict__,
                "paths": {
                    "original_root": str(original_root),
                    "defended_root": str(defended_root),
                    "output_dir": str(output_dir),
                },
                "epoch": epoch,
                "val_metrics": val_m,
            }
            torch.save(ckpt, output_dir / config.checkpoint_name)
        mask_score = val_m.get("keep_f1", val_m["mask_acc"])
        if math.isfinite(mask_score) and mask_score > best_mask_acc:
            best_mask_acc = mask_score
            ckpt_mask = {
                "model_state": raw_state_dict(model),
                "config": config.__dict__,
                "paths": {
                    "original_root": str(original_root),
                    "defended_root": str(defended_root),
                    "output_dir": str(output_dir),
                },
                "epoch": epoch,
                "val_metrics": val_m,
            }
            torch.save(ckpt_mask, output_dir / "daae_best_mask.pt")

    if args.skip_test:
        test_m = {"skipped": True}
        print("Skipping held-out test sweep (--skip-test)", flush=True)
    else:
        test_m = eval_epoch(
            model, test_loader, device, config, use_amp, support_bank=support_bank, amp_dtype=amp_dtype
        )
        print(
            f"Test | loss={test_m['loss']:.4f} mask_acc={test_m['mask_acc']:.4f} "
            f"keep_f1={test_m.get('keep_f1', 0):.4f} rec={test_m.get('keep_recall', 0):.3f} "
            f"prec={test_m.get('keep_prec', 0):.3f}"
        )

    with (output_dir / "train_history.json").open("w") as f:
        json.dump(history, f, indent=2)
    with (output_dir / "test_metrics.json").open("w") as f:
        json.dump(test_m, f, indent=2)


if __name__ == "__main__":
    main()
