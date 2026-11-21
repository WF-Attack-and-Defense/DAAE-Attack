"""Loss functions for DAAE / FS-DAAE training."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def make_valid_mask(defended_len: torch.Tensor, seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(seq_len, device=defended_len.device)
    return (positions.unsqueeze(0) < defended_len.unsqueeze(1)).to(dtype)


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    # Keep BCE/focal in fp32 — fp16 logits easily overflow under AMP.
    logits = logits.float().clamp(-40.0, 40.0)
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    pt = prob * targets + (1.0 - prob) * (1.0 - targets)
    if valid is not None:
        keep_frac = (targets * valid).sum() / valid.sum().clamp(min=1.0)
        # Inverse-frequency: rare keep packets (Tamaraw ~0.26) get higher weight.
        alpha = float((1.0 - keep_frac).clamp(0.35, 0.80).item())
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return alpha_t * ((1.0 - pt).clamp(min=1e-6) ** gamma) * bce


def burst_consistency_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    directions: torch.Tensor | None,
) -> torch.Tensor:
    """Same-direction neighbors that share a keep/dummy label should score alike."""
    if directions is None or logits.shape[1] < 2:
        return torch.zeros((), device=logits.device, dtype=torch.float32)
    logits = logits.float().clamp(-40.0, 40.0)
    prob = torch.sigmoid(logits)
    dirs = directions.float()
    same = (dirs[:, :-1] * dirs[:, 1:]) > 0
    both = valid[:, :-1] * valid[:, 1:]
    agree = 1.0 - (targets[:, :-1] - targets[:, 1:]).abs()
    w = same.float() * both * agree
    diff = (prob[:, :-1] - prob[:, 1:]).abs()
    return (diff * w).sum() / w.sum().clamp(min=1.0)


def keep_rank_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Push original-packet logits above dummy logits (better recover ranking)."""
    logits = logits.float().clamp(-40.0, 40.0)
    keep = targets * valid
    dummy = (1.0 - targets) * valid
    keep_n = keep.sum(dim=1).clamp(min=1.0)
    dummy_n = dummy.sum(dim=1).clamp(min=1.0)
    keep_mean = (logits * keep).sum(dim=1) / keep_n
    dummy_mean = (logits * dummy).sum(dim=1) / dummy_n
    has_both = (keep.sum(dim=1) > 0) & (dummy.sum(dim=1) > 0)
    if not bool(has_both.any()):
        return torch.zeros((), device=logits.device, dtype=torch.float32)
    return F.relu(margin - (keep_mean - dummy_mean))[has_both].mean()


def dice_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    logits = logits.float().clamp(-40.0, 40.0)
    targets = targets.float()
    valid = valid.float()
    prob = torch.sigmoid(logits)
    inter = (prob * targets * valid).sum()
    denom = (prob * valid).sum() + (targets * valid).sum()
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def length_ratio_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Penalize mismatch in keep/dummy fraction — critical for WF-length traces."""
    logits = logits.float().clamp(-40.0, 40.0)
    targets = targets.float()
    valid = valid.float()
    prob = torch.sigmoid(logits)
    denom = valid.sum(dim=1).clamp(min=1.0)
    pred_ratio = (prob * valid).sum(dim=1) / denom
    true_ratio = (targets * valid).sum(dim=1) / denom
    return F.l1_loss(pred_ratio, true_ratio)


def daae_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    trace_format: str = "direction",
    mask_weight: float = 1.0,
    recon_weight: float = 1.0,
    focal_gamma: float = 2.0,
    use_focal: bool = True,
    dice_weight: float = 0.5,
    front_mask_weight: float = 2.0,
    length_ratio_weight: float = 0.0,
    burst_weight: float = 0.4,
    rank_weight: float = 0.5,
    emit_weight: float = 1.0,
    orig_len_weight: float = 0.5,
    distill_weight: float = 0.0,
    teacher_logits: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    # Always accumulate loss in fp32 (AMP autocast often leaves logits in fp16).
    mask_logits = outputs["mask_logits"].float()
    recon = outputs["recon"].float()
    dummy_mask = batch["dummy_mask"].float()
    recon_target = batch["recon_target"].float()
    defended_len = batch["defended_len"]

    b, seq_len = mask_logits.shape
    valid = make_valid_mask(defended_len, seq_len, torch.float32)

    if use_focal and focal_gamma > 0:
        bce = focal_bce_with_logits(mask_logits, dummy_mask, gamma=focal_gamma, valid=valid)
    else:
        bce = F.binary_cross_entropy_with_logits(
            mask_logits.clamp(-40.0, 40.0), dummy_mask, reduction="none"
        )

    mask_loss = (bce * valid).sum() / valid.sum().clamp(min=1.0)
    if front_mask_weight > 1.0:
        # DF reads the leading ±1 packets; dummy errors at the front hurt most.
        front_end = min(seq_len, 2000)
        region = torch.ones_like(valid)
        region[:, :front_end] = front_mask_weight
        mask_loss = (bce * valid * region).sum() / (valid * region).sum().clamp(min=1.0)

    dice_val = 0.0
    if dice_weight > 0:
        dice = dice_loss_with_logits(mask_logits, dummy_mask, valid)
        mask_loss = mask_loss + dice_weight * dice
        dice_val = float(dice.item()) if torch.isfinite(dice) else float("nan")

    recon_loss_map = F.mse_loss(recon, recon_target, reduction="none")
    recon_weight_map = dummy_mask * valid
    recon_loss = (recon_loss_map * recon_weight_map).sum() / recon_weight_map.sum().clamp(min=1.0)

    length_loss = torch.zeros((), device=mask_logits.device, dtype=torch.float32)
    length_val = 0.0
    if length_ratio_weight > 0:
        length_loss = length_ratio_loss(mask_logits, dummy_mask, valid)
        length_val = float(length_loss.item()) if torch.isfinite(length_loss) else float("nan")

    burst_val = 0.0
    burst_loss = torch.zeros((), device=mask_logits.device, dtype=torch.float32)
    if burst_weight > 0:
        directions = batch.get("defended")
        if directions is not None:
            directions = directions[:, 0] if directions.dim() == 3 else directions
        burst_loss = burst_consistency_loss(mask_logits, dummy_mask, valid, directions)
        burst_val = float(burst_loss.item()) if torch.isfinite(burst_loss) else float("nan")

    rank_val = 0.0
    rank_loss = torch.zeros((), device=mask_logits.device, dtype=torch.float32)
    if rank_weight > 0:
        rank_loss = keep_rank_loss(mask_logits, dummy_mask, valid)
        rank_val = float(rank_loss.item()) if torch.isfinite(rank_loss) else float("nan")

    emit_val = 0.0
    emit_loss = torch.zeros((), device=mask_logits.device, dtype=torch.float32)
    emit_logits = outputs.get("emit_logits")
    if emit_weight > 0 and emit_logits is not None:
        # 0=dummy, 1=keep outgoing, 2=keep incoming — no CTC repeat-collapse
        # (collapse would erase DF burst lengths).
        recon_t = recon_target
        emit_tgt = torch.zeros_like(dummy_mask, dtype=torch.long)
        keep = dummy_mask > 0.5
        emit_tgt = torch.where(keep & (recon_t > 0), torch.ones_like(emit_tgt), emit_tgt)
        emit_tgt = torch.where(keep & (recon_t < 0), torch.full_like(emit_tgt, 2), emit_tgt)
        ce = F.cross_entropy(emit_logits.float(), emit_tgt, reduction="none")
        emit_loss = (ce * valid).sum() / valid.sum().clamp(min=1.0)
        emit_val = float(emit_loss.item()) if torch.isfinite(emit_loss) else float("nan")

    olen_val = 0.0
    olen_loss = torch.zeros((), device=mask_logits.device, dtype=torch.float32)
    if orig_len_weight > 0 and "orig_len_pred" in outputs and "original_len" in batch:
        pred = torch.sigmoid(outputs["orig_len_pred"].float())
        tgt = batch["original_len"].float().clamp(min=1.0) / float(seq_len)
        olen_loss = F.l1_loss(pred, tgt)
        olen_val = float(olen_loss.item()) if torch.isfinite(olen_loss) else float("nan")

    distill_val = 0.0
    distill_loss = torch.zeros((), device=mask_logits.device, dtype=torch.float32)
    if distill_weight > 0 and teacher_logits is not None:
        t = teacher_logits.float().detach()
        s = mask_logits.float()
        distill_loss = F.kl_div(
            F.log_softmax(torch.stack([-s, s], dim=-1), dim=-1),
            F.softmax(torch.stack([-t, t], dim=-1), dim=-1),
            reduction="none",
        ).sum(dim=-1)
        distill_loss = (distill_loss * valid).sum() / valid.sum().clamp(min=1.0)
        distill_val = float(distill_loss.item()) if torch.isfinite(distill_loss) else float("nan")

    total = (
        mask_weight * mask_loss
        + recon_weight * recon_loss
        + length_ratio_weight * length_loss
        + burst_weight * burst_loss
        + rank_weight * rank_loss
        + emit_weight * emit_loss
        + orig_len_weight * olen_loss
        + distill_weight * distill_loss
    )
    loss_f = float(total.item()) if torch.isfinite(total) else float("nan")
    return total, {
        "loss": loss_f,
        "mask_loss": float(mask_loss.item()) if torch.isfinite(mask_loss) else float("nan"),
        "recon_loss": float(recon_loss.item()) if torch.isfinite(recon_loss) else float("nan"),
        "dice": dice_val,
        "length_ratio_loss": length_val,
        "burst_loss": burst_val,
        "rank_loss": rank_val,
        "emit_loss": emit_val,
        "orig_len_loss": olen_val,
        "distill_loss": distill_val,
    }
