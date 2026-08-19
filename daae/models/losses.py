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
) -> torch.Tensor:
    # Keep BCE/focal in fp32 — fp16 logits easily overflow under AMP.
    logits = logits.float().clamp(-40.0, 40.0)
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    pt = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return alpha_t * ((1.0 - pt).clamp(min=1e-6) ** gamma) * bce


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
    crf_weight: float = 0.15,
    dice_weight: float = 0.5,
    front_mask_weight: float = 1.0,
    length_ratio_weight: float = 0.0,
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
        bce = focal_bce_with_logits(mask_logits, dummy_mask, gamma=focal_gamma)
    else:
        bce = F.binary_cross_entropy_with_logits(
            mask_logits.clamp(-40.0, 40.0), dummy_mask, reduction="none"
        )

    mask_loss = (bce * valid).sum() / valid.sum().clamp(min=1.0)
    if front_mask_weight > 1.0:
        front_end = max(1, seq_len // 3)
        region = torch.ones_like(valid)
        region[:, :front_end] = front_mask_weight
        mask_loss = (bce * valid * region).sum() / (valid * region).sum().clamp(min=1.0)

    dice_val = 0.0
    if dice_weight > 0:
        dice = dice_loss_with_logits(mask_logits, dummy_mask, valid)
        mask_loss = mask_loss + dice_weight * dice
        dice_val = float(dice.item()) if torch.isfinite(dice) else float("nan")

    crf_loss = torch.zeros((), device=mask_logits.device, dtype=torch.float32)
    crf_mod = outputs.get("crf_module")
    if crf_weight > 0 and crf_mod is not None and "emissions" in outputs:
        emissions = outputs["emissions"].float()
        ds_len = emissions.shape[1]
        if ds_len != seq_len:
            tags_ds = F.adaptive_max_pool1d(dummy_mask.unsqueeze(1), ds_len).squeeze(1).round().long()
            valid_ds = F.adaptive_max_pool1d(valid.unsqueeze(1), ds_len).squeeze(1)
        else:
            tags_ds = dummy_mask.long()
            valid_ds = valid
        crf_loss = crf_mod.neg_log_likelihood(emissions, tags_ds, valid_ds)

    recon_loss_map = F.mse_loss(recon, recon_target, reduction="none")
    recon_weight_map = dummy_mask * valid
    recon_loss = (recon_loss_map * recon_weight_map).sum() / recon_weight_map.sum().clamp(min=1.0)

    length_loss = torch.zeros((), device=mask_logits.device, dtype=torch.float32)
    length_val = 0.0
    if length_ratio_weight > 0:
        length_loss = length_ratio_loss(mask_logits, dummy_mask, valid)
        length_val = float(length_loss.item()) if torch.isfinite(length_loss) else float("nan")

    total = (
        mask_weight * mask_loss
        + recon_weight * recon_loss
        + crf_weight * crf_loss
        + length_ratio_weight * length_loss
    )
    loss_f = float(total.item()) if torch.isfinite(total) else float("nan")
    return total, {
        "loss": loss_f,
        "mask_loss": float(mask_loss.item()) if torch.isfinite(mask_loss) else float("nan"),
        "recon_loss": float(recon_loss.item()) if torch.isfinite(recon_loss) else float("nan"),
        "crf_loss": float(crf_loss.item()) if torch.isfinite(crf_loss) else float("nan"),
        "dice": dice_val,
        "length_ratio_loss": length_val,
    }
