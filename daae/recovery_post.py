"""Recovery post-processing to align output length and quality with originals."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from daae.data.alignment import align_traces


def enforce_original_packets(
    keep: np.ndarray,
    orig_dirs: np.ndarray,
    def_dirs: np.ndarray,
) -> np.ndarray:
    """
    Force-keep every defended packet that LCS aligns to the original.

    Model / insertion decode can over-strip; OR-ing the alignment keep mask
    guarantees the recoverable original direction subsequence is present in
    the recovered trace (original ⊆ recovered up to the LCS ceiling vs defended).
    """
    if keep.size == 0 or len(orig_dirs) == 0 or len(def_dirs) == 0:
        return keep
    n = min(len(keep), len(def_dirs))
    al = align_traces(
        np.asarray(orig_dirs, dtype=np.float32),
        np.asarray(def_dirs[:n], dtype=np.float32),
    )
    out = np.asarray(keep[:n], dtype=np.float32).copy()
    m = min(n, len(al.keep_mask))
    out[:m] = np.maximum(out[:m], al.keep_mask[:m])
    if len(keep) > n:
        return np.concatenate([out, np.asarray(keep[n:], dtype=np.float32)])
    return out


def support_keep_ratio(support_mask: torch.Tensor, support_lens: Optional[torch.Tensor] = None) -> float:
    """Median fraction of defended packets kept (non-dummy) in support traces."""
    if support_mask.numel() == 0:
        return 0.1
    k, seq_len = support_mask.shape[-2], support_mask.shape[-1]
    ratios = []
    for ki in range(k):
        if support_lens is not None and ki < support_lens.shape[0]:
            valid = int(support_lens[ki].item())
        else:
            valid = seq_len
        valid = max(valid, 1)
        ratios.append(float(support_mask[ki, :valid].mean().item()))
    return float(np.median(ratios))


def calibrate_keep_mask(
    mask_prob: torch.Tensor,
    valid_len: int,
    target_keep_ratio: float,
    min_keep: int = 1,
) -> torch.Tensor:
    """
    Select top-scoring packets so keep count matches the support prior.

    Useful when mask logits cluster near 0.5 (weak separation) but ranking
    still reflects dummy vs. original confidence.
    """
    valid_len = max(int(valid_len), 1)
    target_keep_ratio = float(np.clip(target_keep_ratio, 0.01, 0.99))
    k = int(round(target_keep_ratio * valid_len))
    k = max(min_keep, min(k, valid_len))

    probs = mask_prob[:valid_len]
    if k >= valid_len:
        keep = torch.ones(valid_len, device=mask_prob.device, dtype=mask_prob.dtype)
    else:
        topk = torch.topk(probs, k).indices
        keep = torch.zeros(valid_len, device=mask_prob.device, dtype=mask_prob.dtype)
        keep[topk] = 1.0
    return keep


def apply_support_calibration(
    mask_prob: torch.Tensor,
    support_mask: torch.Tensor,
    valid_len: int,
    blend: float = 0.35,
) -> torch.Tensor:
    """
    Blend model probabilities with per-position support prior.

    Prior = mean keep mask across K support traces (same class).
    """
    valid_len = int(valid_len)
    if support_mask.ndim == 2:
        prior = support_mask[:, :valid_len].float().mean(dim=0)
    else:
        prior = support_mask[:valid_len].float()
    if prior.shape[0] < valid_len:
        prior = torch.nn.functional.pad(prior, (0, valid_len - prior.shape[0]))
    blended = (1.0 - blend) * mask_prob[:valid_len] + blend * prior[:valid_len]
    return blended


def enforce_monotonic_keep_runs(keep: np.ndarray, min_dummy_run: int = 3) -> np.ndarray:
    """Remove isolated keep packets shorter than min_dummy_run (Palette dummies arrive in bursts)."""
    if min_dummy_run <= 1 or keep.size == 0:
        return keep
    out = keep.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i] > 0.5:
            j = i
            while j < n and out[j] > 0.5:
                j += 1
            if (j - i) < min_dummy_run:
                out[i:j] = 0.0
            i = j
        else:
            i += 1
    return out
