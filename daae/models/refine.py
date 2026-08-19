"""Subsequence-constrained mask refinement using support direction prototypes."""

from __future__ import annotations

from typing import List

import numpy as np
import torch

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _proto_from_support(
    support_defended: torch.Tensor,
    support_mask: torch.Tensor,
    bi: int,
    ki: int,
    seq_len: int,
) -> np.ndarray:
    valid = seq_len
    dirs = support_defended[bi, ki, 0, :valid].detach().cpu().numpy()
    keep = support_mask[bi, ki, :valid].detach().cpu().numpy() > 0.5
    return dirs[keep].astype(np.int8)


if _HAS_NUMBA:

    @njit(cache=True)
    def _subsequence_viterbi_numba(
        def_dirs: np.ndarray,
        logits: np.ndarray,
        proto_dirs: np.ndarray,
        valid_len: int,
        skip_bias: float,
    ) -> np.ndarray:
        n = valid_len
        t = len(proto_dirs)
        keep = np.zeros(n, dtype=np.float32)
        if t == 0 or n == 0:
            return keep

        neg = -1e18
        dp = np.full((t + 1, n + 1), neg, dtype=np.float64)
        bt = np.zeros((t + 1, n + 1), dtype=np.int8)
        dp[0, :] = 0.0

        for j in range(1, n + 1):
            idx = j - 1
            for i in range(t + 1):
                skip_score = dp[i, j - 1] + skip_bias
                match_score = neg
                if i > 0 and def_dirs[idx] == proto_dirs[i - 1]:
                    match_score = dp[i - 1, j - 1] + logits[idx]
                if match_score >= skip_score:
                    dp[i, j] = match_score
                    bt[i, j] = 1
                else:
                    dp[i, j] = skip_score

        i = t
        for j in range(n, 0, -1):
            if bt[i, j] == 1:
                keep[j - 1] = 1.0
                i -= 1
        return keep


def subsequence_refine_mask(
    def_dirs: np.ndarray,
    logits: np.ndarray,
    proto_dirs: np.ndarray,
    valid_len: int,
    skip_bias: float = -0.05,
) -> np.ndarray:
    valid_len = min(valid_len, len(def_dirs), len(logits))
    proto_dirs = proto_dirs.astype(np.int8)
    def_dirs = def_dirs[:valid_len].astype(np.int8)
    logits = logits[:valid_len].astype(np.float64)

    if proto_dirs.size == 0:
        return (logits >= 0.0).astype(np.float32)

    if _HAS_NUMBA:
        return _subsequence_viterbi_numba(def_dirs, logits, proto_dirs, valid_len, skip_bias)

    t = len(proto_dirs)
    n = valid_len
    neg = -1e18
    dp = np.full((t + 1, n + 1), neg)
    bt = np.zeros((t + 1, n + 1), dtype=np.int8)
    dp[0, :] = 0.0
    for j in range(1, n + 1):
        idx = j - 1
        for i in range(t + 1):
            skip_score = dp[i, j - 1] + skip_bias
            match_score = neg
            if i > 0 and def_dirs[idx] == proto_dirs[i - 1]:
                match_score = dp[i - 1, j - 1] + logits[idx]
            if match_score >= skip_score:
                dp[i, j] = match_score
                bt[i, j] = 1
            else:
                dp[i, j] = skip_score
    i = t
    keep = np.zeros(n, dtype=np.float32)
    for j in range(n, 0, -1):
        if bt[i, j] == 1:
            keep[j - 1] = 1.0
            i -= 1
    return keep


def _mask_score(logits: np.ndarray, mask: np.ndarray) -> float:
    m = mask > 0.5
    return float(logits[m].sum() - logits[~m].sum())


def refine_batch_masks(
    defended: torch.Tensor,
    mask_logits: torch.Tensor,
    support_defended: torch.Tensor,
    support_mask: torch.Tensor,
    defended_len: torch.Tensor,
) -> torch.Tensor:
    """
    Best-of-K subsequence decode: try each support's direction prototype and
    keep the mask with highest logit agreement (Palette cluster patterns vary
  per instance; median prototype was too aggressive).
    """
    b, k, _, seq_len = support_defended.shape
    logits_np = mask_logits.detach().cpu().numpy()
    dirs_np = defended[:, 0].detach().cpu().numpy()
    lens = defended_len.detach().cpu().numpy()
    raw = (mask_logits.detach().cpu().numpy() >= 0.0).astype(np.float32)
    refined = np.zeros((b, seq_len), dtype=np.float32)

    for bi in range(b):
        valid = int(lens[bi])
        best_mask = raw[bi, :valid]
        best_score = _mask_score(logits_np[bi, :valid], best_mask)

        for ki in range(k):
            proto = _proto_from_support(support_defended, support_mask, bi, ki, valid)
            if proto.size == 0:
                continue
            cand = subsequence_refine_mask(dirs_np[bi], logits_np[bi], proto, valid)
            score = _mask_score(logits_np[bi, :valid], cand)
            if score > best_score:
                best_score = score
                best_mask = cand

        refined[bi, :valid] = best_mask
    return torch.from_numpy(refined).to(mask_logits.device)
