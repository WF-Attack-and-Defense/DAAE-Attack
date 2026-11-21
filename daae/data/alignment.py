"""Defense-agnostic alignment between original and defended traces (LCS only)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

# Bump when alignment logic changes (invalidates disk cache).
ALIGNMENT_VERSION = "align_marker_or_lcs_v1"


@dataclass
class AlignmentResult:
    """Result of aligning a defended trace onto an original trace."""

    keep_mask: np.ndarray  # 1 = original packet, 0 = dummy (shape: L_defended)
    orig_indices: np.ndarray  # defended index -> original index (-1 if dummy)
    match_ratio: float
    hamming_distance: float  # defended positions not matched to original (dummy + skip)
    method: str = "lcs"


def _lcs_direction_align_python(orig_dirs: np.ndarray, def_dirs: np.ndarray) -> AlignmentResult:
    n, m = len(orig_dirs), len(def_dirs)
    keep_mask = np.zeros(m, dtype=np.float32)
    orig_indices = np.full(m, -1, dtype=np.int32)

    if n == 0:
        return AlignmentResult(
            keep_mask=keep_mask,
            orig_indices=orig_indices,
            match_ratio=0.0,
            hamming_distance=float(m),
            method="lcs_py",
        )

    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if orig_dirs[i - 1] == def_dirs[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])

    i, j = n, m
    matched = 0
    while i > 0 and j > 0:
        if orig_dirs[i - 1] == def_dirs[j - 1]:
            keep_mask[j - 1] = 1.0
            orig_indices[j - 1] = i - 1
            matched += 1
            i -= 1
            j -= 1
        elif dp[i - 1, j] >= dp[i, j - 1]:
            i -= 1
        else:
            j -= 1

    return AlignmentResult(
        keep_mask=keep_mask,
        orig_indices=orig_indices,
        match_ratio=float(dp[n, m]) / float(n),
        hamming_distance=float(m - matched),
        method="lcs_py",
    )


if _HAS_NUMBA:

    @njit(cache=True)
    def _lcs_direction_align_numba(orig_dirs, def_dirs):
        n = len(orig_dirs)
        m = len(def_dirs)
        dp = np.zeros((n + 1, m + 1), dtype=np.int32)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if orig_dirs[i - 1] == def_dirs[j - 1]:
                    dp[i, j] = dp[i - 1, j - 1] + 1
                else:
                    a = dp[i - 1, j]
                    b = dp[i, j - 1]
                    dp[i, j] = a if a >= b else b
        keep = np.zeros(m, dtype=np.float32)
        orig_idx = np.full(m, -1, dtype=np.int32)
        i, j = n, m
        matched = 0
        while i > 0 and j > 0:
            if orig_dirs[i - 1] == def_dirs[j - 1]:
                keep[j - 1] = 1.0
                orig_idx[j - 1] = i - 1
                matched += 1
                i -= 1
                j -= 1
            elif dp[i - 1, j] >= dp[i, j - 1]:
                i -= 1
            else:
                j -= 1
        ratio = matched / n if n > 0 else 0.0
        hamming = float(m - matched)
        return keep, orig_idx, ratio, hamming


def _lcs_direction_align(orig_dirs: np.ndarray, def_dirs: np.ndarray) -> AlignmentResult:
    orig_dirs = orig_dirs.astype(np.int8)
    def_dirs = def_dirs.astype(np.int8)

    if _HAS_NUMBA:
        keep, orig_idx, ratio, hamming = _lcs_direction_align_numba(orig_dirs, def_dirs)
        return AlignmentResult(
            keep_mask=keep,
            orig_indices=orig_idx,
            match_ratio=float(ratio),
            hamming_distance=float(hamming),
            method="lcs_numba",
        )

    return _lcs_direction_align_python(orig_dirs, def_dirs)


def label_keep_packets(
    orig_dirs: np.ndarray,
    def_raw: np.ndarray,
) -> AlignmentResult:
    """Label dummy positions from defense dump markers, else LCS."""
    from daae.data.cell_io import has_dummy_markers, marker_keep_mask

    raw = np.asarray(def_raw, dtype=np.float32).reshape(-1)
    if has_dummy_markers(raw):
        keep = marker_keep_mask(raw)
        orig_indices = np.full(len(raw), -1, dtype=np.int32)
        orig_indices[keep > 0.5] = np.arange(int(keep.sum()), dtype=np.int32)
        matched = int(keep.sum())
        n = len(np.asarray(orig_dirs).reshape(-1))
        return AlignmentResult(
            keep_mask=keep,
            orig_indices=orig_indices,
            match_ratio=float(matched) / float(max(n, 1)),
            hamming_distance=float(len(raw) - matched),
            method="marker",
        )
    orig_signed = np.where(np.asarray(orig_dirs) > 0, 1.0, -1.0).astype(np.float32)
    def_signed = np.where(raw > 0, 1.0, -1.0).astype(np.float32)
    return _lcs_direction_align(orig_signed, def_signed)


def align_traces(orig_dirs: np.ndarray, def_dirs: np.ndarray) -> AlignmentResult:
    """
    Locate original packet positions inside defended traces via longest common
    subsequence (LCS) on packet directions. Unmatched defended positions are
    labeled dummy; original packets may be skipped when directions diverge.
    """
    return _lcs_direction_align(orig_dirs, def_dirs)


def recover_from_mask(
    timestamps: np.ndarray,
    directions: np.ndarray,
    keep_mask: np.ndarray,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove dummy packets by applying a binary keep mask."""
    mask = keep_mask >= threshold
    return timestamps[mask], directions[mask]
