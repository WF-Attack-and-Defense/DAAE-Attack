"""Defense-agnostic alignment between original and defended traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

AlignmentMethod = Literal["hamming", "lcs", "greedy"]

# Bump when alignment logic changes (invalidates disk cache).
# Bump when alignment inputs/semantics change (e.g. truncate-to-max_len before DP).
ALIGNMENT_VERSION = "align_trunc_maxlen_v2"


@dataclass
class AlignmentResult:
    """Result of aligning a defended trace onto an original trace."""

    keep_mask: np.ndarray  # 1 = original packet, 0 = dummy (shape: L_defended)
    orig_indices: np.ndarray  # defended index -> original index (-1 if dummy)
    match_ratio: float
    hamming_distance: float  # dummy + mismatch count on defended trace
    method: str = "hamming"


def _greedy_direction_align(orig_dirs: np.ndarray, def_dirs: np.ndarray) -> AlignmentResult:
    n, m = len(orig_dirs), len(def_dirs)
    keep_mask = np.zeros(m, dtype=np.float32)
    orig_indices = np.full(m, -1, dtype=np.int32)

    if n == 0:
        return AlignmentResult(
            keep_mask=keep_mask,
            orig_indices=orig_indices,
            match_ratio=0.0,
            hamming_distance=float(m),
            method="greedy",
        )

    i, j, matched = 0, 0, 0
    while i < n and j < m:
        if orig_dirs[i] == def_dirs[j]:
            keep_mask[j] = 1.0
            orig_indices[j] = i
            matched += 1
            i += 1
            j += 1
        else:
            j += 1

    hamming = float(m - matched)
    return AlignmentResult(
        keep_mask=keep_mask,
        orig_indices=orig_indices,
        match_ratio=matched / n,
        hamming_distance=hamming,
        method="greedy",
    )


def _hamming_insertion_align_python(orig_dirs: np.ndarray, def_dirs: np.ndarray) -> AlignmentResult:
    """
  Minimum-Hamming insertion alignment for dummy-packet defenses.

  Defended traces are modeled as the original direction subsequence with
  injected dummy packets. Only defended positions may be skipped (dummy);
  original packets are never dropped from the alignment path.
    """
    n, m = len(orig_dirs), len(def_dirs)
    keep_mask = np.zeros(m, dtype=np.float32)
    orig_indices = np.full(m, -1, dtype=np.int32)

    if n == 0:
        return AlignmentResult(
            keep_mask=keep_mask,
            orig_indices=orig_indices,
            match_ratio=0.0,
            hamming_distance=float(m),
            method="hamming_py",
        )

    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        for j in range(1, m + 1):
            best = dp[i, j - 1]
            if i > 0 and orig_dirs[i - 1] == def_dirs[j - 1]:
                best = max(best, dp[i - 1, j - 1] + 1)
            dp[i, j] = best

    i, j = n, m
    matched = 0
    while j > 0:
        if (
            i > 0
            and orig_dirs[i - 1] == def_dirs[j - 1]
            and dp[i, j] == dp[i - 1, j - 1] + 1
        ):
            keep_mask[j - 1] = 1.0
            orig_indices[j - 1] = i - 1
            matched += 1
            i -= 1
            j -= 1
        else:
            j -= 1

    hamming = float(m - matched)
    return AlignmentResult(
        keep_mask=keep_mask,
        orig_indices=orig_indices,
        match_ratio=float(matched) / float(n),
        hamming_distance=hamming,
        method="hamming_py",
    )


if _HAS_NUMBA:

    @njit(cache=True)
    def _hamming_insertion_align_numba(orig_dirs, def_dirs):
        n = len(orig_dirs)
        m = len(def_dirs)
        dp = np.zeros((n + 1, m + 1), dtype=np.int32)
        for i in range(n + 1):
            for j in range(1, m + 1):
                best = dp[i, j - 1]
                if i > 0 and orig_dirs[i - 1] == def_dirs[j - 1]:
                    v = dp[i - 1, j - 1] + 1
                    if v > best:
                        best = v
                dp[i, j] = best

        keep = np.zeros(m, dtype=np.float32)
        orig_idx = np.full(m, -1, dtype=np.int32)
        i, j = n, m
        matched = 0
        while j > 0:
            if (
                i > 0
                and orig_dirs[i - 1] == def_dirs[j - 1]
                and dp[i, j] == dp[i - 1, j - 1] + 1
            ):
                keep[j - 1] = 1.0
                orig_idx[j - 1] = i - 1
                matched += 1
                i -= 1
                j -= 1
            else:
                j -= 1

        ratio = matched / n if n > 0 else 0.0
        hamming = float(m - matched)
        return keep, orig_idx, ratio, hamming


def _hamming_insertion_align(orig_dirs: np.ndarray, def_dirs: np.ndarray) -> AlignmentResult:
    orig_dirs = orig_dirs.astype(np.int8)
    def_dirs = def_dirs.astype(np.int8)

    if _HAS_NUMBA:
        keep, orig_idx, ratio, hamming = _hamming_insertion_align_numba(orig_dirs, def_dirs)
        return AlignmentResult(
            keep_mask=keep,
            orig_indices=orig_idx,
            match_ratio=float(ratio),
            hamming_distance=float(hamming),
            method="hamming_numba",
        )

    if len(orig_dirs) * len(def_dirs) > 2_000_000:
        return _greedy_direction_align(orig_dirs, def_dirs)

    return _hamming_insertion_align_python(orig_dirs, def_dirs)


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

    if len(orig_dirs) * len(def_dirs) > 2_000_000:
        return _greedy_direction_align(orig_dirs, def_dirs)

    return _lcs_direction_align_python(orig_dirs, def_dirs)


def align_traces(
    orig_dirs: np.ndarray,
    def_dirs: np.ndarray,
    method: AlignmentMethod = "hamming",
) -> AlignmentResult:
    """
    Locate original packet positions inside defended traces.

    Default ``hamming``: insertion-only DP that minimizes Hamming distance
    (count of dummy packets) while embedding the original direction sequence
    in the defended trace. Suitable when defenses only inject dummies.

    ``lcs``: longest common subsequence alignment (may skip original packets).

    ``greedy``: fast left-to-right direction subsequence match.
    """
    orig_dirs = orig_dirs.astype(np.int8)
    def_dirs = def_dirs.astype(np.int8)

    if method == "greedy":
        return _greedy_direction_align(orig_dirs, def_dirs)
    if method == "lcs":
        return _lcs_direction_align(orig_dirs, def_dirs)
    return _hamming_insertion_align(orig_dirs, def_dirs)


def recover_from_mask(
    timestamps: np.ndarray,
    directions: np.ndarray,
    keep_mask: np.ndarray,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove dummy packets by applying a binary keep mask."""
    mask = keep_mask >= threshold
    return timestamps[mask], directions[mask]
