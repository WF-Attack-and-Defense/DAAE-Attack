"""Insertion-only mask decoding for dummy-packet defenses.

Palette-style defenses only insert dummy packets; the original direction
sequence is embedded as a subsequence of the defended trace. Recovery must
therefore select a *subsequence* of defended packets, not an arbitrary mask.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def support_keep_ratio(support_masks: np.ndarray, support_lens: Optional[np.ndarray] = None) -> float:
    """Median keep fraction across K support dummy masks."""
    ratios: List[float] = []
    k = support_masks.shape[0]
    for ki in range(k):
        if support_lens is not None:
            valid = int(support_lens[ki])
        else:
            valid = support_masks.shape[1]
        valid = max(valid, 1)
        ratios.append(float(support_masks[ki, :valid].mean()))
    return float(np.median(ratios)) if ratios else 0.1


def prototype_directions(def_dirs: np.ndarray, keep_mask: np.ndarray, valid_len: int) -> np.ndarray:
    """Direction subsequence kept by a support trace."""
    n = min(valid_len, len(def_dirs), len(keep_mask))
    return def_dirs[:n][keep_mask[:n] > 0.5].astype(np.int8)


def _viterbi_insertion_python(
    def_dirs: np.ndarray,
    logits: np.ndarray,
    proto_dirs: np.ndarray,
    valid_len: int,
    skip_bias: float,
) -> np.ndarray:
    """Rolling-column DP (O(T) memory) with int8 backtrace."""
    n = valid_len
    t = len(proto_dirs)
    keep = np.zeros(n, dtype=np.float32)
    if t == 0 or n == 0:
        return keep

    neg = -1e18
    dp_prev = np.full(t + 1, neg, dtype=np.float64)
    dp_curr = np.full(t + 1, neg, dtype=np.float64)
    dp_prev[0] = 0.0
    bt = np.zeros((t + 1, n), dtype=np.uint8)

    for j in range(n):
        d = def_dirs[j]
        logit = logits[j]
        dp_curr[0] = dp_prev[0] + skip_bias
        for i in range(1, t + 1):
            skip_score = dp_prev[i] + skip_bias
            match_score = neg
            if d == proto_dirs[i - 1]:
                match_score = dp_prev[i - 1] + logit
            if match_score >= skip_score:
                dp_curr[i] = match_score
                bt[i, j] = 1
            else:
                dp_curr[i] = skip_score
        dp_prev, dp_curr = dp_curr, dp_prev

    # Do not require consuming the full prototype. A support proto can be
    # much longer than this query (T >> n); backtracing from i=T then yields
    # an all-zero keep mask and empty .cell files.
    i = 0
    best = dp_prev[0]
    for ii in range(1, t + 1):
        if dp_prev[ii] > best:
            best = dp_prev[ii]
            i = ii
    for j in range(n - 1, -1, -1):
        if i > 0 and bt[i, j] == 1:
            keep[j] = 1.0
            i -= 1
    return keep


if _HAS_NUMBA:

    @njit(cache=True, nogil=True)
    def _viterbi_insertion(
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
        dp_prev = np.full(t + 1, neg, dtype=np.float64)
        dp_curr = np.full(t + 1, neg, dtype=np.float64)
        dp_prev[0] = 0.0
        bt = np.zeros((t + 1, n), dtype=np.uint8)

        for j in range(n):
            d = def_dirs[j]
            logit = logits[j]
            dp_curr[0] = dp_prev[0] + skip_bias
            for i in range(1, t + 1):
                skip_score = dp_prev[i] + skip_bias
                match_score = neg
                if d == proto_dirs[i - 1]:
                    match_score = dp_prev[i - 1] + logit
                if match_score >= skip_score:
                    dp_curr[i] = match_score
                    bt[i, j] = np.uint8(1)
                else:
                    dp_curr[i] = skip_score
            tmp = dp_prev
            dp_prev = dp_curr
            dp_curr = tmp

        # Best reachable prototype prefix — full T is often unreachable.
        i = 0
        best = dp_prev[0]
        for ii in range(1, t + 1):
            if dp_prev[ii] > best:
                best = dp_prev[ii]
                i = ii
        for j in range(n - 1, -1, -1):
            if i > 0 and bt[i, j] == 1:
                keep[j] = 1.0
                i -= 1
        return keep

    @njit(cache=True, nogil=True)
    def _calibrated_viterbi(
        def_dirs: np.ndarray,
        logits: np.ndarray,
        proto_dirs: np.ndarray,
        target_ratio: float,
        max_steps: int,
        low: float,
        high: float,
    ) -> np.ndarray:
        """Binary-search skip_bias inside nopython so calibration is one GIL release."""
        n = len(def_dirs)
        best_keep = np.zeros(n, dtype=np.float32)
        best_err = 1e9
        if n == 0 or len(proto_dirs) == 0:
            return best_keep

        target = target_ratio
        if target < 0.01:
            target = 0.01
        if target > 0.99:
            target = 0.99
        inv_n = 1.0 / float(n)

        for _ in range(max_steps):
            mid = 0.5 * (low + high)
            keep = _viterbi_insertion(def_dirs, logits, proto_dirs, n, mid)
            ratio = 0.0
            for j in range(n):
                ratio += keep[j]
            ratio *= inv_n
            err = ratio - target
            if err < 0.0:
                err = -err
            if err < best_err:
                best_err = err
                best_keep = keep
                if err < 0.015:
                    break
            if ratio > target:
                high = mid
            else:
                low = mid
        return best_keep

else:
    _viterbi_insertion = _viterbi_insertion_python  # type: ignore[misc,assignment]

    def _calibrated_viterbi(
        def_dirs: np.ndarray,
        logits: np.ndarray,
        proto_dirs: np.ndarray,
        target_ratio: float,
        max_steps: int,
        low: float,
        high: float,
    ) -> np.ndarray:
        n = len(def_dirs)
        best_keep = np.zeros(n, dtype=np.float32)
        best_err = 1e9
        if n == 0 or len(proto_dirs) == 0:
            return best_keep
        target = float(np.clip(target_ratio, 0.01, 0.99))
        for _ in range(max_steps):
            mid = 0.5 * (low + high)
            keep = _viterbi_insertion_python(def_dirs, logits, proto_dirs, n, mid)
            ratio = float(keep.mean())
            err = abs(ratio - target)
            if err < best_err:
                best_err = err
                best_keep = keep
                if err < 0.015:
                    break
            if ratio > target:
                high = mid
            else:
                low = mid
        return best_keep


def viterbi_insertion(
    def_dirs: np.ndarray,
    logits: np.ndarray,
    proto_dirs: np.ndarray,
    valid_len: int,
    skip_bias: float = 0.0,
) -> np.ndarray:
    valid_len = min(valid_len, len(def_dirs), len(logits))
    proto_dirs = np.ascontiguousarray(proto_dirs, dtype=np.int8)
    def_dirs = np.ascontiguousarray(def_dirs[:valid_len], dtype=np.int8)
    logits = np.ascontiguousarray(logits[:valid_len], dtype=np.float64)

    if proto_dirs.size == 0:
        return (logits >= 0.0).astype(np.float32)
    return _viterbi_insertion(def_dirs, logits, proto_dirs, valid_len, skip_bias)


def _fallback_keep(logits: np.ndarray) -> np.ndarray:
    """Keep positive-logit packets; if none, keep the whole defended trace."""
    keep = (logits >= 0.0).astype(np.float32)
    if float(keep.sum()) <= 0.0:
        return np.ones(len(logits), dtype=np.float32)
    return keep


def _logit_score(logits: np.ndarray, mask: np.ndarray) -> float:
    m = mask > 0.5
    return float(logits[m].sum() - logits[~m].sum())


def _initial_skip_bounds(logits: np.ndarray, target_ratio: float) -> tuple[float, float]:
    """Center the skip-bias search on a logit quantile so fewer steps suffice."""
    target = float(np.clip(target_ratio, 0.01, 0.99))
    init = float(np.quantile(logits, 1.0 - target))
    return init - 2.0, init + 2.0


def calibrate_skip_bias(
    def_dirs: np.ndarray,
    logits: np.ndarray,
    proto_dirs: np.ndarray,
    valid_len: int,
    target_ratio: float,
    low: float = -2.0,
    high: float = 2.0,
    max_steps: int = 18,
) -> tuple[float, np.ndarray]:
    """Binary-search skip_bias so keep fraction is near target_ratio."""
    valid_len = min(valid_len, len(def_dirs), len(logits))
    def_dirs = np.ascontiguousarray(def_dirs[:valid_len], dtype=np.int8)
    logits_f = np.ascontiguousarray(logits[:valid_len], dtype=np.float64)
    proto_dirs = np.ascontiguousarray(proto_dirs, dtype=np.int8)
    mask = _calibrated_viterbi(
        def_dirs, logits_f, proto_dirs, float(target_ratio), int(max_steps), float(low), float(high)
    )
    return 0.0, mask


def _pick_support_indices(
    support_keep_masks: Sequence[np.ndarray],
    valid_len: int,
    target_ratio: float,
    max_candidates: int = 3,
) -> List[int]:
    """Prefer the longest support keep sequences (avoids under-keeping originals)."""
    del target_ratio  # length-first selection; ratio still used for skip-bias calibration
    scored: List[tuple[float, int]] = []
    for ki, km in enumerate(support_keep_masks):
        sl = min(len(km), valid_len)
        kept = float(km[:sl].sum())
        scored.append((-kept, ki))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [ki for _, ki in scored[:max_candidates]]


def decode_insertion_mask(
    def_dirs: np.ndarray,
    logits: np.ndarray,
    support_def_dirs: Sequence[np.ndarray],
    support_keep_masks: Sequence[np.ndarray],
    support_lens: Optional[Sequence[int]] = None,
    target_ratio: Optional[float] = None,
    calibrate: bool = True,
    calibrate_steps: int = 3,
    max_candidates: int = 1,
    precomputed_protos: Optional[Sequence[np.ndarray]] = None,
) -> np.ndarray:
    """
    Best-of-K insertion decode using support direction prototypes.

    Prototypes are chosen by kept-length proximity to the median support keep
    ratio so decode is not forced to keep an overly long support subsequence.
    """
    valid_len = len(def_dirs)
    def_dirs_c = np.ascontiguousarray(def_dirs[:valid_len], dtype=np.int8)
    logits_c = np.ascontiguousarray(logits[:valid_len], dtype=np.float64)
    if valid_len == 0:
        return np.zeros(0, dtype=np.float32)
    if not np.isfinite(logits_c).any():
        return np.ones(valid_len, dtype=np.float32)
    logits_c = np.nan_to_num(logits_c, nan=-1e9, posinf=1e9, neginf=-1e9)

    if target_ratio is None:
        ratios = []
        for ki, km in enumerate(support_keep_masks):
            sl = support_lens[ki] if support_lens is not None else valid_len
            sl = min(sl, valid_len, len(km))
            ratios.append(float(km[:sl].mean()))
        target_ratio = float(np.max(ratios)) if ratios else 0.1

    if precomputed_protos is not None and len(precomputed_protos) > 0:
        proto_list = []
        for p in precomputed_protos[:max_candidates]:
            arr = np.ascontiguousarray(p, dtype=np.int8)
            if arr.size > valid_len:
                arr = arr[:valid_len]
            if arr.size > 0:
                proto_list.append(arr)
    else:
        candidate_ids = _pick_support_indices(
            support_keep_masks, valid_len, target_ratio, max_candidates=max_candidates
        )
        proto_list = []
        for ki in candidate_ids:
            km = support_keep_masks[ki]
            sl = support_lens[ki] if support_lens is not None else valid_len
            sl = min(sl, valid_len, len(support_def_dirs[ki]), len(km))
            proto = prototype_directions(support_def_dirs[ki], km, sl)
            if proto.size > valid_len:
                proto = proto[:valid_len]
            if proto.size > 0:
                proto_list.append(np.ascontiguousarray(proto, dtype=np.int8))

    if not proto_list:
        return _fallback_keep(logits_c)

    low, high = _initial_skip_bounds(logits_c, target_ratio)
    best_mask = None
    best_score = -1e18

    for proto in proto_list:
        if proto.size == 0:
            continue
        if calibrate:
            cand = _calibrated_viterbi(
                def_dirs_c, logits_c, proto, float(target_ratio), int(calibrate_steps), low, high
            )
        else:
            cand = _viterbi_insertion(def_dirs_c, logits_c, proto, valid_len, 0.0)
        score = _logit_score(logits_c, cand)
        if best_mask is None or score > best_score:
            best_score = score
            best_mask = cand

    if best_mask is None or float(best_mask.sum()) <= 0.0:
        return _fallback_keep(logits_c)
    return best_mask


def warmup_decode_kernels() -> None:
    """Compile Numba kernels once so the first recover batch is not stalled."""
    dirs = np.array([1, -1, 1, -1], dtype=np.int8)
    logits = np.zeros(4, dtype=np.float64)
    proto = np.array([1, -1], dtype=np.int8)
    _viterbi_insertion(dirs, logits, proto, 4, 0.0)
    _calibrated_viterbi(dirs, logits, proto, 0.5, 2, -1.0, 1.0)
