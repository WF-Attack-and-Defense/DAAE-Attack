"""Tor trace feature extraction (direction by default; timing optional/legacy)."""

from __future__ import annotations

import numpy as np

from daae.data.cell_io import CellTrace, dummy_prior_from_raw


def inter_packet_gaps(timestamps: np.ndarray) -> np.ndarray:
    """Log-scaled inter-packet gaps; first packet gap is 0."""
    if timestamps.size == 0:
        return timestamps.astype(np.float32)
    gaps = np.zeros_like(timestamps, dtype=np.float32)
    if timestamps.size > 1:
        raw = np.diff(timestamps.astype(np.float64))
        raw = np.clip(raw, 0.0, None)
        gaps[1:] = np.log1p(raw).astype(np.float32)
    return gaps


def in_channels_for_format(trace_format: str) -> int:
    mapping = {
        "direction": 1,
        "paired_dir": 5,
        "direction_timing": 2,
        "direction_timing_gap": 3,
    }
    return mapping.get(trace_format, 1)


def _sign_dirs(directions: np.ndarray) -> np.ndarray:
    d = np.asarray(directions, dtype=np.float32).reshape(-1)
    return np.where(d > 0.0, 1.0, -1.0).astype(np.float32)


def paired_direction_features(directions: np.ndarray, max_len: int) -> tuple[np.ndarray, int]:
    """Defense-aware features from column 2 (no timing).

    Channels:
      0: signed direction (±1)
      1: dummy prior (1 real / 0 marked dummy / 0.5 unknown)
      2: burst-run fraction
      3: outgoing CUMUL
      4: incoming CUMUL
    """
    raw = np.asarray(directions, dtype=np.float32).reshape(-1)
    dirs = _sign_dirs(raw)
    n = min(len(dirs), max_len)
    feat = np.zeros((5, max_len), dtype=np.float32)
    if n == 0:
        return feat, 0
    d = dirs[:n]
    feat[0, :n] = d
    feat[1, :n] = dummy_prior_from_raw(raw[:n])

    burst = np.ones(n, dtype=np.float32)
    run = 1.0
    for i in range(1, n):
        if d[i] == d[i - 1]:
            run += 1.0
        else:
            run = 1.0
        burst[i] = run
    i = n - 1
    while i >= 0:
        j = i
        while j > 0 and d[j - 1] == d[i]:
            j -= 1
        burst[j : i + 1] = burst[i]
        i = j - 1
    feat[2, :n] = burst / max(float(burst.max()), 1.0)

    out_c = np.cumsum(d > 0)
    in_c = np.cumsum(d < 0)
    feat[3, :n] = out_c / max(float(n), 1.0)
    feat[4, :n] = in_c / max(float(n), 1.0)
    return feat, n


def normalize_timing(timestamps: np.ndarray) -> np.ndarray:
    """Per-trace min-max normalization of packet timing to [0, 1]."""
    if timestamps.size == 0:
        return timestamps.astype(np.float32)
    t_min = float(timestamps.min())
    t_max = float(timestamps.max())
    span = t_max - t_min
    if span <= 0:
        return np.zeros_like(timestamps, dtype=np.float32)
    return ((timestamps - t_min) / span).astype(np.float32)


def extract_features_from_arrays(
    timestamps: np.ndarray,
    directions: np.ndarray,
    max_len: int,
    trace_format: str = "direction",
) -> tuple[np.ndarray, int]:
    """Build feature tensor (C, max_len) from raw arrays.

    Default ``direction`` uses only packet directions (±1) and ignores timing.
    """
    if trace_format == "paired_dir":
        return paired_direction_features(directions, max_len)

    directions = _sign_dirs(directions)
    valid_len = min(len(directions), max_len)
    n_ch = in_channels_for_format(trace_format)
    features = np.zeros((n_ch, max_len), dtype=np.float32)

    n = valid_len
    features[0, :n] = directions[:n]
    if n_ch >= 2:
        timing = normalize_timing(timestamps.astype(np.float32))
        features[1, :n] = timing[:n]
    if n_ch >= 3:
        gaps = inter_packet_gaps(timestamps.astype(np.float32))
        span = float(gaps[:n].max()) if n else 1.0
        if span > 0:
            features[2, :n] = gaps[:n] / span

    return features, valid_len


def extract_trace_features(
    trace: CellTrace,
    max_len: int,
    trace_format: str = "direction",
) -> tuple[np.ndarray, int]:
    """Build Tor WF features from a trace (direction-only by default)."""
    raw = trace.raw_directions if trace.raw_directions is not None else trace.directions
    return extract_features_from_arrays(
        trace.timestamps, raw, max_len, trace_format
    )


def extract_directions(trace: CellTrace) -> np.ndarray:
    return trace.directions.astype(np.float32)
