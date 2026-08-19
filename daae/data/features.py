"""Tor trace feature extraction: direction and timing."""

from __future__ import annotations

import numpy as np

from daae.data.cell_io import CellTrace


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
        "direction_timing": 2,
        "direction_timing_gap": 3,
    }
    return mapping.get(trace_format, 2)


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
    trace_format: str = "direction_timing",
) -> tuple[np.ndarray, int]:
    """Build feature tensor (C, max_len) from raw arrays."""
    directions = directions.astype(np.float32)
    valid_len = min(len(directions), max_len)
    channels = max_len if trace_format == "direction" else max_len
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
    trace_format: str = "direction_timing",
) -> tuple[np.ndarray, int]:
    """Build Tor WF features from a trace."""
    return extract_features_from_arrays(
        trace.timestamps, trace.directions, max_len, trace_format
    )


def extract_directions(trace: CellTrace) -> np.ndarray:
    """Packet direction sequence in {-1, 1}."""
    return trace.directions.astype(np.float32)
