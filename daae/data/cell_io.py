"""Utilities for reading and writing .cell trace files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# WFZoo Tamaraw / RegulaTor dump dummy cells as ±888; real cells stay ±1.
_DUMMY_ABS = 1.5


def has_dummy_markers(raw: np.ndarray) -> bool:
    """True when col-2 mixes real ±1 cells with |value| dummy markers."""
    a = np.abs(np.asarray(raw, dtype=np.float64).reshape(-1))
    return bool(a.size > 0 and np.any(a > _DUMMY_ABS) and np.any(a <= _DUMMY_ABS))


def dummy_prior_from_raw(raw: np.ndarray) -> np.ndarray:
    """1 = real cell, 0 = marked dummy, 0.5 = unknown (Palette-style ±1 only)."""
    a = np.abs(np.asarray(raw, dtype=np.float32).reshape(-1))
    if a.size == 0:
        return a
    if np.any(a > _DUMMY_ABS) and np.any(a <= _DUMMY_ABS):
        return (a <= _DUMMY_ABS).astype(np.float32)
    return np.full(a.shape, 0.5, dtype=np.float32)


def marker_keep_mask(raw: np.ndarray) -> np.ndarray:
    """Keep packets whose raw col-2 magnitude is a real cell (±1)."""
    a = np.abs(np.asarray(raw, dtype=np.float32).reshape(-1))
    return (a <= _DUMMY_ABS).astype(np.float32)


@dataclass
class CellTrace:
    """
    A single Tor website fingerprinting trace.

    Each .cell row has two columns:
      - timestamps: packet timing (seconds)
      - directions: 1 for outgoing, -1 for incoming
    Packet sizes are not used in Tor WF.
    """

    timestamps: np.ndarray  # shape (L,)
    directions: np.ndarray  # shape (L,), values in {-1, 1}
    raw_directions: np.ndarray | None = None
    label: int = -1
    filename: str = ""

    @property
    def length(self) -> int:
        return int(self.directions.shape[0])


def _ids_from_cell_stem(stem: str) -> Tuple[int, int]:
    """Parse ``{class}-{inst}`` when present; otherwise (-1, -1)."""
    if "-" not in stem:
        return -1, -1
    cls_s, inst_s = stem.split("-", 1)
    try:
        return int(cls_s), int(inst_s)
    except ValueError:
        return -1, -1


def normalize_direction(value: float) -> float:
    """
    Map column-2 values to Tor packet directions.

    - value > 0  ->  1  (outgoing)
    - value < 0  -> -1  (incoming)
    - value == 0 -> -1  (fallback)
    """
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return -1.0


def parse_cell_file(path: str | Path) -> CellTrace:
    """Parse a .cell file using vectorized NumPy I/O."""
    path = Path(path)
    label, _ = _ids_from_cell_stem(path.stem)

    try:
        data = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except (ValueError, OSError):
        data = np.zeros((0, 2), dtype=np.float64)

    if data.size == 0:
        timestamps = np.zeros(0, dtype=np.float32)
        raw_directions = np.zeros(0, dtype=np.float32)
        directions = np.zeros(0, dtype=np.float32)
    else:
        timestamps = data[:, 0].astype(np.float32)
        raw_directions = data[:, 1].astype(np.float32)
        directions = np.where(raw_directions > 0, 1.0, -1.0).astype(np.float32)

    return CellTrace(
        timestamps=timestamps,
        directions=directions,
        raw_directions=raw_directions,
        label=label,
        filename=path.name,
    )


def write_cell_file(
    path: str | Path,
    timestamps: np.ndarray,
    directions: np.ndarray,
    label: int | None = None,
    instance_id: int | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = min(len(timestamps), len(directions))
    if n == 0:
        path.write_text("", encoding="utf-8")
    else:
        arr = np.empty((n, 2), dtype=np.float64)
        arr[:, 0] = np.asarray(timestamps[:n], dtype=np.float64)
        arr[:, 1] = np.where(np.asarray(directions[:n]) > 0, 1.0, -1.0)
        np.savetxt(path, arr, fmt="%.6f\t%.1f")

    if label is not None and instance_id is not None:
        target = path.parent / f"{label}-{instance_id}.cell"
        if target != path:
            path.rename(target)


def discover_trace_pairs(
    original_dir: str | Path,
    defended_dir: str | Path,
    num_classes: Optional[int] = None,
    num_instances: Optional[int] = None,
) -> List[Tuple[Path, Path, int, int]]:
    """
    Discover paired traces sharing the same filename.

    When num_classes and num_instances are set, paths are constructed directly
    as ``{class}-{instance}.cell`` (O(num_classes * num_instances)) instead
    of scanning the full directory.

    Returns list of (original_path, defended_path, class_id, instance_id).
    """
    original_dir = Path(original_dir)
    defended_dir = Path(defended_dir)
    pairs: List[Tuple[Path, Path, int, int]] = []

    if num_classes is not None and num_instances is not None:
        for cls_id in range(num_classes):
            for inst_id in range(num_instances):
                name = f"{cls_id}-{inst_id}.cell"
                orig_path = original_dir / name
                def_path = defended_dir / name
                if orig_path.is_file() and def_path.is_file():
                    pairs.append((orig_path, def_path, cls_id, inst_id))
        return pairs

    # Fallback: directory scan (slower)
    for orig_path in sorted(original_dir.glob("*.cell")):
        stem = orig_path.stem
        if "-" not in stem:
            continue
        cls_s, inst_s = stem.split("-", 1)
        cls_id, inst_id = int(cls_s), int(inst_s)
        if num_classes is not None and cls_id >= num_classes:
            continue
        if num_instances is not None and inst_id >= num_instances:
            continue
        def_path = defended_dir / orig_path.name
        if def_path.is_file():
            pairs.append((orig_path, def_path, cls_id, inst_id))

    return pairs


def discover_defended_traces(
    defended_dir: str | Path,
    num_classes: Optional[int] = None,
    num_instances: Optional[int] = None,
) -> List[Tuple[Path, int, int]]:
    """
    Discover defended-only traces (no originals required).

    With no class/instance limits, every ``*.cell`` under ``defended_dir``
    (including subfolders) is returned. Returns (defended_path, class_id, instance_id).
    """
    defended_dir = Path(defended_dir)
    traces: List[Tuple[Path, int, int]] = []

    if num_classes is not None and num_instances is not None:
        for cls_id in range(num_classes):
            for inst_id in range(num_instances):
                def_path = defended_dir / f"{cls_id}-{inst_id}.cell"
                if def_path.is_file():
                    traces.append((def_path, cls_id, inst_id))
        return traces

    for def_path in sorted(defended_dir.rglob("*.cell")):
        if not def_path.is_file():
            continue
        cls_id, inst_id = _ids_from_cell_stem(def_path.stem)
        if num_classes is not None and cls_id >= 0 and cls_id >= num_classes:
            continue
        if num_instances is not None and inst_id >= 0 and inst_id >= num_instances:
            continue
        traces.append((def_path, cls_id, inst_id))

    return traces
