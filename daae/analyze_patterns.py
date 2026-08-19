"""Analyze Palette / dummy-packet noise patterns in paired traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from daae.cli import add_dataset_scope_arguments, add_path_arguments, add_alignment_arguments, resolve_paths
from daae.config import DAAEConfig
from daae.data.alignment import align_traces
from daae.data.cell_io import discover_trace_pairs, parse_cell_file
from daae.data.features import extract_directions, inter_packet_gaps


def _run_lengths(mask: np.ndarray, value: float) -> list[int]:
    runs, cur = [], 0
    for v in mask:
        if v == value:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def analyze_pair(orig_path: Path, def_path: Path, alignment_method: str = "hamming") -> dict:
    orig = parse_cell_file(orig_path)
    defended = parse_cell_file(def_path)
    od, dd = extract_directions(orig), extract_directions(defended)
    align = align_traces(od, dd, method=alignment_method)
    km = align.keep_mask
    n = len(km)
    thirds = n // 3 if n else 1
    gaps = inter_packet_gaps(defended.timestamps.astype(np.float32))

    dummy_runs = _run_lengths(km, 0.0)
    keep_runs = _run_lengths(km, 1.0)
    gap_dummy = [float(gaps[i]) for i in range(min(len(gaps), n)) if km[i] < 0.5]
    gap_keep = [float(gaps[i]) for i in range(min(len(gaps), n)) if km[i] >= 0.5]

    return {
        "orig_len": len(od),
        "def_len": len(dd),
        "dummy_ratio": float(1.0 - km.mean()) if n else 0.0,
        "match_ratio": float(align.match_ratio),
        "dummy_run_median": float(np.median(dummy_runs)) if dummy_runs else 0.0,
        "dummy_run_p90": float(np.percentile(dummy_runs, 90)) if dummy_runs else 0.0,
        "dummy_front_frac": float(1.0 - km[:thirds].mean()) if n else 0.0,
        "dummy_mid_frac": float(1.0 - km[thirds : 2 * thirds].mean()) if n else 0.0,
        "dummy_tail_frac": float(1.0 - km[2 * thirds :].mean()) if n else 0.0,
        "gap_dummy_mean": float(np.mean(gap_dummy)) if gap_dummy else 0.0,
        "gap_keep_mean": float(np.mean(gap_keep)) if gap_keep else 0.0,
        "keep_run_median": float(np.median(keep_runs)) if keep_runs else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze defense trace noise patterns")
    add_path_arguments(parser)
    add_dataset_scope_arguments(parser)
    add_alignment_arguments(parser, default="hamming")
    args = parser.parse_args()
    original_root, defended_root, output_dir = resolve_paths(args)

    pairs = discover_trace_pairs(original_root, defended_root, args.num_classes, args.num_instances)
    stats = [analyze_pair(o, d, args.alignment_method) for o, d, _, _ in pairs]

    arr = {k: np.array([s[k] for s in stats], dtype=np.float64) for k in stats[0]}
    summary = {
        "alignment_method": args.alignment_method,
        "defense": defended_root.name,
        "num_pairs": len(stats),
        "dummy_ratio_mean": float(arr["dummy_ratio"].mean()),
        "match_ratio_mean": float(arr["match_ratio"].mean()),
        "dummy_run_p90_mean": float(arr["dummy_run_p90"].mean()),
        "front_heavy_dummy": float(arr["dummy_front_frac"].mean() - arr["dummy_tail_frac"].mean()),
        "gap_separability": float(arr["gap_dummy_mean"].mean() - arr["gap_keep_mean"].mean()),
        "recommendations": [
            "Use direction_timing_gap features (dummy slots have distinct inter-packet gaps).",
            "Model long dummy bursts with BiLSTM+CRF (Palette super-matrix tail padding).",
            "Few-shot K support traces per class; best-of-K subsequence decode at inference.",
            "Front-third traces are noisiest — consider higher mask loss weight on first third.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "pattern_analysis.json"
    with out.open("w") as f:
        json.dump({"summary": summary, "per_trace": stats[:50]}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
