"""Evaluate FS-DAAE recovery quality against original traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from daae.cli import add_dataset_scope_arguments, add_path_arguments, resolve_paths
from daae.data.alignment import align_traces
from daae.data.cell_io import discover_trace_pairs, parse_cell_file
from daae.data.dataset import split_pairs_by_class
from daae.data.features import extract_directions
from daae.models.factory import load_model_from_checkpoint as load_model


def compute_recovery_metrics(
    pairs: List[Tuple[Path, Path, int, int]],
    recovered_dir: Path,
) -> Dict[str, float]:
    dir_matches, length_ratios = [], []
    for orig_path, _, _, inst_id in pairs:
        cls_id = int(orig_path.stem.split("-")[0])
        rec_path = recovered_dir / f"{cls_id}-{inst_id}.cell"
        if not rec_path.exists():
            continue
        orig = parse_cell_file(orig_path)
        rec = parse_cell_file(rec_path)
        orig_dirs = extract_directions(orig)
        rec_dirs = extract_directions(rec)
        align = align_traces(orig_dirs, rec_dirs)
        dir_matches.append(align.match_ratio)
        length_ratios.append(len(rec_dirs) / max(len(orig_dirs), 1))

    if not dir_matches:
        return {"direction_match": 0.0, "length_ratio": 0.0, "n": 0}

    return {
        "direction_match": float(np.mean(dir_matches)),
        "length_ratio": float(np.mean(length_ratios)),
        "n": len(dir_matches),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate FS-DAAE recovered traces")
    parser.add_argument("--checkpoint", required=True, type=Path)
    add_path_arguments(parser)
    add_dataset_scope_arguments(parser)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    original_root, defended_root, output_dir = resolve_paths(args)
    recovered_dir = output_dir / "recovered"

    device = torch.device(args.device)
    _, config = load_model(str(args.checkpoint.resolve()), device)
    config.num_classes = args.num_classes
    config.num_instances = args.num_instances

    pairs = discover_trace_pairs(
        original_root, defended_root, config.num_classes, config.num_instances
    )
    split = split_pairs_by_class(pairs, config.train_ratio, config.val_ratio, config.seed)

    recovery_metrics = compute_recovery_metrics(split.test, recovered_dir)
    print(f"Recovery metrics: {recovery_metrics}")

    out_path = output_dir / "evaluation.json"
    with out_path.open("w") as f:
        json.dump({"recovery": recovery_metrics}, f, indent=2)
    print(f"Saved evaluation to {out_path}")


if __name__ == "__main__":
    main()
