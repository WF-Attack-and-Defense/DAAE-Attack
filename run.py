#!/usr/bin/env python3
"""End-to-end DAAE pipeline: train -> recover -> evaluate."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from daae.cli import add_dataset_scope_arguments, add_path_arguments, add_alignment_arguments, resolve_paths


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run full DAAE pipeline")
    add_path_arguments(parser)
    add_dataset_scope_arguments(parser)
    parser.add_argument("--train-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    add_alignment_arguments(parser, default="hamming")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    original_root, defended_root, output_dir = resolve_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or (output_dir / "daae_best.pt")

    py = sys.executable
    paths = [
        "--original-root", str(original_root),
        "--defended-root", str(defended_root),
        "--output-dir", str(output_dir),
        "--num-classes", str(args.num_classes),
        "--num-instances", str(args.num_instances),
    ]

    if not args.skip_train:
        run([
            py, "-m", "daae.train",
            *paths,
            "--train-ratio", str(args.train_ratio),
            "--epochs", str(args.epochs),
            "--model", "fs_daae",
            "--alignment-method", args.alignment_method,
        ])

    align_flag = ["--alignment-method", args.alignment_method]
    run([py, "-m", "daae.recover", "--checkpoint", str(checkpoint), *paths, *align_flag, "--split", "test"])
    run([py, "-m", "daae.evaluate", "--checkpoint", str(checkpoint), *paths, *align_flag])


if __name__ == "__main__":
    main()
