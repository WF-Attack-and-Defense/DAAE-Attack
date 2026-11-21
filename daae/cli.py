"""Shared CLI argument helpers."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_path_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--original-root",
        required=True,
        type=Path,
        help="Path to original (undefended) .cell trace directory",
    )
    parser.add_argument(
        "--defended-root",
        required=True,
        type=Path,
        help="Path to defended .cell trace directory (same filenames as original)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for checkpoints, recovered traces, and evaluation results",
    )


def add_dataset_scope_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--num-classes",
        type=int,
        required=True,
        help="Number of website classes to load (classes 0 .. num_classes-1)",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        required=True,
        help="Number of trace instances per class to load (instances 0 .. num_instances-1)",
    )


def resolve_paths(args) -> tuple[Path, Path, Path]:
    return (
        args.original_root.resolve(),
        args.defended_root.resolve(),
        args.output_dir.resolve(),
    )
