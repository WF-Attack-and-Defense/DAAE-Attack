"""PyTorch dataset for paired original/defended traces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from daae.config import DAAEConfig
from daae.data.alignment import align_traces
from daae.data.cache import StackedPairedDataset, preload_pairs_parallel
from daae.data.few_shot import build_fs_dataloader
from daae.data.cell_io import discover_trace_pairs, parse_cell_file
from daae.data.features import extract_features_from_arrays


@dataclass
class TraceSplit:
    train: List[Tuple[Path, Path, int, int]]
    val: List[Tuple[Path, Path, int, int]]
    test: List[Tuple[Path, Path, int, int]]


def split_pairs_by_class(
    pairs: List[Tuple[Path, Path, int, int]],
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
) -> TraceSplit:
    """Split trace pairs per class; train_ratio controls training fraction per class."""
    rng = np.random.default_rng(seed)
    by_class: Dict[int, List[Tuple[Path, Path, int, int]]] = {}
    for item in pairs:
        by_class.setdefault(item[2], []).append(item)

    train, val, test = [], [], []
    for cls_id in sorted(by_class):
        items = sorted(by_class[cls_id], key=lambda x: x[3])
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(0, int(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            n_train = max(1, n - n_val - n_test)

        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    return TraceSplit(train=train, val=val, test=test)


def _pad_1d(arr: np.ndarray, max_len: int, pad_value: float = 0.0) -> np.ndarray:
    out = np.full(max_len, pad_value, dtype=np.float32)
    n = min(len(arr), max_len)
    out[:n] = arr[:n]
    return out


class LazyPairedDataset(Dataset):
    """On-demand loading for splits that are not preloaded."""

    def __init__(
        self,
        pairs: List[Tuple[Path, Path, int, int]],
        max_len: int = 5000,
        trace_format: str = "direction_timing",
        alignment_method: str = "hamming",
    ):
        self.pairs = pairs
        self.max_len = max_len
        self.trace_format = trace_format
        self.alignment_method = alignment_method

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        orig_path, def_path, cls_id, _ = self.pairs[idx]
        orig = parse_cell_file(orig_path)
        defended = parse_cell_file(def_path)

        # Truncate before alignment (same as preload) to bound LCS/hamming cost.
        orig_dirs = orig.directions.astype(np.float32)[: self.max_len]
        def_dirs = defended.directions.astype(np.float32)[: self.max_len]
        alignment = align_traces(orig_dirs, def_dirs, method=self.alignment_method)

        recon_target = np.zeros(len(def_dirs), dtype=np.float32)
        valid_idx = alignment.orig_indices >= 0
        recon_target[valid_idx] = orig_dirs[alignment.orig_indices[valid_idx]]

        def_features, def_len = extract_features_from_arrays(
            defended.timestamps[: self.max_len],
            defended.directions[: self.max_len],
            self.max_len,
            self.trace_format,
        )

        return {
            "defended": torch.from_numpy(def_features),
            "recon_target": torch.from_numpy(_pad_1d(recon_target, self.max_len)),
            "dummy_mask": torch.from_numpy(_pad_1d(alignment.keep_mask, self.max_len)),
            "defended_len": torch.tensor(def_len, dtype=torch.long),
            "original_len": torch.tensor(min(len(orig_dirs), self.max_len), dtype=torch.long),
            "label": torch.tensor(cls_id, dtype=torch.long),
        }


def _build_stacked_dataset(
    pairs: List[Tuple[Path, Path, int, int]],
    config: DAAEConfig,
    split_name: str,
    cache_dir: Optional[Path],
    tensor_device: Optional[torch.device],
) -> Dataset:
    if not pairs:
        return StackedPairedDataset(
            {
                "defended": np.zeros((0, config.in_channels, config.max_len), np.float32),
                "recon_target": np.zeros((0, config.max_len), np.float32),
                "dummy_mask": np.zeros((0, config.max_len), np.float32),
                "defended_len": np.zeros(0, np.int64),
                "original_len": np.zeros(0, np.int64),
                "labels": np.zeros(0, np.int64),
            },
            device=tensor_device,
        )
    arrays = preload_pairs_parallel(
        pairs,
        config.max_len,
        config.trace_format,
        config.preload_workers,
        split_name=split_name,
        cache_dir=cache_dir,
        use_cache=config.use_cache,
        alignment_method=config.alignment_method,
    )
    return StackedPairedDataset(arrays, device=tensor_device)


def build_dataloaders(
    original_root: str | Path,
    defended_root: str | Path,
    config: DAAEConfig,
    split: Optional[TraceSplit] = None,
    cache_dir: Optional[Path] = None,
    tensor_device: Optional[torch.device] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, TraceSplit, Optional[StackedPairedDataset]]:
    print(
        f"Discovering pairs under {original_root} / {defended_root} "
        f"(classes={config.num_classes}, instances={config.num_instances})…",
        flush=True,
    )
    pairs = discover_trace_pairs(
        original_root,
        defended_root,
        config.num_classes,
        config.num_instances,
    )
    if not pairs:
        raise FileNotFoundError(
            f"No paired traces found under {original_root} and {defended_root} "
            f"(num_classes={config.num_classes}, num_instances={config.num_instances})"
        )

    if split is None:
        split = split_pairs_by_class(
            pairs, config.train_ratio, config.val_ratio, config.seed
        )

    print(
        f"Dataset scope: {config.num_classes} classes x {config.num_instances} instances "
        f"-> {len(pairs)} pairs | train={len(split.train)} val={len(split.val)} test={len(split.test)}",
        flush=True,
    )

    on_gpu = tensor_device is not None and tensor_device.type == "cuda"
    train_stacked: Optional[StackedPairedDataset] = None

    if config.preload:
        train_stacked = _build_stacked_dataset(
            split.train, config, "train", cache_dir, tensor_device
        )
        train_ds: Dataset = train_stacked
        if config.preload_val and split.val:
            val_ds = _build_stacked_dataset(
                split.val, config, "val", cache_dir, tensor_device
            )
        else:
            val_ds = LazyPairedDataset(
                split.val, config.max_len, config.trace_format, config.alignment_method
            )
    else:
        train_ds = LazyPairedDataset(
            split.train, config.max_len, config.trace_format, config.alignment_method
        )
        val_ds = LazyPairedDataset(
            split.val, config.max_len, config.trace_format, config.alignment_method
        )

    test_ds = LazyPairedDataset(
        split.test, config.max_len, config.trace_format, config.alignment_method
    )

    train_loader = build_fs_dataloader(train_ds, config, shuffle=True, eval_mode=False)
    val_loader = build_fs_dataloader(
        val_ds, config, shuffle=False, eval_mode=True, support_dataset=train_stacked
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0 if on_gpu else config.num_workers,
        pin_memory=torch.cuda.is_available() and not on_gpu,
    )
    return train_loader, val_loader, test_loader, split, train_stacked
