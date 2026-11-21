"""Hyperparameter configuration for DAAE experiments (no dataset paths)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def in_channels_for_format(trace_format: str) -> int:
    """Return model input channels for the selected trace feature mode."""
    from daae.data.features import in_channels_for_format as _ch

    return _ch(trace_format)


def default_batch_size() -> int:
    return 128


def default_preload_workers() -> int:
    return min(32, os.cpu_count() or 8)


@dataclass
class DAAEConfig:
    # Dataset scope (None = scan entire directory — slower)
    num_classes: Optional[int] = None
    num_instances: Optional[int] = None

    # Data split: train_ratio of instances per class used for DAAE training
    train_ratio: float = 0.2
    val_ratio: float = 0.1
    seed: int = 42

    # Trace features: direction ±1, or paired_dir (dir+burst+CUMUL, no timing)
    trace_format: str = "paired_dir"
    max_len: int = 10000
    in_channels: int = field(default=1)

    # Loading (192GB RAM: preload train + val; 98GB VRAM: optional GPU tensors)
    preload: bool = True
    preload_val: bool = True
    preload_workers: int = field(default_factory=default_preload_workers)
    use_cache: bool = True
    gpu_tensors: bool = True

    # Model (FS-DAAE U-Net)
    model_type: str = "fs_daae"
    support_shots: int = 5
    focal_gamma: float = 2.0
    use_focal: bool = True
    dice_weight: float = 0.5
    front_mask_weight: float = 2.5
    burst_weight: float = 0.4
    rank_weight: float = 0.5
    latent_dim: int = 256
    hidden_channels: tuple = (64, 128, 256, 512)
    encoder_blocks: int = 4
    dropout: float = 0.2

    # Training (RTX PRO 6000: large batch + AMP)
    batch_size: int = field(default_factory=default_batch_size)
    epochs: int = 50
    lr: float = 5e-4
    weight_decay: float = 1e-5
    mask_loss_weight: float = 2.0
    recon_loss_weight: float = 1.0
    length_ratio_weight: float = 1.0
    use_insertion_decode: bool = True
    num_workers: int = 0
    use_amp: bool = True
    compile_model: bool = True

    # Checkpoint
    checkpoint_name: str = "daae_best.pt"

    def __post_init__(self) -> None:
        self.in_channels = in_channels_for_format(self.trace_format)
