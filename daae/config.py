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

    # Alignment labels for dummy masks: hamming (insertion-only) | lcs | greedy
    # Palette: use lcs — hamming labels keep only ~8% of packets vs ~44% true ratio.
    alignment_method: str = "lcs"

    # Trace features (Tor: timing + direction; packet size is not used)
    trace_format: str = "direction_timing_gap"
    max_len: int = 10000
    in_channels: int = field(default=2)

    # Loading (192GB RAM: preload train + val; 98GB VRAM: optional GPU tensors)
    preload: bool = True
    preload_val: bool = True
    preload_workers: int = field(default_factory=default_preload_workers)
    use_cache: bool = True
    gpu_tensors: bool = True

    # Model
    model_type: str = "fs_daae"  # daae | fs_daae (few-shot meta-learning, default)
    architecture: str = "unet"  # unet | bilstm_crf (legacy checkpoints)
    support_shots: int = 5
    focal_gamma: float = 2.0
    use_focal: bool = True
    use_crf: bool = False
    use_subsequence_refine: bool = False
    bilstm_hidden: int = 256
    dice_weight: float = 0.3
    front_mask_weight: float = 1.0
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

    # Evaluation classifier
    clf_epochs: int = 30
    clf_batch_size: int = 128

    def __post_init__(self) -> None:
        self.in_channels = in_channels_for_format(self.trace_format)
        if self.alignment_method not in ("hamming", "lcs", "greedy"):
            raise ValueError(f"alignment_method must be hamming, lcs, or greedy; got {self.alignment_method!r}")
        if self.architecture not in ("unet", "bilstm_crf"):
            raise ValueError(f"architecture must be unet or bilstm_crf; got {self.architecture!r}")
