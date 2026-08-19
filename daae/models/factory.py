"""Model factory for DAAE and few-shot FS-DAAE."""

from __future__ import annotations

from typing import Any, Dict, Union

import torch
import torch.nn as nn

from daae.config import DAAEConfig
from daae.models.daae import DAAE
from daae.models.fs_daae import FS_DAAE
from daae.models.fs_daae_legacy import FS_DAAE_Legacy

ModelType = Union[DAAE, FS_DAAE, FS_DAAE_Legacy]


def infer_architecture(state_dict: Dict[str, Any]) -> str:
    """Detect FS-DAAE variant from checkpoint keys."""
    keys = state_dict.keys()
    if any(k.startswith("unet.") or k.startswith("mask_head.branches") for k in keys):
        return "unet"
    if any(k.startswith("mask_head.lstm") or k.startswith("encoder.net") for k in keys):
        return "bilstm_crf"
    return "unet"


def create_model(config: DAAEConfig) -> ModelType:
    common = dict(
        in_channels=config.in_channels,
        latent_dim=config.latent_dim,
        hidden_channels=config.hidden_channels,
        max_len=config.max_len,
        dropout=config.dropout,
        trace_format=config.trace_format,
    )
    if config.model_type == "fs_daae":
        arch = getattr(config, "architecture", "unet")
        if arch == "bilstm_crf":
            return FS_DAAE_Legacy(
                **common,
                support_shots=config.support_shots,
                encoder_blocks=config.encoder_blocks,
                use_crf=config.use_crf,
                use_subsequence_refine=config.use_subsequence_refine,
                bilstm_hidden=config.bilstm_hidden,
            )
        return FS_DAAE(
            **common,
            support_shots=config.support_shots,
            encoder_blocks=config.encoder_blocks,
        )
    return DAAE(**common)


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ModelType, DAAEConfig]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt["config"])
    state = ckpt["model_state"]

    if cfg_dict.get("model_type") == "fs_daae" and "architecture" not in cfg_dict:
        cfg_dict["architecture"] = infer_architecture(state)

    config = DAAEConfig(**{k: v for k, v in cfg_dict.items() if k in DAAEConfig.__dataclass_fields__})
    if cfg_dict.get("architecture"):
        config.architecture = cfg_dict["architecture"]

    model = create_model(config).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Checkpoint load: {len(missing)} missing keys (architecture={getattr(config, 'architecture', '?')})")
        if len(missing) > 10:
            print("  WARNING: large key mismatch — recovery quality may be poor. Retrain or use a matching checkpoint.")
    if unexpected:
        print(f"Checkpoint load: {len(unexpected)} unexpected keys ignored")
    model.eval()
    return model, config


def raw_state_dict(model: nn.Module) -> Dict[str, Any]:
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.state_dict()
    return model.state_dict()


def forward_model(
    model: ModelType,
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if isinstance(model, (FS_DAAE, FS_DAAE_Legacy)):
        return model(
            batch["defended"],
            batch.get("support_defended"),
            batch.get("support_mask"),
            batch.get("defended_len"),
        )
    return model(batch["defended"])
