"""Model factory for FS-DAAE."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from daae.config import DAAEConfig
from daae.models.fs_daae import FS_DAAE


def create_model(config: DAAEConfig) -> FS_DAAE:
    return FS_DAAE(
        in_channels=config.in_channels,
        latent_dim=config.latent_dim,
        hidden_channels=config.hidden_channels,
        max_len=config.max_len,
        dropout=config.dropout,
        support_shots=config.support_shots,
        trace_format=config.trace_format,
        encoder_blocks=config.encoder_blocks,
    )


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[FS_DAAE, DAAEConfig]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt["config"])
    state = ckpt["model_state"]

    config = DAAEConfig(**{k: v for k, v in cfg_dict.items() if k in DAAEConfig.__dataclass_fields__})
    model = create_model(config).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Checkpoint load: {len(missing)} missing keys")
        if len(missing) > 10:
            print("  WARNING: large key mismatch — use an FS-DAAE U-Net checkpoint.")
    if unexpected:
        print(f"Checkpoint load: {len(unexpected)} unexpected keys ignored")
    model.eval()
    return model, config


def raw_state_dict(model: nn.Module) -> Dict[str, Any]:
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.state_dict()
    return model.state_dict()


def forward_model(model: FS_DAAE, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return model(
        batch["defended"],
        batch.get("support_defended"),
        batch.get("support_mask"),
        batch.get("defended_len"),
        original=batch.get("original") if batch.get("use_original") else None,
    )
