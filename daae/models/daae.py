"""Defense-Aware Autoencoder (DAAE) model."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, stride: int = 2, dropout: float = 0.1):
        super().__init__()
        padding = kernel // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TransposeConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, stride: int = 2, dropout: float = 0.1):
        super().__init__()
        padding = kernel // 2
        output_padding = stride - 1
        self.block = nn.Sequential(
            nn.ConvTranspose1d(
                in_ch, out_ch, kernel, stride=stride,
                padding=padding, output_padding=output_padding,
            ),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DAAE(nn.Module):
    """
    Defense-Aware Autoencoder for website fingerprinting trace recovery.

    The encoder compresses defended traces into a latent representation.
    The decoder produces:
      1) per-packet dummy mask logits (identify injected dummy packets)
      2) per-packet recovered trace values (original traffic estimate)
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 256,
        hidden_channels: Tuple[int, ...] = (64, 128, 256, 512),
        max_len: int = 5000,
        dropout: float = 0.2,
        trace_format: str = "direction",
    ):
        super().__init__()
        self.max_len = max_len
        self.trace_format = trace_format
        self.latent_dim = latent_dim

        enc_layers = []
        ch_in = in_channels
        for ch_out in hidden_channels:
            enc_layers.append(ConvBlock(ch_in, ch_out, dropout=dropout))
            ch_in = ch_out
        self.encoder = nn.Sequential(*enc_layers)

        # Global latent from encoded features
        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels[-1], latent_dim),
            nn.GELU(),
        )

        # Decoder: latent + skip from encoded sequence
        self.from_latent = nn.Linear(latent_dim, hidden_channels[-1])

        dec_layers = []
        rev_channels = list(hidden_channels)[::-1]
        for i in range(len(rev_channels) - 1):
            dec_layers.append(
                TransposeConvBlock(rev_channels[i], rev_channels[i + 1], dropout=dropout)
            )
        self.decoder = nn.Sequential(*dec_layers)

        final_ch = rev_channels[-1]
        self.mask_head = nn.Conv1d(final_ch, 1, kernel_size=7, padding=3)
        self.recon_head = nn.Conv1d(final_ch, 1, kernel_size=7, padding=3)

        self._downsample_factor = 2 ** len(hidden_channels)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, L) defended trace
        Returns:
            latent: (B, latent_dim)
            encoded: (B, C', L')
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        encoded = self.encoder(x)
        latent = self.to_latent(encoded)
        return latent, encoded

    def decode(self, latent: torch.Tensor, target_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        b = latent.shape[0]
        h = self.from_latent(latent).unsqueeze(-1)  # (B, C', 1)
        # Broadcast latent seed across temporal dimension before upsampling
        h = h.expand(-1, -1, max(1, target_len // self._downsample_factor))
        h = self.decoder(h)

        # Interpolate to exact target length
        if h.shape[-1] != target_len:
            h = F.interpolate(h, size=target_len, mode="linear", align_corners=False)

        mask_logits = self.mask_head(h).squeeze(1)   # (B, L)
        recon = self.recon_head(h).squeeze(1)         # (B, L)
        return mask_logits, recon

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        target_len = x.shape[-1]
        latent, _ = self.encode(x)
        mask_logits, recon = self.decode(latent, target_len)
        return {
            "mask_logits": mask_logits,
            "recon": recon,
            "latent": latent,
        }

    @torch.no_grad()
    def recover(
        self,
        defended: torch.Tensor,
        defended_len: torch.Tensor | None = None,
        mask_threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Recover original traces by removing predicted dummy packets.

        Returns keep_mask, recovered values/timestamps placeholder, and recon.
        """
        self.eval()
        out = self.forward(defended)
        mask_prob = torch.sigmoid(out["mask_logits"])
        keep_mask = (mask_prob >= mask_threshold).float()

        if defended_len is not None:
            # Zero out padding region in mask
            for i, ln in enumerate(defended_len.tolist()):
                keep_mask[i, ln:] = 0.0

        return {
            "keep_mask": keep_mask,
            "mask_prob": mask_prob,
            "recon": out["recon"],
            "latent": out["latent"],
        }


# Backward-compatible re-export
from daae.models.losses import daae_loss  # noqa: E402, F401
