"""Few-Shot U-Net DAAE — full-resolution mask head with skip connections."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from daae.models.daae import ConvBlock, TransposeConvBlock

if TYPE_CHECKING:
    from daae.recover_utils import ClassPrototypeCache


class TraceEncoderWithSkips(nn.Module):
    """1D CNN encoder returning bottleneck and per-scale skip features."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Tuple[int, ...] = (64, 128, 256, 512),
        dropout: float = 0.2,
        num_blocks: Optional[int] = None,
    ):
        super().__init__()
        if num_blocks is not None:
            hidden_channels = hidden_channels[:num_blocks]
        ch_in = in_channels
        self.blocks = nn.ModuleList()
        for ch_out in hidden_channels:
            self.blocks.append(ConvBlock(ch_in, ch_out, dropout=dropout))
            ch_in = ch_out
        self.out_channels = hidden_channels[-1]
        self.channel_list = list(hidden_channels)
        self.downsample_factor = 2 ** len(hidden_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        skips = [x]
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        return x, skips


class SupportQueryFusion(nn.Module):
    """Few-shot fusion via prototype FiLM + support-query cross-attention."""

    def __init__(self, channels: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mask_embed = nn.Conv1d(1, channels, kernel_size=1)
        self.proto_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels * 2),
        )
        self.attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )

    def forward(
        self,
        query_feat: torch.Tensor,
        support_defended: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, c, lq = query_feat.shape
        k = support_defended.shape[1]

        sm = support_mask
        if sm.shape[-1] != lq:
            sm = F.interpolate(
                sm.reshape(b * k, 1, sm.shape[-1]), size=lq, mode="nearest"
            ).view(b, k, lq)
        mask_bias = self.mask_embed(sm.reshape(b * k, 1, lq)).view(b, k, c, lq)

        support_feat = support_defended + mask_bias
        proto = support_feat.mean(dim=1)
        film = self.proto_proj(proto)
        gamma, beta = film.chunk(2, dim=-1)
        # Bound FiLM scale/shift — unbounded γ/β under AMP quickly yields Inf/NaN.
        gamma = torch.tanh(gamma)
        beta = torch.tanh(beta)
        query_tokens = query_feat.permute(0, 2, 1)
        query_tokens = query_tokens * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        support_tokens = support_feat.permute(0, 1, 3, 2).reshape(b, k * lq, c)
        # Cross-attn over K·L tokens is fp16-fragile; run it in fp32.
        dtype = query_tokens.dtype
        attn_out, _ = self.attn(
            query_tokens.float(), support_tokens.float(), support_tokens.float()
        )
        attn_out = attn_out.to(dtype)
        fused = self.norm(query_tokens + attn_out)
        fused = fused + self.ffn(fused)
        return fused.permute(0, 2, 1)


class UNetMaskDecoder(nn.Module):
    """Upsample fused bottleneck with encoder skips to full-resolution mask features."""

    def __init__(self, channels: Tuple[int, ...], in_channels: int, dropout: float = 0.2):
        super().__init__()
        self.channels = channels
        rev = list(reversed(channels))
        self.up_convs = nn.ModuleList()
        for i in range(len(rev) - 1):
            skip_ch = rev[i + 1]
            self.up_convs.append(
                nn.Sequential(
                    TransposeConvBlock(rev[i], skip_ch, dropout=dropout),
                    nn.Conv1d(skip_ch * 2, skip_ch, kernel_size=5, padding=2),
                    nn.BatchNorm1d(skip_ch),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        finest = rev[-1]
        self.to_full = TransposeConvBlock(finest, finest, dropout=dropout)
        self.fuse_input = nn.Sequential(
            nn.Conv1d(finest + in_channels + 1, finest, kernel_size=5, padding=2),
            nn.BatchNorm1d(finest),
            nn.GELU(),
        )

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: List[torch.Tensor],
        query: torch.Tensor,
        support_prior: Optional[torch.Tensor],
        target_len: int,
    ) -> torch.Tensor:
        """
        skips[0]=input, skips[i]=feature after encoder block i-1.
        bottleneck = skips[-1] after fusion.
        """
        h = bottleneck
        n_blocks = len(self.channels)
        for i, up in enumerate(self.up_convs):
            level = n_blocks - 1 - i
            skip = skips[level]
            h = up[0](h)
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode="linear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = up[1:](h)

        h = self.to_full(h)
        if h.shape[-1] != target_len:
            h = F.interpolate(h, size=target_len, mode="linear", align_corners=False)

        if query.shape[-1] != target_len:
            query = F.interpolate(query, size=target_len, mode="linear", align_corners=False)
        if support_prior is None:
            support_prior = torch.zeros(query.shape[0], 1, target_len, device=query.device, dtype=query.dtype)
        elif support_prior.shape[-1] != target_len:
            support_prior = F.interpolate(support_prior, size=target_len, mode="linear", align_corners=False)

        h = torch.cat([h, query, support_prior], dim=1)
        return self.fuse_input(h)


class DilatedMaskHead(nn.Module):
    """Multi-scale dilated convolutions for per-packet boundary refinement."""

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(channels, channels // 2, kernel_size=3, padding=d, dilation=d)
            for d in (1, 2, 4, 8)
        ])
        self.proj = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [branch(x) for branch in self.branches]
        h = torch.cat(parts, dim=1)
        return self.proj(h).squeeze(1)


class FS_DAAE(nn.Module):
    """
    Few-Shot U-Net DAAE (FS-DAAE).

    Improvements over conv-only / BiLSTM heads:
    - U-Net decoder with skip connections for full-resolution mask logits
    - Support keep-mask prior fused at finest scale
    - Multi-scale dilated mask refinement head
    - Few-shot cross-attention at encoder bottleneck only
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 256,
        hidden_channels: Tuple[int, ...] = (64, 128, 256, 512),
        max_len: int = 10000,
        dropout: float = 0.2,
        support_shots: int = 8,
        trace_format: str = "direction_timing_gap",
        encoder_blocks: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.max_len = max_len
        self.support_shots = support_shots
        self.trace_format = trace_format

        enc_ch = hidden_channels[:encoder_blocks]
        self.encoder = TraceEncoderWithSkips(in_channels, hidden_channels, dropout, encoder_blocks)
        ch = self.encoder.out_channels
        finest = enc_ch[0]
        self.fusion = SupportQueryFusion(ch, dropout=dropout)
        self.query_only = nn.Conv1d(ch, ch, kernel_size=1)

        self.unet = UNetMaskDecoder(tuple(enc_ch), in_channels, dropout)
        self.mask_head = DilatedMaskHead(finest, dropout)

        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(ch, latent_dim), nn.GELU(),
        )
        self.recon_head = nn.Conv1d(finest, 1, kernel_size=7, padding=3)

    def _support_prior(self, support_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if support_mask is None:
            return None
        return support_mask.float().mean(dim=1, keepdim=True).clamp(1e-4, 1.0 - 1e-4)

    def _encode_support(self, support_defended: torch.Tensor) -> torch.Tensor:
        b, k, c, seq_len = support_defended.shape
        enc, _ = self.encoder(support_defended.view(b * k, c, seq_len))
        _, c_out, l_out = enc.shape
        return enc.view(b, k, c_out, l_out)

    def forward(
        self,
        query: torch.Tensor,
        support_defended: Optional[torch.Tensor] = None,
        support_mask: Optional[torch.Tensor] = None,
        defended_len: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if query.dim() == 2:
            query = query.unsqueeze(1)
        target_len = query.shape[-1]

        bottleneck, skips = self.encoder(query)
        if support_defended is not None and support_mask is not None:
            s_feat = self._encode_support(support_defended)
            h = self.fusion(bottleneck, s_feat, support_mask)
        else:
            h = bottleneck + self.query_only(bottleneck)

        skips = skips[:-1] + [h]
        prior = self._support_prior(support_mask)
        mask_feat = self.unet(h, skips, query, prior, target_len)
        mask_logits = self.mask_head(mask_feat)
        recon = self.recon_head(mask_feat).squeeze(1)

        return {
            "mask_logits": mask_logits,
            "recon": recon,
            "latent": self.to_latent(h),
        }

    def _insertion_keep_mask(
        self,
        defended: torch.Tensor,
        mask_logits: torch.Tensor,
        support_defended: torch.Tensor,
        support_mask: torch.Tensor,
        defended_len: torch.Tensor,
        prototype_caches: Optional[List[Optional[ClassPrototypeCache]]] = None,
        calibrate_steps: int = 6,
    ) -> torch.Tensor:
        """Insertion-only decode: output is a defended subsequence aligned to support prototypes."""
        from daae.recover_utils import batch_insertion_decode, build_prototype_cache

        dirs_np = defended[:, 0].detach().cpu().contiguous().numpy()
        logits_np = mask_logits.detach().float().cpu().contiguous().numpy()
        lens = defended_len.detach().cpu().numpy()

        cache = None
        if prototype_caches:
            cache = prototype_caches[0]
        if cache is None:
            sm_np = support_mask.detach().cpu().numpy()
            sd_np = support_defended[:, :, 0].detach().cpu().numpy()
            k = sd_np.shape[1]
            slens = np.full(k, sm_np.shape[-1], dtype=np.int32)
            cache = build_prototype_cache(sd_np[0], sm_np[0], slens)

        keep_np = batch_insertion_decode(
            dirs_np, logits_np, lens, cache,
            calibrate_steps=calibrate_steps,
        )
        return torch.from_numpy(keep_np).to(device=mask_logits.device, dtype=mask_logits.dtype)

    @torch.no_grad()
    def recover(
        self,
        defended: torch.Tensor,
        defended_len: torch.Tensor | None = None,
        support_defended: Optional[torch.Tensor] = None,
        support_mask: Optional[torch.Tensor] = None,
        mask_threshold: float = 0.5,
        insertion_decode: bool = True,
        prototype_caches: Optional[List[Optional["ClassPrototypeCache"]]] = None,
        calibrate_steps: int = 6,
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        if defended.dim() == 2:
            defended = defended.unsqueeze(1)
        out = self.forward(defended, support_defended, support_mask, defended_len)
        mask_logits = out["mask_logits"]
        mask_prob = torch.sigmoid(mask_logits)

        if (
            insertion_decode
            and support_defended is not None
            and support_mask is not None
            and defended_len is not None
        ):
            keep_mask = self._insertion_keep_mask(
                defended, mask_logits, support_defended, support_mask, defended_len,
                prototype_caches=prototype_caches,
                calibrate_steps=calibrate_steps,
            )
        else:
            keep_mask = (mask_prob >= mask_threshold).float()

        if defended_len is not None:
            seq_len = keep_mask.shape[1]
            valid = torch.arange(seq_len, device=keep_mask.device).unsqueeze(0) < defended_len.unsqueeze(1)
            keep_mask = keep_mask * valid.float()
        return {
            "keep_mask": keep_mask,
            "mask_prob": mask_prob,
            "mask_logits": mask_logits,
            "recon": out["recon"],
            "latent": out["latent"],
        }
