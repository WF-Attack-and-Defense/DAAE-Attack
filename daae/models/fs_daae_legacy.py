"""Legacy FS-DAAE: BiLSTM+CRF mask head (checkpoints before U-Net refactor)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from daae.models.crf import BinaryLinearCRF
from daae.models.daae import ConvBlock, TransposeConvBlock
from daae.models.decode import decode_insertion_mask, support_keep_ratio as _support_keep_ratio_np
from daae.models.refine import refine_batch_masks


class TraceEncoder(nn.Module):
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
        layers = []
        ch_in = in_channels
        for ch_out in hidden_channels:
            layers.append(ConvBlock(ch_in, ch_out, dropout=dropout))
            ch_in = ch_out
        self.net = nn.Sequential(*layers)
        self.out_channels = hidden_channels[-1]
        self.downsample_factor = 2 ** len(hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.net(x)


class SupportQueryFusion(nn.Module):
    def __init__(self, channels: int, support_shots: int = 5, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.support_shots = support_shots
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
        query_tokens = query_feat.permute(0, 2, 1)
        query_tokens = query_tokens * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        support_tokens = support_feat.permute(0, 1, 3, 2).reshape(b, k * lq, c)
        attn_out, _ = self.attn(query_tokens, support_tokens, support_tokens)
        fused = self.norm(query_tokens + attn_out)
        fused = fused + self.ffn(fused)
        return fused.permute(0, 2, 1)


class BiLSTMCRFHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        dropout: float = 0.2,
        use_crf: bool = True,
    ):
        super().__init__()
        self.use_crf = use_crf
        self.input_proj = nn.Linear(in_dim, hidden)
        self.lstm = nn.LSTM(
            hidden,
            hidden // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.emit = nn.Linear(hidden, 2)
        self.crf = BinaryLinearCRF() if use_crf else None

    def forward(
        self,
        feats: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = self.input_proj(feats)
        x, _ = self.lstm(x)
        raw_emit = self.emit(x)
        emissions = F.log_softmax(raw_emit, dim=-1)
        mask_logits = raw_emit[..., 1] - raw_emit[..., 0]
        out: Dict[str, torch.Tensor] = {
            "mask_logits": mask_logits,
            "emissions": emissions,
            "crf_module": self.crf,
        }
        if self.crf is not None and valid_mask is not None:
            tags = (mask_logits > 0).long()
            out["crf_loss"] = self.crf.neg_log_likelihood(emissions, tags, valid_mask)
        return out


class FS_DAAE_Legacy(nn.Module):
    """BiLSTM+CRF FS-DAAE used for fs-v2-full and earlier checkpoints."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 256,
        hidden_channels: Tuple[int, ...] = (64, 128, 256, 512),
        max_len: int = 5000,
        dropout: float = 0.2,
        support_shots: int = 5,
        trace_format: str = "direction_timing_gap",
        use_crf: bool = True,
        use_subsequence_refine: bool = True,
        bilstm_hidden: int = 256,
        encoder_blocks: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.max_len = max_len
        self.support_shots = support_shots
        self.trace_format = trace_format
        self.latent_dim = latent_dim
        self.use_crf = use_crf
        self.use_subsequence_refine = use_subsequence_refine

        enc_channels = hidden_channels[:encoder_blocks]
        self.encoder = TraceEncoder(in_channels, hidden_channels, dropout, num_blocks=encoder_blocks)
        ch = self.encoder.out_channels
        self.fusion = SupportQueryFusion(ch, support_shots=support_shots, dropout=dropout)

        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(ch, latent_dim),
            nn.GELU(),
        )

        rev_channels = list(enc_channels)[::-1]
        dec_layers = []
        for i in range(len(rev_channels) - 1):
            dec_layers.append(TransposeConvBlock(rev_channels[i], rev_channels[i + 1], dropout=dropout))
        self.decoder = nn.Sequential(*dec_layers)

        skip_dim = in_channels + rev_channels[-1]
        self.skip_proj = nn.Conv1d(ch, rev_channels[-1], kernel_size=1)
        self.local_mask = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
        )
        self.mask_head = BiLSTMCRFHead(skip_dim, hidden=bilstm_hidden, dropout=dropout, use_crf=use_crf)
        self.recon_head = nn.Conv1d(rev_channels[-1], 1, kernel_size=7, padding=3)
        self.query_only = nn.Conv1d(ch, ch, kernel_size=1)

    def _encode_support(self, support_defended: torch.Tensor) -> torch.Tensor:
        b, k, c, seq_len = support_defended.shape
        enc = self.encoder(support_defended.view(b * k, c, seq_len))
        _, c_out, l_out = enc.shape
        return enc.view(b, k, c_out, l_out)

    def _decode_mask(
        self,
        h: torch.Tensor,
        query: torch.Tensor,
        target_len: int,
        defended_len: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        enc_len = h.shape[-1]
        skip_enc = self.skip_proj(h)
        x = self.decoder(h)
        recon = self.recon_head(x).squeeze(1)
        if recon.shape[-1] != target_len:
            recon = F.interpolate(
                recon.unsqueeze(1), size=target_len, mode="linear", align_corners=False
            ).squeeze(1)

        local_logits = self.local_mask(query).squeeze(1)
        if local_logits.shape[-1] != target_len:
            local_logits = F.interpolate(
                local_logits.unsqueeze(1), size=target_len, mode="linear", align_corners=False
            ).squeeze(1)

        skip_up = F.interpolate(skip_enc, size=enc_len, mode="linear", align_corners=False)
        query_ds = F.interpolate(query, size=enc_len, mode="linear", align_corners=False)
        combined = torch.cat([query_ds, skip_up], dim=1)
        feats = combined.permute(0, 2, 1)

        valid_mask = None
        if defended_len is not None:
            ds_len = (
                defended_len.float() / self.encoder.downsample_factor
            ).ceil().long().clamp(min=1)
            positions = torch.arange(enc_len, device=feats.device)
            valid_mask = (positions.unsqueeze(0) < ds_len.unsqueeze(1)).to(feats.dtype)

        head_out = self.mask_head(feats, valid_mask)
        coarse_logits = head_out["mask_logits"]
        if coarse_logits.shape[-1] != target_len:
            coarse_logits = F.interpolate(
                coarse_logits.unsqueeze(1), size=target_len, mode="linear", align_corners=False
            ).squeeze(1)
            if "emissions" in head_out:
                em_up = F.interpolate(
                    head_out["emissions"].permute(0, 2, 1),
                    size=target_len,
                    mode="linear",
                    align_corners=False,
                ).permute(0, 2, 1)
                head_out["emissions"] = em_up

        head_out["mask_logits"] = coarse_logits + local_logits
        return {"recon": recon, **head_out}

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

        q_feat = self.encoder(query)
        if support_defended is not None and support_mask is not None:
            h = self.fusion(q_feat, self._encode_support(support_defended), support_mask)
        else:
            h = q_feat + self.query_only(q_feat)

        out = self._decode_mask(h, query, target_len, defended_len)
        out["latent"] = self.to_latent(h)
        return out

    @torch.no_grad()
    def recover(
        self,
        defended: torch.Tensor,
        defended_len: torch.Tensor | None = None,
        support_defended: Optional[torch.Tensor] = None,
        support_mask: Optional[torch.Tensor] = None,
        mask_threshold: float = 0.5,
        use_subsequence_refine: Optional[bool] = None,
        use_crf_decode: Optional[bool] = None,
        crf_keep_bias: float = 0.0,
        prototype_caches: Optional[List] = None,
        calibrate_steps: int = 6,
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        if defended.dim() == 2:
            defended = defended.unsqueeze(1)
        out = self.forward(defended, support_defended, support_mask, defended_len)
        mask_logits = out["mask_logits"]
        mask_prob = torch.sigmoid(mask_logits)

        do_refine = self.use_subsequence_refine if use_subsequence_refine is None else use_subsequence_refine
        do_crf = self.use_crf if use_crf_decode is None else use_crf_decode
        use_insertion = not do_crf and not do_refine

        if (
            use_insertion
            and support_defended is not None
            and support_mask is not None
            and defended_len is not None
        ):
            b, seq_len = mask_logits.shape
            keep_mask = torch.zeros_like(mask_logits)
            dirs_np = defended[:, 0].detach().cpu().numpy()
            logits_np = mask_logits.detach().cpu().numpy()
            lens = defended_len.detach().cpu().numpy()
            sm_np = support_mask.detach().cpu().numpy()
            sd_np = support_defended[:, :, 0].detach().cpu().numpy()
            for bi in range(b):
                valid = int(lens[bi])
                cache = None
                if prototype_caches is not None and bi < len(prototype_caches):
                    cache = prototype_caches[bi]
                if cache is not None:
                    decoded = decode_insertion_mask(
                        dirs_np[bi, :valid],
                        logits_np[bi, :valid],
                        [],
                        cache.proto_masks,
                        target_ratio=cache.target_ratio,
                        calibrate=True,
                        calibrate_steps=calibrate_steps,
                        precomputed_protos=cache.proto_dirs,
                    )
                else:
                    support_dirs = [sd_np[bi, ki] for ki in range(sd_np.shape[1])]
                    support_masks = [sm_np[bi, ki] for ki in range(sm_np.shape[1])]
                    target = _support_keep_ratio_np(sm_np[bi])
                    decoded = decode_insertion_mask(
                        dirs_np[bi, :valid],
                        logits_np[bi, :valid],
                        support_dirs,
                        support_masks,
                        target_ratio=target,
                        calibrate=True,
                        calibrate_steps=calibrate_steps,
                    )
                keep_mask[bi, :valid] = torch.from_numpy(decoded).to(mask_logits.device)
        elif (
            do_refine
            and support_defended is not None
            and support_mask is not None
            and defended_len is not None
        ):
            keep_mask = refine_batch_masks(
                defended, mask_logits, support_defended, support_mask, defended_len
            )
        elif do_crf and self.mask_head.crf is not None and defended_len is not None:
            b, seq_len = mask_logits.shape
            emissions = out["emissions"].clone()
            emissions[..., 1] -= crf_keep_bias
            valid = torch.zeros(b, seq_len, device=mask_logits.device)
            for bi in range(b):
                valid[bi, : int(defended_len[bi].item())] = 1.0
            tags = self.mask_head.crf.viterbi_decode(emissions, valid)
            keep_mask = tags.float()
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
