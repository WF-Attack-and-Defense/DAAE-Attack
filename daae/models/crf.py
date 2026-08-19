"""Linear-chain CRF for per-packet keep/dummy sequence labeling."""

from __future__ import annotations

import torch
import torch.nn as nn


class BinaryLinearCRF(nn.Module):
    """
    Binary CRF (0=dummy, 1=keep) enforcing smooth run-length structure.

    Palette and similar defenses inject long dummy bursts; transition scores
    favor staying in the same state (dummy-run prior from super-matrix padding).
    """

    def __init__(self, dummy_run_bias: float = 0.5, keep_run_bias: float = 0.3):
        super().__init__()
        self.trans = nn.Parameter(torch.zeros(2, 2))
        with torch.no_grad():
            self.trans[0, 0] = dummy_run_bias
            self.trans[1, 1] = keep_run_bias
            self.trans[0, 1] = -0.1
            self.trans[1, 0] = -0.1

    def _score_sentence(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """emissions (B,L,2), tags (B,L), mask (B,L) -> (B,) scores."""
        b, seq_len, _ = emissions.shape
        emit = emissions.gather(2, tags.unsqueeze(-1)).squeeze(-1)
        emit = (emit * mask).sum(dim=1)

        trans_score = torch.zeros(b, device=emissions.device)
        for t in range(1, seq_len):
            prev = tags[:, t - 1]
            cur = tags[:, t]
            step = self.trans[prev, cur]
            trans_score = trans_score + step * mask[:, t]
        return emit + trans_score

    def neg_log_likelihood(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        gold = self._score_sentence(emissions, tags, mask)
        log_z = self._log_partition(emissions, mask)
        nll = log_z - gold
        return (nll * (mask[:, 0] > 0).float()).sum() / (mask[:, 0] > 0).sum().clamp(min=1.0)

    def _log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, seq_len, num_tags = emissions.shape
        alpha = emissions[:, 0]
        for t in range(1, seq_len):
            emit = emissions[:, t].unsqueeze(1)
            trans = self.trans.unsqueeze(0)
            scores = alpha.unsqueeze(2) + trans + emit
            alpha = torch.logsumexp(scores, dim=1)
            alpha = torch.where(mask[:, t].unsqueeze(1) > 0, alpha, alpha)
        return torch.logsumexp(alpha, dim=1)

    @torch.no_grad()
    def viterbi_decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, seq_len, num_tags = emissions.shape
        backpointers = []
        alpha = emissions[:, 0]
        for t in range(1, seq_len):
            emit = emissions[:, t].unsqueeze(1)
            trans = self.trans.unsqueeze(0)
            scores = alpha.unsqueeze(2) + trans
            alpha, bp = scores.max(dim=1)
            alpha = alpha + emit.squeeze(1)
            backpointers.append(bp)
            inactive = mask[:, t] <= 0
            if inactive.any():
                alpha = torch.where(inactive.unsqueeze(1), alpha, alpha)

        tags = torch.zeros(b, seq_len, dtype=torch.long, device=emissions.device)
        tags[:, -1] = alpha.argmax(dim=1)
        for t in range(seq_len - 2, -1, -1):
            tags[:, t] = backpointers[t].gather(1, tags[:, t + 1].unsqueeze(1)).squeeze(1)
        tags = tags * (mask > 0).long()
        return tags
