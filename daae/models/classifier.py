"""Simple 1D-CNN website fingerprinting classifier for evaluation."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class WFClassifier(nn.Module):
    """DF-style 1D CNN classifier over direction/size sequences."""

    def __init__(self, num_classes: int, in_channels: int = 1, max_len: int = 5000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=8, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=4),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=4),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=8, stride=4, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.classifier(self.features(x))


def classifier_loss(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss = F.cross_entropy(logits, labels)
    preds = logits.argmax(dim=-1)
    acc = (preds == labels).float().mean().item()
    return loss, {"loss": float(loss.item()), "acc": acc}
