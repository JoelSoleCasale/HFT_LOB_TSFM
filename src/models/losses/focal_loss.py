"""
Focal Loss implementation for handling class imbalance.
"""

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Focal Loss implementation for handling class imbalance."""

    def __init__(self, alpha: float | list = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.multi_class = False
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        if isinstance(self.alpha, list):
            self.alpha = torch.tensor(self.alpha, dtype=torch.float32)
            self.multi_class = True

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        if self.multi_class:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets)
            focal_loss = at * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
