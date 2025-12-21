"""
Expected Return Loss for financial time series prediction.

This loss function is designed to maximize expected returns while penalizing
opportunity costs (missing profitable trades).
"""

import torch
import torch.nn as nn


class ExpectedReturnLoss(nn.Module):
    """
    Loss function based on expected returns for financial predictions.

    Computes loss as the negative expected return:
    Loss = -(λ * y^T A₁ ŷ - θ * y^T A₂ ŷ - ε * y^T A₃ ŷ)

    where:
    - λ (lambda): Horizontal barrier distance (profit/loss per correct/incorrect prediction)
    - θ (theta): Commission rate (transaction cost)
    - ε (epsilon): Opportunity cost penalty for predicting neutral when true label is actionable
    - y: One-hot encoded true label
    - ŷ: Predicted probability distribution
    - A₁: Directional returns matrix
    - A₂: Commission matrix
    - A₃: Opportunity cost matrix

    The combined return matrix is:
    [[λ-θ,  -ε,  -λ-θ],   # True=-1: correct→λ-θ, neutral→-ε, opposite→-λ-θ
     [-θ,    0,   -θ],    # True=0:  always pay commission only
     [-λ-θ, -ε,   λ-θ]]   # True=+1: opposite→-λ-θ, neutral→-ε, correct→λ-θ
    """

    def __init__(
        self,
        lambda_value: float = 1.0,
        theta_value: float = 0.0,
        epsilon_value: float = 0.0,
        reduction: str = "mean",
    ):
        """
        Initialize Expected Return Loss.

        Args:
            lambda_value: Horizontal barrier distance (profit/loss magnitude)
            theta_value: Commission rate (transaction cost)
            epsilon_value: Opportunity cost penalty for missing trades
            reduction: Specifies reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.lambda_value = lambda_value
        self.theta_value = theta_value
        self.epsilon_value = epsilon_value
        self.reduction = reduction

        # Precompute combined return matrix A = λ*A₁ - θ*A₂ - ε*A₃
        # This combines directional returns, commissions, and opportunity costs
        # into a single matrix for efficient computation
        self.register_buffer(
            "A",
            torch.tensor(
                [
                    [lambda_value - theta_value, -epsilon_value, -lambda_value - theta_value],
                    [-theta_value, 0.0, -theta_value],
                    [-lambda_value - theta_value, -epsilon_value, lambda_value - theta_value],
                ],
                dtype=torch.float32,
            ),
        )

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the expected return loss.

        Args:
            predictions: Model predictions (logits) of shape (N, 3)
            targets: True labels of shape (N,) with values {0, 1, 2}

        Returns:
            Loss value (scalar if reduction='mean' or 'sum', tensor of shape (N,) if 'none')
        """
        # Get probabilities
        probs = torch.softmax(predictions, dim=1)  # (N, 3)

        # Create one-hot encoded true labels
        N = targets.size(0)
        y_onehot = torch.zeros(N, 3, device=predictions.device, dtype=predictions.dtype)
        y_onehot.scatter_(1, targets.unsqueeze(1), 1.0)  # (N, 3)

        # Compute expected return: y^T A ŷ for each sample
        # A already contains λ*A₁ - θ*A₂ - ε*A₃, so this is a single operation
        # Shape: (N, 3) @ (3, 3) @ (N, 3).T → (N,)
        expected_return = torch.einsum("ni,ij,nj->n", y_onehot, self.A, probs)

        # Loss = -expected_return (we want to maximize ER, so minimize -ER)
        loss = -expected_return

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss

    def extra_repr(self) -> str:
        """String representation of the loss parameters."""
        return (
            f"lambda={self.lambda_value}, theta={self.theta_value}, "
            f"epsilon={self.epsilon_value}, reduction={self.reduction}"
        )
