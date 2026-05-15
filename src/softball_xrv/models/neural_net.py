"""Neural network architecture for pitch classification.
Defines a configurable feed-forward classifier that maps input features to raw class logits.
"""

import torch.nn as nn
from collections.abc import Sequence
import torch


class PitchNN(nn.Module):
    """Configurable multilayer perceptron for pitch class prediction."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.20,
        use_layer_norm: bool = True,
    ) -> None:
        """Initialize hidden layers, normalization, dropout, and output projection."""

        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")

        if num_classes <= 1:
            raise ValueError(f"num_classes must be greater than 1, got {num_classes}")

        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        # Build the hidden feature extractor from the configured layer sizes.
        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError(
                    f"hidden dimensions must be positive, got {hidden_dim}"
                )

            layers.append(nn.Linear(previous_dim, hidden_dim))

            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            previous_dim = hidden_dim

        # Final layer returns logits; activation is handled by the loss function.
        layers.append(nn.Linear(previous_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits for one batch of input features."""

        return self.net(x)
