"""Distance-aware loss utilities for pitch classification.
Provides helper functions for class-distance penalties and soft target distributions.
"""

import torch
import torch.nn.functional as F
from collections.abc import Sequence


def make_xrv_distance_matrix(
    class_values: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Build a pairwise absolute-distance matrix from ordered class values."""

    class_values_tensor = torch.tensor(class_values, dtype=torch.float32)

    distance_matrix = torch.abs(
        class_values_tensor[:, None] - class_values_tensor[None, :]
    )

    return distance_matrix


def make_soft_target_matrix(
    distance_matrix: torch.Tensor,
    tau: float = 0.05,
) -> torch.Tensor:
    """Convert class distances into row-normalized soft target distributions."""

    # Smaller tau creates sharper targets; larger tau spreads probability to nearby classes.
    raw_scores = torch.exp(-(distance_matrix**2) / tau)

    soft_targets = raw_scores / raw_scores.sum(dim=1, keepdim=True)

    return soft_targets


def hard_ce_plus_expected_distance_loss(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    distance_matrix: torch.Tensor,
    lambda_distance: float = 0.20,
) -> torch.Tensor:
    """Combine hard-label cross entropy with an expected class-distance penalty."""

    ce_loss = F.cross_entropy(logits, y_true)

    probs = torch.softmax(logits, dim=1)

    row_distances = distance_matrix[y_true]

    expected_distance = (probs * row_distances).sum(dim=1).mean()

    total_loss = ce_loss + lambda_distance * expected_distance

    return total_loss


def soft_label_cross_entropy_loss(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    soft_target_matrix: torch.Tensor,
) -> torch.Tensor:
    """Compute cross entropy against distance-aware soft target distributions."""

    log_probs = F.log_softmax(logits, dim=1)

    q = soft_target_matrix[y_true]

    loss = -(q * log_probs).sum(dim=1).mean()

    return loss
