import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=1)
        logits = features @ features.T
        logits = logits / self.temperature

        batch_size = features.shape[0]
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
        logits = logits.masked_fill(self_mask, float("-inf"))

        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        positive_counts = positive_mask.sum(dim=1)
        valid_anchor_mask = positive_counts > 0
        masked_log_prob = torch.where(
            positive_mask,
            log_prob,
            torch.zeros_like(log_prob),
        )
        positive_log_prob = masked_log_prob.sum(dim=1) / positive_counts.clamp_min(1)
        return -positive_log_prob[valid_anchor_mask].mean()


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"Expected features with shape [N, D], got {tuple(features.shape)}.")
        if labels.ndim != 1:
            raise ValueError(f"Expected labels with shape [N], got {tuple(labels.shape)}.")
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Features and labels must have matching first dimension, got "
                f"{features.shape[0]} and {labels.shape[0]}."
            )

        valid_labels = labels >= 0
        features = features[valid_labels]
        labels = labels[valid_labels]
        if features.shape[0] < 2:
            return features.sum() * 0.0

        features = F.normalize(features, dim=1)
        distances = 1.0 - features @ features.T

        batch_size = features.shape[0]
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
        negative_mask = labels[:, None].ne(labels[None, :])
        valid_anchor_mask = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        if not valid_anchor_mask.any():
            return features.sum() * 0.0

        hardest_positive = distances.masked_fill(~positive_mask, float("-inf")).max(dim=1).values
        hardest_negative = distances.masked_fill(~negative_mask, float("inf")).min(dim=1).values
        losses = F.relu(hardest_positive - hardest_negative + self.margin)
        return losses[valid_anchor_mask].mean()
