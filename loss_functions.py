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