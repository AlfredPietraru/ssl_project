import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        self.temperature = temperature

    def _compute_terms(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = F.normalize(features, dim=1)
        logits = features @ features.T
        logits = logits / self.temperature

        batch_size = features.shape[0]
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
        negative_mask = labels[:, None].ne(labels[None, :])
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

        similarity = features @ features.T
        positive_similarities = similarity[positive_mask]
        negative_similarities = similarity[negative_mask]

        return {
            "positive_log_prob": positive_log_prob,
            "valid_anchor_mask": valid_anchor_mask,
            "positive_counts": positive_counts,
            "positive_similarities": positive_similarities,
            "negative_similarities": negative_similarities,
        }

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        terms = self._compute_terms(features, labels)
        return -terms["positive_log_prob"][terms["valid_anchor_mask"]].mean()

    def diagnostics(self, features: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            terms = self._compute_terms(features, labels)
            valid_anchor_mask = terms["valid_anchor_mask"]
            positive_counts = terms["positive_counts"]
            positive_similarities = terms["positive_similarities"]
            negative_similarities = terms["negative_similarities"]

            valid_anchor_ratio = float(valid_anchor_mask.float().mean().item())
            mean_positive_count = float(positive_counts[valid_anchor_mask].float().mean().item()) if valid_anchor_mask.any() else 0.0
            mean_positive_similarity = float(positive_similarities.mean().item()) if positive_similarities.numel() > 0 else 0.0
            mean_negative_similarity = float(negative_similarities.mean().item()) if negative_similarities.numel() > 0 else 0.0

            return {
                "valid_anchor_ratio": valid_anchor_ratio,
                "mean_positive_count": mean_positive_count,
                "mean_positive_similarity": mean_positive_similarity,
                "mean_negative_similarity": mean_negative_similarity,
                "similarity_gap": mean_positive_similarity - mean_negative_similarity,
            }


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def _compute_terms(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
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
        hardest_positive = distances.masked_fill(~positive_mask, float("-inf")).max(dim=1).values
        hardest_negative = distances.masked_fill(~negative_mask, float("inf")).min(dim=1).values
        losses = F.relu(hardest_positive - hardest_negative + self.margin)
        return {
            "features": features,
            "distances": distances,
            "positive_mask": positive_mask,
            "negative_mask": negative_mask,
            "valid_anchor_mask": valid_anchor_mask,
            "hardest_positive": hardest_positive,
            "hardest_negative": hardest_negative,
            "losses": losses,
        }

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        terms = self._compute_terms(features, labels)
        valid_anchor_mask = terms["valid_anchor_mask"]
        if not valid_anchor_mask.any():
            return terms["features"].sum() * 0.0
        losses = terms["losses"]
        return losses[valid_anchor_mask].mean()

    def diagnostics(self, features: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            terms = self._compute_terms(features, labels)
            valid_anchor_mask = terms["valid_anchor_mask"]
            distances = terms["distances"]
            positive_mask = terms["positive_mask"]
            negative_mask = terms["negative_mask"]
            hardest_positive = terms["hardest_positive"]
            hardest_negative = terms["hardest_negative"]
            losses = terms["losses"]

            valid_anchor_ratio = float(valid_anchor_mask.float().mean().item())
            mean_positive_count = float(positive_mask.sum(dim=1)[valid_anchor_mask].float().mean().item()) if valid_anchor_mask.any() else 0.0
            positive_distances = distances[positive_mask]
            negative_distances = distances[negative_mask]
            mean_positive_distance = float(positive_distances.mean().item()) if positive_distances.numel() > 0 else 0.0
            mean_negative_distance = float(negative_distances.mean().item()) if negative_distances.numel() > 0 else 0.0
            mean_hardest_positive = float(hardest_positive[valid_anchor_mask].mean().item()) if valid_anchor_mask.any() else 0.0
            mean_hardest_negative = float(hardest_negative[valid_anchor_mask].mean().item()) if valid_anchor_mask.any() else 0.0
            active_triplet_ratio = float((losses[valid_anchor_mask] > 0).float().mean().item()) if valid_anchor_mask.any() else 0.0

            return {
                "valid_anchor_ratio": valid_anchor_ratio,
                "mean_positive_count": mean_positive_count,
                "mean_positive_distance": mean_positive_distance,
                "mean_negative_distance": mean_negative_distance,
                "distance_gap": mean_negative_distance - mean_positive_distance,
                "mean_hardest_positive": mean_hardest_positive,
                "mean_hardest_negative": mean_hardest_negative,
                "hard_margin_gap": mean_hardest_negative - mean_hardest_positive,
                "active_triplet_ratio": active_triplet_ratio,
            }
