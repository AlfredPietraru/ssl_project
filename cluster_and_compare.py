import logging
from collections import Counter

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score

from main_utils import normalize_rows
import torch
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("STEP03_TEST_CLUSTERING")


class ClusterAndCompare:
    def __init__(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        metadata: list[dict[str, Any]],
    ) -> None:
        self.embeddings = embeddings.detach().cpu()
        self.labels = labels.detach().cpu()
        self.metadata = metadata
        self.identity_ids = {
            int(item["identity"])
            for item in self.metadata
        }

        if self.embeddings.ndim != 2:
            raise ValueError(f"Expected embeddings with shape [N, D], got {tuple(self.embeddings.shape)}.")
        if self.labels.ndim != 1:
            raise ValueError(f"Expected labels with shape [N], got {tuple(self.labels.shape)}.")
        if self.embeddings.shape[0] != self.labels.shape[0]:
            raise ValueError(
                "Embeddings and labels must have the same number of rows, got "
                f"{self.embeddings.shape[0]} and {self.labels.shape[0]}."
            )
        if not set(self.labels.numpy().astype(int).tolist()).issubset(self.identity_ids):
            raise ValueError("Labels contain identity ids that are not present in metadata.")

    def map_clusters_to_identities(
        self,
        cluster_labels: np.ndarray,
    ) -> dict[int, int]:
        label_values = self.labels.numpy().astype(int)
        cluster_to_identity: dict[int, int] = {}

        for cluster_id in sorted(set(cluster_labels.tolist())):
            member_indices = np.flatnonzero(cluster_labels == cluster_id)
            if len(member_indices) == 0:
                continue
            member_labels = label_values[member_indices]
            dominant_label, _ = Counter(member_labels.tolist()).most_common(1)[0]
            cluster_to_identity[int(cluster_id)] = int(dominant_label)
        return cluster_to_identity

    def compute_accuracy(
        self,
        cluster_labels: np.ndarray,
        cluster_to_identity: dict[int, int],
    ) -> float:
        true_identities = [int(label) for label in self.labels.tolist()]
        predicted_identities = [
            cluster_to_identity[int(cluster_id)]
            for cluster_id in cluster_labels.tolist()
        ]
        correct = sum(
            int(true_identity == predicted_identity)
            for true_identity, predicted_identity in zip(true_identities, predicted_identities)
        )
        return correct / max(len(true_identities), 1)

    def compare(
        self,
        min_cluster_size: int = 2,
        min_samples: int = 2,
    ) -> dict[str, Any]:
        embeddings = self.embeddings.numpy().astype(np.float32)
        embeddings = normalize_rows(embeddings)
        similarity = embeddings @ embeddings.T
        distance = 1.0 - similarity
        np.fill_diagonal(distance, 0.0)

        clustering = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="precomputed",
        )
        cluster_labels = clustering.fit_predict(distance)
        cluster_to_identity = self.map_clusters_to_identities(cluster_labels)
        accuracy = self.compute_accuracy(cluster_labels, cluster_to_identity)
        ari = adjusted_rand_score(self.labels.numpy(), cluster_labels)

        return {
            "accuracy": float(accuracy),
            "adjusted_rand_index": float(ari),
            "num_samples": int(len(cluster_labels)),
            "num_clusters": int(len(set(cluster_labels.tolist()))),
            "num_noise": int(np.sum(cluster_labels == -1)),
            "cluster_to_identity": cluster_to_identity,
            "cluster_labels": cluster_labels,
        }

    def sweep_eps(
        self,
        min_samples: int = 2,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        min_cluster_sizes = [2, 3]
        for min_cluster_size in min_cluster_sizes:
            result = self.compare(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )
            result["min_cluster_size"] = int(min_cluster_size)
            results.append(result)
        
        for result in results:
            logger.info(
            (
                "HDBSCAN sweep | min_cluster_size=%d | accuracy=%.4f | ari=%.4f | "
                "num_clusters=%d | num_noise=%d"
            ),
            result["min_cluster_size"],
            result["accuracy"],
            result["adjusted_rand_index"],
            result["num_clusters"],
            result["num_noise"],
        )
        return results
