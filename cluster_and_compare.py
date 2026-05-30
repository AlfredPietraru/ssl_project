import logging
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
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

    def plot_projection(
        self,
        output_path: str | Path,
        *,
        title: str,
        cluster_labels: np.ndarray | None = None,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        embeddings = self.embeddings.numpy().astype(np.float32)
        embeddings = normalize_rows(embeddings)
        projection = PCA(n_components=2, random_state=42).fit_transform(embeddings)

        if cluster_labels is None:
            color_values = self.labels.numpy().astype(int)
            colorbar_label = "True identity id"
        else:
            if len(cluster_labels) != len(self.labels):
                raise ValueError(
                    "cluster_labels must have the same number of rows as embeddings."
                )
            color_values = np.asarray(cluster_labels, dtype=int)
            colorbar_label = "Predicted cluster id"

        unique_values, color_ids = np.unique(color_values, return_inverse=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            projection[:, 0],
            projection[:, 1],
            c=color_ids,
            cmap="tab20",
            s=20,
            alpha=0.75,
            edgecolors="none",
        )
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)

        if len(unique_values) <= 30:
            colorbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
            colorbar.set_label(colorbar_label)

        fig.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved embedding projection to %s", output_path)
        return output_path
