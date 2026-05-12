import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

from config import CFG
from main_utils import normalize_rows, require_split_artifacts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("STEP03_TEST_CLUSTERING")


def similarity_to_distance(similarity: np.ndarray) -> np.ndarray:
    similarity = np.maximum(similarity, 0.0)
    max_similarity = float(np.max(similarity))
    if max_similarity <= 0.0:
        return np.ones_like(similarity, dtype=np.float64)
    distance = (max_similarity - similarity) / max_similarity
    np.fill_diagonal(distance, 0.0)
    return distance.astype(np.float64)


def relabel_noise_as_singletons(labels: np.ndarray) -> np.ndarray:
    labels = labels.copy()
    noise_indices = np.flatnonzero(labels == -1)
    next_label = int(labels.max()) + 1 if len(labels) else 0
    for offset, index in enumerate(noise_indices):
        labels[index] = next_label + offset
    return labels


def cluster_dataset(
    dataset_name: str,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    min_cluster_size: int = 2,
) -> tuple[pd.DataFrame, dict[str, object]]:
    similarity = embeddings @ embeddings.T
    distance = similarity_to_distance(similarity)

    clustering = HDBSCAN(min_cluster_size=min_cluster_size, metric="precomputed")
    raw_labels = clustering.fit_predict(distance)
    labels = relabel_noise_as_singletons(raw_labels)

    assignments = pd.DataFrame(
        {
            "embedding_index": metadata["embedding_index"].to_numpy(dtype=int),
            "image_id": metadata["image_id"].to_numpy(dtype=int),
            "dataset": dataset_name,
            "cluster_number": labels.astype(int),
            "cluster": [f"cluster_{dataset_name}_{int(label)}" for label in labels],
            "hdbscan_raw_label": raw_labels.astype(int),
        }
    )

    cluster_sizes = assignments["cluster"].value_counts()
    summary = {
        "dataset": dataset_name,
        "samples": int(len(assignments)),
        "clusters": int(cluster_sizes.size),
        "noise_singletons": int((raw_labels == -1).sum()),
        "largest_cluster_size": int(cluster_sizes.max()) if len(cluster_sizes) else 0,
        "median_cluster_size": float(cluster_sizes.median()) if len(cluster_sizes) else 0.0,
        "min_cluster_size": int(min_cluster_size),
    }
    return assignments, summary


def cluster_test_embeddings(cfg: CFG) -> None:
    embeddings_dir = Path(cfg.embeddings_output_dir)
    output_dir = Path(cfg.final_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    require_split_artifacts(
        embeddings_dir,
        ["test"],
        step_name="Step 03 test clustering",
        producer_step="step 02 embedding extraction",
    )

    embeddings = normalize_rows(np.load(embeddings_dir / "test_embeddings.npy"))
    metadata = pd.read_csv(embeddings_dir / "test_metadata.csv")
    if len(embeddings) != len(metadata):
        raise ValueError(
            "Test embeddings and metadata length mismatch: "
            f"{len(embeddings)} vs {len(metadata)}"
        )

    all_assignments: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    for dataset_name in sorted(metadata["dataset"].astype(str).unique()):
        dataset_mask = metadata["dataset"].astype(str).eq(dataset_name).to_numpy()
        dataset_embeddings = embeddings[dataset_mask]
        dataset_metadata = metadata.loc[dataset_mask].reset_index(drop=True)

        logger.info(
            "Clustering %s test images for %s",
            len(dataset_metadata),
            dataset_name,
        )
        assignments, summary = cluster_dataset(
            dataset_name=dataset_name,
            embeddings=dataset_embeddings,
            metadata=dataset_metadata,
            min_cluster_size=2,
        )
        all_assignments.append(assignments)
        summaries.append(summary)
        logger.info(
            "%s | clusters=%d | noise_singletons=%d | largest_cluster=%d",
            dataset_name,
            summary["clusters"],
            summary["noise_singletons"],
            summary["largest_cluster_size"],
        )

    assignments = (
        pd.concat(all_assignments, ignore_index=True)
        .sort_values("embedding_index")
        .reset_index(drop=True)
    )
    submission = assignments[["image_id", "cluster"]].copy()

    assignments_path = output_dir / "test_clustering_assignments.csv"
    submission_path = output_dir / "test_clustering_submission.csv"
    summary_path = output_dir / "test_clustering_summary.json"
    report_path = output_dir / "test_clustering_report.txt"

    assignments.to_csv(assignments_path, index=False)
    submission.to_csv(submission_path, index=False)

    summary = {
        "samples": int(len(assignments)),
        "clusters": int(assignments["cluster"].nunique()),
        "datasets": summaries,
        "method": "HDBSCAN on per-dataset test-test cosine distance",
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("Test embedding clustering\n")
        handle.write("=========================\n\n")
        handle.write(f"Samples: {summary['samples']}\n")
        handle.write(f"Clusters: {summary['clusters']}\n")
        handle.write(f"Method: {summary['method']}\n\n")
        handle.write("Per-dataset results\n")
        handle.write("-------------------\n")
        for dataset_summary in summaries:
            handle.write(
                f"{dataset_summary['dataset']}: "
                f"samples={dataset_summary['samples']}, "
                f"clusters={dataset_summary['clusters']}, "
                f"noise_singletons={dataset_summary['noise_singletons']}, "
                f"largest_cluster_size={dataset_summary['largest_cluster_size']}, "
                f"median_cluster_size={dataset_summary['median_cluster_size']:.1f}\n"
            )

    logger.info("Saved test clustering assignments to %s", assignments_path)
    logger.info("Saved test clustering submission to %s", submission_path)
    logger.info("Saved test clustering summary to %s", summary_path)
    logger.info("Saved test clustering report to %s", report_path)


if __name__ == "__main__":
    cfg = CFG("config.yaml")
    cluster_test_embeddings(cfg)
