import json
import logging
from pathlib import Path
import re

import numpy as np
import pandas as pd

from config import CFG
from main_utils import normalize_rows

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("STEP11_TRAIN_VERIFY")

CLUSTER_PREFIXES = {
    "LynxID2025",
    "SalamanderID2025",
    "SeaTurtleID2022",
    "TexasHornedLizards",
}


def format_cluster(value: object) -> str:
    identity = str(value).strip()
    match = re.match(r"^([^_]+)_(?:.*?)(\d+)$", identity)
    if match and match.group(1) in CLUSTER_PREFIXES:
        return f"cluster_{match.group(1)}_{int(match.group(2))}"
    raise ValueError(f"Unsupported train identity format: '{identity}'")


def topk_indices(similarities: np.ndarray, k: int) -> np.ndarray:
    k = min(k, similarities.shape[1])
    if k == similarities.shape[1]:
        indices = np.argsort(similarities, axis=1)[:, ::-1]
    else:
        partition = np.argpartition(similarities, -k, axis=1)[:, -k:]
        scores = np.take_along_axis(similarities, partition, axis=1)
        order = np.argsort(scores, axis=1)[:, ::-1]
        indices = np.take_along_axis(partition, order, axis=1)
    return indices


def evaluate_dataset(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    dataset_name: str,
    top_k: int = 5,
    batch_size: int = 512,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    dataset_mask = metadata["dataset"].astype(str).eq(dataset_name).to_numpy()
    dataset_indices = np.flatnonzero(dataset_mask)
    dataset_embeddings = embeddings[dataset_indices]
    dataset_metadata = metadata.iloc[dataset_indices].reset_index(drop=True)
    identities = dataset_metadata["identity"].astype(str).to_numpy()
    image_ids = dataset_metadata["image_id"].to_numpy()
    identity_counts = pd.Series(identities).value_counts()
    verifiable = np.array([identity_counts[identity] > 1 for identity in identities], dtype=bool)

    rows: list[dict[str, object]] = []

    for start in range(0, len(dataset_embeddings), batch_size):
        end = min(start + batch_size, len(dataset_embeddings))
        query_embeddings = dataset_embeddings[start:end]
        similarities = query_embeddings @ dataset_embeddings.T

        local_query_indices = np.arange(start, end)
        similarities[np.arange(end - start), local_query_indices] = -np.inf

        nearest = topk_indices(similarities, top_k)

        for offset, neighbor_indices in enumerate(nearest):
            local_index = start + offset
            nearest_index = int(neighbor_indices[0])
            top_identities = identities[neighbor_indices]
            correct_top1 = bool(identities[local_index] == identities[nearest_index])
            correct_topk = bool((top_identities == identities[local_index]).any())

            rows.append(
                {
                    "embedding_index": int(dataset_metadata.iloc[local_index]["embedding_index"]),
                    "image_id": int(image_ids[local_index]),
                    "dataset": dataset_name,
                    "identity": identities[local_index],
                    "expected_cluster": format_cluster(identities[local_index]),
                    "nearest_image_id": int(image_ids[nearest_index]),
                    "nearest_identity": identities[nearest_index],
                    "predicted_cluster": format_cluster(identities[nearest_index]),
                    "top1_similarity": float(similarities[offset, nearest_index]),
                    f"top{top_k}_contains_same_identity": correct_topk,
                    "top1_correct": correct_top1,
                    "verifiable_identity": bool(verifiable[local_index]),
                }
            )

    result = pd.DataFrame(rows)
    verifiable_result = result[result["verifiable_identity"]]
    summary = {
        "dataset": dataset_name,
        "samples": int(len(result)),
        "identities": int(identity_counts.size),
        "singleton_samples": int((~verifiable).sum()),
        "top1_accuracy_all": float(result["top1_correct"].mean()) if len(result) else 0.0,
        f"top{top_k}_accuracy_all": float(result[f"top{top_k}_contains_same_identity"].mean()) if len(result) else 0.0,
        "top1_accuracy_verifiable": (
            float(verifiable_result["top1_correct"].mean()) if len(verifiable_result) else 0.0
        ),
        f"top{top_k}_accuracy_verifiable": (
            float(verifiable_result[f"top{top_k}_contains_same_identity"].mean())
            if len(verifiable_result)
            else 0.0
        ),
    }
    return rows, summary


def verify_train_clustering(cfg: CFG) -> None:
    embeddings_dir = Path(cfg.embeddings_output_dir)
    output_dir = Path(cfg.final_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = normalize_rows(np.load(embeddings_dir / "train_embeddings.npy"))
    metadata = pd.read_csv(embeddings_dir / "train_metadata.csv")
    if len(embeddings) != len(metadata):
        raise ValueError(
            "Train embeddings and metadata length mismatch: "
            f"{len(embeddings)} vs {len(metadata)}"
        )

    all_rows: list[dict[str, object]] = []
    dataset_summaries: list[dict[str, object]] = []

    for dataset_name in sorted(metadata["dataset"].astype(str).unique()):
        logger.info("Verifying train nearest-neighbor clustering for %s", dataset_name)
        rows, summary = evaluate_dataset(
            embeddings=embeddings,
            metadata=metadata,
            dataset_name=dataset_name,
            top_k=5,
        )
        all_rows.extend(rows)
        dataset_summaries.append(summary)
        logger.info(
            "%s | top1_verifiable=%.4f | top5_verifiable=%.4f | samples=%d",
            dataset_name,
            summary["top1_accuracy_verifiable"],
            summary["top5_accuracy_verifiable"],
            summary["samples"],
        )

    results = pd.DataFrame(all_rows).sort_values("embedding_index").reset_index(drop=True)
    verifiable_results = results[results["verifiable_identity"]]
    summary = {
        "samples": int(len(results)),
        "verifiable_samples": int(len(verifiable_results)),
        "top1_accuracy_all": float(results["top1_correct"].mean()) if len(results) else 0.0,
        "top5_accuracy_all": (
            float(results["top5_contains_same_identity"].mean()) if len(results) else 0.0
        ),
        "top1_accuracy_verifiable": (
            float(verifiable_results["top1_correct"].mean()) if len(verifiable_results) else 0.0
        ),
        "top5_accuracy_verifiable": (
            float(verifiable_results["top5_contains_same_identity"].mean())
            if len(verifiable_results)
            else 0.0
        ),
        "datasets": dataset_summaries,
    }

    results_path = output_dir / "train_verification_nearest_neighbor.csv"
    summary_path = output_dir / "train_verification_summary.json"
    report_path = output_dir / "train_verification_report.txt"
    results.to_csv(results_path, index=False)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("Train nearest-neighbor verification\n")
        handle.write("===================================\n\n")
        handle.write(f"Samples: {summary['samples']}\n")
        handle.write(f"Verifiable samples: {summary['verifiable_samples']}\n")
        handle.write(f"Top-1 accuracy, all samples: {summary['top1_accuracy_all']:.4f}\n")
        handle.write(f"Top-5 accuracy, all samples: {summary['top5_accuracy_all']:.4f}\n")
        handle.write(f"Top-1 accuracy, verifiable samples: {summary['top1_accuracy_verifiable']:.4f}\n")
        handle.write(f"Top-5 accuracy, verifiable samples: {summary['top5_accuracy_verifiable']:.4f}\n\n")
        handle.write("Per-dataset results\n")
        handle.write("-------------------\n")
        for dataset_summary in dataset_summaries:
            handle.write(
                f"{dataset_summary['dataset']}: "
                f"samples={dataset_summary['samples']}, "
                f"identities={dataset_summary['identities']}, "
                f"singleton_samples={dataset_summary['singleton_samples']}, "
                f"top1_verifiable={dataset_summary['top1_accuracy_verifiable']:.4f}, "
                f"top5_verifiable={dataset_summary['top5_accuracy_verifiable']:.4f}\n"
            )

    logger.info("Saved train verification rows to %s", results_path)
    logger.info("Saved train verification summary to %s", summary_path)
    logger.info("Saved train verification report to %s", report_path)
    logger.info(
        "Overall train verification | top1_verifiable=%.4f | top5_verifiable=%.4f",
        summary["top1_accuracy_verifiable"],
        summary["top5_accuracy_verifiable"],
    )


if __name__ == "__main__":
    cfg = CFG("config.yaml")
    verify_train_clustering(cfg)
