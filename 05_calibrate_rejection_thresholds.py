import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import CFG
from main_utils import normalize_rows, require_split_artifacts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CALIBRATE_THRESHOLDS")


def load_split(output_dir: Path, name: str) -> tuple[np.ndarray, pd.DataFrame]:
    embeddings_path = output_dir / f"{name}_embeddings.npy"
    metadata_path = output_dir / f"{name}_metadata.csv"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: '{embeddings_path}'")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: '{metadata_path}'")

    embeddings = np.load(embeddings_path)
    metadata = pd.read_csv(metadata_path)

    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Split '{name}' has mismatched embeddings/metadata lengths: "
            f"{len(embeddings)} vs {len(metadata)}"
        )

    return normalize_rows(embeddings), metadata


def cosine_similarity_matrix(queries: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    return queries @ gallery.T


def score_queries_against_gallery(
    query_embeddings: np.ndarray,
    query_metadata: pd.DataFrame,
    gallery_prototypes: np.ndarray,
    gallery_prototype_metadata: pd.DataFrame,
    gallery_images: np.ndarray,
    gallery_image_metadata: pd.DataFrame,
    split_name: str,
    query_should_be_known: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for index in range(len(query_embeddings)):
        query_embedding = query_embeddings[index:index + 1]
        query_row = query_metadata.iloc[index]

        if "species" in query_metadata.columns and "species" in gallery_prototype_metadata.columns:
            species_value = query_row.get("species")
            prototype_mask = gallery_prototype_metadata["species"].eq(species_value).to_numpy()
            image_mask = gallery_image_metadata["species"].eq(species_value).to_numpy()
        else:
            prototype_mask = np.ones(len(gallery_prototypes), dtype=bool)
            image_mask = np.ones(len(gallery_images), dtype=bool)

        if not prototype_mask.any():
            prototype_mask = np.ones(len(gallery_prototypes), dtype=bool)
        if not image_mask.any():
            image_mask = np.ones(len(gallery_images), dtype=bool)

        prototype_similarities = cosine_similarity_matrix(
            query_embedding,
            gallery_prototypes[prototype_mask],
        )[0]
        image_similarities = cosine_similarity_matrix(
            query_embedding,
            gallery_images[image_mask],
        )[0]

        sorted_proto = np.sort(prototype_similarities)
        top1_similarity = float(sorted_proto[-1])
        top2_similarity = float(sorted_proto[-2]) if len(sorted_proto) > 1 else float("-inf")
        margin = float(top1_similarity - top2_similarity) if np.isfinite(top2_similarity) else float("inf")

        best_proto_local_index = int(np.argmax(prototype_similarities))
        best_image_local_index = int(np.argmax(image_similarities))

        filtered_proto_metadata = gallery_prototype_metadata.loc[prototype_mask].reset_index(drop=True)
        filtered_image_metadata = gallery_image_metadata.loc[image_mask].reset_index(drop=True)

        row: dict[str, object] = {
            "split": split_name,
            "should_be_known": bool(query_should_be_known),
            "query_index": int(index),
            "query_identity": str(query_row["identity"]),
            "top1_similarity": top1_similarity,
            "top2_similarity": top2_similarity,
            "top1_top2_margin": margin,
            "best_prototype_identity": str(filtered_proto_metadata.iloc[best_proto_local_index]["identity"]),
            "best_image_identity": str(filtered_image_metadata.iloc[best_image_local_index]["identity"]),
            "best_image_similarity": float(image_similarities[best_image_local_index]),
        }

        for candidate in ["species", "dataset", "image_id", "path"]:
            if candidate in query_metadata.columns:
                row[f"query_{candidate}"] = query_row[candidate]

        rows.append(row)

    return pd.DataFrame(rows)


def compute_classification_metrics(actual_known: np.ndarray, predicted_known: np.ndarray) -> dict[str, float]:
    tp = int(np.logical_and(predicted_known, actual_known).sum())
    tn = int(np.logical_and(~predicted_known, ~actual_known).sum())
    fp = int(np.logical_and(predicted_known, ~actual_known).sum())
    fn = int(np.logical_and(~predicted_known, actual_known).sum())

    known_recall = tp / max(tp + fn, 1)
    unknown_recall = tn / max(tn + fp, 1)
    known_precision = tp / max(tp + fp, 1)
    accuracy = (tp + tn) / max(len(actual_known), 1)
    balanced_accuracy = 0.5 * (known_recall + unknown_recall)

    return {
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "known_precision": known_precision,
        "known_recall": known_recall,
        "unknown_recall": unknown_recall,
    }


def search_thresholds(scores: pd.DataFrame, search_steps: int) -> dict[str, float]:
    if scores.empty:
        raise ValueError("Cannot calibrate thresholds from an empty score table.")

    actual_known = scores["should_be_known"].to_numpy(dtype=bool)
    top1_values = scores["top1_similarity"].to_numpy(dtype=float)
    margin_values = scores["top1_top2_margin"].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    finite_margin_values = margin_values[np.isfinite(margin_values)]

    similarity_grid = np.linspace(top1_values.min(), top1_values.max(), num=search_steps)
    if len(finite_margin_values) == 0:
        margin_grid = np.array([0.0], dtype=float)
    else:
        margin_grid = np.linspace(finite_margin_values.min(), finite_margin_values.max(), num=search_steps)

    best_result: dict[str, float] | None = None

    for similarity_threshold in similarity_grid:
        similarity_accept = top1_values >= similarity_threshold
        for margin_threshold in margin_grid:
            margin_accept = np.where(np.isfinite(margin_values), margin_values >= margin_threshold, True)
            predicted_known = similarity_accept & margin_accept
            metrics = compute_classification_metrics(actual_known, predicted_known)
            candidate = {
                "similarity_threshold": float(similarity_threshold),
                "margin_threshold": float(margin_threshold),
                **metrics,
            }

            if best_result is None:
                best_result = candidate
                continue

            better = candidate["balanced_accuracy"] > best_result["balanced_accuracy"]
            tie_balanced = candidate["balanced_accuracy"] == best_result["balanced_accuracy"]
            better_accuracy = candidate["accuracy"] > best_result["accuracy"]
            tie_accuracy = candidate["accuracy"] == best_result["accuracy"]
            smaller_similarity = candidate["similarity_threshold"] < best_result["similarity_threshold"]
            smaller_margin = candidate["margin_threshold"] < best_result["margin_threshold"]

            if better or (tie_balanced and better_accuracy) or (
                tie_balanced and tie_accuracy and (smaller_similarity or smaller_margin)
            ):
                best_result = candidate

    if best_result is None:
        raise ValueError("Threshold search failed to produce a candidate.")

    return best_result


def calibrate_thresholds(cfg: CFG) -> None:
    input_dir = Path(cfg.gallery_validation_output_dir)
    output_dir = Path(cfg.thresholds_output_dir)
    require_split_artifacts(
        input_dir,
        ["gallery_prototypes", "gallery_images", "known_val", "unseen_val"],
        step_name="Step 05 threshold calibration",
        producer_step="step 03/04 gallery and validation builder",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    gallery_prototypes, gallery_prototype_metadata = load_split(input_dir, "gallery_prototypes")
    gallery_images, gallery_image_metadata = load_split(input_dir, "gallery_images")
    known_val_embeddings, known_val_metadata = load_split(input_dir, "known_val")
    unseen_val_embeddings, unseen_val_metadata = load_split(input_dir, "unseen_val")

    known_scores = score_queries_against_gallery(
        query_embeddings=known_val_embeddings,
        query_metadata=known_val_metadata,
        gallery_prototypes=gallery_prototypes,
        gallery_prototype_metadata=gallery_prototype_metadata,
        gallery_images=gallery_images,
        gallery_image_metadata=gallery_image_metadata,
        split_name="known_val",
        query_should_be_known=True,
    )
    unseen_scores = score_queries_against_gallery(
        query_embeddings=unseen_val_embeddings,
        query_metadata=unseen_val_metadata,
        gallery_prototypes=gallery_prototypes,
        gallery_prototype_metadata=gallery_prototype_metadata,
        gallery_images=gallery_images,
        gallery_image_metadata=gallery_image_metadata,
        split_name="unseen_val",
        query_should_be_known=False,
    )

    score_table = pd.concat([known_scores, unseen_scores], ignore_index=True)
    thresholds = search_thresholds(score_table, search_steps=cfg.threshold_search_steps)

    score_table_path = output_dir / "rejection_calibration_scores.csv"
    thresholds_path = output_dir / "rejection_thresholds.json"

    score_table.to_csv(score_table_path, index=False)
    with thresholds_path.open("w", encoding="utf-8") as handle:
        json.dump(thresholds, handle, indent=2, sort_keys=True)

    logger.info("Saved calibration score table to %s", score_table_path)
    logger.info("Saved calibrated rejection thresholds to %s", thresholds_path)


if __name__ == "__main__":
    cfg = CFG("config.yaml")
    calibrate_thresholds(cfg)
