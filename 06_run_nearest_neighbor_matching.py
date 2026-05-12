import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import CFG
from main_utils import normalize_rows, require_split_artifacts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("STEP06_MATCHING")


def load_split(base_dir: Path, name: str) -> tuple[np.ndarray, pd.DataFrame]:
    embeddings = np.load(base_dir / f"{name}_embeddings.npy")
    metadata = pd.read_csv(base_dir / f"{name}_metadata.csv")
    if len(embeddings) != len(metadata):
        raise ValueError(f"Split '{name}' has mismatched embeddings and metadata lengths.")
    return normalize_rows(embeddings), metadata


def top_identity_scores(
    similarities: np.ndarray,
    metadata: pd.DataFrame,
    top_k: int,
) -> tuple[str, float, float]:
    if len(similarities) == 0:
        return "unknown", float("-inf"), float("-inf")

    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_rows = metadata.iloc[top_indices].copy()
    top_rows["similarity"] = similarities[top_indices]

    grouped = (
        top_rows.groupby("identity", sort=False)["similarity"]
        .mean()
        .sort_values(ascending=False)
    )
    best_identity = str(grouped.index[0])
    best_score = float(grouped.iloc[0])
    second_score = float(grouped.iloc[1]) if len(grouped) > 1 else float("-inf")
    return best_identity, best_score, second_score


def run_matching(cfg: CFG) -> None:
    gallery_dir = Path(cfg.gallery_validation_output_dir)
    embeddings_dir = Path(cfg.embeddings_output_dir)
    output_dir = Path(cfg.matching_output_dir)
    require_split_artifacts(
        gallery_dir,
        ["full_gallery_prototypes", "full_gallery_images"],
        step_name="Step 06 nearest-neighbor matching",
        producer_step="step 03/04 gallery and validation builder",
    )
    require_split_artifacts(
        embeddings_dir,
        ["test"],
        step_name="Step 06 nearest-neighbor matching",
        producer_step="step 02 embedding extraction",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    gallery_prototypes, gallery_proto_meta = load_split(gallery_dir, "full_gallery_prototypes")
    gallery_images, gallery_image_meta = load_split(gallery_dir, "full_gallery_images")
    test_embeddings, test_metadata = load_split(embeddings_dir, "test")

    rows: list[dict[str, object]] = []

    for query_index in range(len(test_embeddings)):
        query_embedding = test_embeddings[query_index : query_index + 1]
        query_row = test_metadata.iloc[query_index]

        if "species" in test_metadata.columns and "species" in gallery_proto_meta.columns:
            species_value = query_row.get("species")
            proto_mask = gallery_proto_meta["species"].eq(species_value).to_numpy()
            image_mask = gallery_image_meta["species"].eq(species_value).to_numpy()
        else:
            proto_mask = np.ones(len(gallery_prototypes), dtype=bool)
            image_mask = np.ones(len(gallery_images), dtype=bool)

        if not proto_mask.any():
            proto_mask = np.ones(len(gallery_prototypes), dtype=bool)
        if not image_mask.any():
            image_mask = np.ones(len(gallery_images), dtype=bool)

        filtered_proto_embeddings = gallery_prototypes[proto_mask]
        filtered_proto_meta = gallery_proto_meta.loc[proto_mask].reset_index(drop=True)
        filtered_image_embeddings = gallery_images[image_mask]
        filtered_image_meta = gallery_image_meta.loc[image_mask].reset_index(drop=True)

        proto_similarities = (query_embedding @ filtered_proto_embeddings.T)[0]
        image_similarities = (query_embedding @ filtered_image_embeddings.T)[0]

        best_identity, top1_similarity, top2_similarity = top_identity_scores(
            similarities=image_similarities,
            metadata=filtered_image_meta,
            top_k=cfg.matching_top_k,
        )

        best_proto_index = int(np.argmax(proto_similarities))
        best_image_index = int(np.argmax(image_similarities))
        best_proto_identity = str(filtered_proto_meta.iloc[best_proto_index]["identity"])
        best_image_identity = str(filtered_image_meta.iloc[best_image_index]["identity"])
        best_proto_similarity = float(proto_similarities[best_proto_index])
        best_image_similarity = float(image_similarities[best_image_index])
        margin = float(top1_similarity - top2_similarity) if np.isfinite(top2_similarity) else float("inf")

        row: dict[str, object] = {
            "query_index": int(query_index),
            "predicted_identity": best_identity,
            "top1_similarity": top1_similarity,
            "top2_similarity": top2_similarity,
            "top1_top2_margin": margin,
            "best_prototype_identity": best_proto_identity,
            "best_prototype_similarity": best_proto_similarity,
            "best_image_identity": best_image_identity,
            "best_image_similarity": best_image_similarity,
            "prototype_image_agreement": bool(best_proto_identity == best_image_identity),
        }

        for candidate in ["identity", "species", "dataset", "image_id", "path"]:
            if candidate in test_metadata.columns:
                row[f"query_{candidate}"] = query_row[candidate]
        rows.append(row)

    scores = pd.DataFrame(rows)
    output_path = output_dir / "match_scores.csv"
    scores.to_csv(output_path, index=False)
    logger.info("Saved nearest-neighbor match scores to %s", output_path)


if __name__ == "__main__":
    cfg = CFG("config.yaml")
    run_matching(cfg)
