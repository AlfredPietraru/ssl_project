import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import CFG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("STEP09_REFINEMENT")


def refine_with_incremental_enrollment(cfg: CFG) -> None:
    gallery_dir = Path(cfg.gallery_validation_output_dir)
    rejection_dir = Path(cfg.rejection_output_dir)
    clustering_dir = Path(cfg.clustering_output_dir)
    output_dir = Path(cfg.refinement_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gallery_proto_embeddings = np.load(gallery_dir / "full_gallery_prototypes_embeddings.npy")
    gallery_proto_metadata = pd.read_csv(gallery_dir / "full_gallery_prototypes_metadata.csv")
    known_matches = pd.read_csv(rejection_dir / "known_matches.csv")
    cluster_assignments = pd.read_csv(clustering_dir / "unknown_cluster_assignments.csv")
    cluster_proto_embeddings = np.load(clustering_dir / "unknown_cluster_prototypes_embeddings.npy")
    cluster_proto_metadata = pd.read_csv(clustering_dir / "unknown_cluster_prototypes_metadata.csv")

    eligible_clusters = cluster_proto_metadata[
        (cluster_proto_metadata["cluster_size"] >= cfg.refinement_min_cluster_size)
        & (cluster_proto_metadata["mean_internal_similarity"] >= cfg.refinement_min_cluster_similarity)
    ].reset_index(drop=True).copy()

    eligible_indices = eligible_clusters["prototype_index"].to_numpy(dtype=int) if not eligible_clusters.empty else np.array([], dtype=int)
    eligible_embeddings = (
        cluster_proto_embeddings[eligible_indices]
        if len(eligible_indices) > 0
        else np.empty((0, gallery_proto_embeddings.shape[1]), dtype=gallery_proto_embeddings.dtype)
    )

    eligible_clusters["identity"] = eligible_clusters["cluster_id"]
    eligible_clusters["num_images"] = eligible_clusters["cluster_size"]

    augmented_proto_embeddings = (
        np.concatenate([gallery_proto_embeddings, eligible_embeddings], axis=0)
        if len(eligible_embeddings) > 0
        else gallery_proto_embeddings.copy()
    )
    augmented_proto_metadata = pd.concat(
        [
            gallery_proto_metadata,
            eligible_clusters[[column for column in gallery_proto_metadata.columns if column in eligible_clusters.columns]].copy(),
        ],
        ignore_index=True,
    )
    augmented_proto_metadata["prototype_index"] = np.arange(len(augmented_proto_metadata))

    high_conf_known = known_matches[
        known_matches["top1_similarity"] >= cfg.refinement_known_confidence_threshold
    ].reset_index(drop=True).copy()
    high_conf_known["refined_identity"] = high_conf_known["predicted_identity"]
    high_conf_known["assignment_source"] = "known_high_confidence"

    enrolled_unknowns = cluster_assignments[
        cluster_assignments["cluster_id"].isin(set(eligible_clusters["cluster_id"].astype(str)))
    ].reset_index(drop=True).copy()
    if not enrolled_unknowns.empty:
        enrolled_unknowns["refined_identity"] = enrolled_unknowns["cluster_id"]
        enrolled_unknowns["assignment_source"] = "incremental_enrollment"

    refined_assignments = pd.concat(
        [
            high_conf_known,
            enrolled_unknowns,
        ],
        ignore_index=True,
        sort=False,
    )

    np.save(output_dir / "augmented_gallery_prototypes_embeddings.npy", augmented_proto_embeddings)
    augmented_proto_metadata.to_csv(output_dir / "augmented_gallery_prototypes_metadata.csv", index=False)
    refined_assignments.to_csv(output_dir / "refined_assignments.csv", index=False)

    summary = {
        "eligible_enrolled_clusters": int(len(eligible_clusters)),
        "eligible_enrolled_samples": int(len(enrolled_unknowns)),
        "high_confidence_known_matches": int(len(high_conf_known)),
        "refinement_min_cluster_size": cfg.refinement_min_cluster_size,
        "refinement_min_cluster_similarity": cfg.refinement_min_cluster_similarity,
        "refinement_known_confidence_threshold": cfg.refinement_known_confidence_threshold,
    }
    with (output_dir / "refinement_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    logger.info("Saved augmented gallery prototypes to %s", output_dir / "augmented_gallery_prototypes_metadata.csv")
    logger.info("Saved refined assignments to %s", output_dir / "refined_assignments.csv")


if __name__ == "__main__":
    cfg = CFG("config.yaml")
    refine_with_incremental_enrollment(cfg)
