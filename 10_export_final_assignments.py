import logging
from pathlib import Path

import pandas as pd

from config import CFG
from main_utils import require_existing_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("STEP10_EXPORT")


def export_final_assignments(cfg: CFG) -> None:
    rejection_dir = Path(cfg.rejection_output_dir)
    clustering_dir = Path(cfg.clustering_output_dir)
    refinement_dir = Path(cfg.refinement_output_dir)
    output_dir = Path(cfg.final_output_dir)
    require_existing_paths(
        [
            (rejection_dir / "known_matches.csv", "step 07 low-confidence rejection"),
            (clustering_dir / "unknown_cluster_assignments.csv", "step 08 clustering rejected unknowns"),
        ],
        step_name="Step 10 final export",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    refined_assignments_path = refinement_dir / "refined_assignments.csv"
    known_matches = pd.read_csv(rejection_dir / "known_matches.csv")
    unknown_clusters = pd.read_csv(clustering_dir / "unknown_cluster_assignments.csv")

    known_export = known_matches.copy()
    known_export["assigned_identity"] = known_export["predicted_identity"]
    known_export["assignment_source"] = "known_match"

    unknown_export = unknown_clusters.copy()
    if not unknown_export.empty:
        unknown_export["assigned_identity"] = unknown_export["cluster_id"]
        unknown_export["assignment_source"] = "unknown_cluster"

    final_assignments = pd.concat([known_export, unknown_export], ignore_index=True, sort=False)

    if refined_assignments_path.exists():
        refined_assignments = pd.read_csv(refined_assignments_path)
        if not refined_assignments.empty:
            refined_subset = refined_assignments[["query_index", "refined_identity", "assignment_source"]].copy()
            refined_subset = refined_subset.rename(columns={"refined_identity": "assigned_identity"})
            final_assignments = final_assignments.drop(columns=["assignment_source"], errors="ignore")
            final_assignments = final_assignments.merge(
                refined_subset,
                on="query_index",
                how="left",
                suffixes=("", "_refined"),
            )
            final_assignments["assigned_identity"] = final_assignments["assigned_identity_refined"].fillna(
                final_assignments["assigned_identity"]
            )
            final_assignments["assignment_source"] = final_assignments["assignment_source"].fillna("base_pipeline")
            final_assignments = final_assignments.drop(columns=["assigned_identity_refined"])

    preferred_columns = [
        "query_index",
        "query_image_id",
        "query_path",
        "query_species",
        "assigned_identity",
        "assignment_source",
    ]
    ordered_columns = [column for column in preferred_columns if column in final_assignments.columns]
    trailing_columns = [column for column in final_assignments.columns if column not in ordered_columns]
    final_assignments = final_assignments[ordered_columns + trailing_columns]
    final_assignments = final_assignments.sort_values("query_index").reset_index(drop=True)

    output_path = output_dir / cfg.final_assignments_filename
    final_assignments.to_csv(output_path, index=False)
    logger.info("Saved final assignments to %s", output_path)


if __name__ == "__main__":
    cfg = CFG("config.yaml")
    export_final_assignments(cfg)
