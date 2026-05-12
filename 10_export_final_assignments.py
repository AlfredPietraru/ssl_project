import logging
from pathlib import Path
import re

import pandas as pd

from config import CFG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("STEP10_EXPORT")

CLUSTER_PREFIXES = {
    "LynxID2025",
    "SalamanderID2025",
    "SeaTurtleID2022",
    "TexasHornedLizards",
}


def format_cluster(value: object, dataset: object | None = None) -> str:
    cluster = str(value).strip()
    if not cluster or cluster.lower() == "nan":
        raise ValueError("Cannot export an empty cluster assignment.")

    if cluster.startswith("cluster_"):
        cluster = cluster.removeprefix("cluster_")

    known_match = re.match(r"^([^_]+)_(?:.*?)(\d+)$", cluster)
    if known_match and known_match.group(1) in CLUSTER_PREFIXES:
        return f"cluster_{known_match.group(1)}_{int(known_match.group(2))}"

    new_match = re.match(r"^new_[^_]+_0*(\d+)$", cluster)
    if new_match:
        dataset_name = str(dataset).strip() if dataset is not None else ""
        if dataset_name not in CLUSTER_PREFIXES:
            raise ValueError(f"Cannot format new cluster '{cluster}' without a valid dataset.")
        return f"cluster_{dataset_name}_{int(new_match.group(1))}"

    raise ValueError(f"Unsupported cluster assignment format: '{cluster}'")


def build_test_assignments(
    known_matches: pd.DataFrame,
    unknown_clusters: pd.DataFrame,
    refined_assignments_path: Path,
) -> pd.DataFrame:
    known_export = known_matches[["query_index", "query_dataset", "predicted_identity"]].copy()
    known_export = known_export.rename(columns={"predicted_identity": "cluster"})

    unknown_export = unknown_clusters[["query_index", "query_dataset", "cluster_id"]].copy()
    unknown_export = unknown_export.rename(columns={"cluster_id": "cluster"})

    assignments = pd.concat([known_export, unknown_export], ignore_index=True)

    if refined_assignments_path.exists():
        refined_assignments = pd.read_csv(refined_assignments_path)
        if not refined_assignments.empty:
            refined_subset = refined_assignments[["query_index", "refined_identity"]].copy()
            refined_subset = refined_subset.rename(columns={"refined_identity": "refined_cluster"})
            assignments = assignments.merge(refined_subset, on="query_index", how="left")
            assignments["cluster"] = assignments["refined_cluster"].fillna(assignments["cluster"])
            assignments = assignments.drop(columns=["refined_cluster"])

    duplicate_count = int(assignments["query_index"].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate test query assignments.")

    assignments["cluster"] = assignments.apply(
        lambda row: format_cluster(row["cluster"], row["query_dataset"]),
        axis=1,
    )
    return assignments


def export_train_submission(embeddings_dir: Path, output_dir: Path) -> Path:
    train_metadata = pd.read_csv(embeddings_dir / "train_metadata.csv")
    submission = pd.DataFrame(
        {
            "image_id": train_metadata["image_id"],
            "cluster": train_metadata.apply(
                lambda row: format_cluster(row["identity"], row["dataset"]),
                axis=1,
            ),
        }
    )
    submission = submission.sort_values("image_id").reset_index(drop=True)

    output_path = output_dir / "train_submission.csv"
    submission.to_csv(output_path, index=False)
    return output_path


def export_test_submission(
    embeddings_dir: Path,
    output_dir: Path,
    assignments: pd.DataFrame,
) -> Path:
    test_metadata = pd.read_csv(embeddings_dir / "test_metadata.csv")
    submission = test_metadata[["embedding_index", "image_id"]].merge(
        assignments,
        left_on="embedding_index",
        right_on="query_index",
        how="left",
    )

    missing = submission["cluster"].isna()
    if missing.any():
        missing_ids = submission.loc[missing, "image_id"].head(10).tolist()
        raise ValueError(
            "Missing cluster assignments for "
            f"{int(missing.sum())} test images. First missing image_id values: {missing_ids}"
        )

    submission = submission[["image_id", "cluster"]].sort_values("image_id").reset_index(drop=True)

    output_path = output_dir / "submission.csv"
    submission.to_csv(output_path, index=False)
    return output_path


def export_final_assignments(cfg: CFG) -> None:
    embeddings_dir = Path(cfg.embeddings_output_dir)
    rejection_dir = Path(cfg.rejection_output_dir)
    clustering_dir = Path(cfg.clustering_output_dir)
    refinement_dir = Path(cfg.refinement_output_dir)
    output_dir = Path(cfg.final_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    refined_assignments_path = refinement_dir / "refined_assignments.csv"
    known_matches = pd.read_csv(rejection_dir / "known_matches.csv")
    unknown_clusters = pd.read_csv(clustering_dir / "unknown_cluster_assignments.csv")

    assignments = build_test_assignments(
        known_matches=known_matches,
        unknown_clusters=unknown_clusters,
        refined_assignments_path=refined_assignments_path,
    )

    train_output_path = export_train_submission(embeddings_dir, output_dir)
    test_output_path = export_test_submission(embeddings_dir, output_dir, assignments)

    logger.info("Saved train submission to %s", train_output_path)
    logger.info("Saved test submission to %s", test_output_path)


if __name__ == "__main__":
    cfg = CFG("config.yaml")
    export_final_assignments(cfg)
