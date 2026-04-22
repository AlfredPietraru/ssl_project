import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_submission(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find {name} submission: {path}")

    df = pd.read_csv(path)
    required_columns = {"image_id", "cluster"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {sorted(missing_columns)}")

    return df[["image_id", "cluster"]].copy()


def compare_submissions(root_path: Path, artifacts_path: Path) -> pd.DataFrame:
    root_df = load_submission(root_path, "root").rename(columns={"cluster": "root_cluster"})
    artifacts_df = load_submission(artifacts_path, "artifacts").rename(
        columns={"cluster": "artifacts_cluster"}
    )

    comparison = root_df.merge(artifacts_df, on="image_id", how="outer", indicator=True)
    comparison["status"] = "same"
    comparison.loc[comparison["_merge"] == "left_only", "status"] = "missing_from_artifacts"
    comparison.loc[comparison["_merge"] == "right_only", "status"] = "missing_from_root"

    present_in_both = comparison["_merge"].eq("both")
    different_cluster = comparison["root_cluster"].ne(comparison["artifacts_cluster"])
    comparison.loc[present_in_both & different_cluster, "status"] = "different_cluster"

    return comparison.drop(columns=["_merge"]).sort_values(["status", "image_id"])


def pairwise_grouping_summary(comparison: pd.DataFrame) -> dict[str, float | int]:
    paired = comparison.dropna(subset=["root_cluster", "artifacts_cluster"]).copy()
    n = len(paired)
    if n < 2:
        return {
            "common_image_ids": n,
            "pairs_total": 0,
            "pairs_same_in_both": 0,
            "pairs_same_in_root_only": 0,
            "pairs_same_in_artifacts_only": 0,
            "pairs_different_in_both": 0,
            "pairwise_grouping_agreement": 1.0,
        }

    root_codes = pd.Categorical(paired["root_cluster"]).codes
    artifacts_codes = pd.Categorical(paired["artifacts_cluster"]).codes
    upper_triangle = np.triu(np.ones((n, n), dtype=bool), k=1)

    root_same = (root_codes[:, None] == root_codes[None, :])[upper_triangle]
    artifacts_same = (artifacts_codes[:, None] == artifacts_codes[None, :])[upper_triangle]

    same_in_both = root_same & artifacts_same
    same_in_root_only = root_same & ~artifacts_same
    same_in_artifacts_only = ~root_same & artifacts_same
    different_in_both = ~root_same & ~artifacts_same
    agreement = same_in_both | different_in_both

    return {
        "common_image_ids": n,
        "pairs_total": int(len(root_same)),
        "pairs_same_in_both": int(same_in_both.sum()),
        "pairs_same_in_root_only": int(same_in_root_only.sum()),
        "pairs_same_in_artifacts_only": int(same_in_artifacts_only.sum()),
        "pairs_different_in_both": int(different_in_both.sum()),
        "pairwise_grouping_agreement": float(agreement.mean()),
    }


def label_mapping_summary(comparison: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    paired = comparison.dropna(subset=["root_cluster", "artifacts_cluster"]).copy()
    if paired.empty:
        empty = pd.DataFrame(
            columns=["artifacts_cluster", "mapped_root_cluster", "count", "artifacts_cluster_size", "purity"]
        )
        return empty, {"mapped_rows": 0, "common_image_ids": 0, "mapping_accuracy": 0.0}

    counts = (
        paired.groupby(["artifacts_cluster", "root_cluster"], sort=True)
        .size()
        .reset_index(name="count")
    )
    best_mapping = counts.sort_values(
        ["artifacts_cluster", "count", "root_cluster"],
        ascending=[True, False, True],
    ).drop_duplicates("artifacts_cluster")
    cluster_sizes = paired["artifacts_cluster"].value_counts().rename("artifacts_cluster_size")
    best_mapping = best_mapping.join(cluster_sizes, on="artifacts_cluster")
    best_mapping["purity"] = best_mapping["count"] / best_mapping["artifacts_cluster_size"]
    best_mapping = best_mapping.rename(columns={"root_cluster": "mapped_root_cluster"})

    mapped_rows = int(best_mapping["count"].sum())
    summary = {
        "common_image_ids": int(len(paired)),
        "mapped_rows": mapped_rows,
        "mapping_accuracy": float(mapped_rows / len(paired)),
        "artifact_clusters": int(paired["artifacts_cluster"].nunique()),
        "root_clusters": int(paired["root_cluster"].nunique()),
    }
    return best_mapping.sort_values(["purity", "artifacts_cluster"]), summary


def print_summary(comparison: pd.DataFrame) -> None:
    counts = comparison["status"].value_counts().sort_index()
    print("Exact label comparison")
    print("======================")
    print(f"Total image_ids: {len(comparison)}")
    for status, count in counts.items():
        print(f"{status}: {count}")

    print("\nPairwise grouping comparison")
    print("============================")
    grouping_summary = pairwise_grouping_summary(comparison)
    for key, value in grouping_summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print("\nBest artifact-label to root-label mapping")
    print("=========================================")
    _, mapping_summary = label_mapping_summary(comparison)
    for key, value in mapping_summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare root submission.csv against an artifacts submission CSV."
    )
    parser.add_argument("--root-submission", default="artifacts/groundtruth_submission.csv", help="Root submission CSV path.")
    parser.add_argument(
        "--artifacts-submission",
        default="artifacts/submission.csv",
        help="Artifacts submission CSV path.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/submission_diff.csv",
        help="Where to write the row-level comparison.",
    )
    parser.add_argument(
        "--mapping-output",
        default="artifacts/submission_label_mapping.csv",
        help="Where to write best artifact-cluster to root-cluster label mapping.",
    )
    parser.add_argument(
        "--only-differences",
        action="store_true",
        help="Write only rows whose cluster/status differs.",
    )
    args = parser.parse_args()

    comparison = compare_submissions(
        Path(args.root_submission),
        Path(args.artifacts_submission),
    )
    print_summary(comparison)

    output_df = comparison
    if args.only_differences:
        output_df = comparison[comparison["status"] != "same"].copy()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Wrote comparison to {output_path}")

    mapping_df, _ = label_mapping_summary(comparison)
    mapping_output_path = Path(args.mapping_output)
    mapping_output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(mapping_output_path, index=False)
    print(f"Wrote label mapping to {mapping_output_path}")


if __name__ == "__main__":
    main()
