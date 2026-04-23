import logging
import os
import re
import importlib.util
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd

from main_utils import (
    agglomerative_labels_from_similarity,
    build_identity_prototypes,
    load_embedding_artifacts,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EXPERIMENT")


def load_train_experiment_module():
    module_path = Path(__file__).with_name("01_train_experiment.py")
    spec = importlib.util.spec_from_file_location("train_experiment_01", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load train experiment module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_label_part(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"^cluster_", "", text)
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def make_cluster_label(*parts: object) -> str:
    return "cluster_" + "_".join(safe_label_part(part) for part in parts)


def canonical_known_identity_label(identity: object, dataset: object, species: object) -> str:
    identity_part = safe_label_part(identity)
    dataset_part = safe_label_part(dataset)
    species_part = safe_label_part(species).lower()
    species_tokens = {
        "lynx",
        "salamander",
        "turtle",
        "loggerhead",
        "loggerhead_turtle",
        "lizard",
        species_part,
    }
    if "_" in species_part:
        species_tokens.update(token for token in species_part.split("_") if token)

    prefix = f"{dataset_part}_"
    if not identity_part.startswith(prefix):
        return identity_part

    suffix_parts = [part for part in identity_part[len(prefix):].split("_") if part]
    suffix_parts = [
        part for part in suffix_parts
        if part.lower() not in species_tokens
    ]
    if len(suffix_parts) == 1 and re.fullmatch(r"t\d+", suffix_parts[0], flags=re.IGNORECASE):
        suffix_parts[0] = str(int(suffix_parts[0][1:]))

    return "_".join([dataset_part, *suffix_parts]) if suffix_parts else dataset_part


def match_known_identities(
    test_embeddings: np.ndarray,
    prototypes: np.ndarray,
    prototype_identities: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(prototype_identities) == 0:
        n = len(test_embeddings)
        return np.full(n, False), np.array([""] * n, dtype=object), np.full(n, np.nan)

    similarity = test_embeddings @ prototypes.T
    best_indices = similarity.argmax(axis=1)
    best_scores = similarity[np.arange(len(test_embeddings)), best_indices]
    best_identities = prototype_identities[best_indices]
    accepted = best_scores >= threshold
    return accepted, best_identities, best_scores


def match_known_identities_by_neighbors(
    test_embeddings: np.ndarray,
    train_embeddings: np.ndarray,
    train_df: pd.DataFrame,
    threshold: float,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    known_df = train_df[train_df["identity"] != "unknown"].copy()
    if known_df.empty:
        n = len(test_embeddings)
        return np.full(n, False), np.array([""] * n, dtype=object), np.full(n, np.nan)

    known_embeddings = train_embeddings[known_df["_embedding_index"].to_numpy()]
    known_identities = known_df["identity"].to_numpy()
    similarity = test_embeddings @ known_embeddings.T
    neighbors = min(neighbors, similarity.shape[1])

    top_indices = np.argpartition(-similarity, kth=neighbors - 1, axis=1)[:, :neighbors]
    best_identities = []
    best_scores = []
    for row_index, row_top_indices in enumerate(top_indices):
        identity_scores: dict[str, list[float]] = {}
        for train_index in row_top_indices:
            identity = known_identities[train_index]
            score = float(similarity[row_index, train_index])
            identity_scores.setdefault(identity, []).append(score)

        identity, score = max(
            (
                (identity, max(scores))
                for identity, scores in identity_scores.items()
            ),
            key=lambda item: item[1],
        )
        best_identities.append(identity)
        best_scores.append(score)

    best_identities = np.asarray(best_identities, dtype=object)
    best_scores = np.asarray(best_scores, dtype=np.float32)
    accepted = best_scores >= threshold
    return accepted, best_identities, best_scores


def match_known_identities_hierarchical(
    test_embeddings: np.ndarray,
    species_model: dict[str, object] | None,
    threshold: float,
    coarse_top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if species_model is None:
        n = len(test_embeddings)
        return (
            np.full(n, False),
            np.array([""] * n, dtype=object),
            np.full(n, np.nan),
            np.full(n, -1, dtype=int),
        )

    hierarchical_model = species_model.get("hierarchical_model", {})
    coarse_centroids = hierarchical_model.get("coarse_centroids")
    coarse_cluster_ids = hierarchical_model.get("coarse_cluster_ids")
    identity_models = hierarchical_model.get("identity_models", {})
    if coarse_centroids is None or len(coarse_centroids) == 0:
        n = len(test_embeddings)
        return (
            np.full(n, False),
            np.array([""] * n, dtype=object),
            np.full(n, np.nan),
            np.full(n, -1, dtype=int),
        )

    coarse_similarity = test_embeddings @ coarse_centroids.T
    coarse_top_k = min(coarse_top_k, coarse_similarity.shape[1])
    coarse_top_indices = np.argpartition(
        -coarse_similarity,
        kth=coarse_top_k - 1,
        axis=1,
    )[:, :coarse_top_k]

    best_identities = []
    best_scores = []
    best_coarse_clusters = []
    for row_index, candidate_coarse_indices in enumerate(coarse_top_indices):
        row_best_identity = ""
        row_best_score = -np.inf
        row_best_coarse_cluster = -1

        for coarse_index in candidate_coarse_indices:
            coarse_cluster = int(coarse_cluster_ids[coarse_index])
            identity_model = identity_models[coarse_cluster]
            prototypes = identity_model["prototypes"]
            prototype_identities = identity_model["prototype_identities"]
            if len(prototype_identities) == 0:
                continue

            identity_similarity = test_embeddings[row_index : row_index + 1] @ prototypes.T
            best_index = int(identity_similarity.argmax(axis=1)[0])
            score = float(identity_similarity[0, best_index])
            if score > row_best_score:
                row_best_score = score
                row_best_identity = prototype_identities[best_index]
                row_best_coarse_cluster = coarse_cluster

        best_identities.append(row_best_identity)
        best_scores.append(row_best_score if np.isfinite(row_best_score) else np.nan)
        best_coarse_clusters.append(row_best_coarse_cluster)

    best_identities = np.asarray(best_identities, dtype=object)
    best_scores = np.asarray(best_scores, dtype=np.float32)
    best_coarse_clusters = np.asarray(best_coarse_clusters, dtype=int)
    accepted = best_scores >= threshold
    return accepted, best_identities, best_scores, best_coarse_clusters


def parse_thresholds(raw_value: str | None) -> dict[str, float]:
    thresholds = {}
    if not raw_value:
        return thresholds

    for item in raw_value.split(","):
        if not item.strip():
            continue
        name, value = item.split("=", maxsplit=1)
        thresholds[name.strip()] = float(value)
    return thresholds


def run_species_experiment(
    species: str,
    species_model: dict[str, object] | None,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    known_threshold: float,
    cluster_threshold: float,
    min_cluster_size: int,
    known_match_strategy: str,
    knn_neighbors: int,
    coarse_top_k: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    logger.info(
        "[%s] train=%s test=%s strategy=%s known_threshold=%.3f cluster_threshold=%.3f",
        species,
        len(train_df),
        len(test_df),
        known_match_strategy,
        known_threshold,
        cluster_threshold,
    )

    prototypes, prototype_identities = build_identity_prototypes(train_embeddings, train_df)
    best_coarse_clusters = np.full(len(test_embeddings), -1, dtype=int)
    if len(prototype_identities) == 0:
        known_mask = np.full(len(test_embeddings), False)
        best_identities = np.array([""] * len(test_embeddings), dtype=object)
        best_scores = np.full(len(test_embeddings), np.nan)
    elif known_match_strategy == "prototype":
        known_mask, best_identities, best_scores = match_known_identities(
            test_embeddings,
            prototypes,
            prototype_identities,
            known_threshold,
        )
    elif known_match_strategy == "knn":
        known_mask, best_identities, best_scores = match_known_identities_by_neighbors(
            test_embeddings,
            train_embeddings,
            train_df,
            known_threshold,
            knn_neighbors,
        )
    elif known_match_strategy == "hierarchical":
        known_mask, best_identities, best_scores, best_coarse_clusters = (
            match_known_identities_hierarchical(
                test_embeddings,
                species_model,
                known_threshold,
                coarse_top_k,
            )
        )
    else:
        raise ValueError(f"Unknown known_match_strategy: {known_match_strategy}")

    assignments = np.array([""] * len(test_df), dtype=object)
    assignment_type = np.array(["unknown"] * len(test_df), dtype=object)
    known_rows = test_df.loc[known_mask, ["dataset", "species"]].reset_index(drop=True)
    assignments[known_mask] = [
        make_cluster_label(canonical_known_identity_label(identity, row.dataset, row.species))
        for identity, row in zip(best_identities[known_mask], known_rows.itertuples(index=False))
    ]
    assignment_type[known_mask] = "known_match"

    unknown_indices = np.where(~known_mask)[0]
    unknown_labels = agglomerative_labels_from_similarity(
        test_embeddings[unknown_indices],
        cluster_threshold,
        min_cluster_size,
    )
    for local_index, cluster_id in zip(unknown_indices, unknown_labels):
        dataset_name = test_df.iloc[local_index]["dataset"]
        assignments[local_index] = make_cluster_label(dataset_name, cluster_id)

    result = test_df[["image_id", "dataset", "species"]].copy()
    result["cluster"] = assignments
    result["assignment_type"] = assignment_type
    result["best_known_identity"] = best_identities
    result["best_known_similarity"] = best_scores
    result["best_coarse_cluster"] = best_coarse_clusters

    summary = {
        "species": species,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_identities": int(train_df["identity"].nunique()),
        "prototype_identities": int(len(prototype_identities)),
        "known_matches": int(known_mask.sum()),
        "unknown_clustered": int((~known_mask).sum()),
        "unknown_clusters": int(len(np.unique(unknown_labels))) if len(unknown_labels) else 0,
        "known_threshold": known_threshold,
        "cluster_threshold": cluster_threshold,
        "known_match_strategy": known_match_strategy,
        "knn_neighbors": knn_neighbors,
        "coarse_top_k": coarse_top_k,
    }
    return result, summary


def main() -> None:
    config = {
        "embeddings_dir": "data_embeddings",
        "model_name": "mega",
        "output": "artifacts/submission.csv",
        "details_output": "artifacts/experiment_assignments.csv",
        "summary_output": "artifacts/experiment_summary.csv",
        "train_eval_output": "artifacts/train_prototype_eval.csv",
        "known_threshold": 0.72,
        "cluster_threshold": 0.68,
        "known_thresholds": None,
        "cluster_thresholds": None,
        "min_cluster_size": 2,
        "known_match_strategy": "hierarchical",
        "knn_neighbors": 1,
        "coarse_cluster_threshold": 0.68,
        "coarse_min_cluster_size": 2,
        "coarse_top_k": 3,
        "no_normalize": False
    }

    output_path = Path(config["output"])
    details_output_path = Path(config["details_output"])
    summary_output_path = Path(config["summary_output"])
    train_eval_output_path = Path(config["train_eval_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    details_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    train_eval_output_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings_dir = Path(config["embeddings_dir"])
    train_experiment = load_train_experiment_module()
    train_model = train_experiment.get_general_train_model(
        embeddings_dir=embeddings_dir,
        model_name=config["model_name"],
        normalize=not config["no_normalize"],
        coarse_cluster_threshold=float(config["coarse_cluster_threshold"]),
        coarse_min_cluster_size=int(config["coarse_min_cluster_size"]),
    )
    train_embeddings = train_model["train_embeddings"]
    train_df = train_model["train_df"]
    test_embeddings, test_df = load_embedding_artifacts(
        embeddings_dir,
        config["model_name"],
        "test",
        normalize=not config["no_normalize"],
    )

    known_thresholds = parse_thresholds(config["known_thresholds"])
    cluster_thresholds = parse_thresholds(config["cluster_thresholds"])

    results = []
    summaries = []
    train_eval_df = pd.DataFrame()
    train_eval_summary_by_species = {}
    if len(train_df):
        train_eval_summary_df, train_eval_df = train_experiment.test_train_model(
            train_model=train_model,
            known_threshold=float(config["known_threshold"]),
        )
        train_eval_summary_by_species = {
            row["species"]: row.drop(labels=["species"]).to_dict()
            for _, row in train_eval_summary_df.iterrows()
            if row["species"] != "overall"
        }

    for species in sorted(test_df["species"].unique()):
        species_model = train_model["species_models"].get(species)
        if species_model is None:
            species_train_df = train_df[train_df["species"] == species].copy()
            species_train_embeddings = train_embeddings[
                species_train_df["_embedding_index"].to_numpy()
            ]
            species_train_df["_embedding_index"] = np.arange(len(species_train_df))
        else:
            species_train_df = species_model["train_df"].copy()
            species_train_embeddings = species_model["train_embeddings"]
        species_test_df = test_df[test_df["species"] == species].copy()

        species_test_embeddings = test_embeddings[species_test_df["_embedding_index"].to_numpy()]
        species_test_df["_embedding_index"] = np.arange(len(species_test_df))
        known_threshold = known_thresholds.get(species, float(config["known_threshold"]))
        cluster_threshold = cluster_thresholds.get(species, float(config["cluster_threshold"]))

        result, summary = run_species_experiment(
            species=species,
            species_model=species_model,
            train_df=species_train_df,
            test_df=species_test_df,
            train_embeddings=species_train_embeddings,
            test_embeddings=species_test_embeddings,
            known_threshold=known_threshold,
            cluster_threshold=cluster_threshold,
            min_cluster_size=config["min_cluster_size"],
            known_match_strategy=config["known_match_strategy"],
            knn_neighbors=config["knn_neighbors"],
            coarse_top_k=config["coarse_top_k"],
        )
        train_eval_summary = train_eval_summary_by_species.get(
            species,
            {
                "train_eval_samples": 0,
                "inclusive_top1_accuracy": np.nan,
                "inclusive_accepted_accuracy": np.nan,
                "inclusive_accept_rate": np.nan,
                "leave_one_out_top1_accuracy": np.nan,
                "leave_one_out_accepted_accuracy": np.nan,
                "leave_one_out_accept_rate": np.nan,
                "singleton_identity_samples": 0,
            },
        )
        summary.update(train_eval_summary)
        results.append(result)
        summaries.append(summary)

    details = pd.concat(results, ignore_index=True).sort_values("image_id")
    submission = details[["image_id", "cluster"]].copy()
    summary_df = pd.DataFrame(summaries).sort_values("species")
    if not train_eval_df.empty:
        train_eval_df = train_eval_df.sort_values("image_id")

    submission.to_csv(output_path, index=False)
    details.to_csv(details_output_path, index=False)
    summary_df.to_csv(summary_output_path, index=False)
    train_eval_df.to_csv(train_eval_output_path, index=False)

    logger.info("Wrote submission to %s", output_path)
    logger.info("Wrote detailed assignments to %s", details_output_path)
    logger.info("Wrote summary to %s", summary_output_path)
    logger.info("Wrote train prototype evaluation to %s", train_eval_output_path)


if __name__ == "__main__":
    main()
