import logging
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TRAIN_EXPERIMENT")


def load_test_experiment_module():
    module_path = Path(__file__).with_name("02_test_experiment.py")
    spec = importlib.util.spec_from_file_location("test_experiment_02", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load test experiment module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_test_experiment = load_test_experiment_module()
build_identity_prototypes = _test_experiment.build_identity_prototypes
evaluate_identity_prototypes = _test_experiment.evaluate_identity_prototypes
load_embedding_artifacts = _test_experiment.load_embedding_artifacts
normalize_rows = _test_experiment.normalize_rows
agglomerative_labels_from_similarity = _test_experiment.agglomerative_labels_from_similarity


def build_prototype_table(
    species: str,
    train_df: pd.DataFrame,
    train_embeddings: np.ndarray,
) -> pd.DataFrame:
    known_df = train_df[train_df["identity"] != "unknown"].copy()
    if known_df.empty:
        return pd.DataFrame(
            columns=[
                "species",
                "identity",
                "prototype_index",
                "train_samples",
                "prototype_norm",
            ]
        )

    prototypes, prototype_identities = build_identity_prototypes(train_embeddings, train_df)
    sample_counts = known_df.groupby("identity").size()

    return pd.DataFrame(
        {
            "species": species,
            "identity": prototype_identities,
            "prototype_index": np.arange(len(prototype_identities)),
            "train_samples": [int(sample_counts[identity]) for identity in prototype_identities],
            "prototype_norm": np.linalg.norm(prototypes, axis=1),
        }
    )


def build_hierarchical_species_model(
    species: str,
    train_df: pd.DataFrame,
    train_embeddings: np.ndarray,
    coarse_cluster_threshold: float,
    coarse_min_cluster_size: int,
) -> dict[str, object]:
    known_df = train_df[train_df["identity"] != "unknown"].copy()
    if known_df.empty:
        return {
            "coarse_centroids": np.empty((0, train_embeddings.shape[1]), dtype=train_embeddings.dtype),
            "coarse_cluster_ids": np.array([], dtype=int),
            "coarse_table": pd.DataFrame(
                columns=[
                    "species",
                    "coarse_cluster",
                    "train_samples",
                    "identity_count",
                    "coarse_centroid_norm",
                ]
            ),
            "identity_models": {},
        }

    known_embeddings = train_embeddings[known_df["_embedding_index"].to_numpy()]
    known_df = known_df.reset_index(drop=True)
    known_df["_local_embedding_index"] = np.arange(len(known_df))

    coarse_labels = agglomerative_labels_from_similarity(
        known_embeddings,
        similarity_threshold=coarse_cluster_threshold,
        min_cluster_size=coarse_min_cluster_size,
    )
    coarse_cluster_ids = np.asarray(sorted(np.unique(coarse_labels)), dtype=int)

    coarse_centroids = []
    coarse_rows = []
    identity_models = {}
    for coarse_cluster in coarse_cluster_ids:
        local_indices = np.where(coarse_labels == coarse_cluster)[0]
        cluster_embeddings = known_embeddings[local_indices]
        cluster_df = known_df.iloc[local_indices].reset_index(drop=True).copy()
        cluster_df["_embedding_index"] = np.arange(len(cluster_df))

        coarse_centroid = normalize_rows(cluster_embeddings.mean(axis=0, keepdims=True))[0]
        prototypes, prototype_identities = build_identity_prototypes(
            cluster_embeddings,
            cluster_df,
        )

        coarse_centroids.append(coarse_centroid)
        coarse_rows.append(
            {
                "species": species,
                "coarse_cluster": int(coarse_cluster),
                "train_samples": int(len(cluster_df)),
                "identity_count": int(cluster_df["identity"].nunique()),
                "coarse_centroid_norm": float(np.linalg.norm(coarse_centroid)),
            }
        )
        identity_models[int(coarse_cluster)] = {
            "train_df": cluster_df,
            "train_embeddings": cluster_embeddings,
            "prototypes": prototypes,
            "prototype_identities": prototype_identities,
        }

    return {
        "coarse_centroids": np.vstack(coarse_centroids),
        "coarse_cluster_ids": coarse_cluster_ids,
        "coarse_table": pd.DataFrame(coarse_rows),
        "identity_models": identity_models,
    }


def summarize_overall(summary_df: pd.DataFrame) -> pd.DataFrame:
    eval_df = summary_df[summary_df["train_eval_samples"] > 0].copy()
    if eval_df.empty:
        return pd.DataFrame()

    weights = eval_df["train_eval_samples"].to_numpy(dtype=float)
    weighted_columns = [
        "inclusive_top1_accuracy",
        "inclusive_accept_rate",
        "leave_one_out_top1_accuracy",
        "leave_one_out_accept_rate",
    ]

    row = {
        "species": "overall",
        "train_samples": int(eval_df["train_samples"].sum()),
        "train_identities": int(eval_df["train_identities"].sum()),
        "prototype_identities": int(eval_df["prototype_identities"].sum()),
        "train_eval_samples": int(eval_df["train_eval_samples"].sum()),
        "singleton_identity_samples": int(eval_df["singleton_identity_samples"].sum()),
    }
    for column in weighted_columns:
        row[column] = float(np.average(eval_df[column].to_numpy(dtype=float), weights=weights))

    accepted_correct = (
        eval_df["inclusive_accepted_accuracy"] * eval_df["inclusive_accept_rate"] * eval_df["train_eval_samples"]
    ).sum()
    accepted_total = (eval_df["inclusive_accept_rate"] * eval_df["train_eval_samples"]).sum()
    row["inclusive_accepted_accuracy"] = (
        float(accepted_correct / accepted_total) if accepted_total else np.nan
    )

    loo_accepted_correct = (
        eval_df["leave_one_out_accepted_accuracy"]
        * eval_df["leave_one_out_accept_rate"]
        * eval_df["train_eval_samples"]
    ).sum()
    loo_accepted_total = (eval_df["leave_one_out_accept_rate"] * eval_df["train_eval_samples"]).sum()
    row["leave_one_out_accepted_accuracy"] = (
        float(loo_accepted_correct / loo_accepted_total) if loo_accepted_total else np.nan
    )

    return pd.DataFrame([row])


def get_general_train_model(
    embeddings_dir: str | Path,
    model_name: str,
    normalize: bool,
    coarse_cluster_threshold: float = 0.68,
    coarse_min_cluster_size: int = 1,
) -> dict[str, object]:
    train_embeddings, train_df = load_embedding_artifacts(
        Path(embeddings_dir),
        model_name,
        "train",
        normalize=normalize,
    )

    species_models = {}
    prototype_frames = []
    coarse_frames = []
    for species in sorted(train_df["species"].unique()):
        species_train_df = train_df[train_df["species"] == species].copy()
        species_train_embeddings = train_embeddings[
            species_train_df["_embedding_index"].to_numpy()
        ]
        species_train_df["_embedding_index"] = np.arange(len(species_train_df))

        prototypes, prototype_identities = build_identity_prototypes(
            species_train_embeddings,
            species_train_df,
        )
        prototype_table = build_prototype_table(
            species=species,
            train_df=species_train_df,
            train_embeddings=species_train_embeddings,
        )
        hierarchical_model = build_hierarchical_species_model(
            species=species,
            train_df=species_train_df,
            train_embeddings=species_train_embeddings,
            coarse_cluster_threshold=coarse_cluster_threshold,
            coarse_min_cluster_size=coarse_min_cluster_size,
        )

        species_models[species] = {
            "species": species,
            "train_df": species_train_df,
            "train_embeddings": species_train_embeddings,
            "prototypes": prototypes,
            "prototype_identities": prototype_identities,
            "prototype_table": prototype_table,
            "hierarchical_model": hierarchical_model,
        }
        prototype_frames.append(prototype_table)
        coarse_frames.append(hierarchical_model["coarse_table"])

    prototypes_df = pd.concat(prototype_frames, ignore_index=True).sort_values(
        ["species", "identity"]
    )
    coarse_df = pd.concat(coarse_frames, ignore_index=True).sort_values(
        ["species", "coarse_cluster"]
    )
    return {
        "train_df": train_df,
        "train_embeddings": train_embeddings,
        "species_models": species_models,
        "prototypes_df": prototypes_df,
        "coarse_df": coarse_df,
    }


def test_train_model(
    train_model: dict[str, object],
    known_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames = []
    summary_rows = []

    for species, species_model in train_model["species_models"].items():
        species_train_df = species_model["train_df"]
        species_train_embeddings = species_model["train_embeddings"]

        predictions, metrics = evaluate_identity_prototypes(
            species=species,
            train_df=species_train_df,
            train_embeddings=species_train_embeddings,
            known_threshold=known_threshold,
        )

        metrics.update(
            {
                "species": species,
                "train_samples": int(len(species_train_df)),
                "train_identities": int(species_train_df["identity"].nunique()),
                "prototype_identities": int(len(species_model["prototype_table"])),
                "known_threshold": known_threshold,
            }
        )
        summary_rows.append(metrics)
        prediction_frames.append(predictions)

    summary_df = pd.DataFrame(summary_rows).sort_values("species")
    overall_df = summarize_overall(summary_df)
    if not overall_df.empty:
        summary_df = pd.concat([summary_df, overall_df], ignore_index=True)

    predictions_df = pd.concat(prediction_frames, ignore_index=True).sort_values("image_id")
    return summary_df, predictions_df


def main() -> None:
    config = {
        "embeddings_dir": "data_embeddings",
        "model_name": "mega",
        "known_threshold": 0.72,
        "coarse_cluster_threshold": 0.68,
        "coarse_min_cluster_size": 1,
        "no_normalize": False,
        "summary_output": "artifacts/train_experiment_summary.csv",
        "predictions_output": "artifacts/train_experiment_predictions.csv",
        "prototypes_output": "artifacts/train_experiment_prototypes.csv",
        "coarse_output": "artifacts/train_experiment_coarse_clusters.csv",
    }

    summary_output_path = Path(config["summary_output"])
    predictions_output_path = Path(config["predictions_output"])
    prototypes_output_path = Path(config["prototypes_output"])
    coarse_output_path = Path(config["coarse_output"])
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_output_path.parent.mkdir(parents=True, exist_ok=True)
    prototypes_output_path.parent.mkdir(parents=True, exist_ok=True)
    coarse_output_path.parent.mkdir(parents=True, exist_ok=True)

    train_model = get_general_train_model(
        embeddings_dir=config["embeddings_dir"],
        model_name=config["model_name"],
        normalize=not config["no_normalize"],
        coarse_cluster_threshold=float(config["coarse_cluster_threshold"]),
        coarse_min_cluster_size=int(config["coarse_min_cluster_size"]),
    )
    summary_df, predictions_df = test_train_model(
        train_model=train_model,
        known_threshold=float(config["known_threshold"]),
    )
    prototypes_df = train_model["prototypes_df"]
    coarse_df = train_model["coarse_df"]

    summary_df.to_csv(summary_output_path, index=False)
    predictions_df.to_csv(predictions_output_path, index=False)
    prototypes_df.to_csv(prototypes_output_path, index=False)
    coarse_df.to_csv(coarse_output_path, index=False)

    logger.info("Wrote train experiment summary to %s", summary_output_path)
    logger.info("Wrote train experiment predictions to %s", predictions_output_path)
    logger.info("Wrote train experiment prototypes to %s", prototypes_output_path)
    logger.info("Wrote train experiment coarse clusters to %s", coarse_output_path)


if __name__ == "__main__":
    main()
