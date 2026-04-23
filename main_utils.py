import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from wildlife_datasets.datasets import AnimalCLEF2026

from dotenv import load_dotenv
load_dotenv()
import kagglehub


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MAIN")


def download_dataset():
    if os.path.isdir("data"):
        logger.info("`data` folder already exists. Skipping download.")
        return AnimalCLEF2026("data")
    
    logger.info("`data` folder not found. Downloading dataset...")
    try:
        path_of_files = kagglehub.competition_download("animal-clef-2026")
    except Exception:
        logger.exception("Dataset download failed in kagglehub.competition_download().")
        return None

    try:
        shutil.move(path_of_files, "data")
    except Exception:
        logger.exception("Dataset download succeeded, but moving files to `data` failed.")
        return
    logger.info("Dataset downloaded and moved to `data`.")
    return AnimalCLEF2026("data")


def download_mega_descriptor_model_feature_extraction():
    import timm
    try:
        m = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
    except Exception:
        logger.exception("Model for embeddings failed downloading...")
        raise
    m = m.eval()
    return m

def download_wildlife_pretraining():
    try:
        path = kagglehub.dataset_download("wildlifedatasets/wildlifereid-10k")
        shutil.move(path, "pretrained_data")
    except Exception:
        logger.exception("Failed downloading other moving dataset to the right path")


def clean_string_series(values: pd.Series, missing_value: str = "unknown") -> pd.Series:
    series = pd.Series(values, dtype="object")
    series = series.where(series.notna(), missing_value)
    series = series.astype(str).str.strip()
    return series.mask(series.eq("") | series.str.lower().isin({"nan", "none"}), missing_value)


def infer_species_from_text(row: pd.Series) -> str:
    for column in ("identity", "dataset", "path", "image_id"):
        value = str(row.get(column, "")).lower()
        if "salamander" in value:
            return "salamander"
        if "lynx" in value:
            return "lynx"
        if "turtle" in value:
            return "loggerhead turtle"
        if "lizard" in value:
            return "lizard"
    return "unknown"


def prepare_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_columns = {"image_id", "identity", "split", "species", "dataset"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset metadata is missing required columns: {sorted(missing_columns)}")

    df["identity"] = clean_string_series(df["identity"])
    df["split"] = clean_string_series(df["split"])
    df["dataset"] = clean_string_series(df["dataset"])
    df["species"] = clean_string_series(df["species"])

    missing_species = df["species"].eq("unknown")
    if missing_species.any():
        df.loc[missing_species, "species"] = df.loc[missing_species].apply(
            infer_species_from_text,
            axis=1,
        )
    return df


def normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def load_embedding_artifacts(
    embeddings_dir: Path,
    model_name: str,
    split: str,
    normalize: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    embeddings_path = embeddings_dir / f"{model_name}_{split}_embeddings.npy"
    metadata_path = embeddings_dir / f"{model_name}_{split}_metadata.csv"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Could not find {embeddings_path}. Run embedding_extraction.py first."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Could not find {metadata_path}. Run embedding_extraction.py first."
        )

    embeddings = np.load(embeddings_path)
    metadata = prepare_metadata(pd.read_csv(metadata_path))

    if len(metadata) != len(embeddings):
        raise ValueError(
            f"{split} artifact row mismatch: {len(metadata)} metadata rows but "
            f"{len(embeddings)} embeddings."
        )

    if "embedding_index" in metadata.columns:
        expected = np.arange(len(metadata))
        actual = metadata["embedding_index"].to_numpy()
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"{metadata_path} has non-contiguous embedding_index values. "
                "The CSV rows must align exactly with the .npy rows."
            )
    else:
        metadata.insert(0, "embedding_index", np.arange(len(metadata)))

    if normalize:
        embeddings = normalize_rows(embeddings)

    metadata["_embedding_index"] = np.arange(len(metadata))
    logger.info("Loaded %s embeddings from %s with shape %s", split, embeddings_path, embeddings.shape)
    logger.info("Loaded %s metadata from %s", split, metadata_path)
    return embeddings.astype(np.float32), metadata


def build_identity_prototypes(
    train_embeddings: np.ndarray,
    train_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    known_df = train_df[train_df["identity"] != "unknown"].copy()
    if known_df.empty:
        return np.empty((0, train_embeddings.shape[1]), dtype=train_embeddings.dtype), np.array([])

    prototypes = []
    identities = []
    for identity, identity_df in known_df.groupby("identity", sort=True):
        centroid = train_embeddings[identity_df["_embedding_index"].to_numpy()].mean(axis=0, keepdims=True)
        prototypes.append(normalize_rows(centroid)[0])
        identities.append(identity)

    return np.vstack(prototypes), np.asarray(identities, dtype=object)


def evaluate_identity_prototypes(
    species: str,
    train_df: pd.DataFrame,
    train_embeddings: np.ndarray,
    known_threshold: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    known_df = train_df[train_df["identity"] != "unknown"].copy()
    if known_df.empty:
        empty = pd.DataFrame(
            columns=[
                "image_id",
                "species",
                "identity",
                "prototype_train_count",
                "inclusive_best_identity",
                "inclusive_best_similarity",
                "inclusive_correct",
                "inclusive_accepted",
                "leave_one_out_best_identity",
                "leave_one_out_best_similarity",
                "leave_one_out_correct",
                "leave_one_out_accepted",
            ]
        )
        return empty, {
            "train_eval_samples": 0,
            "inclusive_top1_accuracy": np.nan,
            "inclusive_accepted_accuracy": np.nan,
            "inclusive_accept_rate": np.nan,
            "leave_one_out_top1_accuracy": np.nan,
            "leave_one_out_accepted_accuracy": np.nan,
            "leave_one_out_accept_rate": np.nan,
            "singleton_identity_samples": 0,
        }

    known_embeddings = train_embeddings[known_df["_embedding_index"].to_numpy()]
    known_df = known_df.reset_index(drop=True)
    known_df["_local_embedding_index"] = np.arange(len(known_df))

    prototypes, prototype_identities = build_identity_prototypes(known_embeddings, known_df)
    similarity = known_embeddings @ prototypes.T
    inclusive_best_indices = similarity.argmax(axis=1)
    inclusive_best_scores = similarity[np.arange(len(known_embeddings)), inclusive_best_indices]
    inclusive_best_identities = prototype_identities[inclusive_best_indices]

    identity_to_prototype_index = {
        identity: index for index, identity in enumerate(prototype_identities)
    }
    own_prototype_indices = known_df["identity"].map(identity_to_prototype_index).to_numpy()
    identity_counts = known_df.groupby("identity").size()
    train_counts = known_df["identity"].map(identity_counts).to_numpy()

    leave_one_out_similarity = similarity.copy()
    for identity, identity_df in known_df.groupby("identity", sort=True):
        prototype_index = identity_to_prototype_index[identity]
        local_indices = identity_df["_local_embedding_index"].to_numpy()
        if len(local_indices) == 1:
            leave_one_out_similarity[local_indices, prototype_index] = -np.inf
            continue

        identity_sum = known_embeddings[local_indices].sum(axis=0, keepdims=True)
        leave_one_out_centroids = identity_sum - known_embeddings[local_indices]
        leave_one_out_centroids = normalize_rows(leave_one_out_centroids)
        leave_one_out_similarity[local_indices, prototype_index] = np.sum(
            known_embeddings[local_indices] * leave_one_out_centroids,
            axis=1,
        )

    leave_one_out_best_indices = leave_one_out_similarity.argmax(axis=1)
    leave_one_out_best_scores = leave_one_out_similarity[
        np.arange(len(known_embeddings)),
        leave_one_out_best_indices,
    ]
    leave_one_out_best_identities = prototype_identities[leave_one_out_best_indices]

    true_identities = known_df["identity"].to_numpy()
    inclusive_correct = inclusive_best_identities == true_identities
    leave_one_out_correct = leave_one_out_best_identities == true_identities
    inclusive_accepted = inclusive_best_scores >= known_threshold
    leave_one_out_accepted = leave_one_out_best_scores >= known_threshold

    def accepted_accuracy(correct: np.ndarray, accepted: np.ndarray) -> float:
        if not accepted.any():
            return np.nan
        return float(correct[accepted].mean())

    result = known_df[["image_id", "species", "identity"]].copy()
    result["prototype_train_count"] = train_counts
    result["inclusive_best_identity"] = inclusive_best_identities
    result["inclusive_best_similarity"] = inclusive_best_scores
    result["inclusive_correct"] = inclusive_correct
    result["inclusive_accepted"] = inclusive_accepted
    result["leave_one_out_best_identity"] = leave_one_out_best_identities
    result["leave_one_out_best_similarity"] = leave_one_out_best_scores
    result["leave_one_out_correct"] = leave_one_out_correct
    result["leave_one_out_accepted"] = leave_one_out_accepted

    summary = {
        "train_eval_samples": int(len(known_df)),
        "inclusive_top1_accuracy": float(inclusive_correct.mean()),
        "inclusive_accepted_accuracy": accepted_accuracy(inclusive_correct, inclusive_accepted),
        "inclusive_accept_rate": float(inclusive_accepted.mean()),
        "leave_one_out_top1_accuracy": float(leave_one_out_correct.mean()),
        "leave_one_out_accepted_accuracy": accepted_accuracy(
            leave_one_out_correct,
            leave_one_out_accepted,
        ),
        "leave_one_out_accept_rate": float(leave_one_out_accepted.mean()),
        "singleton_identity_samples": int((train_counts == 1).sum()),
    }
    logger.info(
        "[%s] train prototype eval inclusive_top1=%.4f leave_one_out_top1=%.4f",
        species,
        summary["inclusive_top1_accuracy"],
        summary["leave_one_out_top1_accuracy"],
    )
    return result, summary


def agglomerative_labels_from_similarity(
    embeddings: np.ndarray,
    similarity_threshold: float,
    min_cluster_size: int,
) -> np.ndarray:
    n_samples = len(embeddings)
    if n_samples == 0:
        return np.array([], dtype=int)
    if n_samples == 1:
        return np.array([0], dtype=int)

    distance = np.clip(1.0 - embeddings @ embeddings.T, 0.0, 2.0)
    distance_threshold = 1.0 - similarity_threshold

    try:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=distance_threshold,
        )
    except TypeError:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage="average",
            distance_threshold=distance_threshold,
        )

    labels = clustering.fit_predict(distance)
    labels = split_small_clusters(labels, min_cluster_size)
    return relabel_contiguously(labels)


def split_small_clusters(labels: np.ndarray, min_cluster_size: int) -> np.ndarray:
    labels = labels.copy()
    next_label = labels.max() + 1 if len(labels) else 0
    for label in sorted(np.unique(labels)):
        indices = np.where(labels == label)[0]
        if len(indices) >= min_cluster_size:
            continue
        for index in indices:
            labels[index] = next_label
            next_label += 1
    return labels


def relabel_contiguously(labels: np.ndarray) -> np.ndarray:
    mapping = {label: idx for idx, label in enumerate(sorted(np.unique(labels)))}
    return np.asarray([mapping[label] for label in labels], dtype=int)
    

if __name__ == "__main__":
    dataset = download_dataset()
    if dataset is None:
        print("empty dataset")
        exit(1)
    print(dataset.metadata)
