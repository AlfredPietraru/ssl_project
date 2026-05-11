import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import CFG
from main_utils import normalize_rows

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GALLERY_VALIDATION")


def load_train_embeddings(embeddings_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    embeddings_path = embeddings_dir / "train_embeddings.npy"
    metadata_path = embeddings_dir / "train_metadata.csv"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Could not find train embeddings at '{embeddings_path}'")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find train metadata at '{metadata_path}'")

    embeddings = np.load(embeddings_path)
    metadata = pd.read_csv(metadata_path)

    if len(embeddings) != len(metadata):
        raise ValueError(
            "Train embeddings and train metadata have different lengths: "
            f"{len(embeddings)} vs {len(metadata)}"
        )

    required_columns = {"identity", "embedding_index"}
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        raise ValueError(
            f"Train metadata is missing required columns: {sorted(missing_columns)}"
        )

    return embeddings, metadata


def filter_known_training_identities(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    identity_series = metadata["identity"].astype(str).str.strip()
    known_mask = ~identity_series.str.lower().eq("unknown")
    filtered_metadata = metadata.loc[known_mask].reset_index(drop=True).copy()
    filtered_embeddings = embeddings[known_mask.to_numpy()]

    if filtered_metadata.empty:
        raise ValueError("No known training identities remained after filtering 'unknown'.")

    filtered_metadata["embedding_index"] = np.arange(len(filtered_metadata))
    return filtered_embeddings, filtered_metadata

def build_identity_gallery(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    gallery_rows: list[dict[str, object]] = []
    prototype_vectors: list[np.ndarray] = []

    grouped = metadata.groupby("identity", sort=True)
    for identity, group in grouped:
        indices = group["embedding_index"].to_numpy(dtype=int)
        identity_embeddings = embeddings[indices]
        prototype = normalize_rows(identity_embeddings.mean(axis=0, keepdims=True))[0]
        prototype_vectors.append(prototype)

        row: dict[str, object] = {
            "identity": identity,
            "prototype_index": len(prototype_vectors) - 1,
            "num_images": int(len(group)),
        }
        for candidate in ["species", "dataset"]:
            if candidate in group.columns:
                row[candidate] = group.iloc[0][candidate]
        gallery_rows.append(row)

    gallery_metadata = pd.DataFrame(gallery_rows)
    gallery_embeddings = np.stack(prototype_vectors, axis=0)
    return gallery_embeddings, gallery_metadata


def split_identities_for_open_set_validation(
    metadata: pd.DataFrame,
    known_val_ratio: float,
    unseen_val_ratio: float,
    random_seed: int,
) -> tuple[set[str], set[str], set[str]]:
    if known_val_ratio < 0 or unseen_val_ratio < 0:
        raise ValueError("Validation ratios must be non-negative.")
    if known_val_ratio + unseen_val_ratio >= 1.0:
        raise ValueError("known_val_ratio + unseen_val_ratio must be smaller than 1.0.")

    rng = np.random.default_rng(random_seed)
    species_column = "species" if "species" in metadata.columns else None

    known_train_ids: set[str] = set()
    known_val_ids: set[str] = set()
    unseen_val_ids: set[str] = set()

    if species_column is None:
        species_groups = [("all", metadata)]
    else:
        species_groups = list(metadata.groupby(species_column, sort=True))

    for _, species_df in species_groups:
        identities = species_df["identity"].astype(str).drop_duplicates().to_numpy()
        if len(identities) < 3:
            raise ValueError(
                "Each species group needs at least 3 identities to create "
                "known_train, known_val, and unseen_val splits."
            )

        shuffled = identities.copy()
        rng.shuffle(shuffled)

        unseen_count = max(1, int(round(len(shuffled) * unseen_val_ratio)))
        known_val_count = max(1, int(round(len(shuffled) * known_val_ratio)))

        if unseen_count + known_val_count >= len(shuffled):
            unseen_count = 1
            known_val_count = 1

        unseen_ids = set(shuffled[:unseen_count])
        known_val_start = unseen_count
        known_val_end = unseen_count + known_val_count
        known_val_species_ids = set(shuffled[known_val_start:known_val_end])
        known_train_species_ids = set(shuffled[known_val_end:])

        if not known_train_species_ids:
            moved_identity = known_val_species_ids.pop()
            known_train_species_ids.add(moved_identity)

        known_train_ids.update(known_train_species_ids)
        known_val_ids.update(known_val_species_ids)
        unseen_val_ids.update(unseen_ids)

    return known_train_ids, known_val_ids, unseen_val_ids


def slice_split(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    identities: set[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    mask = metadata["identity"].astype(str).isin(identities).to_numpy()
    split_embeddings = embeddings[mask]
    split_metadata = metadata.loc[mask].reset_index(drop=True).copy()
    split_metadata["embedding_index"] = np.arange(len(split_metadata))
    return split_embeddings, split_metadata


def save_split(
    name: str,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / f"{name}_embeddings.npy"
    metadata_path = output_dir / f"{name}_metadata.csv"
    np.save(embeddings_path, embeddings)
    metadata.to_csv(metadata_path, index=False)
    return embeddings_path, metadata_path


def build_gallery_and_validation(
    cfg: CFG,
) -> None:
    output_dir = Path(cfg.gallery_validation_output_dir)
    embeddings_dir = Path(cfg.embeddings_output_dir)
    embeddings, metadata = load_train_embeddings(embeddings_dir)
    embeddings, metadata = filter_known_training_identities(embeddings, metadata)

    full_gallery_embeddings, full_gallery_metadata = build_identity_gallery(
        embeddings=embeddings,
        metadata=metadata,
    )
    save_split("full_gallery_prototypes", full_gallery_embeddings, full_gallery_metadata, output_dir)
    save_split("full_gallery_images", embeddings, metadata, output_dir)

    known_train_ids, known_val_ids, unseen_val_ids = split_identities_for_open_set_validation(
        metadata=metadata,
        known_val_ratio=cfg.known_val_ratio,
        unseen_val_ratio=cfg.unseen_val_ratio,
        random_seed=cfg.validation_random_seed,
    )

    known_train_embeddings, known_train_metadata = slice_split(
        embeddings=embeddings,
        metadata=metadata,
        identities=known_train_ids,
    )
    known_val_embeddings, known_val_metadata = slice_split(
        embeddings=embeddings,
        metadata=metadata,
        identities=known_val_ids,
    )
    unseen_val_embeddings, unseen_val_metadata = slice_split(
        embeddings=embeddings,
        metadata=metadata,
        identities=unseen_val_ids,
    )

    gallery_embeddings, gallery_metadata = build_identity_gallery(
        embeddings=known_train_embeddings,
        metadata=known_train_metadata,
    )

    save_split("gallery_prototypes", gallery_embeddings, gallery_metadata, output_dir)
    save_split("gallery_images", known_train_embeddings, known_train_metadata, output_dir)
    save_split("known_val", known_val_embeddings, known_val_metadata, output_dir)
    save_split("unseen_val", unseen_val_embeddings, unseen_val_metadata, output_dir)

    summary = {
        "random_seed": cfg.validation_random_seed,
        "known_val_ratio": cfg.known_val_ratio,
        "unseen_val_ratio": cfg.unseen_val_ratio,
        "train_samples_total": int(len(metadata)),
        "full_gallery_image_samples": int(len(metadata)),
        "full_gallery_identity_count": int(full_gallery_metadata["identity"].nunique()),
        "gallery_image_samples": int(len(known_train_metadata)),
        "gallery_identity_count": int(gallery_metadata["identity"].nunique()),
        "known_val_samples": int(len(known_val_metadata)),
        "known_val_identities": int(known_val_metadata["identity"].nunique()),
        "unseen_val_samples": int(len(unseen_val_metadata)),
        "unseen_val_identities": int(unseen_val_metadata["identity"].nunique()),
    }
    summary_path = output_dir / "validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    logger.info("Saved gallery and validation artifacts to %s", output_dir)
    logger.info("Validation summary written to %s", summary_path)


if __name__ == "__main__":
    cfg = CFG("config.yaml")
    build_gallery_and_validation(cfg)
