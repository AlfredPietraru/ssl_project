import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from wildlife_datasets.datasets import AnimalCLEF2026

from main_utils import download_mega_descriptor_model_feature_extraction


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EMBEDDING_EXTRACTION")


def build_transform(image_size: int = 384) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


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
    required_columns = {"image_id", "identity", "split", "species", "dataset", "path"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset metadata is missing required columns: {sorted(missing_columns)}")

    df["identity"] = clean_string_series(df["identity"])
    df["split"] = clean_string_series(df["split"])
    df["species"] = clean_string_series(df["species"])
    df["dataset"] = clean_string_series(df["dataset"])

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


def extract_embeddings_from_batch(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        if hasattr(model, "forward_features"):
            features = model.forward_features(batch)
            if hasattr(model, "forward_head"):
                try:
                    features = model.forward_head(features, pre_logits=True)
                except TypeError:
                    features = model.forward_head(features)
        else:
            features = model(batch)

    if isinstance(features, (tuple, list)):
        features = features[0]
    if features.ndim > 2:
        features = torch.flatten(features, start_dim=1)
    return features


def compute_embeddings(
    dataset: AnimalCLEF2026,
    model: torch.nn.Module,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    normalize: bool,
) -> np.ndarray:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    model = model.to(device)
    model.eval()

    all_embeddings = []
    for step, (images, _) in enumerate(dataloader, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        embeddings = extract_embeddings_from_batch(model, images)
        all_embeddings.append(embeddings.detach().cpu())
        if step % 20 == 0 or step == len(dataloader):
            logger.info("Processed %s/%s batches", step, len(dataloader))

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    if normalize:
        embeddings = normalize_rows(embeddings)
    return embeddings.astype(np.float32)


def build_split_dataset(
    dataset: AnimalCLEF2026,
    metadata: pd.DataFrame,
    split: str,
    max_samples: int | None,
) -> tuple[AnimalCLEF2026, pd.DataFrame]:
    split_indices = metadata.index[metadata["split"] == split].to_numpy()
    if max_samples is not None:
        split_indices = split_indices[:max_samples]

    split_dataset = dataset.get_subset(split_indices.tolist())
    split_dataset.set_transform(build_transform())

    split_metadata = metadata.loc[split_indices].reset_index(drop=True).copy()
    split_metadata.insert(0, "embedding_index", np.arange(len(split_metadata)))
    return split_dataset, split_metadata


def metadata_output_columns(metadata: pd.DataFrame) -> list[str]:
    preferred_columns = [
        "embedding_index",
        "image_id",
        "split",
        "species",
        "identity",
        "dataset",
        "path",
        "date",
        "orientation",
    ]
    return [column for column in preferred_columns if column in metadata.columns]


def save_split_artifacts(
    split: str,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: Path,
    model_name: str,
) -> None:
    embeddings_path = output_dir / f"{model_name}_{split}_embeddings.npy"
    metadata_path = output_dir / f"{model_name}_{split}_metadata.csv"

    np.save(embeddings_path, embeddings)
    metadata[metadata_output_columns(metadata)].to_csv(metadata_path, index=False)

    logger.info("Saved %s embeddings to %s with shape %s", split, embeddings_path, embeddings.shape)
    logger.info("Saved %s metadata to %s", split, metadata_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract AnimalCLEF2026 image embeddings once and save row-aligned .npy + metadata CSV artifacts."
    )
    parser.add_argument("--root", default="data", help="Dataset root directory.")
    parser.add_argument("--output-dir", default="data_embeddings", help="Directory for embedding artifacts.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers for image loading.")
    parser.add_argument(
        "--split",
        choices=("train", "test", "both"),
        default="both",
        help="Which split to extract.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional quick-test limit applied to each selected split.",
    )
    parser.add_argument(
        "--model-name",
        default="mega",
        help="Prefix used in saved artifact filenames.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Store raw embeddings instead of L2-normalized embeddings.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not recompute a split if both its .npy and metadata CSV already exist.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root '{root}' does not exist.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = AnimalCLEF2026(
        str(root),
        transform=None,
        load_label=True,
        factorize_label=True,
        check_files=False,
    )
    metadata = prepare_metadata(dataset.df)

    selected_splits = ("train", "test") if args.split == "both" else (args.split,)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = None
    for split in selected_splits:
        embeddings_path = output_dir / f"{args.model_name}_{split}_embeddings.npy"
        metadata_path = output_dir / f"{args.model_name}_{split}_metadata.csv"
        if args.skip_existing and embeddings_path.exists() and metadata_path.exists():
            logger.info("Skipping %s because %s and %s already exist", split, embeddings_path, metadata_path)
            continue

        split_dataset, split_metadata = build_split_dataset(
            dataset,
            metadata,
            split,
            args.max_samples,
        )
        logger.info("Extracting %s embeddings for %s samples", split, len(split_dataset))

        if model is None:
            model = download_mega_descriptor_model_feature_extraction()

        embeddings = compute_embeddings(
            split_dataset,
            model,
            args.batch_size,
            args.num_workers,
            device,
            normalize=not args.no_normalize,
        )
        save_split_artifacts(split, embeddings, split_metadata, output_dir, args.model_name)


if __name__ == "__main__":
    main()
