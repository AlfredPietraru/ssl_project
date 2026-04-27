import argparse
import logging
import os
from pathlib import Path
import torchvision.transforms as T

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import torch
import timm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from wildlife_datasets.datasets import AnimalCLEF2026


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EMBEDDING_COMPARISON")


def clean_string_array(values, missing_value: str = "unknown") -> np.ndarray:
    series = pd.Series(values, dtype="object")
    series = series.where(series.notna(), missing_value)
    series = series.astype(str).str.strip()
    series = series.mask(
        series.eq("") | series.str.lower().isin({"nan", "none"}),
        missing_value,
    )
    return series.to_numpy(dtype=str)


def build_dataset(root: str, split: str, max_samples: int | None):
    transform = T.Compose([
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    dataset = AnimalCLEF2026(
        root,
        transform=transform,
        load_label=True,
        factorize_label=True,
        check_files=False,
    )

    dataset = dataset.get_subset(dataset.df["split"] == split)

    if max_samples is not None:
        dataset = dataset.get_subset(dataset.df.index[:max_samples])

    return dataset


def normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def build_model_default(device: torch.device) -> torch.nn.Module:
    logger.info("Loading default MegaDescriptor-L-384 from Hugging Face...")
    model = timm.create_model(
        "hf-hub:BVRA/MegaDescriptor-L-384",
        pretrained=True,
    )
    model.eval()
    model.to(device)
    return model


def build_model_finetuned(model_path: Path, device: torch.device) -> torch.nn.Module:
    logger.info("Loading fine-tuned embedding model from %s", model_path)

    obj = torch.load(model_path, map_location=device)

    if isinstance(obj, torch.nn.Module):
        model = obj
        model.eval()
        model.to(device)
        return model

    model = timm.create_model(
        "hf-hub:BVRA/MegaDescriptor-L-384",
        pretrained=False,
    )

    if isinstance(obj, dict) and "backbone_state_dict" in obj:
        model.load_state_dict(obj["backbone_state_dict"])
    elif isinstance(obj, dict) and "model_state_dict" in obj:
        state_dict = obj["model_state_dict"]

        backbone_state_dict = {
            key.replace("backbone.", ""): value
            for key, value in state_dict.items()
            if key.startswith("backbone.")
        }

        if not backbone_state_dict:
            raise ValueError(
                "Checkpoint has model_state_dict, but no keys starting with 'backbone.'."
            )

        model.load_state_dict(backbone_state_dict)
    elif isinstance(obj, dict):
        model.load_state_dict(obj)
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(obj)}")

    model.eval()
    model.to(device)
    return model


def extract_metadata(dataset) -> pd.DataFrame:
    df = dataset.df.reset_index(drop=True).copy()

    if "identity" in df.columns:
        df["label_string"] = clean_string_array(df["identity"])
    elif "label" in df.columns:
        df["label_string"] = clean_string_array(df["label"])
    else:
        df["label_string"] = "unknown"

    for candidate in ["species", "species_string", "category", "dataset"]:
        if candidate in df.columns:
            df["species_string"] = clean_string_array(df[candidate])
            break
    else:
        df["species_string"] = "unknown"

    df.insert(0, "embedding_index", np.arange(len(df)))
    return df


def extract_embeddings(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    all_embeddings = []

    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        for batch_idx, batch in enumerate(loader, start=1):
            if isinstance(batch, dict):
                images = batch["image"]
            else:
                images = batch[0]

            images = images.to(device, non_blocking=True)

            embeddings = model(images)

            if isinstance(embeddings, tuple):
                embeddings = embeddings[0]

            all_embeddings.append(embeddings.detach().cpu().float().numpy())

            if batch_idx % 20 == 0:
                logger.info("Processed batch %d/%d", batch_idx, len(loader))

    return np.concatenate(all_embeddings, axis=0)


def run_pca(embeddings: np.ndarray, normalize: bool) -> tuple[np.ndarray, PCA]:
    if normalize:
        embeddings = normalize_rows(embeddings)

    pca = PCA(n_components=2, random_state=42)
    pca_embeddings = pca.fit_transform(embeddings)

    logger.info(
        "PCA variance: PC1=%.4f, PC2=%.4f, total=%.4f",
        pca.explained_variance_ratio_[0],
        pca.explained_variance_ratio_[1],
        pca.explained_variance_ratio_.sum(),
    )

    return pca_embeddings, pca


def plot_pca(
    pca_embeddings: np.ndarray,
    metadata: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    labels = clean_string_array(metadata["label_string"])
    species = clean_string_array(metadata["species_string"])

    unique_labels, label_ids = np.unique(labels, return_inverse=True)
    unique_species, species_ids = np.unique(species, return_inverse=True)

    fig, ax = plt.subplots(figsize=(13, 10))

    scatter = ax.scatter(
        pca_embeddings[:, 0],
        pca_embeddings[:, 1],
        c=label_ids,
        cmap="tab20",
        s=24,
        alpha=0.75,
        edgecolors="none",
    )

    for species_id, species_name in enumerate(unique_species):
        mask = species_ids == species_id
        center = pca_embeddings[mask].mean(axis=0)

        ax.scatter(
            center[0],
            center[1],
            marker="X",
            s=180,
            color="black",
            linewidths=0.8,
        )
        ax.text(
            center[0],
            center[1],
            species_name,
            fontsize=10,
            ha="left",
            va="bottom",
        )

    if len(unique_labels) <= 30:
        colorbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
        colorbar.set_label("Identity index")

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pca_csv(
    pca_embeddings: np.ndarray,
    metadata: pd.DataFrame,
    output_path: Path,
) -> None:
    df = metadata.reset_index(drop=True).copy()
    df["pc1"] = pca_embeddings[:, 0]
    df["pc2"] = pca_embeddings[:, 1]
    df.to_csv(output_path, index=False)


def process_model(
    model_type: str,
    model: torch.nn.Module,
    dataset,
    metadata: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    normalize: bool,
    output_dir: Path,
    split: str,
) -> None:
    logger.info("Extracting embeddings for model: %s", model_type)

    embeddings = extract_embeddings(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    logger.info("%s embeddings shape: %s", model_type, embeddings.shape)
    pca_embeddings, _ = run_pca(embeddings, normalize=normalize)

    plot_pca(
        pca_embeddings=pca_embeddings,
        metadata=metadata,
        title=f"AnimalCLEF2026 {split} PCA - {model_type}",
        output_path=output_dir / f"{model_type}_{split}_pca.png",
    )

    # save_pca_csv(
    #     pca_embeddings=pca_embeddings,
    #     metadata=metadata,
    #     output_path=output_dir / f"{model_type}_{split}_pca.csv",
    # )

    logger.info("Saved outputs for model: %s", model_type)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare default MegaDescriptor and fine-tuned MegaDescriptor embeddings with PCA."
    )

    parser.add_argument("--root", default="data")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--finetuned-model-path", default="artifacts/embedding_checkpoints/embedding_backbone.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="artifacts/embedding_comparison",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    dataset = build_dataset(
        root=args.root,
        split=args.split,
        max_samples=args.max_samples,
    )
    metadata = extract_metadata(dataset)

    logger.info("Loaded %d samples from split=%s", len(dataset), args.split)

    default_model = build_model_default(device)
    process_model(
        model_type="default_mega",
        model=default_model,
        dataset=dataset,
        metadata=metadata,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=args.normalize,
        output_dir=output_dir,
        split=args.split,
    )

    del default_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    finetuned_model = build_model_finetuned(
        model_path=Path(args.finetuned_model_path),
        device=device,
    )
    process_model(
        model_type="finetuned_mega",
        model=finetuned_model,
        dataset=dataset,
        metadata=metadata,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=args.normalize,
        output_dir=output_dir,
        split=args.split,
    )


if __name__ == "__main__":
    main()