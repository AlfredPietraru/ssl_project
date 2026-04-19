import argparse
import logging
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from wildlife_datasets.datasets import AnimalCLEF2026

from main_utils import download_mega_descriptor_model_feature_extraction


# Keep matplotlib cache writable when running from restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EMBEDDINGS_TEST")


def build_transform(image_size: int = 384) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_test_dataset(root: str, max_samples: int | None = None) -> AnimalCLEF2026:
    dataset_full = AnimalCLEF2026(
        root,
        transform=None,
        load_label=True,
        factorize_label=True,
        check_files=False,
    )
    dataset_test = dataset_full.get_subset(dataset_full.df["split"] == "test")
    dataset_test.set_transform(build_transform())

    if max_samples is not None:
        max_samples = min(max_samples, len(dataset_test))
        dataset_test = dataset_test.get_subset(list(range(max_samples)))

    return dataset_test


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


def resolve_label_strings(dataset: AnimalCLEF2026) -> np.ndarray:
    if hasattr(dataset, "labels_string"):
        return np.asarray(dataset.labels_string)

    if "identity" in dataset.df.columns:
        return dataset.df["identity"].astype(str).to_numpy()

    return dataset.df.iloc[:, 0].astype(str).to_numpy()


def resolve_species_strings(dataset: AnimalCLEF2026) -> np.ndarray:
    for candidate in ("species", "category", "dataset"):
        if candidate in dataset.df.columns:
            return dataset.df[candidate].astype(str).to_numpy()
    return np.array(["unknown"] * len(dataset))


def compute_embeddings(
    dataset: AnimalCLEF2026,
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = model.to(device)
    model.eval()

    all_embeddings = []
    for step, (images, _) in enumerate(dataloader, start=1):
        images = images.to(device)
        embeddings = extract_embeddings_from_batch(model, images)
        all_embeddings.append(embeddings.detach().cpu())
        if step % 10 == 0 or step == len(dataloader):
            logger.info("Processed %s/%s batches", step, len(dataloader))

    return torch.cat(all_embeddings, dim=0).numpy()


def plot_embeddings(
    pca_embeddings: np.ndarray,
    labels: np.ndarray,
    species: np.ndarray,
    output_path: Path,
) -> None:
    unique_labels, label_ids = np.unique(labels, return_inverse=True)
    unique_species, species_ids = np.unique(species, return_inverse=True)

    fig, ax = plt.subplots(figsize=(13, 10))
    scatter = ax.scatter(
        pca_embeddings[:, 0],
        pca_embeddings[:, 1],
        c=label_ids,
        cmap="tab20",
        s=24,
        alpha=0.78,
        edgecolors="none",
    )

    ax.set_title("AnimalCLEF2026 Test Embeddings Projected with PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)

    # Mark species centroids so the plot stays readable even with many identities.
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
        ax.text(center[0], center[1], species_name, fontsize=10, ha="left", va="bottom")

    # Only show a compact colorbar when the number of identities is reasonable.
    if len(unique_labels) <= 30:
        colorbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
        colorbar.set_label("Identity index")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_class_separation():
    csv_path = Path("artifacts/animalclef2026_test_pca.csv")
    output_plot = Path("artifacts/animalclef2026_test_pca_by_class.png")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}. Generate the PCA CSV first or point this function to an existing file."
        )

    df = pd.read_csv(csv_path)
    required_columns = {"pc1", "pc2", "species"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")

    plot_df = df.dropna(subset=["pc1", "pc2", "species"]).copy()
    if plot_df.empty:
        raise ValueError("No valid PCA rows with class labels were found in the CSV.")

    classes = sorted(plot_df["species"].astype(str).unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

    fig, ax = plt.subplots(figsize=(12, 9))
    class_centroids = {}
    within_class_scatter = {}

    for color, class_name in zip(colors, classes):
        class_df = plot_df[plot_df["species"].astype(str) == class_name]
        points = class_df[["pc1", "pc2"]].to_numpy()
        centroid = points.mean(axis=0)
        class_centroids[class_name] = centroid

        distances = np.linalg.norm(points - centroid, axis=1)
        within_class_scatter[class_name] = float(distances.mean())

        ax.scatter(
            points[:, 0],
            points[:, 1],
            label=f"{class_name} (n={len(points)})",
            s=28,
            alpha=0.72,
            color=color,
            edgecolors="none",
        )
        ax.scatter(
            centroid[0],
            centroid[1],
            marker="X",
            s=220,
            color=color,
            edgecolors="black",
            linewidths=1.0,
        )
        ax.text(
            centroid[0],
            centroid[1],
            f" {class_name}",
            fontsize=11,
            ha="left",
            va="bottom",
        )

    ax.set_title("AnimalCLEF2026 Test PCA Colored by Class")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_plot, dpi=200, bbox_inches="tight")
    plt.close(fig)

    centroid_rows = []
    class_names = list(class_centroids.keys())
    for i, class_a in enumerate(class_names):
        for class_b in class_names[i + 1:]:
            distance = float(np.linalg.norm(class_centroids[class_a] - class_centroids[class_b]))
            centroid_rows.append(
                {"class_a": class_a, "class_b": class_b, "centroid_distance": distance}
            )

    separation_df = pd.DataFrame(centroid_rows).sort_values("centroid_distance", ascending=False)
    separation_path = Path("artifacts/animalclef2026_class_separation.csv")
    separation_df.to_csv(separation_path, index=False)

    logger.info("Saved class-only PCA plot to %s", output_plot)
    logger.info("Saved class centroid distances to %s", separation_path)
    logger.info("Average within-class scatter: %s", within_class_scatter)
    if not separation_df.empty:
        logger.info("Centroid distances between classes:\n%s", separation_df.to_string(index=False))

def save_projection_table(
    dataset: AnimalCLEF2026,
    pca_embeddings: np.ndarray,
    labels: np.ndarray,
    species: np.ndarray,
    output_path: Path,
) -> None:
    df = dataset.df.reset_index(drop=True).copy()
    df["label_string"] = labels
    df["species_string"] = species
    df["pc1"] = pca_embeddings[:, 0]
    df["pc2"] = pca_embeddings[:, 1]
    df.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MegaDescriptor embeddings for AnimalCLEF2026 test images and visualize them with PCA."
    )
    parser.add_argument("--root", default="data", help="Dataset root directory.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for quick experiments on a subset of the test split.",
    )
    parser.add_argument(
        "--output-plot",
        default="artifacts/animalclef2026_test_pca.png",
        help="Path to the saved PCA scatter plot.",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/animalclef2026_test_pca.csv",
        help="Path to the saved table with PCA coordinates.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root '{root}' does not exist. Run download_dataset() first so the AnimalCLEF2026 files are available."
        )

    output_plot = Path(args.output_plot)
    output_csv = Path(args.output_csv)
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    dataset_test = load_test_dataset(str(root), max_samples=args.max_samples)
    labels = resolve_label_strings(dataset_test)
    species = resolve_species_strings(dataset_test)
    logger.info("Loaded %s test samples", len(dataset_test))
    logger.info("Found %s unique identities and %s unique species groups", len(np.unique(labels)), len(np.unique(species)))

    logger.info("Started downloading the model.")
    model = download_mega_descriptor_model_feature_extraction()
    logger.info("Model download was succesfull.")
    embeddings = compute_embeddings(dataset_test, model, args.batch_size, device)
    logger.info("Embedding matrix shape: %s", embeddings.shape)

    pca = PCA(n_components=2, random_state=42)
    pca_embeddings = pca.fit_transform(embeddings)
    logger.info(
        "Explained variance ratio: PC1=%.4f, PC2=%.4f, total=%.4f",
        pca.explained_variance_ratio_[0],
        pca.explained_variance_ratio_[1],
        pca.explained_variance_ratio_.sum(),
    )

    plot_embeddings(pca_embeddings, labels, species, output_plot)
    save_projection_table(dataset_test, pca_embeddings, labels, species, output_csv)

    logger.info("Saved PCA plot to %s", output_plot)
    logger.info("Saved PCA coordinates to %s", output_csv)


if __name__ == "__main__":
    # main()
    compute_class_separation()
