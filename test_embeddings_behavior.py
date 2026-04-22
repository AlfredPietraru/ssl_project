import argparse
import logging
import os
from pathlib import Path

# Keep matplotlib cache writable when running from restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EMBEDDINGS_TEST")


def clean_string_array(values, missing_value: str = "unknown") -> np.ndarray:
    series = pd.Series(values, dtype="object")
    series = series.where(series.notna(), missing_value)
    series = series.astype(str).str.strip()
    series = series.mask(series.eq("") | series.str.lower().isin({"nan", "none"}), missing_value)
    return series.to_numpy(dtype=str)


def normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def load_embedding_artifacts(
    embeddings_dir: Path,
    model_name: str,
    split: str,
    max_samples: int | None = None,
    normalize: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    embeddings_path = embeddings_dir / f"{model_name}_{split}_embeddings.npy"
    metadata_path = embeddings_dir / f"{model_name}_{split}_metadata.csv"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Could not find {embeddings_path}. Run 00_embedding_extraction.py first."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Could not find {metadata_path}. Run 00_embedding_extraction.py first."
        )

    embeddings = np.load(embeddings_path)
    metadata = pd.read_csv(metadata_path)

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

    if max_samples is not None:
        max_samples = min(max_samples, len(metadata))
        embeddings = embeddings[:max_samples]
        metadata = metadata.iloc[:max_samples].reset_index(drop=True).copy()

    if normalize:
        embeddings = normalize_rows(embeddings)

    logger.info("Loaded %s embeddings from %s with shape %s", split, embeddings_path, embeddings.shape)
    logger.info("Loaded %s metadata from %s", split, metadata_path)
    return embeddings.astype(np.float32), metadata


def resolve_label_strings(metadata: pd.DataFrame) -> np.ndarray:
    if "identity" in metadata.columns:
        return clean_string_array(metadata["identity"])
    if "label_string" in metadata.columns:
        return clean_string_array(metadata["label_string"])
    return np.array(["unknown"] * len(metadata), dtype=str)


def resolve_species_strings(metadata: pd.DataFrame) -> np.ndarray:
    for candidate in ("species", "species_string", "category", "dataset"):
        if candidate in metadata.columns:
            return clean_string_array(metadata[candidate])
    return np.array(["unknown"] * len(metadata), dtype=str)


def plot_embeddings(
    pca_embeddings: np.ndarray,
    labels: np.ndarray,
    species: np.ndarray,
    output_path: Path,
    my_type : str
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

    ax.set_title(f"AnimalCLEF2026 {my_type} Embeddings Projected with PCA")
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


def compute_class_separation(
    my_type: str,
    csv_path: Path,
    output_plot: Path,
    separation_path: Path,
) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}. Generate the PCA CSV first or point this function to an existing file."
        )

    df = pd.read_csv(csv_path)
    class_column = "species_string" if "species_string" in df.columns else "species"
    required_columns = {"pc1", "pc2", class_column}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")

    plot_df = df.dropna(subset=["pc1", "pc2", class_column]).copy()
    if plot_df.empty:
        raise ValueError("No valid PCA rows with class labels were found in the CSV.")

    plot_df[class_column] = clean_string_array(plot_df[class_column])
    classes = sorted(plot_df[class_column].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

    fig, ax = plt.subplots(figsize=(12, 9))
    class_centroids = {}
    within_class_scatter = {}

    for color, class_name in zip(colors, classes):
        class_df = plot_df[plot_df[class_column] == class_name]
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

    ax.set_title(f"AnimalCLEF2026 {my_type} PCA Colored by Class")
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

    separation_df = pd.DataFrame(
        centroid_rows,
        columns=["class_a", "class_b", "centroid_distance"],
    ).sort_values("centroid_distance", ascending=False)
    separation_df.to_csv(separation_path, index=False)

    logger.info("Saved class-only PCA plot to %s", output_plot)
    logger.info("Saved class centroid distances to %s", separation_path)
    logger.info("Average within-class scatter: %s", within_class_scatter)
    if not separation_df.empty:
        logger.info("Centroid distances between classes:\n%s", separation_df.to_string(index=False))

def save_projection_table(
    metadata: pd.DataFrame,
    pca_embeddings: np.ndarray,
    labels: np.ndarray,
    species: np.ndarray,
    output_path: Path,
) -> None:
    df = metadata.reset_index(drop=True).copy()
    df["label_string"] = labels
    df["species_string"] = species
    df["pc1"] = pca_embeddings[:, 0]
    df["pc2"] = pca_embeddings[:, 1]
    df.to_csv(output_path, index=False)


def run_split(
    my_type: str,
    embeddings_dir: Path,
    model_name: str,
    max_samples: int | None,
    normalize: bool,
    output_plot: Path,
    output_csv: Path,
) -> None:
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    embeddings, metadata = load_embedding_artifacts(
        embeddings_dir=embeddings_dir,
        model_name=model_name,
        split=my_type,
        max_samples=max_samples,
        normalize=normalize,
    )
    labels = resolve_label_strings(metadata)
    species = resolve_species_strings(metadata)
    logger.info("Loaded %s %s samples", len(metadata), my_type)
    logger.info(
        "Found %s unique identities and %s unique species groups",
        len(np.unique(labels)),
        len(np.unique(species)),
    )
    logger.info("Embedding matrix shape: %s", embeddings.shape)

    pca = PCA(n_components=2, random_state=42)
    pca_embeddings = pca.fit_transform(embeddings)
    logger.info(
        "Explained variance ratio: PC1=%.4f, PC2=%.4f, total=%.4f",
        pca.explained_variance_ratio_[0],
        pca.explained_variance_ratio_[1],
        pca.explained_variance_ratio_.sum(),
    )

    plot_embeddings(pca_embeddings, labels, species, output_plot, my_type)
    save_projection_table(metadata, pca_embeddings, labels, species, output_csv)

    logger.info("Saved PCA plot to %s", output_plot)
    logger.info("Saved PCA coordinates to %s", output_csv)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize precomputed AnimalCLEF2026 embeddings from data_embeddings with PCA."
    )
    parser.add_argument(
        "--embeddings-dir",
        default="data_embeddings",
        help="Directory containing row-aligned .npy embeddings and metadata CSV artifacts.",
    )
    parser.add_argument(
        "--model-name",
        default="mega",
        help="Prefix used in artifact filenames, e.g. mega_train_embeddings.npy.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "both"),
        default="both",
        help="Dataset split to process.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional row limit for quick experiments on each selected split.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize loaded embeddings before PCA.",
    )
    parser.add_argument(
        "--output-plot",
        default=None,
        help="Path to the saved PCA scatter plot. Only valid when --split is train or test.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path to the saved table with PCA coordinates. Only valid when --split is train or test.",
    )
    args = parser.parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings directory '{embeddings_dir}' does not exist.")

    selected_splits = ("train", "test") if args.split == "both" else (args.split,)
    if len(selected_splits) > 1 and (args.output_plot or args.output_csv):
        raise ValueError("--output-plot and --output-csv can only be used with a single split.")

    for split in selected_splits:
        output_plot = Path(args.output_plot or f"artifacts/animalclef2026_{split}_pca.png")
        output_csv = Path(args.output_csv or f"artifacts/animalclef2026_{split}_pca.csv")
        run_split(
            split,
            embeddings_dir,
            args.model_name,
            args.max_samples,
            args.normalize,
            output_plot,
            output_csv,
        )

if __name__ == "__main__":
    main()
