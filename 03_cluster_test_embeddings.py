import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import HDBSCAN
from torch.utils.data import DataLoader
from wildlife_datasets.datasets import AnimalCLEF2026

from config import CFG
from main_utils import normalize_rows
from model import load_trained_embedding_model
from transformations import build_cpu_testing_transform


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("STEP03_TEST_CLUSTERING")


def build_test_dataset(root: str, image_size: int):
    dataset = AnimalCLEF2026(
        root,
        transform=build_cpu_testing_transform(image_size),
        load_label=True,
        factorize_label=True,
        check_files=False,
    )
    return dataset.get_subset(dataset.df["split"] == "test")


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
        persistent_workers=num_workers > 0,
    )

    all_embeddings: list[np.ndarray] = []
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        for batch_idx, batch in enumerate(loader, start=1):
            if isinstance(batch, dict):
                images = batch["image"]
            else:
                images = batch[0]

            images = images.to(device, non_blocking=True)
            embeddings = model(images)
            all_embeddings.append(embeddings.detach().cpu().float().numpy())

            if batch_idx % 20 == 0:
                logger.info("Processed batch %d/%d", batch_idx, len(loader))

    if not all_embeddings:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(all_embeddings, axis=0)


def cluster_embeddings_like_notebook(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        return np.empty((0,), dtype=int)

    embeddings = normalize_rows(embeddings.astype(np.float32))
    similarity = embeddings @ embeddings.T
    positive_similarity = np.maximum(similarity, 0.0)
    max_similarity = float(np.max(similarity))
    if max_similarity <= 0.0:
        distance = 1.0 - positive_similarity
    else:
        distance = (max_similarity - positive_similarity) / max_similarity

    clustering = HDBSCAN(min_cluster_size=2, metric="precomputed")
    labels = clustering.fit_predict(distance)

    negative_indices = np.where(labels == -1)[0]
    if negative_indices.size > 0:
        next_label = int(labels[labels >= 0].max() + 1) if np.any(labels >= 0) else 0
        labels = labels.copy()
        labels[negative_indices] = np.arange(next_label, next_label + negative_indices.size)
    return labels.astype(int)


def build_submission(test_dataset, dataset_clusters: dict[str, np.ndarray]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset_name, cluster_labels in dataset_clusters.items():
        subset = test_dataset.get_subset(test_dataset.df["dataset"] == dataset_name)
        subset_df = subset.df.reset_index(drop=True)
        if len(subset_df) != len(cluster_labels):
            raise ValueError(
                f"Dataset '{dataset_name}' produced {len(cluster_labels)} cluster labels "
                f"for {len(subset_df)} rows."
            )
        frames.append(
            pd.DataFrame(
                {
                    "image_id": subset_df["image_id"].astype(str).tolist(),
                    "cluster": [f"cluster_{dataset_name}_{cluster_id}" for cluster_id in cluster_labels],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster AnimalCLEF test images per dataset using a fine-tuned embedding model."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--root", default=None, help="Dataset root. Defaults to config root.")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/full_model_checkpoints/contrastive_model_salamander.pt",
        help="Fine-tuned training checkpoint to load.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/final/submission.csv",
        help="Where to write the notebook-style submission CSV.",
    )
    args = parser.parse_args()

    cfg = CFG(args.config)
    root = cfg.root
    device = cfg.device

    logger.info("Loading fine-tuned model from %s", args.checkpoint)
    model = load_trained_embedding_model(args.checkpoint, device=device, eval_mode=True)

    logger.info("Building test dataset from %s", root)
    test_dataset = build_test_dataset(root=root, image_size=cfg.image_size)
    dataset_names = sorted(test_dataset.df["dataset"].dropna().astype(str).unique().tolist())
    logger.info("Found %d test datasets: %s", len(dataset_names), dataset_names)

    clustered_per_dataset: dict[str, np.ndarray] = {}
    for dataset_name in dataset_names:
        subset = test_dataset.get_subset(test_dataset.df["dataset"] == dataset_name)
        logger.info("Extracting embeddings for dataset=%s | samples=%d", dataset_name, len(subset))
        embeddings = extract_embeddings(
            model=model,
            dataset=subset,
            device=device,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        logger.info("Clustering dataset=%s | embeddings_shape=%s", dataset_name, tuple(embeddings.shape))
        clustered_per_dataset[dataset_name] = cluster_embeddings_like_notebook(embeddings)

    submission = build_submission(test_dataset, clustered_per_dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    logger.info("Wrote submission with %d rows to %s", len(submission), output_path)

for all elements in test:
imagine -> classificator -> clasa -> modelul_de_embedding_asociat_cu_clasa 
imagini de test -> embedding de test 
embedding de test-> algoritm de clusterizare

if __name__ == "__main__":
    main()
