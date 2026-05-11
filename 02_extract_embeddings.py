import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from config import CFG

from animal_dataset import (
    build_simclr_data,
    testing_transform
)
from model import load_embedding_backbone_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EXTRACT_EMBEDDINGS")

def extract_split_embeddings(
    cfg : CFG,
    model: torch.nn.Module,
    loader : DataLoader
) -> np.ndarray:
    all_embeddings = []
    with torch.inference_mode(), torch.autocast("cuda", enabled=cfg.device.type == "cuda"):
        for batch_idx, batch in enumerate(loader, start=1):
            if isinstance(batch, dict):
                images = batch["image"]
            else:
                images = batch[0]

            images = images.to(cfg.device, non_blocking=True)
            embeddings = model(images)

            if isinstance(embeddings, tuple):
                embeddings = embeddings[0]

            all_embeddings.append(embeddings.detach().cpu().float().numpy())

            if batch_idx % 20 == 0 or batch_idx == len(loader):
                logger.info("Processed batch %d/%d", batch_idx, len(loader))

    if not all_embeddings:
        raise ValueError("No embeddings were extracted because the dataloader is empty.")

    embeddings_array = np.concatenate(all_embeddings, axis=0)
    if cfg.normalize_embeddings:
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_array = embeddings_array / np.maximum(norms, 1e-12)
    return embeddings_array


def save_split_outputs(
    split: str,
    embeddings: np.ndarray,
    metadata,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / f"{split}_embeddings.npy"
    metadata_path = output_dir / f"{split}_metadata.csv"

    np.save(embeddings_path, embeddings)
    metadata.to_csv(metadata_path, index=False)
    return embeddings_path, metadata_path


def extract_embeddings_model(cfg : CFG) -> None:
    root = Path(cfg.root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root '{root}' does not exist.")

    logger.info("Loading fine-tuned embedding backbone from %s", cfg.embedding_checkpoint_path)
    model = load_embedding_backbone_checkpoint(
        checkpoint_path=cfg.embedding_checkpoint_path,
        device=cfg.device,
        eval_mode=True,
    )
    train_dataset, eval_dataset, train_dataloader, eval_dataloader = build_simclr_data(
        cfg, False, testing_transform, testing_transform)

    for dataset, loader in zip([train_dataset, eval_dataset], [train_dataloader, eval_dataloader]):
        split = dataset.split
        logger.info("Building %s dataset from %s", split, root)

        metadata = dataset.metadata.copy()

        logger.info("Extracting %s embeddings for %d samples", split, len(metadata))
        embeddings = extract_split_embeddings(cfg=cfg, model=model, loader=loader)

        if len(metadata) != len(embeddings):
            raise ValueError(
                f"Metadata rows and embeddings rows differ for split '{split}': "
                f"{len(metadata)} vs {len(embeddings)}"
            )

        embeddings_path, metadata_path = save_split_outputs(
                split=dataset.split,
                embeddings=embeddings,
                metadata=metadata,
                output_dir=Path(cfg.embeddings_output_dir),
        )
        logger.info("Saved %s embeddings to %s", split, embeddings_path)
        logger.info("Saved %s metadata to %s", split, metadata_path)
        logger.info("Final %s embedding shape: %s", split, tuple(embeddings.shape))


if __name__ == "__main__":
    cfg = CFG("config.yaml")
    extract_embeddings_model(cfg)
