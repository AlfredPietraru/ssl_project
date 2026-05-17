import logging
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from animal_dataset import (
    build_pk_train_val_datasets_and_loaders,
    build_simple_train_val_datasets_and_loaders
)
from transformations import (
    build_cpu_testing_transform,
    build_cpu_training_transform,
)

from cluster_and_compare import ClusterAndCompare
from data_fetcher import DataFetcher

import csv
from trainer import EmbeddingModelTrainer

from config import CFG
from model import ContrastiveEmbeddingModel
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SIMCLR")
METADATA_PATH = Path("metadata.csv")


def get_train_species(metadata_path: Path) -> list[str]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    species = set()
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("split") != "train":
                continue

            name = (row.get("species") or "").strip()
            if name:
                species.add(name)

    return sorted(species)


TRAIN_SPECIES = (
    "lynx",
    "salamander",
    "loggerhead turtle",
)

def main(config: CFG) -> None:
    animal = TRAIN_SPECIES[1]
    data_fetcher = DataFetcher()
    train_metadata, val_metadata = data_fetcher.get_train_split_animal(animal=animal)
    train_samples = sum(len(item["paths"]) for item in train_metadata)
    val_samples = sum(len(item["paths"]) for item in val_metadata)
    train_identities = len(train_metadata)
    val_identities = len(val_metadata)
    train_dt, val_dt, train_loader, val_loader = build_pk_train_val_datasets_and_loaders(
        train_metadata=train_metadata,
        val_metadata=val_metadata,
        cfg=config,
        train_transform=build_cpu_training_transform(config.image_size),
        val_transform=build_cpu_testing_transform(config.image_size),
    )
    simple_train_dt, simple_val_dt, simple_train_loader, simple_val_loader = build_simple_train_val_datasets_and_loaders(
        train_metadata=train_metadata,
        val_metadata=val_metadata,
        cfg=config,
        train_transform=build_cpu_testing_transform(config.image_size),
        val_transform=build_cpu_testing_transform(config.image_size),
    )
    logger.info(
        "Built %s identity split | train_samples=%d | val_samples=%d | train_identities=%d | val_identities=%d",
        animal,
        train_samples,
        val_samples,
        train_identities,
        val_identities,
    )

    trainer_salamander = EmbeddingModelTrainer(
        cfg=config,
        train_loader=train_loader,
        val_loader=val_loader,
        simple_train_loader=simple_train_loader,
        simple_val_loader=simple_val_loader,
        animal_name=animal,
        model=ContrastiveEmbeddingModel().to(config.device)
    )
    best_model = trainer_salamander.train()

    embeddings, labels = trainer_salamander.get_embeddings(trained_model=best_model,
                                                            loader=simple_val_loader)
    logger.info("Validation clustering | embeddings_shape=%s", tuple(embeddings.shape))
    comparator = ClusterAndCompare(
        embeddings=embeddings,
        labels=labels,
        metadata=val_metadata,
    )
    comparator.sweep_eps(min_samples=2)

    logger.info("Try for training.")
    embeddings, labels = trainer_salamander.get_embeddings(trained_model=best_model,
                                                            loader=simple_train_loader)
    logger.info("Train clustering | embeddings_shape=%s", tuple(embeddings.shape))
    comparator = ClusterAndCompare(
        embeddings=embeddings,
        labels=labels,
        metadata=train_metadata,
    )
    comparator.sweep_eps(min_samples=2)

if __name__ == "__main__":
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    main(CFG())
