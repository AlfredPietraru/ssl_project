import logging
import time
import warnings
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from animal_dataset import (
    build_pk_train_val_datasets_and_loaders,
    build_simple_train_val_datasets_and_loaders
)
from transformations import (
    training_transform,
    testing_transform,
    build_transformations
)

from data_fetcher import DataFetcher

import csv
from pathlib import Path
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

def main(
    config: CFG,
    backbone_weights_path: str,
    allow_download: bool = False,
) -> None:
    animal = TRAIN_SPECIES[1]
    data_fetcher = DataFetcher()
    train_metadata, val_metadata = data_fetcher.get_train_split_animal(animal=animal)
    train_samples = sum(len(item["paths"]) for item in train_metadata)
    val_samples = sum(len(item["paths"]) for item in val_metadata)
    train_identities = len(train_metadata)
    val_identities = len(val_metadata)
    train_dt, val_dt, train_loader, val_loader = build_simple_train_val_datasets_and_loaders(
        train_metadata=train_metadata,
        val_metadata=val_metadata,
        cfg=config,
        train_transform=training_transform,
        val_transform=testing_transform,
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
        transform=build_transformations(config=config),
        animal_name=animal,
        backbone_weights_path=backbone_weights_path,
        allow_download=allow_download,
    )
    # trainer_salamander.train()
    embeddings, labels = trainer_salamander.get_embeddings(split='validation')
    print(embeddings.shape)


if __name__ == "__main__":
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="Train the contrastive embedding model.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--backbone-weights-path",
        default="artifacts/mega_descriptor_t_cnn_288.pth",
        help="Path to the backbone weights file used to initialize the model.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow downloading backbone weights if the provided path does not exist.",
    )
    args = parser.parse_args()

    config = CFG(args.config)
    train_species = get_train_species(METADATA_PATH)

    print("Species found in the train set:")
    for species_name in train_species:
        print(f"- {species_name}")

    main(
        config,
        backbone_weights_path=args.backbone_weights_path,
        allow_download=args.allow_download,
    )
