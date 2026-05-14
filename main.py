import logging
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from animal_dataset import (
    build_simclr_train_val_by_identity,
    build_transformations,
)

import csv
from pathlib import Path
from trainer import EmbeddingModelTrainer

from config import CFG
from model import ContrastiveEmbeddingModel
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SIMCLR")



METADATA_PATH = Path("data/metadata.csv")


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
    train_dataset, validation_dataset, train_loader, validation_loader = build_simclr_train_val_by_identity(
        config=config,
        shuffle_training=True,
        class_name=animal,
        instances_per_identity=2,
    )
    logger.info(
        "Built %s identity split | train_samples=%d | val_samples=%d | train_identities=%d | val_identities=%d",
        animal,
        len(train_dataset),
        len(validation_dataset),
        train_dataset.metadata["identity"].nunique(),
        validation_dataset.metadata["identity"].nunique(),
    )

    trainer_salamander = EmbeddingModelTrainer(
        cfg=config,
        train_loader=train_loader,
        val_loader=validation_loader,
        transform=build_transformations(config),
    )
    model: ContrastiveEmbeddingModel = trainer_salamander.train()

if __name__ == "__main__":
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    config = CFG("config.yaml")
    train_species = get_train_species(METADATA_PATH)

    print("Species found in the train set:")
    for species_name in train_species:
        print(f"- {species_name}")

    main(config)
