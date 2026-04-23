import logging
import time
import warnings

import torch
from torch.utils.data import DataLoader

from animal_dataset import AnimalSimCLRDataset


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SIMCLR")


def summarize_metadata(dataset: AnimalSimCLRDataset) -> dict[str, int]:
    metadata = dataset.metadata
    return {
        "samples": int(len(metadata)),
        "species": int(metadata["species"].nunique()),
        "datasets": int(metadata["dataset"].nunique()),
        "identities": int(metadata["identity"].nunique()),
    }


def inspect_one_train_batch(train_loader: DataLoader) -> None:
    start_time = time.perf_counter()
    (view_1, view_2), labels = next(iter(train_loader))
    elapsed_seconds = time.perf_counter() - start_time

    images = torch.cat([view_1, view_2], dim=0)
    contrastive_labels = torch.cat([labels, labels], dim=0)

    logger.info("One train batch load time: %.3f seconds", elapsed_seconds)
    logger.info("view_1 shape: %s", tuple(view_1.shape))
    logger.info("view_2 shape: %s", tuple(view_2.shape))
    logger.info("labels shape: %s", tuple(labels.shape))
    logger.info("SimCLR images shape after concat: %s", tuple(images.shape))
    logger.info("SimCLR labels shape after concat: %s", tuple(contrastive_labels.shape))
    logger.info("First batch labels: %s", labels.tolist())


def main() -> None:
    config = {
        "root": "data",
        "image_size": 384,
        "batch_size":  64,
        "num_workers":  4,
        "max_train_samples": None,
    }

    train_dataset = AnimalSimCLRDataset(
        root=config["root"],
        split="train",
        image_size=int(config["image_size"]),
        training=True,
        max_samples=config["max_train_samples"],
        drop_unknown_identity=True,
    )
    eval_dataset = AnimalSimCLRDataset(
        root=config["root"],
        split="test",
        image_size=int(config["image_size"]),
        training=False,
        max_samples=config["max_train_samples"],
        drop_unknown_identity=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(config["num_workers"]) > 0,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=int(config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(config["num_workers"]) > 0,
    )

    logger.info("Train loader batches: %s", len(train_loader))
    logger.info("Eval loader batches: %s", len(eval_loader))
    logger.info("Train metadata summary: %s", summarize_metadata(train_dataset))
    logger.info("Eval metadata summary: %s", summarize_metadata(eval_dataset))
    
    for i in range(10):
        inspect_one_train_batch(train_loader)


if __name__ == "__main__":
    main()
