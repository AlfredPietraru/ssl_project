import argparse
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
from model import ContrastiveEmbeddingModel, load_trained_embedding_model
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SIMCLR")
METADATA_PATH = Path("metadata.csv")


TRAIN_SPECIES = (
    "lynx",
    "salamander",
    "loggerhead turtle",
)

def main(
    config: CFG,
    *,
    animal: str,
) -> None:
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

    trainer = EmbeddingModelTrainer(
        cfg=config,
        train_loader=train_loader,
        val_loader=val_loader,
        simple_train_loader=simple_train_loader,
        simple_val_loader=simple_val_loader,
        animal_name=animal,
        model=ContrastiveEmbeddingModel(
            num_classes=train_identities,
            freeze_backbone=config.freeze_backbone,
            unfreeze_last_backbone_block=config.unfreeze_last_backbone_block,
            projection_dim=(
                config.projection_dim if config.mode == "train" else None
            ),
        ).to(config.device)
    )

    if config.mode == "train":
        best_model = trainer.train()
    elif config.mode == "load":
        resolved_checkpoint = config.checkpoint_path or (
            Path("artifacts") / "full_model_checkpoints" / f"contrastive_model_{animal.replace(' ', '_')}.pt"
        )
        logger.info("Loading trained model from %s", resolved_checkpoint)
        best_model = load_trained_embedding_model(
            resolved_checkpoint,
            device=config.device,
            eval_mode=True,
        )
    elif config.mode in {"pretrained", "untrained"}:
        logger.info(
            "Using base pretrained model without fine-tuning checkpoint for %s",
            animal,
        )
        best_model = trainer.model.eval()
    else:
        raise ValueError(
            f"Unsupported mode '{config.mode}'. Expected 'train', 'load', 'pretrained', or 'untrained'."
        )

    embeddings, labels = trainer.get_embeddings(trained_model=best_model,
                                                loader=simple_val_loader)
    logger.info("Validation clustering | embeddings_shape=%s", tuple(embeddings.shape))
    comparator = ClusterAndCompare(
        embeddings=embeddings,
        labels=labels,
        metadata=val_metadata,
    )
    validation_results = comparator.sweep_eps(min_samples=2)
    plots_dir = Path("artifacts") / "embedding_plots"
    animal_slug = animal.replace(" ", "_")
    plot_mode_slug = str(config.mode).replace(" ", "_")
    comparator.plot_projection(
        plots_dir / f"{animal_slug}_{plot_mode_slug}_val_true_labels.png",
        title=f"{animal} validation embeddings by true identity ({config.mode})",
    )
    for result in validation_results:
        comparator.plot_projection(
            plots_dir / f"{animal_slug}_{plot_mode_slug}_val_clusters_mcs_{result['min_cluster_size']}.png",
            title=(
                f"{animal} validation embeddings by predicted cluster "
                f"(min_cluster_size={result['min_cluster_size']}, mode={config.mode})"
            ),
            cluster_labels=result["cluster_labels"],
        )

    logger.info("Try for training.")
    embeddings, labels = trainer.get_embeddings(trained_model=best_model,
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
    parser = argparse.ArgumentParser(
        description="Train or load an animal re-identification embedding model and evaluate clustering."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--animal",
        choices=TRAIN_SPECIES,
        default=TRAIN_SPECIES[2],
        help="Which species to train or evaluate.",
    )
    args = parser.parse_args()
    gc.collect()
    torch.cuda.empty_cache()
    main(
        CFG(args.config),
        animal=args.animal,
    )
