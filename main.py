import logging
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from animal_dataset import (
    build_simclr_data,
    build_transformations,
    training_transform,
    testing_transform
)
from config import CFG
from model import ContrastiveEmbeddingModel
from loss_functions import BatchHardTripletLoss, SupervisedContrastiveLoss
from main_utils import (
    plot_loss,
    set_seed
)


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SIMCLR")


def train_one_epoch(
    epoch: int,
    config: CFG,
    model: ContrastiveEmbeddingModel,
    train_loader,
    gpu_transform,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.GradScaler
) -> float:
    model.train()

    epoch_loss = 0.0
    valid_batches = 0
    start_time = time.perf_counter()

    for batch_idx, (images, labels) in enumerate(train_loader, start=1):
        images = images.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", enabled=(config.device.type == "cuda")):
            if config.training_loss == "triplet":
                triplet_images = gpu_transform(images)
                embeddings = model.extract_embeddings(triplet_images)
                loss = loss_fn(embeddings, labels)
            else:
                view_1 = gpu_transform(images)
                view_2 = gpu_transform(images)

                simclr_images = torch.cat([view_1, view_2], dim=0)
                simclr_labels = torch.cat([labels, labels], dim=0)

                projections = model(simclr_images)

                if isinstance(projections, tuple):
                    projections = projections[-1]

                loss = loss_fn(projections, simclr_labels)

        if not torch.isfinite(loss):
            logger.warning("Skipping non-finite loss at batch %d: %s", batch_idx, loss.item())
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += float(loss.detach().cpu())
        valid_batches += 1

        if batch_idx % 20 == 0:
            logger.info(
                "Epoch %d | Batch %d/%d | Loss %.6f",
                epoch,
                batch_idx,
                len(train_loader),
                float(loss.detach().cpu()),
            )

    avg_loss = epoch_loss / max(valid_batches, 1)
    elapsed = time.perf_counter() - start_time

    logger.info(
        "Epoch %d/%d finished | avg_loss=%.6f | time=%.1fs",
        epoch,
        config.epochs,
        avg_loss,
        elapsed,
    )
    return avg_loss


def train(
    config: CFG,
    model: ContrastiveEmbeddingModel,
    train_loader,
    gpu_transform,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    train_losses = []
    best_loss = float("inf")
    scaler = torch.GradScaler("cuda", enabled=(config.device.type == "cuda"))

    for epoch in range(1, config.epochs + 1):
        avg_loss = train_one_epoch(
            epoch=epoch,
            config=config,
            model=model,
            train_loader=train_loader,
            gpu_transform=gpu_transform,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler
        )
        train_losses.append(avg_loss)

        plot_loss(train_losses, save_name=f"{config.training_loss}_loss.png")

        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_full_checkpoint(
                checkpoint_data={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "loss": avg_loss,
                    "config": config.to_dict(),
                }
            )
            model.save_embedding_checkpoint()
            logger.info("Saved new best checkpoint with loss %.6f", best_loss)



def main(config: CFG) -> None:
    train_dataset, eval_dataset, train_loader, test_loader = build_simclr_data(
        config,
        shuffle_training=True,
        training_transform=training_transform,
        testing_transform=testing_transform
    )
    gpu_transform = build_transformations(config)

    model = ContrastiveEmbeddingModel(
        projection_dim=config.projection_dim,
        projection_hidden_dim=config.projection_hidden_dim,
        dropout=config.projection_dropout,
        allow_download=True,
    ).to(config.device)

    if config.training_loss == "triplet":
        loss_fn = BatchHardTripletLoss(
            margin=config.triplet_margin
        ).to(config.device)
    elif config.training_loss == "supcon":
        loss_fn = SupervisedContrastiveLoss(
            temperature=config.temperature
        ).to(config.device)
    else:
        raise ValueError(f"Unsupported training_loss: {config.training_loss}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    train(
        config=config,
        model=model,
        train_loader=train_loader,
        gpu_transform=gpu_transform,
        loss_fn=loss_fn,
        optimizer=optimizer
    )


if __name__ == "__main__":
    config = CFG("config.yaml")
    set_seed()
    main(config)
