import logging
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from animal_dataset import SimCLRGPUTransform, build_simclr_data
from model import ContrastiveEmbeddingModel
from main_utils import (
    plot_loss
)


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SIMCLR")

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"Expected features with shape [2N, D], got {tuple(features.shape)}.")
        if labels.ndim != 1:
            raise ValueError(f"Expected labels with shape [2N], got {tuple(labels.shape)}.")
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Features and labels must have matching first dimension, got "
                f"{features.shape[0]} and {labels.shape[0]}."
            )

        features = F.normalize(features, dim=1)
        logits = features @ features.T
        logits = logits / self.temperature

        batch_size = features.shape[0]
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
        logits = logits.masked_fill(self_mask, float("-inf"))

        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        positive_counts = positive_mask.sum(dim=1)
        valid_anchor_mask = positive_counts > 0
        masked_log_prob = torch.where(
            positive_mask,
            log_prob,
            torch.zeros_like(log_prob),
        )
        positive_log_prob = masked_log_prob.sum(dim=1) / positive_counts.clamp_min(1)
        return -positive_log_prob[valid_anchor_mask].mean()


def main(config : dict[str, any]) -> None:
    simclr_data = build_simclr_data(config)
    device = simclr_data["device"]
    gpu_transform = simclr_data["gpu_transform"]
    train_loader = simclr_data["train_loader"]

    model = ContrastiveEmbeddingModel(
        projection_dim=int(config["projection_dim"]),
        projection_hidden_dim=int(config["projection_hidden_dim"]),
        dropout=float(config["projection_dropout"]),
    ).to(device)

    loss_fn = SupervisedContrastiveLoss(
        temperature=float(config["temperature"])
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    train_losses = []
    best_loss = float("inf")

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()

        epoch_loss = 0.0
        valid_batches = 0
        start_time = time.perf_counter()

        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            view_1 = gpu_transform(images)
            view_2 = gpu_transform(images)

            simclr_images = torch.cat([view_1, view_2], dim=0)
            simclr_labels = torch.cat([labels, labels], dim=0)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
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
        train_losses.append(avg_loss)

        elapsed = time.perf_counter() - start_time

        logger.info(
            "Epoch %d/%d finished | avg_loss=%.6f | time=%.1fs",
            epoch,
            config["epochs"],
            avg_loss,
            elapsed,
        )

        plot_loss(train_losses, save_name="simclr_loss.png")

        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_full_checkpoint(
                checkpoint_data={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "loss": avg_loss,
                    "config": config,
                }
            )
            model.save_embedding_checkpoint()
            logger.info("Saved new best checkpoint with loss %.6f", best_loss)


if __name__ == "__main__":
    config = {
        "root": "data",
        "batch_size": 16,
        "num_workers": 4,
        "epochs": 20,
        "lr": 1e-5,
        "weight_decay": 1e-4,
        "temperature": 0.5,
        "projection_dim": 128,
        "projection_hidden_dim": 512,
        "projection_dropout": 0.0,
    }
    main(config)
