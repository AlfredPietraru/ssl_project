import logging
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from animal_dataset import SimCLRGPUTransform, build_simclr_data
from model import ContrastiveEmbeddingModel


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


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def build_one_simclr_batch(
    train_loader: DataLoader,
    gpu_transform: SimCLRGPUTransform,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    start_time = time.perf_counter()
    images, labels = next(iter(train_loader))
    load_seconds = time.perf_counter() - start_time

    start_time = time.perf_counter()
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    view_1 = gpu_transform(images)
    view_2 = gpu_transform(images)
    simclr_images = torch.cat([view_1, view_2], dim=0)
    simclr_labels = torch.cat([labels, labels], dim=0)
    synchronize_if_cuda(device)
    augment_seconds = time.perf_counter() - start_time

    timings = {
        "cpu_load_seconds": load_seconds,
        "gpu_transfer_augment_seconds": augment_seconds,
    }
    return simclr_images, simclr_labels, timings


def main() -> None:
    config = {
        "root": "data",
        "image_size": 384,
        "batch_size": 32,
        "num_workers": 4,
        "max_train_samples": None,
        "max_eval_samples": None,
        "temperature": 0.5,
        "projection_dim": 256,
        "projection_hidden_dim": 512,
        "projection_dropout": 0.0,
        "freeze_backbone": False,
    }

    simclr_data = build_simclr_data(config)
    device = simclr_data["device"]
    gpu_transform = simclr_data["gpu_transform"]
    train_loader = simclr_data["train_loader"]

    model = ContrastiveEmbeddingModel(
        projection_dim=int(config["projection_dim"]),
        projection_hidden_dim=int(config["projection_hidden_dim"]),
        dropout=float(config["projection_dropout"]),
        freeze_backbone=bool(config["freeze_backbone"]),
    ).to(device)
    loss_fn = SupervisedContrastiveLoss(temperature=float(config["temperature"])).to(device)
    for i in range(10):
        simclr_images, simclr_labels, timings = build_one_simclr_batch(
            train_loader=train_loader,
            gpu_transform=gpu_transform,
            device=device,
        )

        embeddings, projections = model(simclr_images)
        loss = loss_fn(projections, simclr_labels)

        logger.info("Device: %s", device)
        logger.info("CPU batch load time: %.3f seconds", timings["cpu_load_seconds"])
        logger.info(
            "GPU transfer+augment time: %.3f seconds",
            timings["gpu_transfer_augment_seconds"],
        )
        logger.info("SimCLR images shape: %s", tuple(simclr_images.shape))
        logger.info("SimCLR labels shape: %s", tuple(simclr_labels.shape))
        logger.info("Backbone embedding shape: %s", tuple(embeddings.shape))
        logger.info("Projection shape: %s", tuple(projections.shape))
        logger.info("Unique labels in concatenated batch: %s", int(simclr_labels.unique().numel()))
        logger.info("Supervised contrastive loss: %.6f", float(loss.detach().cpu()))
        print()


if __name__ == "__main__":
    main()
