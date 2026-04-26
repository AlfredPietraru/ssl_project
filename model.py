import logging
import warnings

import torch
import torch.nn as nn

from animal_dataset import build_simclr_data
from main_utils import download_mega_descriptor_model_feature_extraction


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MODEL")


class ContrastiveEmbeddingModel(nn.Module):
    def __init__(self, projection_dim: int = 256, projection_hidden_dim: int = 512,
                  dropout: float = 0.0):
        super().__init__()
        self.backbone = download_mega_descriptor_model_feature_extraction()
        self.embedding_dim = int(self.backbone.num_features)
        self.projection_head = nn.Sequential(
            nn.Linear(self.embedding_dim, projection_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_hidden_dim, projection_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection_head(self.backbone(images))


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def build_one_simclr_batch(simclr_data: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    device = simclr_data["device"]
    gpu_transform = simclr_data["gpu_transform"]
    train_loader = simclr_data["train_loader"]

    images, labels = next(iter(train_loader))
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    view_1 = gpu_transform(images)
    view_2 = gpu_transform(images)
    simclr_images = torch.cat([view_1, view_2], dim=0)
    simclr_labels = torch.cat([labels, labels], dim=0)
    synchronize_if_cuda(device)
    return simclr_images, simclr_labels


def main() -> None:
    config = {
        "root": "data",
        "image_size": 384,
        "batch_size": 32,
        "num_workers": 4,
        "max_train_samples": None,
        "max_eval_samples": None,
        "projection_dim": 128,
        "projection_hidden_dim": 256,
        "projection_dropout": 0.0,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simclr_data = build_simclr_data(config)
    model = ContrastiveEmbeddingModel(
        projection_dim=int(config["projection_dim"]),
        projection_hidden_dim=int(config["projection_hidden_dim"]),
        dropout=float(config["projection_dropout"]),
    ).to(device=device)

    simclr_images, simclr_labels = build_one_simclr_batch(simclr_data)
    forward_output = model(simclr_images)

    logger.info("Device: %s", device)
    logger.info("SimCLR images shape: %s", tuple(simclr_images.shape))
    logger.info("SimCLR labels shape: %s", tuple(simclr_labels.shape))
    logger.info("Backbone embedding dimension: %s", model.embedding_dim)
    logger.info("Forward output shape: %s", tuple(forward_output.shape))


if __name__ == "__main__":
    main()
