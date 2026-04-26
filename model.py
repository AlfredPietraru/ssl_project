import logging
import warnings

import torch
import torch.nn as nn
from pathlib import Path

from animal_dataset import build_simclr_data
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MODEL")


def download_mega_descriptor_model_feature_extraction():
    import timm
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "mega_descriptor_l_384.pth"

    try:
        # Always create model architecture first (no weights yet)
        model = timm.create_model(
            "hf-hub:BVRA/MegaDescriptor-L-384",
            pretrained=False
        )

        if model_path.exists():
            # Load locally saved weights
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            logger.info("Loaded MegaDescriptor model from local cache.")
        else:
            # Download from HF once
            logger.info("Downloading MegaDescriptor model from HuggingFace...")
            model = timm.create_model(
                "hf-hub:BVRA/MegaDescriptor-L-384",
                pretrained=True
            )

            # Save locally
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved model locally at {model_path}")

        model = model.train()
        return model

    except Exception:
        logger.exception("Model for embeddings failed downloading/loading...")
        raise



class ContrastiveEmbeddingModel(nn.Module):
    def __init__(
        self,
        projection_dim: int = 256,
        projection_hidden_dim: int = 512,
        dropout: float = 0.0,
    ):
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
        

    def extract_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def save_full_checkpoint(self, checkpoint_data : dict, 
        path_name: str = "contrastive_model.pt") -> None:
        save_dir = Path("artifacts") / "full_model_checkpoints"
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_data, save_dir / path_name)
        

    def save_embedding_checkpoint(self, path_name: str = "embedding_backbone.pt") -> None:
        save_dir = Path("artifacts") / "embedding_checkpoints"
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / path_name

        torch.save(
            {
                "backbone_state_dict": self.backbone.state_dict(),
                "embedding_dim": self.embedding_dim,
            },
            save_path,
        )


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
        "batch_size": 4,
        "num_workers": 1,
        "projection_dim": 256,
        "projection_hidden_dim": 512,
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
