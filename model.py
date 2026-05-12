import logging
import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn

from animal_dataset import build_simclr_data

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MODEL")

MEGA_DESCRIPTOR_MODEL_ID = "hf-hub:BVRA/MegaDescriptor-L-384"
MEGA_DESCRIPTOR_LOCAL_WEIGHTS = Path("artifacts") / "mega_descriptor_l_384.pth"
EMBEDDING_CHECKPOINT_PATH = Path("artifacts") / "embedding_checkpoints" / "embedding_backbone.pt"


def load_mega_descriptor_model_feature_extraction(
    weights_path: Path = MEGA_DESCRIPTOR_LOCAL_WEIGHTS,
    allow_download: bool = False,
) -> nn.Module:
    import timm

    try:
        if not allow_download:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        model = timm.create_model(
            MEGA_DESCRIPTOR_MODEL_ID,
            pretrained=False,
        )

        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
            logger.info("Loaded MegaDescriptor model from local weights: %s", weights_path)
        elif allow_download:
            logger.info("Downloading MegaDescriptor model from HuggingFace...")
            model = timm.create_model(
                MEGA_DESCRIPTOR_MODEL_ID,
                pretrained=True,
            )
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), weights_path)
            logger.info("Saved model locally at %s", weights_path)
        else:
            raise FileNotFoundError(
                "Local MegaDescriptor weights were not found at "
                f"'{weights_path}'. Download them once manually or call this loader "
                "with allow_download=True during setup."
            )

        model.train()
        return model

    except Exception:
        logger.exception("Model for embeddings failed downloading/loading...")
        raise


def load_embedding_backbone_checkpoint(
    checkpoint_path: str | Path = EMBEDDING_CHECKPOINT_PATH,
    device: str | torch.device = "cpu",
    allow_download: bool = False,
    eval_mode: bool = True,
) -> nn.Module:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Embedding checkpoint not found: '{checkpoint_path}'")

    model = load_mega_descriptor_model_feature_extraction(
        allow_download=allow_download,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "backbone_state_dict" in checkpoint:
        state_dict = checkpoint["backbone_state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = {
            key.replace("backbone.", ""): value
            for key, value in checkpoint["model_state_dict"].items()
            if key.startswith("backbone.")
        }
        if not state_dict:
            raise ValueError(
                "Checkpoint has model_state_dict, but no keys starting with 'backbone.'."
            )
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

    model.load_state_dict(state_dict)
    model.to(device)
    if eval_mode:
        model.eval()
    else:
        model.train()
    return model


class ContrastiveEmbeddingModel(nn.Module):
    def __init__(
        self,
        projection_dim: int = 256,
        projection_hidden_dim: int = 512,
        dropout: float = 0.0,
        allow_download=False
    ):
        super().__init__()

        self.backbone = load_mega_descriptor_model_feature_extraction(allow_download=allow_download)
        self.embedding_dim = int(self.backbone.num_features)  # type: ignore[attr-defined]
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

    def save_full_checkpoint(
        self,
        checkpoint_data: dict,
        path_name: str = "contrastive_model.pt",
    ) -> None:
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
