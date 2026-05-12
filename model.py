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

MEGA_DESCRIPTOR_MODEL_ID = "hf-hub:BVRA/MegaDescriptor-T-CNN-288"
MEGA_DESCRIPTOR_LOCAL_WEIGHTS = Path("artifacts") / "mega_descriptor_t_cnn_288.pth"
EMBEDDING_CHECKPOINT_PATH = Path("artifacts") / "embedding_checkpoints" / "embedding_backbone.pt"


def _extract_backbone_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

    if "backbone_state_dict" in checkpoint:
        state_dict = checkpoint["backbone_state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = {
            key.replace("backbone.", ""): value
            for key, value in checkpoint["model_state_dict"].items()
            if key.startswith("backbone.")
        }
        if not state_dict:
            raise ValueError(
                "Checkpoint has model_state_dict, but no keys starting with 'backbone.'."
            )
    elif "model" in checkpoint:
        model_state = checkpoint["model"]
        if not isinstance(model_state, dict):
            raise ValueError("Checkpoint field 'model' is not a state dict.")
        if any(key.startswith("backbone.") for key in model_state):
            state_dict = {
                key.replace("backbone.", ""): value
                for key, value in model_state.items()
                if key.startswith("backbone.")
            }
        else:
            state_dict = model_state
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported state_dict format: {type(state_dict)}")

    return state_dict


def _load_backbone_weights(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = set(incompatible.missing_keys)
    unexpected_keys = set(incompatible.unexpected_keys)

    allowed_missing = {"classifier.weight", "classifier.bias"}
    disallowed_missing = missing_keys - allowed_missing

    if disallowed_missing or unexpected_keys:
        raise RuntimeError(
            "Backbone checkpoint is incompatible. "
            f"Missing keys: {sorted(disallowed_missing)}. "
            f"Unexpected keys: {sorted(unexpected_keys)}."
        )


def _extract_model_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_features"):
        features = model.forward_features(images)  # type: ignore[attr-defined]
        if hasattr(model, "forward_head"):
            features = model.forward_head(features, pre_logits=True)  # type: ignore[attr-defined]
        return features
    return model(images)


def load_mega_descriptor_model_feature_extraction(
    weights_path: Path = MEGA_DESCRIPTOR_LOCAL_WEIGHTS,
    allow_download: bool = False,
) -> nn.Module:
    import timm
    from huggingface_hub import hf_hub_download

    try:
        if not allow_download:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        model = timm.create_model(
            MEGA_DESCRIPTOR_MODEL_ID,
            pretrained=False,
        )

        if weights_path.exists():
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
            state_dict = _extract_backbone_state_dict(checkpoint)
            _load_backbone_weights(model, state_dict)
            logger.info("Loaded %s from local weights: %s", MEGA_DESCRIPTOR_MODEL_ID, weights_path)
        elif allow_download:
            logger.info("Downloading %s from HuggingFace...", MEGA_DESCRIPTOR_MODEL_ID)
            repo_id = MEGA_DESCRIPTOR_MODEL_ID.removeprefix("hf-hub:")
            cached_checkpoint = hf_hub_download(
                repo_id=repo_id,
                filename="pytorch_model.bin",
            )
            checkpoint = torch.load(cached_checkpoint, map_location="cpu", weights_only=False)
            state_dict = _extract_backbone_state_dict(checkpoint)
            _load_backbone_weights(model, state_dict)
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
    state_dict = _extract_backbone_state_dict(checkpoint)

    _load_backbone_weights(model, state_dict)
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
        allow_download=False,
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
        return self.projection_head(_extract_model_features(self.backbone, images))

    def extract_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        return _extract_model_features(self.backbone, images)

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
