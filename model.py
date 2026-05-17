import logging
import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from typing import Optional

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


def load_mega_descriptor_model_feature_extraction(
    weights_path: Optional[str] | Optional[Path],
) -> nn.Module:
    import timm
    from huggingface_hub import hf_hub_download

    try:
        previous_hf_offline = os.environ.get("HF_HUB_OFFLINE")
        previous_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")

        try:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            model = timm.create_model(
                MEGA_DESCRIPTOR_MODEL_ID,
                pretrained=False,
            )
        except Exception:
            if previous_hf_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous_hf_offline
            if previous_transformers_offline is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = previous_transformers_offline

            model = timm.create_model(
                MEGA_DESCRIPTOR_MODEL_ID,
                pretrained=False,
            )
        finally:
            if previous_hf_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous_hf_offline
            if previous_transformers_offline is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = previous_transformers_offline

        weights_path = Path(weights_path) if weights_path is not None else None
        if weights_path is not None:
            if not weights_path.exists():
                raise FileNotFoundError(f"Backbone weights not found at: {weights_path}")
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
            state_dict = _extract_backbone_state_dict(checkpoint)
            _load_backbone_weights(model, state_dict)
            logger.info("Loaded %s from local weights: %s", MEGA_DESCRIPTOR_MODEL_ID, weights_path)
        else:
            logger.info("Downloading %s from HuggingFace...", MEGA_DESCRIPTOR_MODEL_ID)
            repo_id = MEGA_DESCRIPTOR_MODEL_ID.removeprefix("hf-hub:")
            cached_checkpoint = hf_hub_download(
                repo_id=repo_id,
                filename="pytorch_model.bin",
            )
            checkpoint = torch.load(cached_checkpoint, map_location="cpu", weights_only=False)
            state_dict = _extract_backbone_state_dict(checkpoint)
            _load_backbone_weights(model, state_dict)

        model.train()
        return model

    except Exception:
        logger.exception("Model for embeddings failed downloading/loading...")
        raise


def load_embedding_backbone_checkpoint(
    checkpoint_path: str | Path = EMBEDDING_CHECKPOINT_PATH,
    device: str | torch.device = "cpu",
    eval_mode: bool = True,
) -> nn.Module:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Embedding checkpoint not found: '{checkpoint_path}'")

    model = load_mega_descriptor_model_feature_extraction()
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
    def __init__(self, weights_path : str | None = None):
        super().__init__()
        self.backbone = load_mega_descriptor_model_feature_extraction(weights_path=weights_path)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(images)
        return self.backbone.forward_head(features, pre_logits=True)


    def save_checkpoints(
        self,
        full_model_path_name: str = "training_checkpoint.pt",
    ) -> Path:
        full_model_dir = Path("artifacts") / "full_model_checkpoints"
        full_model_dir.mkdir(parents=True, exist_ok=True)
        full_model_path = full_model_dir / full_model_path_name
        torch.save(self.state_dict(), full_model_path)
        return full_model_path
