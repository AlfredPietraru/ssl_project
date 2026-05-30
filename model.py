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

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = _extract_backbone_state_dict(checkpoint)
    model = load_mega_descriptor_model_feature_extraction(weights_path=None)

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
        num_classes: int,
        weights_path: str | None = None,
        freeze_backbone: bool = False,
        unfreeze_last_backbone_block: bool = False,
        projection_dim: int | None = None,
    ):
        super().__init__()
        self.backbone = load_mega_descriptor_model_feature_extraction(weights_path=weights_path)
        if not hasattr(self.backbone, "num_features"):
            raise ValueError("Backbone is missing `num_features`, cannot build embedding head.")

        self.embedding_dim = int(self.backbone.num_features)
        if projection_dim is None or projection_dim <= 0:
            self.projection_head: nn.Module = nn.Identity()
            self.projected_dim = self.embedding_dim
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.GELU(),
                nn.Linear(self.embedding_dim, projection_dim),
            )
            self.projected_dim = projection_dim
        self.bn_neck = nn.BatchNorm1d(self.projected_dim)
        self.bn_neck.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.projected_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_backbone_block = (
            unfreeze_last_backbone_block and self.freeze_backbone
        )
        self.set_backbone_trainable(not self.freeze_backbone)

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(images)
        return self.backbone.forward_head(features, pre_logits=True)

    def _project_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        projected_embeddings = self.projection_head(embeddings)
        return self.bn_neck(projected_embeddings)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self._extract_features(images)
        return self._project_embeddings(embeddings)

    def forward_with_logits(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self._extract_features(images)
        projected_embeddings = self._project_embeddings(embeddings)
        logits = self.classifier(projected_embeddings)
        return projected_embeddings, logits

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

        if trainable:
            self.backbone.train()
        else:
            self.backbone.eval()
            if self.unfreeze_last_backbone_block:
                self._unfreeze_last_backbone_block()

    def _unfreeze_last_backbone_block(self) -> None:
        if not hasattr(self.backbone, "blocks") or len(self.backbone.blocks) == 0:
            raise ValueError("Backbone does not expose any blocks to unfreeze.")

        modules_to_unfreeze = [self.backbone.blocks[-1]]
        if hasattr(self.backbone, "conv_head"):
            modules_to_unfreeze.append(self.backbone.conv_head)
        if hasattr(self.backbone, "bn2"):
            modules_to_unfreeze.append(self.backbone.bn2)

        for module in modules_to_unfreeze:
            module.train()
            for parameter in module.parameters():
                parameter.requires_grad = True

    def train(self, mode: bool = True) -> "ContrastiveEmbeddingModel":
        super().train(mode)
        if self.freeze_backbone and mode:
            self.backbone.eval()
            if self.unfreeze_last_backbone_block:
                self._unfreeze_last_backbone_block()
        return self


    def save_checkpoints(
        self,
        full_model_path_name: str = "training_checkpoint.pt",
    ) -> Path:
        full_model_dir = Path("artifacts") / "full_model_checkpoints"
        full_model_dir.mkdir(parents=True, exist_ok=True)
        full_model_path = full_model_dir / full_model_path_name
        torch.save(self.state_dict(), full_model_path)
        return full_model_path


def load_trained_embedding_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    eval_mode: bool = True,
) -> ContrastiveEmbeddingModel:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Training checkpoint not found: '{checkpoint_path}'")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Expected state_dict checkpoint at '{checkpoint_path}', got {type(state_dict)}."
        )
    classifier_weight = state_dict.get("classifier.weight")
    if classifier_weight is None or not isinstance(classifier_weight, torch.Tensor):
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' does not contain 'classifier.weight'; "
            "cannot infer the number of training classes."
        )

    num_classes = int(classifier_weight.shape[0])
    projection_dim = None
    if any(key.startswith("projection_head.") for key in state_dict):
        projection_dim = int(classifier_weight.shape[1])

    model = ContrastiveEmbeddingModel(
        num_classes=num_classes,
        projection_dim=projection_dim,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    if eval_mode:
        model.eval()
    else:
        model.train()
    return model
