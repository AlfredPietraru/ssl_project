from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import torch



class CFG:
    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        data = self._load_yaml_mapping(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root = str(data["root"])
        self.image_size = int(data["image_size"])  # type: ignore[arg-type]
        self.batch_size = int(data["batch_size"])  # type: ignore[arg-type]
        self.instances_per_identity = int(data["instances_per_identity"])  # type: ignore[arg-type]
        self.num_workers = int(data["num_workers"])  # type: ignore[arg-type]
        self.epochs = int(data["epochs"])  # type: ignore[arg-type]
        self.early_stopping_patience = int(data["early_stopping_patience"])  # type: ignore[arg-type]
        self.early_stopping_min_delta = float(data["early_stopping_min_delta"])  # type: ignore[arg-type]
        self.lr = float(data["lr"])  # type: ignore[arg-type]
        self.weight_decay = float(data["weight_decay"])  # type: ignore[arg-type]
        self.temperature = float(data["temperature"])  # type: ignore[arg-type]
        self.triplet_margin = float(data["triplet_margin"])  # type: ignore[arg-type]
        self.projection_dim = int(data["projection_dim"])  # type: ignore[arg-type]
        self.metric_loss_weight = float(data["metric_loss_weight"])  # type: ignore[arg-type]
        self.id_loss_weight = float(data["id_loss_weight"])  # type: ignore[arg-type]
        self.freeze_backbone = bool(data["freeze_backbone"])
        self.unfreeze_last_backbone_block = bool(data["unfreeze_last_backbone_block"])
        self.mode = str(data["mode"])
        raw_checkpoint_path = data["checkpoint_path"]
        self.checkpoint_path = None if raw_checkpoint_path is None else str(raw_checkpoint_path)
        self.checkpoint_dir = str(data["checkpoint_dir"])
        self.embedding_checkpoint_path = str(data["embedding_checkpoint_path"])
        raw_max_samples = data["max_samples"]
        self.max_samples = None if raw_max_samples is None else int(raw_max_samples)  # type: ignore[arg-type]
        
        
    def to_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "instances_per_identity": self.instances_per_identity,
            "num_workers": self.num_workers,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "temperature": self.temperature,
            "triplet_margin": self.triplet_margin,
            "projection_dim": self.projection_dim,
            "metric_loss_weight": self.metric_loss_weight,
            "id_loss_weight": self.id_loss_weight,
            "freeze_backbone": self.freeze_backbone,
            "unfreeze_last_backbone_block": self.unfreeze_last_backbone_block,
            "mode": self.mode,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_dir": self.checkpoint_dir,
            "device": str(self.device),
            "embedding_checkpoint_path": self.embedding_checkpoint_path,
            "max_samples": self.max_samples,
        }

    @staticmethod
    def _load_yaml_mapping(path: str | Path) -> dict[str, object]:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to read CFG objects from YAML files."
            ) from exc

        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        if not isinstance(data, dict):
            raise ValueError(f"Expected YAML object mapping in '{path}', got {type(data)}")

        required_fields = {
            "root",
            "image_size",
            "batch_size",
            "instances_per_identity",
            "num_workers",
            "epochs",
            "early_stopping_patience",
            "early_stopping_min_delta",
            "lr",
            "weight_decay",
            "temperature",
            "triplet_margin",
            "projection_dim",
            "metric_loss_weight",
            "id_loss_weight",
            "freeze_backbone",
            "unfreeze_last_backbone_block",
            "mode",
            "checkpoint_path",
            "checkpoint_dir",
            "embedding_checkpoint_path",
            "max_samples"
        }
        missing_fields = sorted(required_fields - set(data))
        if missing_fields:
            raise ValueError(
                f"Config file '{path}' is missing required fields: {missing_fields}"
            )
        return data

    def to_yaml(self, path: str | Path) -> Path:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to write CFG objects to YAML files."
            ) from exc

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
        return path

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CFG":
        return cls(path)
