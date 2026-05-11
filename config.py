from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass(slots=True, init=False)
class CFG:
    root: str = "data"
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 3
    lr: float = 1e-5
    weight_decay: float = 1e-4
    temperature: float = 0.5
    projection_dim: int = 256
    projection_hidden_dim: int = 512
    projection_dropout: float = 0.0
    checkpoint_dir: str = "checkpoints_simclr"
    device: torch.device
    embedding_checkpoint_path: str = "artifacts/embedding_checkpoints/embedding_backbone.pt"
    embeddings_output_dir: str = "artifacts/embeddings"
    max_samples: int | None = None
    normalize_embeddings: bool = True

    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        data = self._load_yaml_mapping(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root = str(data["root"])
        self.batch_size = int(data["batch_size"])  # type: ignore[arg-type]
        self.num_workers = int(data["num_workers"])  # type: ignore[arg-type]
        self.epochs = int(data["epochs"])  # type: ignore[arg-type]
        self.lr = float(data["lr"])  # type: ignore[arg-type]
        self.weight_decay = float(data["weight_decay"])  # type: ignore[arg-type]
        self.temperature = float(data["temperature"])  # type: ignore[arg-type]
        self.projection_dim = int(data["projection_dim"])  # type: ignore[arg-type]
        self.projection_hidden_dim = int(data["projection_hidden_dim"])  # type: ignore[arg-type]
        self.projection_dropout = float(data["projection_dropout"])  # type: ignore[arg-type]
        self.checkpoint_dir = str(data["checkpoint_dir"])
        self.embedding_checkpoint_path = str(data["embedding_checkpoint_path"])
        self.embeddings_output_dir = str(data["embeddings_output_dir"])
        raw_max_samples = data["max_samples"]
        self.max_samples = None if raw_max_samples is None else int(raw_max_samples)  # type: ignore[arg-type]
        self.normalize_embeddings = bool(data["normalize_embeddings"])

    def to_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "temperature": self.temperature,
            "projection_dim": self.projection_dim,
            "projection_hidden_dim": self.projection_hidden_dim,
            "projection_dropout": self.projection_dropout,
            "checkpoint_dir": self.checkpoint_dir,
            "device": str(self.device),
            "embedding_checkpoint_path": self.embedding_checkpoint_path,
            "embeddings_output_dir": self.embeddings_output_dir,
            "max_samples": self.max_samples,
            "normalize_embeddings": self.normalize_embeddings,
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
            "batch_size",
            "num_workers",
            "epochs",
            "lr",
            "weight_decay",
            "temperature",
            "projection_dim",
            "projection_hidden_dim",
            "projection_dropout",
            "checkpoint_dir",
            "embedding_checkpoint_path",
            "embeddings_output_dir",
            "max_samples",
            "normalize_embeddings",
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
