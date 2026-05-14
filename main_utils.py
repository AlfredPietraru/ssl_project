import logging
import os
import random
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from wildlife_datasets.datasets import AnimalCLEF2026

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"))
import kagglehub
from kagglehub.config import get_kaggle_credentials
from kagglehub.exceptions import KaggleApiHTTPError
from pathlib import Path
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MAIN")


def normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def require_existing_paths(
    requirements: list[tuple[Path, str]],
    *,
    step_name: str,
) -> None:
    missing = [(path, producer_step) for path, producer_step in requirements if not path.exists()]
    if not missing:
        return

    missing_lines = [
        f"- Missing required input `{path}`. Expected from {producer_step}."
        for path, producer_step in missing
    ]
    joined_lines = "\n".join(missing_lines)
    raise FileNotFoundError(
        f"{step_name} cannot start because required outputs from previous steps are missing:\n"
        f"{joined_lines}"
    )


def require_split_artifacts(
    base_dir: Path,
    split_names: list[str],
    *,
    step_name: str,
    producer_step: str,
) -> None:
    requirements: list[tuple[Path, str]] = []
    for split_name in split_names:
        requirements.append((base_dir / f"{split_name}_embeddings.npy", producer_step))
        requirements.append((base_dir / f"{split_name}_metadata.csv", producer_step))
    require_existing_paths(requirements, step_name=step_name)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


class HarryPlotter:
    def __init__(
        self,
        save_name: str = "simclr_loss.png",
        artifacts_dir: str | Path = "artifacts",
        title: str = "SimCLR training loss",
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = self.artifacts_dir / save_name
        self.title = title
        self.train_losses: list[float] = []

    def update(self, loss: float) -> Path:
        self.train_losses.append(float(loss))
        self.save()
        return self.save_path

    def extend(self, losses: Sequence[float]) -> Path:
        self.train_losses.extend(float(loss) for loss in losses)
        self.save()
        return self.save_path

    def save(self) -> Path:
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, label="Train loss")
        plt.xlabel("Epoch")
        plt.ylabel("Supervised contrastive loss")
        plt.title(self.title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.save_path)
        plt.close()
        return self.save_path


def plot_loss(train_losses, save_name="simclr_loss.png"):
    return HarryPlotter(save_name=save_name).extend(train_losses)


def _log_kaggle_auth_state() -> None:
    credentials = get_kaggle_credentials()
    has_token_env = bool(os.getenv("KAGGLE_API_TOKEN"))
    has_username_env = bool(os.getenv("KAGGLE_USERNAME"))
    has_key_env = bool(os.getenv("KAGGLE_KEY"))

    logger.info(
        "Kaggle auth loaded: credentials_present=%s, token_env=%s, username_env=%s, key_env=%s",
        bool(credentials),
        has_token_env,
        has_username_env,
        has_key_env,
    )


def download_dataset():
    if os.path.isdir("data"):
        logger.info("`data` folder already exists. Skipping download.")
        return AnimalCLEF2026("data")

    logger.info("`data` folder not found. Downloading dataset...")
    _log_kaggle_auth_state()

    try:
        kagglehub.competition_download("animal-clef-2026", output_dir="data")
    except KaggleApiHTTPError as exc:
        logger.exception("Dataset download failed in kagglehub.competition_download().")
        if exc.response is not None and exc.response.status_code == 401:
            logger.error(
                "Kaggle returned 401. This usually means one of these is true: "
                "(1) `KAGGLE_API_TOKEN` was not loaded into this process, "
                "(2) the token is invalid or belongs to a different account, or "
                "(3) that Kaggle account has not accepted the AnimalCLEF 2026 competition rules yet."
            )
            logger.error(
                "Open https://www.kaggle.com/competitions/animal-clef-2026/rules while logged into "
                "the same Kaggle account as the API token and accept the rules, then retry."
            )
        return None
    except Exception:
        logger.exception("Dataset download failed in kagglehub.competition_download().")
        return None

    logger.info("Dataset downloaded and moved to `data`.")
    return AnimalCLEF2026("data")


if __name__ == "__main__":
    dataset = download_dataset()
    if dataset is None:
        print("empty dataset")
        exit(1)
    print(dataset.metadata)
