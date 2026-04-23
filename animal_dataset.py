from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
import kornia.augmentation as K
from torch.utils.data import DataLoader, Dataset
from wildlife_datasets.datasets import AnimalCLEF2026
import logging
import time
import warnings
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class AnimalSimCLRDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 384,
        training: bool = True,
        max_samples: int | None = None,
        drop_unknown_identity: bool = True,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root '{self.root}' does not exist.")

        self.dataset = AnimalCLEF2026(
            str(self.root),
            transform=None,
            load_label=True,
            factorize_label=True,
            check_files=False,
        )
        self.metadata = self._select_metadata(
            split=split,
            max_samples=max_samples,
            drop_unknown_identity=drop_unknown_identity,
        )
        self.dataset = self.dataset.get_subset(self.metadata["_source_index"].tolist())
        if training:
            self.dataset.set_transform(T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
            ]))
        else:
            self.dataset.set_transform(T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
            ))
        

        identities = sorted(self.metadata["identity"].astype(str).unique())
        self.identity_to_label = {
            identity: label for label, identity in enumerate(identities)
        }

    def _select_metadata(
        self,
        split: str,
        max_samples: int | None,
        drop_unknown_identity: bool,
    ) -> pd.DataFrame:
        metadata = self.dataset.df.copy()
        mask = metadata["split"].astype(str).eq(split)
        if drop_unknown_identity:
            mask &= ~metadata["identity"].astype(str).eq("unknown")

        source_indices = metadata.index[mask].to_numpy()
        if max_samples is not None:
            source_indices = source_indices[:max_samples]

        selected = metadata.loc[source_indices].reset_index(drop=True).copy()
        selected.insert(0, "_source_index", source_indices)
        selected.insert(1, "embedding_index", np.arange(len(selected)))
        return selected

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, _ = self.dataset[index]
        identity = str(self.metadata.iloc[index]["identity"])
        label = self.identity_to_label[identity]
        return image, torch.tensor(label, dtype=torch.long)


class SimCLRGPUTransform(nn.Module):
    def __init__(self, image_size: int = 384) -> None:
        super().__init__()
        kernel_size = max(3, int(0.1 * image_size) // 2 * 2 + 1)
        self.augment = nn.Sequential(
            K.RandomResizedCrop(size=(image_size, image_size), scale=(0.08, 1.0), p=1.0),
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(
                brightness=0.80,
                contrast=0.80,
                saturation=0.80,
                hue=0.20,
                p=0.8,
            ),
            K.RandomGrayscale(p=0.2),
            K.RandomGaussianBlur(
                kernel_size=(kernel_size, kernel_size),
                sigma=(0.1, 2.0),
                p=0.5,
            ),
            K.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.augment(images)


def build_simclr_data(config: dict[str, object]) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = int(config["image_size"])
    batch_size = int(config["batch_size"])
    num_workers = int(config["num_workers"])

    train_dataset = AnimalSimCLRDataset(
        root="data",
        split="train",
        image_size=image_size,
        training=True,
        max_samples=config.get("max_train_samples"),
        drop_unknown_identity=False
    )
    eval_dataset = AnimalSimCLRDataset(
        root="data",
        split="test",
        image_size=image_size,
        training=False,
        max_samples=config.get("max_eval_samples", config.get("max_train_samples")),
        drop_unknown_identity=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    
    gpu_transform = SimCLRGPUTransform(image_size=image_size).to(device)
    gpu_transform = gpu_transform.eval()

    return {
        "device": device,
        "gpu_transform": gpu_transform,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "train_loader": train_loader,
        "eval_loader": eval_loader,
    }


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SIMCLR")


def summarize_metadata(dataset: AnimalSimCLRDataset) -> dict[str, int]:
    metadata = dataset.metadata
    return {
        "samples": int(len(metadata)),
        "species": int(metadata["species"].nunique()),
        "datasets": int(metadata["dataset"].nunique()),
        "identities": int(metadata["identity"].nunique()),
    }

def inspect_one_train_batch(
    train_loader: DataLoader,
    gpu_transform: SimCLRGPUTransform,
    device: torch.device,
) -> None:
    start_time = time.perf_counter()
    images, labels = next(iter(train_loader))
    load_seconds = time.perf_counter() - start_time

    start_time = time.perf_counter()
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    view_1 = gpu_transform(images)
    view_2 = gpu_transform(images)
    simclr_images = torch.cat([view_1, view_2], dim=0)
    contrastive_labels = torch.cat([labels, labels], dim=0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    augment_seconds = time.perf_counter() - start_time

    logger.info("One train batch CPU load time: %.3f seconds", load_seconds)
    logger.info("One train batch GPU transfer+augment time: %.3f seconds", augment_seconds)
    logger.info("raw images shape: %s", tuple(images.shape))
    logger.info("view_1 shape: %s", tuple(view_1.shape))
    logger.info("view_2 shape: %s", tuple(view_2.shape))
    logger.info("labels shape: %s", tuple(labels.shape))
    logger.info("SimCLR images shape after concat: %s", tuple(simclr_images.shape))
    logger.info("SimCLR labels shape after concat: %s", tuple(contrastive_labels.shape))
    logger.info("First batch labels: %s", labels.cpu().tolist())


def main() -> None:
    config = {
        "root": "data",
        "image_size": 384,
        "batch_size":  64,
        "num_workers":  4,
        "max_train_samples": None,
        "max_eval_samples": None,
    }
    simclr_data = build_simclr_data(config)
    device = simclr_data["device"]
    gpu_transform = simclr_data["gpu_transform"]
    train_dataset = simclr_data["train_dataset"]
    eval_dataset = simclr_data["eval_dataset"]
    train_loader = simclr_data["train_loader"]
    eval_loader = simclr_data["eval_loader"]

    logger.info("Train loader batches: %s", len(train_loader))
    logger.info("Eval loader batches: %s", len(eval_loader))
    logger.info("Train metadata summary: %s", summarize_metadata(train_dataset))
    logger.info("Eval metadata summary: %s", summarize_metadata(eval_dataset))
    
    for i in range(10):
        inspect_one_train_batch(train_loader, gpu_transform, device)


if __name__ == "__main__":
    main()

