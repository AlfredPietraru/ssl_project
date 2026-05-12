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
from config import CFG
import warnings
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
training_transform = T.Compose([
                T.Resize((384, 384)),
                T.ToTensor(),
])


testing_transform = T.Compose([
                T.Resize((384, 384)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]) 


class AnimalSimCLRDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        max_samples: int | None = None,
        drop_unknown_identity: bool = True,
        transform = None
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root '{self.root}' does not exist.")
        
        self.split  = split
        self.dataset = AnimalCLEF2026(
            str(self.root),
            transform=T.Compose([
                T.Resize((384, 384)),
                T.ToTensor()
            ]),
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
        self.dataset.set_transform(transform)
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
        identity = self.metadata.iloc[index]["identity"]
        label = -1 if pd.isna(identity) else self.identity_to_label[str(identity)]
        return image, torch.tensor(label, dtype=torch.long)


class SimCLRGPUTransform(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel_size = max(3, int(0.1 * 384) // 2 * 2 + 1)
        self.augment = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(degrees=10, translate=(0.03, 0.03), scale=(0.95, 1.05), p=0.4),
            K.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.25,
                hue=0.20,
                p=0.8,
            ),
            K.RandomGrayscale(p=0.1),
            # K.RandomGaussianBlur(
            #     kernel_size=(kernel_size, kernel_size),
            #     sigma=(0.1, 2.0),
            #     p=0.5,
            # ),
            K.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.augment(images)
    

def build_transformations(config : CFG):
    transformations = SimCLRGPUTransform()
    transformations = transformations.to(config.device)
    transformations = transformations.eval()
    return transformations

def build_simclr_data(
        config : CFG, 
        shuffle_training : bool = True, 
        training_transform = None,
        testing_transform = None
    ) -> tuple[AnimalSimCLRDataset, AnimalSimCLRDataset, DataLoader, DataLoader]:
    batch_size = config.batch_size
    num_workers = config.num_workers

    train_dataset = AnimalSimCLRDataset(
        root=config.root,
        split="train",
        max_samples=config.max_samples,
        drop_unknown_identity=False,
        transform=training_transform,
    )
    eval_dataset = AnimalSimCLRDataset(
        root=config.root,
        split="test",
        max_samples=config.max_samples,
        drop_unknown_identity=False,
        transform=testing_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_training,
        drop_last=False,
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
    return train_dataset, eval_dataset, train_loader, eval_loader  


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("animal_dataset")


def summarize_metadata(dataset: AnimalSimCLRDataset) -> dict[str, int]:
    metadata = dataset.metadata
    return {
        "samples": int(len(metadata)),
        "species": int(metadata["species"].nunique()),
        "datasets": int(metadata["dataset"].nunique()),
        "identities": int(metadata["identity"].nunique()),
    }


def main() -> None:
    cfg = CFG("config.yaml")
    train_dataset, eval_dataset, train_loader, eval_loader  = build_simclr_data(cfg)
    device = cfg.device
    gpu_transform = build_transformations(cfg)

    logger.info("Device tyoe: %s", device)
    logger.info("Train loader batches: %s", len(train_loader))
    logger.info("Eval loader batches: %s", len(eval_loader))
    logger.info("Train metadata summary: %s", summarize_metadata(train_dataset))
    logger.info("Eval metadata summary: %s", summarize_metadata(eval_dataset))
    print()
    for i in range(20): 
        start_time = time.perf_counter()
        images, labels = next(iter(train_loader))
        load_seconds = time.perf_counter() - start_time

        start_time = time.perf_counter()
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        view_1 = gpu_transform(images)
        view_2 = gpu_transform(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        simclr_images = torch.cat([view_1, view_2], dim=0)
        contrastive_labels = torch.cat([labels, labels], dim=0)
        
        augment_seconds = time.perf_counter() - start_time

        logger.info("One train batch CPU load time: %.3f seconds", load_seconds)
        logger.info("One train batch GPU transfer+augment time: %.3f seconds", augment_seconds)
        logger.info("raw images shape: %s", tuple(images.shape))
        logger.info("view_1 shape: %s", tuple(view_1.shape))
        logger.info("view_2 shape: %s", tuple(view_2.shape))
        logger.info("labels shape: %s", tuple(labels.shape))
        logger.info("SimCLR images shape after concat: %s", tuple(simclr_images.shape))
        logger.info("SimCLR labels shape after concat: %s", tuple(contrastive_labels.shape))

if __name__ == "__main__":
    main()
