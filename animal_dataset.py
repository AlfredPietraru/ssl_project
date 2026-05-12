from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
import kornia.augmentation as K
from torch.utils.data import BatchSampler, DataLoader, Dataset
from wildlife_datasets.datasets import AnimalCLEF2026
import logging
import time
from config import CFG
import warnings
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_cpu_training_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])


def build_cpu_testing_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


training_transform = build_cpu_training_transform(288)
testing_transform = build_cpu_testing_transform(288)


class AnimalSimCLRDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        image_size: int,
        split: str = "train",
        max_samples: int | None = None,
        drop_unknown_identity: bool = True,
        transform = None
    ) -> None:
        self.root = Path(root)
        self.image_size = int(image_size)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root '{self.root}' does not exist.")
        
        self.split  = split
        self.dataset = AnimalCLEF2026(
            str(self.root),
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


class IdentityBalancedBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: AnimalSimCLRDataset,
        identities_per_batch: int,
        images_per_identity: int,
        seed: int = 42,
    ) -> None:
        self.identities_per_batch = int(identities_per_batch)
        self.images_per_identity = int(images_per_identity)
        if self.identities_per_batch <= 0 or self.images_per_identity <= 1:
            raise ValueError("Triplet batches need positive identities_per_batch and images_per_identity > 1.")

        label_to_indices: dict[int, list[int]] = {}
        for index in range(len(dataset)):
            identity = dataset.metadata.iloc[index]["identity"]
            if pd.isna(identity) or str(identity).strip().lower() == "unknown":
                continue
            label = dataset.identity_to_label[str(identity)]
            label_to_indices.setdefault(label, []).append(index)

        self.label_to_indices = {
            label: indices
            for label, indices in label_to_indices.items()
            if len(indices) >= 2
        }
        self.labels = np.array(sorted(self.label_to_indices), dtype=np.int64)
        if len(self.labels) < self.identities_per_batch:
            raise ValueError(
                "Not enough identities with at least two images for balanced triplet batches: "
                f"{len(self.labels)} available, {self.identities_per_batch} requested."
            )

        self.batch_size = self.identities_per_batch * self.images_per_identity
        self.num_batches = max(1, sum(len(indices) for indices in self.label_to_indices.values()) // self.batch_size)
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for _ in range(self.num_batches):
            selected_labels = self.rng.choice(self.labels, size=self.identities_per_batch, replace=False)
            batch: list[int] = []
            for label in selected_labels:
                indices = self.label_to_indices[int(label)]
                replace = len(indices) < self.images_per_identity
                sampled = self.rng.choice(indices, size=self.images_per_identity, replace=replace)
                batch.extend(int(index) for index in sampled)
            self.rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches


class SimCLRGPUTransform(nn.Module):
    def __init__(self, image_size: int) -> None:
        super().__init__()
        kernel_size = max(3, int(0.1 * image_size) // 2 * 2 + 1)
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
    transformations = SimCLRGPUTransform(config.image_size)
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
        image_size=config.image_size,
        split="train",
        max_samples=config.max_samples,
        drop_unknown_identity=False,
        transform=training_transform or build_cpu_training_transform(config.image_size),
    )
    eval_dataset = AnimalSimCLRDataset(
        root=config.root,
        image_size=config.image_size,
        split="test",
        max_samples=config.max_samples,
        drop_unknown_identity=False,
        transform=testing_transform or build_cpu_testing_transform(config.image_size)
    )

    if config.training_loss == "triplet" and shuffle_training:
        train_batch_sampler = IdentityBalancedBatchSampler(
            train_dataset,
            identities_per_batch=config.triplet_identities_per_batch,
            images_per_identity=config.triplet_images_per_identity,
            seed=config.validation_random_seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
    else:
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
