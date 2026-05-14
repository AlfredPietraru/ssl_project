from pathlib import Path
from collections import defaultdict
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
import kornia.augmentation as K
from torch.utils.data import DataLoader, Dataset, Sampler
from wildlife_datasets.datasets import AnimalCLEF2026
import logging
import time
from config import CFG
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("animal_dataset")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomIdentityBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        labels: list[str],
        batch_size: int,
        instances_per_identity: int = 2,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if instances_per_identity <= 0:
            raise ValueError("instances_per_identity must be positive.")
        if batch_size % instances_per_identity != 0:
            raise ValueError(
                "batch_size must be divisible by instances_per_identity for identity-balanced batching."
            )

        self.labels = [str(label) for label in labels]
        self.batch_size = int(batch_size)
        self.instances_per_identity = int(instances_per_identity)
        self.identities_per_batch = self.batch_size // self.instances_per_identity
        self.index_by_label: dict[str, list[int]] = defaultdict(list)
        for index, label in enumerate(self.labels):
            self.index_by_label[label].append(index)

        self.unique_labels = list(self.index_by_label.keys())
        if len(self.unique_labels) < self.identities_per_batch:
            raise ValueError(
                f"Need at least {self.identities_per_batch} unique identities, "
                f"but found only {len(self.unique_labels)}."
            )

        self.num_batches = max(1, math.ceil(len(self.labels) / self.batch_size))

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng()
        for _ in range(self.num_batches):
            chosen_labels = rng.choice(
                self.unique_labels,
                size=self.identities_per_batch,
                replace=False,
            )
            batch_indices: list[int] = []
            for label in chosen_labels.tolist():
                candidates = self.index_by_label[label]
                replace = len(candidates) < self.instances_per_identity
                sampled = rng.choice(
                    candidates,
                    size=self.instances_per_identity,
                    replace=replace,
                )
                batch_indices.extend(int(index) for index in sampled.tolist())
            rng.shuffle(batch_indices)
            yield batch_indices


def split_metadata_by_identity_train_validation(
    metadata: pd.DataFrame,
    train_probability: float = 0.8,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "identity" not in metadata.columns:
        raise ValueError("Expected an 'identity' column in metadata.")
    if not 0.0 < train_probability < 1.0:
        raise ValueError("train_probability must be strictly between 0 and 1.")

    eligible_metadata = metadata.copy()
    eligible_metadata = eligible_metadata.loc[
        ~eligible_metadata["identity"].isna()
        & ~eligible_metadata["identity"].astype(str).eq("unknown")
    ].copy()
    if eligible_metadata.empty:
        raise ValueError("No known identities available for splitting.")

    identities = eligible_metadata["identity"].astype(str).unique()
    rng = np.random.default_rng(random_seed)
    train_identity_mask = rng.random(len(identities)) < train_probability

    if len(identities) > 1:
        if train_identity_mask.all():
            train_identity_mask[rng.integers(len(identities))] = False
        elif (~train_identity_mask).all():
            train_identity_mask[rng.integers(len(identities))] = True

    train_identities = set(identities[train_identity_mask].tolist())
    validation_identities = set(identities[~train_identity_mask].tolist())

    train_rows = eligible_metadata.loc[
        eligible_metadata["identity"].astype(str).isin(train_identities)
    ].reset_index(drop=True)
    validation_rows = eligible_metadata.loc[
        eligible_metadata["identity"].astype(str).isin(validation_identities)
    ].reset_index(drop=True)
    return train_rows, validation_rows

class AnimalSimCLRDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        image_size: int,
        split: str = "train",
        max_samples: int | None = None,
        drop_unknown_identity: bool = True,
        transform = None,
        class_name: str | None = None,
        class_column: str = "species",
        metadata: pd.DataFrame | None = None,
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
        if metadata is None:
            self.metadata = self._select_metadata(
                split=split,
                max_samples=max_samples,
                drop_unknown_identity=drop_unknown_identity,
                class_name=class_name,
                class_column=class_column,
            )
        else:
            self.metadata = self._prepare_external_metadata(metadata=metadata)
        self.dataset = self.dataset.get_subset(self.metadata["_source_index"].tolist())
        self.dataset.set_transform(transform)
        self.cached_images = self._cache_images_in_ram(cache_num_workers) if cache_images_in_ram else None
        identities = sorted(self.metadata["identity"].astype(str).unique())
        self.identity_to_label = {
            identity: label for label, identity in enumerate(identities)
        }

    def _select_metadata(
        self,
        split: str,
        max_samples: int | None,
        drop_unknown_identity: bool,
        class_name: str | None,
        class_column: str,
    ) -> pd.DataFrame:
        metadata = self.dataset.df.copy()
        if class_column not in metadata.columns:
            available_columns = ", ".join(sorted(metadata.columns))
            raise ValueError(
                f"Column '{class_column}' is not present in the dataset metadata. "
                f"Available columns: {available_columns}"
            )

        mask = metadata["split"].astype(str).eq(split)
        if drop_unknown_identity:
            mask &= ~metadata["identity"].astype(str).eq("unknown")
        if class_name is not None:
            normalized_class = str(class_name).strip().lower()
            mask &= metadata[class_column].astype(str).str.strip().str.lower().eq(normalized_class)

        source_indices = metadata.index[mask].to_numpy()
        if max_samples is not None:
            source_indices = source_indices[:max_samples]

        selected = metadata.loc[source_indices].reset_index(drop=True).copy()
        selected.insert(0, "_source_index", source_indices)
        selected.insert(1, "embedding_index", np.arange(len(selected)))
        return selected

    def _prepare_external_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        required_columns = {"identity", "_source_index"}
        missing_columns = sorted(required_columns - set(metadata.columns))
        if missing_columns:
            raise ValueError(
                "Provided metadata is missing required columns: "
                f"{missing_columns}"
            )

        selected = metadata.reset_index(drop=True).copy()
        if "embedding_index" not in selected.columns:
            selected.insert(1, "embedding_index", np.arange(len(selected)))
        return selected

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        if self.cached_images is None:
            image, _ = self.dataset[index]
        else:
            image = self.cached_images[index].clone()
        identity = self.metadata.iloc[index]["identity"]
        label = -1 if pd.isna(identity) else self.identity_to_label[str(identity)]
        return image, torch.tensor(label, dtype=torch.long)

    def _cache_images_in_ram(self, num_workers: int = 0) -> torch.Tensor:
        logger.info("Caching %s %s images in RAM", len(self.dataset), self.split)
        loader = DataLoader(
            self.dataset,
            batch_size=64,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=False,
        )

        batches: list[torch.Tensor] = []
        for batch in tqdm(
            loader,
            desc=f"cache {self.split}",
            leave=False,
        ):
            if isinstance(batch, dict):
                images = batch["image"]
            else:
                images = batch[0]
            if not isinstance(images, torch.Tensor):
                images = torch.stack([T.ToTensor()(image) for image in images], dim=0)
            batches.append(images.cpu())

        cached = torch.cat(batches, dim=0).contiguous()
        gib = cached.numel() * cached.element_size() / (1024 ** 3)
        logger.info(
            "Cached %s %s images in RAM | shape=%s | size=%.2f GiB",
            len(self.dataset),
            self.split,
            tuple(cached.shape),
            gib,
        )
        return cached


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


class AnimalTrainClassDataset(AnimalSimCLRDataset):
    def __init__(
        self,
        root: str | Path,
        image_size: int,
        class_name: str,
        max_samples: int | None = None,
        drop_unknown_identity: bool = True,
        transform = None,
        class_column: str = "species",
    ) -> None:
        super().__init__(
            root=root,
            image_size=image_size,
            split="train",
            max_samples=max_samples,
            drop_unknown_identity=drop_unknown_identity,
            transform=transform,
            class_name=class_name,
            class_column=class_column,
        )


class SimCLRGPUTransform(nn.Module):
    def __init__(self, image_size: int) -> None:
        super().__init__()
        kernel_size = max(3, int(0.1 * image_size) // 2 * 2 + 1)
        self.augment = nn.Sequential(
            # K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(degrees=3, translate=(0.03, 0.03), scale=(0.95, 1.05), p=0.4),
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

TRAIN_SPECIES = (
    "lynx",
    "salamander",
    "loggerhead turtle",
)

def build_transformations(config : CFG):
    transformations = SimCLRGPUTransform(config.image_size)
    transformations = transformations.to(config.device)
    transformations = transformations.eval()
    return transformations


def build_simclr_data(
        config : CFG, 
        shuffle_training : bool = True, 
        training_transform = None,
        testing_transform = None,
        build_eval: bool = True,
    ) -> tuple[AnimalSimCLRDataset, AnimalSimCLRDataset, DataLoader, DataLoader]:
    batch_size = config.batch_size
    cache_num_workers = config.num_workers
    loader_num_workers = 0 if config.cache_images_in_ram else config.num_workers

    train_dataset = AnimalSimCLRDataset(
        root=config.root,
        image_size=config.image_size,
        split="train",
        max_samples=config.max_samples,
        drop_unknown_identity=False,
        transform=training_transform or build_cpu_training_transform(config.image_size),
        cache_images_in_ram=config.cache_images_in_ram,
        cache_num_workers=cache_num_workers,
    )
    eval_dataset = None
    if build_eval:
        eval_dataset = AnimalSimCLRDataset(
            root=config.root,
            image_size=config.image_size,
            split="test",
            max_samples=config.max_samples,
            drop_unknown_identity=False,
            transform=testing_transform or build_cpu_testing_transform(config.image_size),
            cache_images_in_ram=config.cache_images_in_ram,
            cache_num_workers=cache_num_workers,
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


def build_simclr_train_val_by_identity(
        config: CFG,
        shuffle_training: bool = True,
        train_probability: float = 0.8,
        random_seed: int | None = None,
        training_transform=None,
        validation_transform=None,
        class_name: str | None = None,
        class_column: str = "species",
        instances_per_identity: int = 2,
    ) -> tuple[AnimalSimCLRDataset, AnimalSimCLRDataset, DataLoader, DataLoader]:
    batch_size = config.batch_size
    num_workers = config.num_workers
    seed = config.validation_random_seed if random_seed is None else random_seed

    source_dataset = AnimalCLEF2026(
        str(config.root),
        load_label=True,
        factorize_label=True,
        check_files=False,
    )
    source_metadata = source_dataset.df.copy()
    source_metadata = source_metadata.loc[
        source_metadata["split"].astype(str).eq("train")
    ].copy()

    if class_name is not None:
        if class_column not in source_metadata.columns:
            available_columns = ", ".join(sorted(source_metadata.columns))
            raise ValueError(
                f"Column '{class_column}' is not present in the dataset metadata. "
                f"Available columns: {available_columns}"
            )
        available_classes = sorted(
            source_metadata[class_column].dropna().astype(str).str.strip().unique().tolist()
        )
        normalized_class = str(class_name).strip().lower()
        source_metadata = source_metadata.loc[
            source_metadata[class_column].astype(str).str.strip().str.lower().eq(normalized_class)
        ].copy()

    if config.max_samples is not None:
        source_metadata = source_metadata.iloc[: config.max_samples].copy()

    if source_metadata.empty:
        if class_name is not None:
            preview = ", ".join(available_classes[:20])
            raise ValueError(
                f"No training rows found for class '{class_name}'. "
                f"Available {class_column} values include: {preview}"
            )
        raise ValueError("No training rows found.")

    source_indices = source_metadata.index.to_numpy()
    source_metadata = source_metadata.reset_index(drop=True).copy()
    source_metadata.insert(0, "_source_index", source_indices)
    source_metadata.insert(1, "embedding_index", np.arange(len(source_metadata)))

    train_metadata, validation_metadata = split_metadata_by_identity_train_validation(
        metadata=source_metadata,
        train_probability=train_probability,
        random_seed=seed,
    )

    train_dataset = AnimalSimCLRDataset(
        root=config.root,
        image_size=config.image_size,
        split="train",
        max_samples=None,
        drop_unknown_identity=False,
        transform=training_transform or build_cpu_training_transform(config.image_size),
        metadata=train_metadata,
    )
    validation_dataset = AnimalSimCLRDataset(
        root=config.root,
        image_size=config.image_size,
        split="train",
        max_samples=None,
        drop_unknown_identity=False,
        transform=validation_transform or build_cpu_testing_transform(config.image_size),
        metadata=validation_metadata,
    )

    train_loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if shuffle_training:
        train_batch_sampler = RandomIdentityBatchSampler(
            labels=train_dataset.metadata["identity"].astype(str).tolist(),
            batch_size=batch_size,
            instances_per_identity=instances_per_identity,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            **train_loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **train_loader_kwargs,
        )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return train_dataset, validation_dataset, train_loader, validation_loader


def build_species_specific_train_data(
        config: CFG,
        shuffle_training: bool = True,
        training_transform=None,
        species_names: tuple[str, ...] = TRAIN_SPECIES,
    ) -> tuple[
        dict[str, AnimalTrainClassDataset],
        dict[str, DataLoader],
    ]:
    batch_size = config.batch_size
    num_workers = config.num_workers
    transform = training_transform or build_cpu_training_transform(config.image_size)

    train_datasets: dict[str, AnimalTrainClassDataset] = {}
    train_loaders: dict[str, DataLoader] = {}

    for species_name in species_names:
        dataset = AnimalTrainClassDataset(
            root=config.root,
            image_size=config.image_size,
            class_name=species_name,
            max_samples=config.max_samples,
            drop_unknown_identity=False,
            transform=transform,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_training,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
        train_datasets[species_name] = dataset
        train_loaders[species_name] = loader

    return train_datasets, train_loaders
