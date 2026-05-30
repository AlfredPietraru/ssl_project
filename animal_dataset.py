import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import kornia.augmentation as K
from torch.utils.data import DataLoader, Dataset, Sampler, BatchSampler
import logging
from typing import Any
from config import CFG
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("animal_dataset")


class MyAnimalDatasetPK(Dataset):
    def __init__(
        self,
        metadata: list[dict],
        cfg: CFG,
        cache_all_images: bool = False,
        transform=None,
    ):
        super().__init__()
        self.metadata = metadata
        self.cfg = cfg
        self.transform = transform or T.ToTensor()
        self.image_cache: dict[str, Image.Image] = {}

        if len(self.metadata) == 0:
            raise ValueError("The metadata does not exist.")
        lm = self.metadata[0]
        if lm.get("identity") is None or lm.get("paths") is None:
            raise ValueError("The metadata format is not good.")
        if not isinstance(lm["identity"], int):
            raise ValueError("Expected metadata identities to be integers.")
        if not isinstance(lm["paths"], list):
            raise ValueError("Expected metadata paths to be a list of image paths.")

        self.cache_all_images = cache_all_images
        if cache_all_images:
            for item in self.metadata:
                for path in item["paths"]:
                    if path not in self.image_cache:
                        self.image_cache[path] = Image.open(path).convert("RGB")
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        local_metadata = self.metadata[index]
        identity = local_metadata["identity"]
        paths = list(local_metadata["paths"])
        instances_per_identity = int(self.cfg.instances_per_identity)

        if len(paths) == 0:
            raise ValueError(f"Identity '{identity}' has no image paths.")

        sampled_indices = np.random.choice(
            len(paths),
            size=instances_per_identity,
            replace=len(paths) < instances_per_identity,
        )
        sampled_paths = [paths[int(sampled_index)] for sampled_index in sampled_indices.tolist()]

        images: list[torch.Tensor] = []
        for path in sampled_paths:
            if self.cache_all_images and path in self.image_cache:
                image = self.image_cache[path].copy()
            else:
                image = Image.open(path).convert("RGB")
            augmented_image = self.transform(image.copy())
            images.append(augmented_image)

        stacked_images = torch.stack(images, dim=0)
        return stacked_images, torch.tensor(identity, dtype=torch.long)
    

class MyAnimalDatasetSimple(Dataset):
    def __init__(
        self,
        metadata: list[dict],
        cfg: CFG,
        cache_all_images: bool = False,
        transform=None,
    ):
        super().__init__()
        self.metadata = metadata
        self.cfg = cfg
        self.transform = transform or T.ToTensor()
        self.image_cache: dict[str, Image.Image] = {}

        if len(self.metadata) == 0:
            raise ValueError("The metadata does not exist.")
        lm = self.metadata[0]
        if lm.get("identity") is None or lm.get("paths") is None:
            raise ValueError("The metadata format is not good.")
        if not isinstance(lm["identity"], int):
            raise ValueError("Expected metadata identities to be integers.")
        if not isinstance(lm["paths"], list):
            raise ValueError("Expected metadata paths to be a list of image paths.")

        self.flat_entries: list[tuple[str, int]] = []
        for item in self.metadata:
            identity = item["identity"]
            for path in item["paths"]:
                self.flat_entries.append((str(path), identity))

        self.cache_all_images = cache_all_images
        if cache_all_images:
            for path, _ in self.flat_entries:
                if path not in self.image_cache:
                    self.image_cache[path] = Image.open(path).convert("RGB")

    def __len__(self):
        return len(self.flat_entries)

    def __getitem__(self, index):
        path, label = self.flat_entries[index]
        if self.cache_all_images and path in self.image_cache:
            image = self.image_cache[path].copy()
        else:
            image = Image.open(path).convert("RGB")

        transformed_image = self.transform(image)
        return transformed_image, torch.tensor(label, dtype=torch.long)


def build_pk_train_val_datasets_and_loaders(
    train_metadata: list[dict[str, Any]],
    val_metadata: list[dict[str, Any]],
    cfg: CFG,
    train_transform=None,
    val_transform=None,
    cache_all_images: bool = False,
) -> tuple[MyAnimalDatasetPK, MyAnimalDatasetPK, DataLoader, DataLoader]:
    train_dataset = MyAnimalDatasetPK(
        metadata=train_metadata,
        cfg=cfg,
        cache_all_images=cache_all_images,
        transform=train_transform or T.ToTensor(),
    )
    val_dataset = MyAnimalDatasetPK(
        metadata=val_metadata,
        cfg=cfg,
        cache_all_images=cache_all_images,
        transform=val_transform or T.ToTensor(),
    )
    def _pk_collate_fn(batch: list[tuple[torch.Tensor, 
                                         torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        image_groups, labels = zip(*batch)
        images = torch.cat(image_groups, dim=0)
        repeated_labels = torch.cat(
            [
                label.repeat(image_group.shape[0])
                for image_group, label in zip(image_groups, labels)
            ],
            dim=0,
        )
        return images, repeated_labels

    loader_kwargs = {
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": cfg.num_workers > 0,
        "collate_fn": _pk_collate_fn,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def build_simple_train_val_datasets_and_loaders(
    train_metadata: list[dict[str, Any]],
    val_metadata: list[dict[str, Any]],
    cfg: CFG,
    train_transform=None,
    val_transform=None,
    cache_all_images: bool = False,
) -> tuple[MyAnimalDatasetSimple, MyAnimalDatasetSimple, DataLoader, DataLoader]:
    train_dataset = MyAnimalDatasetSimple(
        metadata=train_metadata,
        cfg=cfg,
        cache_all_images=cache_all_images,
        transform=train_transform or T.ToTensor(),
    )
    val_dataset = MyAnimalDatasetSimple(
        metadata=val_metadata,
        cfg=cfg,
        cache_all_images=cache_all_images,
        transform=val_transform or T.ToTensor(),
    )
    loader_kwargs = {
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": cfg.num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_dataset, val_dataset, train_loader, val_loader
