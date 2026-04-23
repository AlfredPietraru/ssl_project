from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from wildlife_datasets.datasets import AnimalCLEF2026


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TwoViewTransform:
    def __init__(self, base_transform: T.Compose) -> None:
        self.base_transform = base_transform

    def __call__(self, image):
        return self.base_transform(image), self.base_transform(image)


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
        transform = build_simclr_transform(image_size) if training else build_eval_transform(image_size)
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
        views, _ = self.dataset[index]
        identity = str(self.metadata.iloc[index]["identity"])
        label = self.identity_to_label[identity]
        return views, torch.tensor(label, dtype=torch.long)


def build_simclr_transform(image_size: int = 384) -> TwoViewTransform:
    color_jitter = T.ColorJitter(
        brightness=0.80,
        contrast=0.80,
        saturation=0.80,
        hue=0.20,
    )
    augmentation = T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=max(3, int(0.1 * image_size) // 2 * 2 + 1), sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return TwoViewTransform(augmentation)


def build_eval_transform(image_size: int = 384) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
