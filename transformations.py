from pathlib import Path
from collections import defaultdict
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
import kornia.augmentation as K
from torch.utils.data import DataLoader, Dataset, Sampler, BatchSampler
from wildlife_datasets.datasets import AnimalCLEF2026
import logging
import time
from config import CFG
import warnings
from tqdm import tqdm
from data_fetcher import DataFetcher

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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

def build_transformations(config : CFG):
    transformations = SimCLRGPUTransform(config.image_size)
    transformations = transformations.to(config.device)
    transformations = transformations.eval()
    return transformations


training_transform = build_cpu_training_transform(288)
testing_transform = build_cpu_testing_transform(288)
