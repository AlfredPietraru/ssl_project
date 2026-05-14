import torch.nn as nn
from config import CFG
from torch.utils.data import DataLoader, Dataset
import torch
from loss_functions import SupervisedContrastiveLoss
from main_utils import (
    HarryPlotter,
    set_seed
)
from typing import Optional
import logging
import time
import warnings
from model import ContrastiveEmbeddingModel
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

class EmbeddingModelTrainer:
    def __init__(self, cfg : CFG, transform : nn.Module, 
                 train_loader : DataLoader, val_loader : Optional[DataLoader] = None) -> None:
        set_seed()
        self.logger = logging.getLogger("embedding_model_trainer")
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = ContrastiveEmbeddingModel(
            projection_dim=self.cfg.projection_dim,
            projection_hidden_dim=self.cfg.projection_hidden_dim,
            dropout=self.cfg.projection_dropout,
            allow_download=True,
        ).to(cfg.device)
        self.scaler = torch.GradScaler("cuda", enabled=(self.cfg.device.type == "cuda"))
        self.gpu_transform = transform
        self.loss_fn = SupervisedContrastiveLoss(
            temperature=cfg.temperature
        ).to(cfg.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),  # type: ignore
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        self.plotter = HarryPlotter(save_name="simclr_loss.png")

    def train(self) -> ContrastiveEmbeddingModel:
        best_loss = float("inf")
        self.model.train()

        for epoch in range(1, self.cfg.epochs + 1):
            avg_loss, elapsed = self._train_one_epoch(epoch)
            self.logger.info(
                "Epoch %d/%d finished | avg_loss=%.6f | time=%.1fs",
                epoch,
                self.cfg.epochs,
                avg_loss,
                elapsed,
            )
            self.plotter.update(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.model.save_checkpoints(
                    checkpoint_data={
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scaler_state_dict": self.scaler.state_dict(),
                        "loss": avg_loss,
                        "config": self.cfg.to_dict(),
                    }
                )
                self.logger.info("Saved new best checkpoint with loss %.6f", best_loss)
        self.model = self.model.eval()
        return self.model

    def _train_one_epoch(self, epoch : int) -> tuple[float, float]:
        epoch_loss = 0.0
        valid_batches = 0
        start_time = time.perf_counter()

        for batch_idx, (images, labels) in enumerate(self.train_loader, start=1):
            images = images.to(self.cfg.device, non_blocking=True)
            labels = labels.to(self.cfg.device, non_blocking=True)

            view_1 = self.gpu_transform(images)
            view_2 = self.gpu_transform(images)

            simclr_images = torch.cat([view_1, view_2], dim=0)
            simclr_labels = torch.cat([labels, labels], dim=0)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast("cuda", enabled=(self.cfg.device.type == "cuda")):
                projections = self.model(simclr_images)
                loss = self.loss_fn(projections, simclr_labels)

            if not torch.isfinite(loss):
                self.logger.warning("Skipping non-finite loss at batch %d: %s", batch_idx, loss.item())
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += float(loss.detach().cpu())
            valid_batches += 1

            if batch_idx % 20 == 0:
                self.logger.info(
                    "Epoch %d | Batch %d/%d | Loss %.6f",
                    epoch,
                    batch_idx,
                    len(self.train_loader),
                    float(loss.detach().cpu()),
                )

        avg_loss = epoch_loss / max(valid_batches, 1)
        elapsed = time.perf_counter() - start_time
        return avg_loss, elapsed
