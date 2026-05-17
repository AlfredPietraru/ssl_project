import torch.nn as nn
from config import CFG
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from loss_functions import (
    SupervisedContrastiveLoss,
    BatchHardTripletLoss
)
from main_utils import (
    HarryPlotter,
    set_seed
)
import logging
import time
import warnings
from pathlib import Path
from model import ContrastiveEmbeddingModel
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

class EmbeddingModelTrainer:
    def __init__(self, cfg : CFG, train_loader : DataLoader, val_loader : DataLoader,
                 animal_name: str, model : ContrastiveEmbeddingModel) -> None:
        set_seed()
        self.logger = logging.getLogger("embedding_model_trainer")
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.animal_name = animal_name
        self.animal_slug = self.animal_name.strip().lower().replace(" ", "_")
        self.train_losses: list[float] = []
        self.validation_losses: list[float] = []
        self.validation_recall_at_1: list[float] = []
        self.validation_recall_at_5: list[float] = []
        self.model = model
        
        self.scaler = torch.GradScaler("cuda", enabled=(self.cfg.device.type == "cuda"))
        # self.loss_fn = SupervisedContrastiveLoss(
        #     temperature=cfg.temperature
        # ).to(cfg.device)
        self.loss_fn = BatchHardTripletLoss().to(device=cfg.device)


        self.optimizer = torch.optim.AdamW(
            model.parameters(),  # type: ignore
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        self.plotter = HarryPlotter(save_name="simclr_loss.png")

    def train(self) -> ContrastiveEmbeddingModel:
        best_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        self.model.train()

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss, train_elapsed = self._train_one_epoch(epoch)
            validation_loss, validation_elapsed, recall_at_1, recall_at_5 = self._validate_one_epoch(epoch)
            self.train_losses.append(train_loss)
            self.validation_losses.append(validation_loss)
            self.validation_recall_at_1.append(recall_at_1)
            self.validation_recall_at_5.append(recall_at_5)
            self.logger.info(
                (
                    "Epoch %d/%d finished | train_loss=%.6f | val_loss=%.6f | "
                    "recall@1=%.4f | recall@5=%.4f | train_time=%.1fs | val_time=%.1fs"
                ),
                epoch,
                self.cfg.epochs,
                train_loss,
                validation_loss,
                recall_at_1,
                recall_at_5,
                train_elapsed,
                validation_elapsed,
            )
            self.plotter.update(train_loss, validation_loss)

            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                self.model.save_checkpoints(
                    full_model_path_name=f"contrastive_model_{self.animal_slug}.pt")
                self.logger.info("Saved new best checkpoint with loss %.6f", best_loss)
            else:
                epochs_without_improvement += 1
                self.logger.info(
                    "No validation improvement for %d epoch(s) | best_val_loss=%.6f at epoch %d",
                    epochs_without_improvement,
                    best_loss,
                    best_epoch,
                )
                if epochs_without_improvement >= self.cfg.early_stopping_patience:
                    self.logger.info(
                        "Early stopping triggered at epoch %d | best_val_loss=%.6f at epoch %d",
                        epoch,
                        best_loss,
                        best_epoch,
                    )
                    break
        self.model = self.model.eval()
        return self.model

    def _train_one_epoch(self, epoch : int) -> tuple[float, float]:
        epoch_loss = 0.0
        valid_batches = 0
        start_time = time.perf_counter()
        diagnostics_totals: dict[str, float] = {}
        last_batch_diagnostics: dict[str, float] = {}

        for batch_idx, (images, labels) in enumerate(self.train_loader, start=1):
            images = images.to(self.cfg.device, non_blocking=True)
            labels = labels.to(self.cfg.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast("cuda", enabled=(self.cfg.device.type == "cuda")):
                projections = self.model(images)
                loss = self.loss_fn(projections, labels)

            if not torch.isfinite(loss):
                self.logger.warning("Skipping non-finite loss at batch %d: %s", batch_idx, loss.item())
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += float(loss.detach().cpu())
            valid_batches += 1
            batch_diagnostics = self.loss_fn.diagnostics(projections.detach(), labels.detach())
            last_batch_diagnostics = batch_diagnostics
            for key, value in batch_diagnostics.items():
                diagnostics_totals[key] = diagnostics_totals.get(key, 0.0) + value

            if batch_idx % 20 == 0:
                diagnostics_text = " | ".join(
                    f"{key}={value:.3f}" for key, value in batch_diagnostics.items()
                )
                self.logger.info(
                    "Epoch %d | Batch %d/%d | Loss %.6f | %s",
                    epoch,
                    batch_idx,
                    len(self.train_loader),
                    float(loss.detach().cpu()),
                    diagnostics_text,
                )

        avg_loss = epoch_loss / max(valid_batches, 1)
        elapsed = time.perf_counter() - start_time
        if valid_batches > 0 and diagnostics_totals:
            averaged_diagnostics = {
                key: value / valid_batches for key, value in diagnostics_totals.items()
            }
            diagnostics_text = " | ".join(
                f"{key}={value:.3f}" for key, value in averaged_diagnostics.items()
            )
            self.logger.info(
                "Epoch %d | Train loss diagnostics | %s",
                epoch,
                diagnostics_text,
            )
        return avg_loss, elapsed

    def _compute_recall_at_k(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        k: int,
    ) -> float:
        valid_mask = labels >= 0
        embeddings = embeddings[valid_mask]
        labels = labels[valid_mask]
        if embeddings.shape[0] <= 1:
            return 0.0

        positive_counts = labels[:, None].eq(labels[None, :]).sum(dim=1) - 1
        eligible_mask = positive_counts > 0
        if not torch.any(eligible_mask):
            return 0.0

        normalized_embeddings = F.normalize(embeddings, dim=1)
        similarities = normalized_embeddings @ normalized_embeddings.T
        similarities.fill_diagonal_(float("-inf"))

        effective_k = min(k, similarities.shape[1] - 1)
        if effective_k <= 0:
            return 0.0

        topk_indices = similarities.topk(k=effective_k, dim=1).indices
        retrieved_labels = labels[topk_indices]
        hits = retrieved_labels.eq(labels[:, None]).any(dim=1)
        return float(hits[eligible_mask].float().mean().item())

    def get_embeddings(
        self, trained_model : ContrastiveEmbeddingModel, loader : DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trained_model.eval()
        collected_embeddings: list[torch.Tensor] = []
        collected_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.cfg.device, non_blocking=True)
                labels = labels.to(self.cfg.device, non_blocking=True)

                with torch.autocast("cuda", enabled=(self.cfg.device.type == "cuda")):
                    projections = trained_model(images)

                collected_embeddings.append(projections.detach().cpu())
                collected_labels.append(labels.detach().cpu())

        trained_model.train()
        if not collected_embeddings:
            return torch.empty((0,)), torch.empty((0,), dtype=torch.long)

        return torch.cat(collected_embeddings, dim=0), torch.cat(collected_labels, dim=0)

    def _validate_one_epoch(self, epoch : int) -> tuple[float, float, float, float]:
        self.model.eval()
        epoch_loss = 0.0
        valid_batches = 0
        start_time = time.perf_counter()
        collected_embeddings: list[torch.Tensor] = []
        collected_labels: list[torch.Tensor] = []
        diagnostics_totals: dict[str, float] = {}

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.val_loader, start=1):
                images = images.to(self.cfg.device, non_blocking=True)
                labels = labels.to(self.cfg.device, non_blocking=True)

                with torch.autocast("cuda", enabled=(self.cfg.device.type == "cuda")):
                    projections = self.model(images)
                    loss = self.loss_fn(projections, labels)

                if not torch.isfinite(loss):
                    self.logger.warning(
                        "Skipping non-finite validation loss at batch %d: %s",
                        batch_idx,
                        loss.item(),
                    )
                    continue

                epoch_loss += float(loss.detach().cpu())
                valid_batches += 1
                collected_embeddings.append(projections.detach().cpu())
                collected_labels.append(labels.detach().cpu())
                batch_diagnostics = self.loss_fn.diagnostics(projections.detach(), labels.detach())
                for key, value in batch_diagnostics.items():
                    diagnostics_totals[key] = diagnostics_totals.get(key, 0.0) + value

                if batch_idx % 20 == 0:
                    diagnostics_text = " | ".join(
                        f"{key}={value:.3f}" for key, value in batch_diagnostics.items()
                    )
                    self.logger.info(
                        "Epoch %d | Validation Batch %d/%d | Loss %.6f | %s",
                        epoch,
                        batch_idx,
                        len(self.val_loader),
                        float(loss.detach().cpu()),
                        diagnostics_text,
                    )

        elapsed = time.perf_counter() - start_time
        self.model.train()
        avg_loss = epoch_loss / max(valid_batches, 1)
        if valid_batches > 0 and diagnostics_totals:
            averaged_diagnostics = {
                key: value / valid_batches for key, value in diagnostics_totals.items()
            }
            diagnostics_text = " | ".join(
                f"{key}={value:.3f}" for key, value in averaged_diagnostics.items()
            )
            self.logger.info(
                "Epoch %d | Validation loss diagnostics | %s",
                epoch,
                diagnostics_text,
            )
        if collected_embeddings:
            all_embeddings = torch.cat(collected_embeddings, dim=0)
            all_labels = torch.cat(collected_labels, dim=0)
            recall_at_1 = self._compute_recall_at_k(all_embeddings, all_labels, k=1)
            recall_at_5 = self._compute_recall_at_k(all_embeddings, all_labels, k=5)
        else:
            recall_at_1 = 0.0
            recall_at_5 = 0.0
        return avg_loss, elapsed, recall_at_1, recall_at_5
