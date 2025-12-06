"""
Training script for reward router model.
"""

import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from ..config import Config
from ..models.reward_router import RewardRouterModel, create_model
from .dataset import create_dataloaders, load_dataset

logger = logging.getLogger(__name__)


class RewardRouterTrainer:
    """
    Trainer for reward router model.
    """

    def __init__(
        self,
        model: RewardRouterModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
        device: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Reward router model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            config: Configuration
            device: Device to use (auto-detect if None)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.train_config = config.training

        # Device
        if device is None:
            device = self.train_config.device
            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"

        self.device = torch.device(device)
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Loss function
        self.criterion = nn.MSELoss()

        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_mse = float("inf")
        self.best_model_path = None
        self.patience_counter = 0

        # History
        self.history = {
            "train_loss": [],
            "train_corr": [],
            "val_loss": [],
            "val_corr": [],
            "lr": [],
        }

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.train_config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.train_config.learning_rate,
        )

        logger.info(f"Created optimizer: AdamW(lr={self.train_config.learning_rate}, "
                   f"weight_decay={self.train_config.weight_decay})")

        return optimizer

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        total_steps = len(self.train_loader) * self.train_config.num_epochs
        warmup_steps = int(total_steps * self.train_config.warmup_ratio)

        if self.train_config.scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
            )
            logger.info(f"Created cosine scheduler with {warmup_steps} warmup steps")
        elif self.train_config.scheduler == "linear":
            scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            logger.info(f"Created linear warmup scheduler with {warmup_steps} steps")
        elif self.train_config.scheduler == "constant":
            scheduler = None
            logger.info("Using constant learning rate")
        else:
            raise ValueError(f"Unknown scheduler: {self.train_config.scheduler}")

        return scheduler

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        total_loss = 0.0
        all_preds = []
        all_targets = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            model_id = batch["model_id"].to(self.device)
            mode_id = batch["mode_id"].to(self.device)
            reward = batch["reward"].to(self.device)

            # Forward
            pred_reward = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                model_id=model_id,
                mode_id=mode_id,
            )

            # Loss
            loss = self.criterion(pred_reward, reward)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.train_config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.gradient_clip_norm,
                )

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Track
            total_loss += loss.item()
            all_preds.extend(pred_reward.detach().cpu().numpy())
            all_targets.extend(reward.detach().cpu().numpy())

            self.global_step += 1

            # Update progress bar
            if batch_idx % self.train_config.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{current_lr:.2e}",
                })

        # Compute metrics
        avg_loss = total_loss / len(self.train_loader)

        # Compute correlation
        try:
            corr, _ = pearsonr(all_preds, all_targets)
        except:
            corr = 0.0

        metrics = {
            "loss": avg_loss,
            "corr": corr,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")

        for batch in pbar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            model_id = batch["model_id"].to(self.device)
            mode_id = batch["mode_id"].to(self.device)
            reward = batch["reward"].to(self.device)

            # Forward
            pred_reward = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                model_id=model_id,
                mode_id=mode_id,
            )

            # Loss
            loss = self.criterion(pred_reward, reward)

            # Track
            total_loss += loss.item()
            all_preds.extend(pred_reward.cpu().numpy())
            all_targets.extend(reward.cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)

        # Compute correlation
        try:
            corr, _ = pearsonr(all_preds, all_targets)
        except:
            corr = 0.0

        metrics = {
            "loss": avg_loss,
            "corr": corr,
        }

        return metrics

    def save_checkpoint(self, path: str, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            is_best: Whether this is the best model so far
        """
        self.model.save(path)

        if is_best:
            self.best_model_path = path
            logger.info(f"Saved best model to: {path}")

    def train(self) -> Dict[str, list]:
        """
        Full training loop.

        Returns:
            Training history
        """
        logger.info("=" * 80)
        logger.info("Starting Training")
        logger.info("=" * 80)
        logger.info(f"Epochs: {self.train_config.num_epochs}")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")
        logger.info(f"Device: {self.device}")
        logger.info("=" * 80)

        for epoch in range(self.train_config.num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            if (epoch + 1) % self.train_config.eval_interval == 0:
                val_metrics = self.validate()
            else:
                val_metrics = {"loss": float("nan"), "corr": float("nan")}

            # Log
            logger.info(
                f"Epoch {epoch + 1}/{self.train_config.num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Corr: {train_metrics['corr']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Corr: {val_metrics['corr']:.4f}, "
                f"LR: {train_metrics['lr']:.2e}"
            )

            # Track history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_corr"].append(train_metrics["corr"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_corr"].append(val_metrics["corr"])
            self.history["lr"].append(train_metrics["lr"])

            # Save checkpoint
            if not np.isnan(val_metrics["loss"]):
                checkpoint_path = self.config.paths.get_checkpoint_path(
                    f"router_epoch_{epoch + 1}.pt"
                )

                is_best = val_metrics["loss"] < self.best_val_mse
                if is_best:
                    self.best_val_mse = val_metrics["loss"]
                    self.patience_counter = 0

                    # Save best model
                    if self.train_config.save_best_only:
                        best_path = self.config.paths.get_checkpoint_path()
                        self.save_checkpoint(best_path, is_best=True)
                else:
                    self.patience_counter += 1

                # Save regular checkpoint
                if not self.train_config.save_best_only:
                    self.save_checkpoint(checkpoint_path, is_best=is_best)

                # Early stopping
                if (
                    self.train_config.early_stopping_patience is not None
                    and self.patience_counter >= self.train_config.early_stopping_patience
                ):
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        logger.info("=" * 80)
        logger.info("Training Complete")
        logger.info(f"Best Val MSE: {self.best_val_mse:.4f}")
        if self.best_model_path:
            logger.info(f"Best model saved to: {self.best_model_path}")
        logger.info("=" * 80)

        return self.history


def train_router(config: Config) -> Tuple[RewardRouterModel, Dict[str, list]]:
    """
    Main training function.

    Args:
        config: Configuration object

    Returns:
        Tuple of (trained_model, history)
    """
    # Set random seeds
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)

    # Load dataset
    dataset_path = config.paths.get_data_path()
    df = load_dataset(dataset_path)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.model.text_encoder_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.text_encoder_name)

    # Create dataloaders
    train_loader, val_loader, test_loader, train_df, val_df, test_df = create_dataloaders(
        df, tokenizer, config.training, config.model
    )

    # Get num_models and num_modes
    num_models = df["model_id"].max() + 1
    num_modes = df["mode_id"].max() + 1

    logger.info(f"Dataset info: {num_models} models, {num_modes} modes")

    # Create model
    model = create_model(config.model, num_models, num_modes)

    # Create trainer
    trainer = RewardRouterTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    # Train
    history = trainer.train()

    return model, history


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Train with default config
    from ..config import Config

    config = Config.default()
    model, history = train_router(config)
