#!/usr/bin/env python3
"""
VLM Router Training Script with Comprehensive W&B Tracking

This script implements the complete training pipeline for the VLM router,
including extensive experiment tracking, evaluation, and visualization.

Usage:
    python train_router.py --config config.yaml
    python train_router.py --batch-size 32 --epochs 15 --lr 1e-4
"""

from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
import argparse
import json
warnings.filterwarnings('ignore')

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup

# Transformers
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    DistilBertModel,
    DistilBertTokenizer,
)

# Image loading
from PIL import Image

# Tracking and visualization
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Progress tracking
from tqdm.auto import tqdm

# Local utilities
from imports.common_utils import return_model_specs, return_model_pricing


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RouterConfig:
    """Configuration for router training."""

    # Data paths
    data_root: Path = Path.cwd().parent.parent.parent / 'dataset' / 'final_dataset' / 'router_final'
    image_root: Path = Path.cwd().parent.parent.parent / 'dataset' / 'which_vlm_data' / 'images'

    # Model architecture
    num_models: int = 5
    vision_encoder_name: str = "openai/clip-vit-base-patch32"
    text_encoder_name: str = "distilbert-base-uncased"
    hidden_dim: int = 384
    num_fusion_layers: int = 4
    num_attention_heads: int = 8
    dropout: float = 0.1
    freeze_vision: bool = True
    freeze_text: bool = False

    # Training hyperparameters
    batch_size: int = 32
    num_epochs: int = 15
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_text_length: int = 256
    gradient_clip: float = 1.0

    # Loss configuration
    use_soft_labels: bool = True
    soft_label_weight: float = 0.3
    label_smoothing: float = 0.0

    # Training configuration
    num_workers: int = 4
    accumulation_steps: int = 1
    eval_every_n_steps: int = 500
    save_total_limit: int = 3

    # W&B tracking
    wandb_project: str = "vlm-router"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    use_wandb: bool = True

    # Output
    checkpoint_dir: Path = Path('./checkpoints')
    log_dir: Path = Path('./logs')

    def __post_init__(self):
        self.data_root = Path(self.data_root)
        self.image_root = Path(self.image_root)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.log_dir = Path(self.log_dir)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Dataset
# ============================================================================

class VLMRouterDataset(Dataset):
    """Dataset for VLM router training."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_processor: CLIPImageProcessor,
        tokenizer: DistilBertTokenizer,
        image_root: Path,
        max_text_length: int = 256,
        use_soft_labels: bool = True,
        model_names: List[str] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_root = Path(image_root)
        self.max_text_length = max_text_length
        self.use_soft_labels = use_soft_labels
        self.model_names = model_names

        # Statistics
        self.missing_images = 0
        self.fallback_count = 0

    def __len__(self):
        return len(self.df)

    def _build_router_text(self, row: pd.Series) -> str:
        """Construct text input combining prompt and metadata."""
        parts = []

        if pd.notna(row.get('router_task')):
            parts.append(f"Task: {row['router_task']}")
        if pd.notna(row.get('source_dataset')):
            parts.append(f"Dataset: {row['source_dataset']}")
        if pd.notna(row.get('txt_question_type')):
            parts.append(f"Type: {row['txt_question_type']}")
        if pd.notna(row.get('txt_has_mc_options')):
            mc_str = "yes" if row['txt_has_mc_options'] else "no"
            parts.append(f"Multiple Choice: {mc_str}")
        if pd.notna(row.get('img_width')) and pd.notna(row.get('img_height')):
            parts.append(f"Image: {int(row['img_width'])}x{int(row['img_height'])}")
        if pd.notna(row.get('img_aspect_ratio')):
            parts.append(f"AR: {row['img_aspect_ratio']:.2f}")
        if pd.notna(row.get('txt_prompt_length_words')):
            parts.append(f"Words: {int(row['txt_prompt_length_words'])}")

        metadata = ". ".join(parts)
        prompt = str(row.get('prompt_raw', ''))

        return f"[{metadata}] {prompt}"

    def _load_image(self, row: pd.Series) -> torch.Tensor:
        """Load and process image."""
        image_path = row.get('image_path')

        if pd.notna(image_path) and image_path:
            try:
                if not Path(image_path).is_absolute():
                    image_path = self.image_root / image_path

                image = Image.open(image_path).convert('RGB')
                pixel_values = self.image_processor(
                    images=image,
                    return_tensors='pt'
                ).pixel_values.squeeze(0)
                return pixel_values
            except Exception:
                self.missing_images += 1

        self.fallback_count += 1
        return torch.zeros((3, 224, 224))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        pixel_values = self._load_image(row)
        router_text = self._build_router_text(row)

        encoding = self.tokenizer(
            router_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt'
        )

        hard_label = int(row['router_best_model_id'])

        sample = {
            'pixel_values': pixel_values,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': hard_label,
            'sample_id': row.get('sample_id', idx),
        }

        if self.use_soft_labels and self.model_names:
            soft_cols = [f'router_soft_p_{name}' for name in self.model_names]
            if all(c in row for c in soft_cols):
                soft_labels = torch.tensor(
                    [row[c] for c in soft_cols],
                    dtype=torch.float32
                )
                sample['soft_labels'] = soft_labels

        sample['router_task'] = row.get('router_task', 'unknown')
        sample['true_perf'] = row.get('router_chosen_perf', 0.0)
        sample['true_cost'] = row.get('router_chosen_cost', 0.0)

        return sample


# ============================================================================
# Model
# ============================================================================

class VLMRouter(nn.Module):
    """Multimodal transformer router for VLM selection."""

    def __init__(self, config: RouterConfig):
        super().__init__()

        self.config = config
        self.num_models = config.num_models
        self.hidden_dim = config.hidden_dim

        # Vision encoder (frozen)
        self.vision_encoder = CLIPVisionModel.from_pretrained(config.vision_encoder_name)
        vision_dim = self.vision_encoder.config.hidden_size

        if config.freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        # Text encoder
        self.text_encoder = DistilBertModel.from_pretrained(config.text_encoder_name)
        text_dim = self.text_encoder.config.hidden_size

        if config.freeze_text:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # Projections
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Fusion transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_fusion_layers,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_models),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in [self.vision_projection, self.text_projection, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = pixel_values.size(0)

        # Vision encoding
        if self.config.freeze_vision:
            with torch.no_grad():
                vision_outputs = self.vision_encoder(pixel_values)
                vision_features = vision_outputs.pooler_output
        else:
            vision_outputs = self.vision_encoder(pixel_values)
            vision_features = vision_outputs.pooler_output

        vision_features = self.vision_projection(vision_features)
        vision_tokens = vision_features.unsqueeze(1)

        # Text encoding
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_features = text_outputs.last_hidden_state
        text_features = self.text_projection(text_features)

        # Concatenate
        fused_tokens = torch.cat([vision_tokens, text_features], dim=1)

        # Attention mask
        vision_mask = torch.ones(
            batch_size, 1,
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        fused_mask = torch.cat([vision_mask, attention_mask], dim=1)

        # Fusion
        fused_output = self.fusion_transformer(
            fused_tokens,
            src_key_padding_mask=(fused_mask == 0),
        )

        # Classification
        pooled_output = fused_output[:, 0, :]
        logits = self.classifier(pooled_output)

        return logits

    def get_num_params(self) -> Dict[str, int]:
        """Get parameter counts."""
        vision_params = sum(p.numel() for p in self.vision_encoder.parameters())
        text_params = sum(p.numel() for p in self.text_encoder.parameters())
        fusion_params = sum(p.numel() for p in self.fusion_transformer.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        projection_params = (
            sum(p.numel() for p in self.vision_projection.parameters()) +
            sum(p.numel() for p in self.text_projection.parameters())
        )

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())

        return {
            'vision': vision_params,
            'text': text_params,
            'fusion': fusion_params,
            'classifier': classifier_params,
            'projection': projection_params,
            'trainable': trainable_params,
            'total': total_params,
        }


# ============================================================================
# Training Utilities
# ============================================================================

def compute_loss(
    logits: torch.Tensor,
    hard_labels: torch.Tensor,
    soft_labels: Optional[torch.Tensor] = None,
    config: RouterConfig = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute training loss."""
    loss_dict = {}

    ce_loss = F.cross_entropy(
        logits,
        hard_labels,
        label_smoothing=config.label_smoothing if config else 0.0
    )
    loss_dict['ce_loss'] = ce_loss.item()

    total_loss = ce_loss

    if soft_labels is not None and config and config.use_soft_labels:
        log_probs = F.log_softmax(logits, dim=-1)
        kl_loss = F.kl_div(log_probs, soft_labels, reduction='batchmean')
        loss_dict['kl_loss'] = kl_loss.item()

        total_loss = (
            (1 - config.soft_label_weight) * ce_loss +
            config.soft_label_weight * kl_loss
        )

    loss_dict['total_loss'] = total_loss.item()

    return total_loss, loss_dict


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Compute evaluation metrics."""
    predictions = logits.argmax(dim=-1)

    accuracy = (predictions == labels).float().mean().item()

    top3_preds = logits.topk(min(3, logits.size(1)), dim=1).indices
    top3_acc = (top3_preds == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    probs = F.softmax(logits, dim=-1)
    confidence = probs.max(dim=-1).values.mean().item()
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()

    return {
        'accuracy': accuracy,
        'top3_accuracy': top3_acc,
        'confidence': confidence,
        'entropy': entropy,
    }


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: RouterConfig,
    epoch: int,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_ce_loss = 0
    total_kl_loss = 0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs} [Train]')

    for step, batch in enumerate(pbar):
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        soft_labels = batch.get('soft_labels')
        if soft_labels is not None:
            soft_labels = soft_labels.to(device)

        logits = model(pixel_values, input_ids, attention_mask)
        loss, loss_dict = compute_loss(logits, labels, soft_labels, config)

        loss = loss / config.accumulation_steps
        loss.backward()

        if (step + 1) % config.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        predictions = logits.argmax(dim=-1)
        correct = (predictions == labels).sum().item()

        batch_size = labels.size(0)
        total_loss += loss_dict['total_loss'] * batch_size
        total_ce_loss += loss_dict['ce_loss'] * batch_size
        if 'kl_loss' in loss_dict:
            total_kl_loss += loss_dict['kl_loss'] * batch_size
        total_correct += correct
        total_samples += batch_size

        current_acc = total_correct / total_samples
        pbar.set_postfix({
            'loss': f"{loss_dict['total_loss']:.4f}",
            'acc': f'{current_acc:.4f}',
            'lr': f"{scheduler.get_last_lr()[0]:.2e}",
        })

        if config.use_wandb and step % 50 == 0:
            global_step = epoch * len(train_loader) + step
            wandb.log({
                'train/loss': loss_dict['total_loss'],
                'train/ce_loss': loss_dict['ce_loss'],
                'train/kl_loss': loss_dict.get('kl_loss', 0.0),
                'train/accuracy': current_acc,
                'train/learning_rate': scheduler.get_last_lr()[0],
                'train/epoch': epoch,
                'train/step': global_step,
            })

    return {
        'loss': total_loss / total_samples,
        'ce_loss': total_ce_loss / total_samples,
        'kl_loss': total_kl_loss / total_samples if total_kl_loss > 0 else 0.0,
        'accuracy': total_correct / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    config: RouterConfig,
    device: torch.device,
    split: str = 'val',
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()

    total_loss = 0
    total_correct = 0
    total_top3 = 0
    total_samples = 0

    all_predictions = []
    all_labels = []
    all_probs = []
    all_tasks = []

    pbar = tqdm(dataloader, desc=f'{split.capitalize()} evaluation')

    for batch in pbar:
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        soft_labels = batch.get('soft_labels')
        if soft_labels is not None:
            soft_labels = soft_labels.to(device)

        logits = model(pixel_values, input_ids, attention_mask)
        loss, _ = compute_loss(logits, labels, soft_labels, config)

        predictions = logits.argmax(dim=-1)
        probs = F.softmax(logits, dim=-1)

        correct = (predictions == labels).sum().item()

        top3_preds = logits.topk(min(3, logits.size(1)), dim=1).indices
        top3_correct = (top3_preds == labels.unsqueeze(1)).any(dim=1).sum().item()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += correct
        total_top3 += top3_correct
        total_samples += batch_size

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_tasks.extend(batch['router_task'])

        pbar.set_postfix({
            'loss': f'{total_loss / total_samples:.4f}',
            'acc': f'{total_correct / total_samples:.4f}',
        })

    metrics = {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'top3_accuracy': total_top3 / total_samples,
    }

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Per-class accuracy
    MODEL_SPECS = return_model_specs()
    model_names = [m['name'] for m in MODEL_SPECS]
    for i, model_name in enumerate(model_names):
        mask = all_labels == i
        if mask.sum() > 0:
            acc = (all_predictions[mask] == i).mean()
            metrics[f'accuracy_{model_name}'] = acc

    confidences = all_probs.max(axis=1)
    metrics['mean_confidence'] = confidences.mean()
    metrics['median_confidence'] = np.median(confidences)

    entropy = -(all_probs * np.log(all_probs + 1e-10)).sum(axis=1)
    metrics['mean_entropy'] = entropy.mean()

    metrics['_predictions'] = all_predictions
    metrics['_labels'] = all_labels
    metrics['_probs'] = all_probs
    metrics['_tasks'] = all_tasks

    return metrics


# ============================================================================
# Main Training Function
# ============================================================================

def main(config: RouterConfig):
    """Main training function."""

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load model specs
    MODEL_SPECS = return_model_specs()
    MODEL_PRICING = return_model_pricing()
    model_names = [m['name'] for m in MODEL_SPECS]
    ID_TO_NAME = {m['id']: m['name'] for m in MODEL_SPECS}

    # Load datasets
    print('\\nLoading datasets...')
    train_df = pd.read_parquet(config.data_root / 'router_train_final.parquet')
    val_df = pd.read_parquet(config.data_root / 'router_val_final.parquet')
    test_df = pd.read_parquet(config.data_root / 'router_test_final.parquet')

    print(f'Train: {len(train_df):,} samples')
    print(f'Val:   {len(val_df):,} samples')
    print(f'Test:  {len(test_df):,} samples')

    # Initialize processors
    print('\\nInitializing processors...')
    image_processor = CLIPImageProcessor.from_pretrained(config.vision_encoder_name)
    tokenizer = DistilBertTokenizer.from_pretrained(config.text_encoder_name)

    # Create datasets
    print('Creating datasets...')
    train_dataset = VLMRouterDataset(
        train_df, image_processor, tokenizer, config.image_root,
        max_text_length=config.max_text_length,
        use_soft_labels=config.use_soft_labels,
        model_names=model_names,
    )
    val_dataset = VLMRouterDataset(
        val_df, image_processor, tokenizer, config.image_root,
        max_text_length=config.max_text_length,
        use_soft_labels=config.use_soft_labels,
        model_names=model_names,
    )
    test_dataset = VLMRouterDataset(
        test_df, image_processor, tokenizer, config.image_root,
        max_text_length=config.max_text_length,
        use_soft_labels=config.use_soft_labels,
        model_names=model_names,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        multiprocessing_context="spawn",
        pin_memory=True if device.type == 'cuda' else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        multiprocessing_context="spawn",
        pin_memory=True if device.type == 'cuda' else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        multiprocessing_context="spawn",
        pin_memory=True if device.type == 'cuda' else False,
    )

    # Initialize model
    print('\\nInitializing model...')
    model = VLMRouter(config).to(device)
    param_counts = model.get_num_params()

    print('\\nModel parameters:')
    for k, v in param_counts.items():
        print(f'  {k:12s}: {v:,}')

    # Initialize optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps,
    )

    # Initialize W&B
    if config.use_wandb:
        run_name = config.wandb_run_name or f"router_h{config.hidden_dim}_l{config.num_fusion_layers}"
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=run_name,
            config=asdict(config),
            tags=['router', 'multimodal'],
        )
        wandb.watch(model, log='all', log_freq=100)

    # Training loop
    print('\\n' + '='*80)
    print('Starting training...')
    print('='*80)

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(config.num_epochs):
        print(f'\\n{\"=\"*80}')
        print(f'Epoch {epoch+1}/{config.num_epochs}')
        print(f'{\"=\"*80}')

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, config, epoch, device)
        print(f"\\nTrain - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")

        # Validate
        val_metrics = evaluate(model, val_loader, config, device, split='val')
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

        # Log to W&B
        if config.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/epoch_loss': train_metrics['loss'],
                'train/epoch_accuracy': train_metrics['accuracy'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'val/top3_accuracy': val_metrics['top3_accuracy'],
            })

        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            checkpoint_path = config.checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
                'config': config,
            }, checkpoint_path)
            print(f'✓ Saved best model (val_acc={val_metrics[\"accuracy\"]:.4f})')

    # Final test evaluation
    print('\\n' + '='*80)
    print('Final test evaluation...')
    print('='*80)

    # PyTorch 2.6+ defaults weights_only=True; we stored full objects (config/path).
    checkpoint = torch.load(
        config.checkpoint_dir / 'best_model.pt',
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, config, device, split='test')

    print(f"\\nTest Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"Test Top-3 Acc:   {test_metrics['top3_accuracy']:.4f}")

    if config.use_wandb:
        wandb.log({
            'test/accuracy': test_metrics['accuracy'],
            'test/top3_accuracy': test_metrics['top3_accuracy'],
        })
        wandb.finish()

    print('\\nTraining complete!')
    return model, history, test_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VLM Router')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=384)
    parser.add_argument('--no-wandb', action='store_true')

    args = parser.parse_args()

    config = RouterConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        use_wandb=not args.no_wandb,
    )

    main(config)
