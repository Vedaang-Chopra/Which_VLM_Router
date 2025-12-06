"""
Training utilities including W&B logging and evaluation.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# Try to import wandb, but make it optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Install with: pip install wandb")


def init_wandb(
    project_name: str = "vlm-router-training",
    run_name: Optional[str] = None,
    config: Optional[Dict] = None,
    tags: Optional[List[str]] = None,
) -> Optional[any]:
    """
    Initialize Weights & Biases logging.

    Args:
        project_name: W&B project name
        run_name: Optional run name (auto-generated if None)
        config: Configuration dictionary to log
        tags: List of tags for this run

    Returns:
        W&B run object or None if wandb not available
    """
    if not WANDB_AVAILABLE:
        logger.warning("W&B not available, skipping initialization")
        return None

    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"

    try:
        run = wandb.init(
            project=project_name,
            name=run_name,
            config=config or {},
            tags=tags or [],
        )
        logger.info(f"✓ Initialized W&B: {project_name}/{run_name}")
        return run
    except Exception as e:
        logger.error(f"Failed to initialize W&B: {e}")
        return None


def train_epoch_reward_router(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip: float = 1.0,
    log_wandb: bool = True,
    epoch: int = 0,
) -> Dict[str, float]:
    """
    Train reward router for one epoch.

    Args:
        model: Router model
        train_loader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        gradient_clip: Gradient clipping value
        log_wandb: Whether to log to W&B
        epoch: Current epoch number

    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    with tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]") as pbar:
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            model_id = batch['model_id'].to(device)
            mode_id = batch['mode_id'].to(device)
            reward = batch['reward'].to(device)

            # Forward
            optimizer.zero_grad()
            pred_reward = model(input_ids, attention_mask, model_id, mode_id)

            # Loss
            loss = criterion(pred_reward, reward)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            # Log to W&B (every 10 batches)
            if log_wandb and WANDB_AVAILABLE and batch_idx % 10 == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/epoch': epoch,
                    'train/batch': epoch * num_batches + batch_idx,
                })

    avg_loss = total_loss / num_batches

    return {'train_loss': avg_loss}


def evaluate_reward_router(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str = "val",
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate reward router on a dataset.

    Args:
        model: Router model
        data_loader: Dataloader for evaluation
        criterion: Loss function
        device: Device to evaluate on
        split_name: Name of split ("val" or "test")

    Returns:
        Tuple of (metrics_dict, predictions, targets)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        with tqdm(data_loader, desc=f"Evaluating {split_name}") as pbar:
            for batch in pbar:
                # Move to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                model_id = batch['model_id'].to(device)
                mode_id = batch['mode_id'].to(device)
                reward = batch['reward'].to(device)

                # Forward
                pred_reward = model(input_ids, attention_mask, model_id, mode_id)

                # Loss
                loss = criterion(pred_reward, reward)
                total_loss += loss.item()

                # Collect predictions
                all_preds.extend(pred_reward.cpu().numpy())
                all_targets.extend(reward.cpu().numpy())

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / len(data_loader)

    # Convert to arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute metrics
    pearson_corr, _ = pearsonr(all_preds, all_targets)
    mse = np.mean((all_preds - all_targets) ** 2)
    mae = np.mean(np.abs(all_preds - all_targets))

    metrics = {
        f'{split_name}_loss': avg_loss,
        f'{split_name}_pearson': pearson_corr,
        f'{split_name}_mse': mse,
        f'{split_name}_mae': mae,
    }

    return metrics, all_preds, all_targets


def compute_routing_accuracy(
    df: pd.DataFrame,
    model_to_id: Dict[str, int],
    id_to_model: Dict[int, str],
    mode_to_id: Dict[str, int],
    id_to_mode: Dict[int, str],
    pred_column: str = 'pred_reward',
    target_column: str = 'reward',
) -> Dict[str, any]:
    """
    Compute routing accuracy: how often does router pick same model as oracle?

    Args:
        df: Dataframe with columns [sample_id, model_id, mode_id, reward, pred_reward]
        model_to_id: Model name to ID mapping
        id_to_model: ID to model name mapping
        mode_to_id: Mode name to ID mapping
        id_to_mode: ID to mode name mapping
        pred_column: Column with predicted rewards
        target_column: Column with true rewards

    Returns:
        Dictionary with routing accuracy metrics
    """
    results = []

    # Group by (sample_id, mode_id)
    grouped = df.groupby(['sample_id', 'mode_id'])

    for (sample_id, mode_id), group in grouped:
        if len(group) < 2:
            continue  # Need at least 2 models to compare

        # Oracle: model with max true reward
        oracle_idx = group[target_column].idxmax()
        oracle_model_id = group.loc[oracle_idx, 'model_id']
        oracle_reward = group.loc[oracle_idx, target_column]

        # Router: model with max predicted reward
        router_idx = group[pred_column].idxmax()
        router_model_id = group.loc[router_idx, 'model_id']
        router_pred_reward = group.loc[router_idx, pred_column]
        router_true_reward = group.loc[router_idx, target_column]

        match = oracle_model_id == router_model_id

        results.append({
            'sample_id': sample_id,
            'mode_id': mode_id,
            'mode_name': id_to_mode[mode_id],
            'oracle_model_id': oracle_model_id,
            'oracle_model': id_to_model[oracle_model_id],
            'oracle_reward': oracle_reward,
            'router_model_id': router_model_id,
            'router_model': id_to_model[router_model_id],
            'router_pred_reward': router_pred_reward,
            'router_true_reward': router_true_reward,
            'match': match,
            'reward_gap': oracle_reward - router_true_reward,
        })

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        logger.warning("No results to compute routing accuracy")
        return {}

    # Overall metrics
    overall_acc = results_df['match'].mean()
    avg_reward_gap = results_df['reward_gap'].mean()
    median_reward_gap = results_df['reward_gap'].median()

    # Per-mode metrics
    mode_accuracy = results_df.groupby('mode_name')['match'].mean().to_dict()
    mode_reward_gap = results_df.groupby('mode_name')['reward_gap'].mean().to_dict()

    # Reward recovery (what % of oracle reward does router get)
    oracle_reward_avg = results_df['oracle_reward'].mean()
    router_reward_avg = results_df['router_true_reward'].mean()
    reward_recovery = router_reward_avg / oracle_reward_avg if oracle_reward_avg > 0 else 0.0

    metrics = {
        'routing_accuracy': overall_acc,
        'avg_reward_gap': avg_reward_gap,
        'median_reward_gap': median_reward_gap,
        'reward_recovery': reward_recovery,
        'oracle_reward_avg': oracle_reward_avg,
        'router_reward_avg': router_reward_avg,
        'num_decisions': len(results_df),
    }

    # Add per-mode metrics
    for mode_name in mode_accuracy:
        metrics[f'routing_accuracy_{mode_name}'] = mode_accuracy[mode_name]
        metrics[f'reward_gap_{mode_name}'] = mode_reward_gap[mode_name]

    return metrics


def log_final_results(
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    val_routing_metrics: Dict[str, float],
    test_routing_metrics: Dict[str, float],
    log_wandb: bool = True,
) -> None:
    """
    Log final results to console and W&B.

    Args:
        train_metrics: Training metrics
        val_metrics: Validation metrics
        test_metrics: Test metrics
        val_routing_metrics: Validation routing metrics
        test_routing_metrics: Test routing metrics
        log_wandb: Whether to log to W&B
    """
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print("\nPrediction Performance:")
    print(f"  Train Loss:        {train_metrics.get('train_loss', 0):.4f}")
    print(f"  Val Loss:          {val_metrics.get('val_loss', 0):.4f}")
    print(f"  Val Pearson:       {val_metrics.get('val_pearson', 0):.4f}")
    print(f"  Test Loss:         {test_metrics.get('test_loss', 0):.4f}")
    print(f"  Test Pearson:      {test_metrics.get('test_pearson', 0):.4f}")

    print("\nRouting Performance (Validation):")
    print(f"  Routing Accuracy:  {100*val_routing_metrics.get('routing_accuracy', 0):.2f}%")
    print(f"  Reward Recovery:   {100*val_routing_metrics.get('reward_recovery', 0):.2f}%")
    print(f"  Avg Reward Gap:    {val_routing_metrics.get('avg_reward_gap', 0):.4f}")

    print("\nRouting Performance (Test):")
    print(f"  Routing Accuracy:  {100*test_routing_metrics.get('routing_accuracy', 0):.2f}%")
    print(f"  Reward Recovery:   {100*test_routing_metrics.get('reward_recovery', 0):.2f}%")
    print(f"  Avg Reward Gap:    {test_routing_metrics.get('avg_reward_gap', 0):.4f}")

    print("\nPer-Mode Routing Accuracy (Test):")
    for key, value in test_routing_metrics.items():
        if key.startswith('routing_accuracy_') and key != 'routing_accuracy':
            mode_name = key.replace('routing_accuracy_', '')
            print(f"  {mode_name:12s}: {100*value:.2f}%")

    print("=" * 80)

    # Log to W&B
    if log_wandb and WANDB_AVAILABLE:
        wandb.log({
            **train_metrics,
            **val_metrics,
            **test_metrics,
            **{f'val_routing/{k}': v for k, v in val_routing_metrics.items()},
            **{f'test_routing/{k}': v for k, v in test_routing_metrics.items()},
        })
        wandb.run.summary.update({
            'final_test_accuracy': test_routing_metrics.get('routing_accuracy', 0),
            'final_test_reward_recovery': test_routing_metrics.get('reward_recovery', 0),
            'final_test_pearson': test_metrics.get('test_pearson', 0),
        })
        logger.info("✓ Logged final results to W&B")
