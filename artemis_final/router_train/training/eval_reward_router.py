"""
Evaluation script for reward router model.

Compares router performance against oracle and baselines.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from ..config import Config
from ..models.reward_router import RewardRouterModel
from .dataset import create_dataloaders, load_dataset

logger = logging.getLogger(__name__)


class RouterEvaluator:
    """
    Evaluator for reward router performance.
    """

    def __init__(
        self,
        model: RewardRouterModel,
        test_df: pd.DataFrame,
        test_loader: DataLoader,
        config: Config,
        device: Optional[str] = None,
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained router model
            test_df: Test dataframe
            test_loader: Test dataloader
            config: Configuration
            device: Device to use
        """
        self.model = model
        self.test_df = test_df
        self.test_loader = test_loader
        self.config = config

        # Device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Evaluator initialized on device: {self.device}")

    @torch.no_grad()
    def predict_rewards(self) -> pd.DataFrame:
        """
        Predict rewards for all test samples.

        Returns:
            Test dataframe with added 'pred_reward' column
        """
        logger.info("Predicting rewards for test set...")

        predictions = []

        for batch in tqdm(self.test_loader, desc="Predicting"):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            model_id = batch["model_id"].to(self.device)
            mode_id = batch["mode_id"].to(self.device)

            # Predict
            pred_reward = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                model_id=model_id,
                mode_id=mode_id,
            )

            predictions.extend(pred_reward.cpu().numpy())

        # Add predictions to dataframe
        df_with_preds = self.test_df.copy()
        df_with_preds["pred_reward"] = predictions

        logger.info(f"Generated {len(predictions)} predictions")

        return df_with_preds

    def evaluate_oracle(self, mode_id: int) -> Dict[str, float]:
        """
        Evaluate oracle (ground truth best model selection).

        Args:
            mode_id: Mode ID to evaluate

        Returns:
            Dictionary with oracle metrics
        """
        mode_df = self.test_df[self.test_df["mode_id"] == mode_id]

        # For each sample, find model with max reward (oracle choice)
        oracle_selections = []

        for sample_id in mode_df["sample_id"].unique():
            sample_df = mode_df[mode_df["sample_id"] == sample_id]

            # Get oracle choice (max true reward)
            oracle_idx = sample_df["reward"].idxmax()
            oracle_row = sample_df.loc[oracle_idx]

            oracle_selections.append({
                "sample_id": sample_id,
                "model_id": oracle_row["model_id"],
                "model_name": oracle_row["model_name"],
                "reward": oracle_row["reward"],
                "primary_acc": oracle_row.get("primary_acc", np.nan),
                "cost_usd": oracle_row.get("cost_usd", np.nan),
                "latency_ms": oracle_row.get("latency_ms", np.nan),
            })

        oracle_df = pd.DataFrame(oracle_selections)

        # Compute aggregate metrics
        metrics = {
            "avg_reward": oracle_df["reward"].mean(),
            "avg_primary_acc": oracle_df["primary_acc"].mean(),
            "avg_cost_usd": oracle_df["cost_usd"].mean(),
            "avg_latency_ms": oracle_df["latency_ms"].mean(),
        }

        return metrics

    def evaluate_router(self, mode_id: int, df_with_preds: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate router model selection.

        Args:
            mode_id: Mode ID to evaluate
            df_with_preds: Dataframe with predicted rewards

        Returns:
            Dictionary with router metrics
        """
        mode_df = df_with_preds[df_with_preds["mode_id"] == mode_id]

        # For each sample, find model with max predicted reward (router choice)
        router_selections = []
        oracle_selections = []

        for sample_id in mode_df["sample_id"].unique():
            sample_df = mode_df[mode_df["sample_id"] == sample_id]

            # Router choice (max predicted reward)
            router_idx = sample_df["pred_reward"].idxmax()
            router_row = sample_df.loc[router_idx]

            # Oracle choice (max true reward)
            oracle_idx = sample_df["reward"].idxmax()
            oracle_row = sample_df.loc[oracle_idx]

            router_selections.append({
                "sample_id": sample_id,
                "model_id": router_row["model_id"],
                "model_name": router_row["model_name"],
                "reward": router_row["reward"],  # TRUE reward of router's choice
                "pred_reward": router_row["pred_reward"],
                "primary_acc": router_row.get("primary_acc", np.nan),
                "cost_usd": router_row.get("cost_usd", np.nan),
                "latency_ms": router_row.get("latency_ms", np.nan),
            })

            oracle_selections.append(oracle_row["model_id"])

        router_df = pd.DataFrame(router_selections)

        # Compute metrics
        metrics = {
            "avg_reward": router_df["reward"].mean(),
            "avg_primary_acc": router_df["primary_acc"].mean(),
            "avg_cost_usd": router_df["cost_usd"].mean(),
            "avg_latency_ms": router_df["latency_ms"].mean(),
            # Routing accuracy: % of times router matches oracle
            "routing_accuracy": (router_df["model_id"] == oracle_selections).mean(),
        }

        return metrics

    def evaluate_baseline_biggest(self, mode_id: int) -> Dict[str, float]:
        """
        Evaluate 'always pick biggest model' baseline.

        Args:
            mode_id: Mode ID to evaluate

        Returns:
            Dictionary with baseline metrics
        """
        mode_df = self.test_df[self.test_df["mode_id"] == mode_id]

        # Get biggest model from ranking
        model_ranking = self.config.evaluation.model_size_ranking
        available_models = mode_df["model_name"].unique()

        # Find biggest available model
        biggest_model = None
        for model in reversed(model_ranking):
            if model in available_models:
                biggest_model = model
                break

        if biggest_model is None:
            # Fallback: pick any model
            biggest_model = available_models[0]

        logger.info(f"Biggest model baseline: {biggest_model}")

        # Select this model for all samples
        baseline_df = mode_df[mode_df["model_name"] == biggest_model]

        # Compute metrics
        metrics = {
            "avg_reward": baseline_df["reward"].mean(),
            "avg_primary_acc": baseline_df.get("primary_acc", pd.Series([np.nan])).mean(),
            "avg_cost_usd": baseline_df.get("cost_usd", pd.Series([np.nan])).mean(),
            "avg_latency_ms": baseline_df.get("latency_ms", pd.Series([np.nan])).mean(),
        }

        return metrics

    def evaluate_baseline_cheapest(self, mode_id: int) -> Dict[str, float]:
        """
        Evaluate 'always pick cheapest model' baseline.

        Args:
            mode_id: Mode ID to evaluate

        Returns:
            Dictionary with baseline metrics
        """
        mode_df = self.test_df[self.test_df["mode_id"] == mode_id]

        # Find cheapest model (by average cost)
        avg_costs = mode_df.groupby("model_name")["cost_usd"].mean()
        cheapest_model = avg_costs.idxmin()

        logger.info(f"Cheapest model baseline: {cheapest_model} (avg cost: ${avg_costs.min():.6f})")

        # Select this model for all samples
        baseline_df = mode_df[mode_df["model_name"] == cheapest_model]

        # Compute metrics
        metrics = {
            "avg_reward": baseline_df["reward"].mean(),
            "avg_primary_acc": baseline_df.get("primary_acc", pd.Series([np.nan])).mean(),
            "avg_cost_usd": baseline_df.get("cost_usd", pd.Series([np.nan])).mean(),
            "avg_latency_ms": baseline_df.get("latency_ms", pd.Series([np.nan])).mean(),
        }

        return metrics

    def evaluate_baseline_random(self, mode_id: int, seed: int = 42) -> Dict[str, float]:
        """
        Evaluate random model selection baseline.

        Args:
            mode_id: Mode ID to evaluate
            seed: Random seed

        Returns:
            Dictionary with baseline metrics
        """
        mode_df = self.test_df[self.test_df["mode_id"] == mode_id]

        # For each sample, randomly pick a model
        np.random.seed(seed)
        random_selections = []

        for sample_id in mode_df["sample_id"].unique():
            sample_df = mode_df[mode_df["sample_id"] == sample_id]
            random_idx = np.random.choice(sample_df.index)
            random_selections.append(sample_df.loc[random_idx])

        baseline_df = pd.DataFrame(random_selections)

        # Compute metrics
        metrics = {
            "avg_reward": baseline_df["reward"].mean(),
            "avg_primary_acc": baseline_df.get("primary_acc", pd.Series([np.nan])).mean(),
            "avg_cost_usd": baseline_df.get("cost_usd", pd.Series([np.nan])).mean(),
            "avg_latency_ms": baseline_df.get("latency_ms", pd.Series([np.nan])).mean(),
        }

        return metrics

    def evaluate_all(self) -> pd.DataFrame:
        """
        Evaluate router and baselines across all modes.

        Returns:
            Summary dataframe with all results
        """
        logger.info("=" * 80)
        logger.info("Evaluating Router Performance")
        logger.info("=" * 80)

        # Get predictions
        df_with_preds = self.predict_rewards()

        # Evaluate each mode
        results = []

        for mode_id, mode_name in enumerate(self.config.evaluation.mode_names):
            logger.info(f"\n--- Mode: {mode_name} (ID={mode_id}) ---")

            # Oracle
            oracle_metrics = self.evaluate_oracle(mode_id)
            logger.info(f"Oracle:  Reward={oracle_metrics['avg_reward']:.4f}, "
                       f"Acc={oracle_metrics['avg_primary_acc']:.4f}, "
                       f"Cost=${oracle_metrics['avg_cost_usd']:.6f}, "
                       f"Latency={oracle_metrics['avg_latency_ms']:.1f}ms")

            results.append({
                "mode": mode_name,
                "method": "oracle",
                **oracle_metrics,
            })

            # Router
            router_metrics = self.evaluate_router(mode_id, df_with_preds)
            logger.info(f"Router:  Reward={router_metrics['avg_reward']:.4f}, "
                       f"Acc={router_metrics['avg_primary_acc']:.4f}, "
                       f"Cost=${router_metrics['avg_cost_usd']:.6f}, "
                       f"Latency={router_metrics['avg_latency_ms']:.1f}ms, "
                       f"RoutingAcc={router_metrics['routing_accuracy']:.2%}")

            results.append({
                "mode": mode_name,
                "method": "router",
                **router_metrics,
            })

            # Baselines
            baseline_biggest = self.evaluate_baseline_biggest(mode_id)
            logger.info(f"Biggest: Reward={baseline_biggest['avg_reward']:.4f}, "
                       f"Acc={baseline_biggest['avg_primary_acc']:.4f}")

            results.append({
                "mode": mode_name,
                "method": "baseline_biggest",
                **baseline_biggest,
            })

            baseline_cheapest = self.evaluate_baseline_cheapest(mode_id)
            logger.info(f"Cheapest: Reward={baseline_cheapest['avg_reward']:.4f}, "
                       f"Cost=${baseline_cheapest['avg_cost_usd']:.6f}")

            results.append({
                "mode": mode_name,
                "method": "baseline_cheapest",
                **baseline_cheapest,
            })

            baseline_random = self.evaluate_baseline_random(mode_id)
            logger.info(f"Random:  Reward={baseline_random['avg_reward']:.4f}")

            results.append({
                "mode": mode_name,
                "method": "baseline_random",
                **baseline_random,
            })

        # Create summary dataframe
        summary_df = pd.DataFrame(results)

        logger.info("\n" + "=" * 80)
        logger.info("Evaluation Complete")
        logger.info("=" * 80)

        return summary_df


def evaluate_router(
    config: Config,
    model_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Main evaluation function.

    Args:
        config: Configuration object
        model_path: Path to trained model (uses best model if None)

    Returns:
        Summary dataframe
    """
    # Load model
    if model_path is None:
        model_path = config.paths.get_checkpoint_path()

    logger.info(f"Loading model from: {model_path}")
    model = RewardRouterModel.load(model_path)

    # Load dataset
    dataset_path = config.paths.get_data_path()
    df = load_dataset(dataset_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.text_encoder_name)

    # Create dataloaders
    train_loader, val_loader, test_loader, train_df, val_df, test_df = create_dataloaders(
        df, tokenizer, config.training, config.model
    )

    # Create evaluator
    evaluator = RouterEvaluator(
        model=model,
        test_df=test_df,
        test_loader=test_loader,
        config=config,
    )

    # Evaluate
    summary_df = evaluator.evaluate_all()

    # Save results
    results_path = config.paths.get_results_path()
    summary_df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to: {results_path}")

    # Generate plots if requested
    if config.evaluation.generate_plots:
        plot_results(summary_df, config)

    return summary_df


def plot_results(summary_df: pd.DataFrame, config: Config):
    """
    Generate evaluation plots.

    Args:
        summary_df: Summary dataframe from evaluation
        config: Configuration
    """
    logger.info("Generating plots...")

    plots_dir = config.paths.get_full_path(config.paths.plots_dir)

    # Plot 1: Reward comparison by mode
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, mode in enumerate(config.evaluation.mode_names):
        mode_df = summary_df[summary_df["mode"] == mode]

        methods = mode_df["method"].tolist()
        rewards = mode_df["avg_reward"].tolist()

        # Color code
        colors = []
        for method in methods:
            if method == "oracle":
                colors.append("green")
            elif method == "router":
                colors.append("blue")
            else:
                colors.append("gray")

        axes[i].bar(range(len(methods)), rewards, color=colors)
        axes[i].set_xticks(range(len(methods)))
        axes[i].set_xticklabels(methods, rotation=45, ha="right")
        axes[i].set_ylabel("Avg Reward")
        axes[i].set_title(f"Mode: {mode}")
        axes[i].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = f"{plots_dir}/reward_comparison.{config.evaluation.plot_format}"
    plt.savefig(plot_path, dpi=config.evaluation.plot_dpi, bbox_inches="tight")
    logger.info(f"Saved plot: {plot_path}")
    plt.close()

    # Plot 2: Cost vs Accuracy tradeoff
    fig, ax = plt.subplots(figsize=(10, 6))

    for mode in config.evaluation.mode_names:
        mode_df = summary_df[summary_df["mode"] == mode]

        for _, row in mode_df.iterrows():
            if row["method"] in ["oracle", "router"]:
                marker = "o" if row["method"] == "oracle" else "^"
                label = f"{mode}-{row['method']}"
                ax.scatter(
                    row["avg_cost_usd"],
                    row["avg_primary_acc"],
                    marker=marker,
                    s=100,
                    label=label,
                    alpha=0.7,
                )

    ax.set_xlabel("Avg Cost (USD)")
    ax.set_ylabel("Avg Accuracy")
    ax.set_title("Cost vs Accuracy Tradeoff")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = f"{plots_dir}/cost_accuracy_tradeoff.{config.evaluation.plot_format}"
    plt.savefig(plot_path, dpi=config.evaluation.plot_dpi, bbox_inches="tight")
    logger.info(f"Saved plot: {plot_path}")
    plt.close()

    logger.info("Plotting complete")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Evaluate with default config
    from ..config import Config

    config = Config.default()
    summary_df = evaluate_router(config)
