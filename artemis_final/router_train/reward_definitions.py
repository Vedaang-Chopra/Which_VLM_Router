"""
Multi-objective reward function definitions for VLM router training.

This module implements different reward modes (accuracy, cheap, fast, balanced)
that combine multiple signals: accuracy, confidence, cost, latency, and hallucination.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from config import RewardWeights

logger = logging.getLogger(__name__)


def normalize_cost_latency(
    df: pd.DataFrame,
    weights: RewardWeights,
) -> Tuple[pd.Series, pd.Series]:
    """
    Normalize cost and latency using quantile-based scaling.

    Values are normalized by dividing by the specified quantile (default 95th percentile)
    and clipping to [0, 1].

    Args:
        df: Profiles dataframe with columns: cost_usd, latency_ms
        weights: Reward weights configuration

    Returns:
        Tuple of (cost_norm, lat_norm) series
    """
    # Cost normalization
    cost_usd = df["cost_usd"].fillna(0.0)
    cost_quantile_value = cost_usd.quantile(weights.cost_quantile)
    if cost_quantile_value > 0:
        cost_norm = cost_usd / cost_quantile_value
    else:
        logger.warning("Cost quantile is 0, setting all cost_norm to 0")
        cost_norm = pd.Series(0.0, index=df.index)
    cost_norm = cost_norm.clip(0.0, 1.0)

    # Latency normalization
    latency_ms = df["latency_ms"].fillna(0.0)
    lat_quantile_value = latency_ms.quantile(weights.latency_quantile)
    if lat_quantile_value > 0:
        lat_norm = latency_ms / lat_quantile_value
    else:
        logger.warning("Latency quantile is 0, setting all lat_norm to 0")
        lat_norm = pd.Series(0.0, index=df.index)
    lat_norm = lat_norm.clip(0.0, 1.0)

    logger.info(f"Cost normalized: quantile={cost_quantile_value:.6f}, mean={cost_norm.mean():.3f}")
    logger.info(f"Latency normalized: quantile={lat_quantile_value:.2f}ms, mean={lat_norm.mean():.3f}")

    return cost_norm, lat_norm


def compute_reward_accuracy(
    A: pd.Series,
    H: pd.Series,
    weights: RewardWeights,
) -> pd.Series:
    """
    Compute accuracy-focused reward: (A^exp) * H

    Maximizes accuracy while penalizing hallucinations.

    Args:
        A: Primary accuracy [0, 1]
        H: Hallucination cleanliness [0, 1]
        weights: Reward weights configuration

    Returns:
        Reward series
    """
    reward = (A ** weights.accuracy_exp) * H
    logger.info(f"Reward (accuracy): mean={reward.mean():.3f}, std={reward.std():.3f}")
    return reward


def compute_reward_cheap(
    A: pd.Series,
    H: pd.Series,
    cost_norm: pd.Series,
    weights: RewardWeights,
) -> pd.Series:
    """
    Compute cost-focused reward: A * H - weight * (cost_norm^exp)

    Balances accuracy/quality with low cost.

    Args:
        A: Primary accuracy [0, 1]
        H: Hallucination cleanliness [0, 1]
        cost_norm: Normalized cost [0, 1]
        weights: Reward weights configuration

    Returns:
        Reward series
    """
    quality_term = A * H
    cost_penalty = weights.cheap_cost_weight * (cost_norm ** weights.cheap_cost_exp)
    reward = quality_term - cost_penalty

    logger.info(f"Reward (cheap): mean={reward.mean():.3f}, std={reward.std():.3f}")
    return reward


def compute_reward_fast(
    A: pd.Series,
    H: pd.Series,
    lat_norm: pd.Series,
    weights: RewardWeights,
) -> pd.Series:
    """
    Compute latency-focused reward: A * H - weight * (lat_norm^exp)

    Balances accuracy/quality with low latency.

    Args:
        A: Primary accuracy [0, 1]
        H: Hallucination cleanliness [0, 1]
        lat_norm: Normalized latency [0, 1]
        weights: Reward weights configuration

    Returns:
        Reward series
    """
    quality_term = A * H
    lat_penalty = weights.fast_lat_weight * (lat_norm ** weights.fast_lat_exp)
    reward = quality_term - lat_penalty

    logger.info(f"Reward (fast): mean={reward.mean():.3f}, std={reward.std():.3f}")
    return reward


def compute_reward_balanced(
    A: pd.Series,
    H: pd.Series,
    C: pd.Series,
    cost_norm: pd.Series,
    lat_norm: pd.Series,
    weights: RewardWeights,
) -> pd.Series:
    """
    Compute balanced reward: (A^a_exp) * H + c_weight * (C^c_exp) - cost_weight * (cost^cost_exp) - lat_weight * (lat^lat_exp)

    Multi-objective optimization balancing quality, confidence, cost, and latency.

    Args:
        A: Primary accuracy [0, 1]
        H: Hallucination cleanliness [0, 1]
        C: Confidence proxy [0, 1]
        cost_norm: Normalized cost [0, 1]
        lat_norm: Normalized latency [0, 1]
        weights: Reward weights configuration

    Returns:
        Reward series
    """
    acc_term = (A ** weights.balanced_acc_exp) * H
    conf_term = weights.balanced_conf_weight * (C ** weights.balanced_conf_exp)
    cost_penalty = weights.balanced_cost_weight * (cost_norm ** weights.balanced_cost_exp)
    lat_penalty = weights.balanced_lat_weight * (lat_norm ** weights.balanced_lat_exp)

    reward = acc_term + conf_term - cost_penalty - lat_penalty

    logger.info(f"Reward (balanced): mean={reward.mean():.3f}, std={reward.std():.3f}")
    return reward


def compute_rewards_real_schema(df: pd.DataFrame, weights: RewardWeights) -> pd.DataFrame:
    """
    Compute all reward modes for the profiles dataframe using REAL schema.

    This version uses glider_score (0-5) as the primary accuracy signal.

    Args:
        df: Profiles dataframe from database with glider_score
        weights: Reward weights configuration

    Returns:
        Modified dataframe with added columns:
            - primary_acc: Primary accuracy metric [0, 1] (from glider_score)
            - cost_norm: Normalized cost [0, 1]
            - lat_norm: Normalized latency [0, 1]
            - H: Hallucination cleanliness [0, 1] (defaulting to 1.0)
            - C: Confidence proxy [0, 1]
            - reward_accuracy: Accuracy-focused reward
            - reward_cheap: Cost-focused reward
            - reward_fast: Latency-focused reward
            - reward_balanced: Balanced multi-objective reward
    """
    logger.info("Computing rewards for all modes (using real schema with glider_score)...")

    df = df.copy()

    # Step 1: Primary accuracy from glider_score (0-5 scale)
    if "glider_score" in df.columns:
        # Normalize glider_score to [0, 1]
        df["primary_acc"] = (df["glider_score"] / 5.0).clip(0, 1)
        logger.info(f"Using glider_score as primary accuracy (mean={df['primary_acc'].mean():.3f})")
    else:
        logger.warning("glider_score not found, defaulting to 0.5")
        df["primary_acc"] = 0.5

    # Step 2: Normalize cost and latency
    df["cost_norm"], df["lat_norm"] = normalize_cost_latency(df, weights)

    # Step 3: Hallucination cleanliness (not available in current schema, default to 1.0)
    df["H"] = 1.0
    logger.info("Hallucination cleanliness H set to 1.0 (no hallucination signal available)")

    # Step 4: Confidence proxy from confidence_score
    if "confidence_score" in df.columns:
        df["C"] = df["confidence_score"].fillna(0.5).clip(0, 1)
        logger.info(f"Using confidence_score as C (mean={df['C'].mean():.3f})")
    else:
        logger.warning("confidence_score not found, using primary_acc as proxy")
        df["C"] = df["primary_acc"]

    # Step 5: Compute rewards for each mode
    A = df["primary_acc"]
    H = df["H"]
    C = df["C"]
    cost_norm = df["cost_norm"]
    lat_norm = df["lat_norm"]

    df["reward_accuracy"] = compute_reward_accuracy(A, H, weights)
    df["reward_cheap"] = compute_reward_cheap(A, H, cost_norm, weights)
    df["reward_fast"] = compute_reward_fast(A, H, lat_norm, weights)
    df["reward_balanced"] = compute_reward_balanced(A, H, C, cost_norm, lat_norm, weights)

    # Log summary statistics
    logger.info("\n=== Reward Summary Statistics ===")
    for mode in ["accuracy", "cheap", "fast", "balanced"]:
        col = f"reward_{mode}"
        logger.info(f"{mode:12s}: mean={df[col].mean():7.3f}, std={df[col].std():7.3f}, "
                   f"min={df[col].min():7.3f}, max={df[col].max():7.3f}")

    return df
