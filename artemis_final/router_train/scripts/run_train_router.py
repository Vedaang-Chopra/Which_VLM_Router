#!/usr/bin/env python3
"""
Script to train reward router model.

Usage:
    python scripts/run_train_router.py [OPTIONS]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from training.train_reward_router import train_router


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train reward router model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset parquet file (overrides config)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay",
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=None,
        help="Warmup ratio for learning rate scheduler",
    )

    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=None,
        help="Gradient clipping norm",
    )

    # Model
    parser.add_argument(
        "--text-encoder",
        type=str,
        default=None,
        help="Text encoder model name",
    )

    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze text encoder parameters",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension for MLP",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout probability",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use for training",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints",
    )

    parser.add_argument(
        "--early-stopping",
        type=int,
        default=None,
        help="Early stopping patience (epochs)",
    )

    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)

    try:
        # Load config
        logger.info("Loading configuration...")
        config = Config.default()

        # Override config with command line arguments
        if args.dataset:
            config.paths.dataset_file = args.dataset

        if args.epochs:
            config.training.num_epochs = args.epochs

        if args.batch_size:
            config.training.batch_size = args.batch_size

        if args.lr:
            config.training.learning_rate = args.lr

        if args.weight_decay:
            config.training.weight_decay = args.weight_decay

        if args.warmup_ratio:
            config.training.warmup_ratio = args.warmup_ratio

        if args.gradient_clip:
            config.training.gradient_clip_norm = args.gradient_clip

        if args.text_encoder:
            config.model.text_encoder_name = args.text_encoder

        if args.freeze_encoder:
            config.model.freeze_text_encoder = True

        if args.hidden_dim:
            config.model.hidden_dim = args.hidden_dim

        if args.dropout:
            config.model.dropout = args.dropout

        if args.device:
            config.training.device = args.device

        if args.checkpoint_dir:
            config.paths.checkpoints_dir = args.checkpoint_dir

        if args.early_stopping:
            config.training.early_stopping_patience = args.early_stopping

        if args.seed:
            config.training.seed = args.seed

        # Print configuration
        logger.info("\n" + "=" * 80)
        logger.info("Training Configuration")
        logger.info("=" * 80)
        logger.info(f"Dataset: {config.paths.get_data_path()}")
        logger.info(f"Epochs: {config.training.num_epochs}")
        logger.info(f"Batch size: {config.training.batch_size}")
        logger.info(f"Learning rate: {config.training.learning_rate}")
        logger.info(f"Weight decay: {config.training.weight_decay}")
        logger.info(f"Text encoder: {config.model.text_encoder_name}")
        logger.info(f"Freeze encoder: {config.model.freeze_text_encoder}")
        logger.info(f"Hidden dim: {config.model.hidden_dim}")
        logger.info(f"Device: {config.training.device}")
        logger.info(f"Checkpoint dir: {config.paths.checkpoints_dir}")
        logger.info(f"Seed: {config.training.seed}")
        logger.info("=" * 80 + "\n")

        # Train model
        model, history = train_router(config)

        logger.info("\n" + "=" * 80)
        logger.info("SUCCESS: Training completed!")
        logger.info(f"Best model saved to: {config.paths.get_checkpoint_path()}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
