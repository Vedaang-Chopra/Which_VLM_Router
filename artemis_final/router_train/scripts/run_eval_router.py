#!/usr/bin/env python3
"""
Script to evaluate trained reward router model.

Usage:
    python scripts/run_eval_router.py [OPTIONS]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from training.eval_reward_router import evaluate_router


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate reward router model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model checkpoint (uses best model if not specified)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset parquet file (overrides config)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results CSV (overrides config)",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation",
    )

    parser.add_argument(
        "--plot-format",
        type=str,
        default=None,
        choices=["png", "pdf", "svg"],
        help="Plot output format",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use for evaluation",
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

        if args.output:
            config.paths.eval_summary_file = args.output

        if args.no_plots:
            config.evaluation.generate_plots = False

        if args.plot_format:
            config.evaluation.plot_format = args.plot_format

        if args.device:
            config.training.device = args.device

        # Model path
        model_path = args.model

        # Print configuration
        logger.info("\n" + "=" * 80)
        logger.info("Evaluation Configuration")
        logger.info("=" * 80)
        logger.info(f"Model: {model_path or config.paths.get_checkpoint_path()}")
        logger.info(f"Dataset: {config.paths.get_data_path()}")
        logger.info(f"Output: {config.paths.get_results_path()}")
        logger.info(f"Generate plots: {config.evaluation.generate_plots}")
        if config.evaluation.generate_plots:
            logger.info(f"Plot format: {config.evaluation.plot_format}")
        logger.info(f"Modes: {config.evaluation.mode_names}")
        logger.info(f"Baselines: {config.evaluation.baselines}")
        logger.info("=" * 80 + "\n")

        # Evaluate
        summary_df = evaluate_router(config, model_path=model_path)

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("Evaluation Summary")
        logger.info("=" * 80)
        print("\n" + summary_df.to_string(index=False))
        logger.info("\n" + "=" * 80)

        logger.info("\n" + "=" * 80)
        logger.info("SUCCESS: Evaluation completed!")
        logger.info(f"Results saved to: {config.paths.get_results_path()}")
        if config.evaluation.generate_plots:
            logger.info(f"Plots saved to: {config.paths.get_full_path(config.paths.plots_dir)}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
