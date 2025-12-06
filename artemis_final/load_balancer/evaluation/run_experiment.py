"""
Main experiment orchestration script for load balancer evaluation.

This script:
1. Loads configuration and statistics
2. Initializes router, load balancer, and loggers
3. Runs traffic simulation across load profiles
4. Logs metrics to W&B and CSV
5. Computes and reports SLA metrics

Can be run from command line:
    python -m load_balancer.evaluation.run_experiment --config my_config.yaml
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Iterator
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parents[2]))

from load_balancer.config import (
    ExperimentConfig,
    default_experiment_config,
    load_capacity_config,
    get_output_dir,
    WANDB_PROJECT,
    WANDB_ENTITY,
)
from load_balancer.types import RouterOutput, SchedulingContext
from load_balancer.stats_registry import StatsRegistry, load_per_task_model_stats
from load_balancer.scheduler import ArtemisLoadBalancer
from load_balancer.sla_monitor import SlaMonitor, print_detailed_summary
from load_balancer.metrics_logger import CsvMetricsLogger, JsonlMetricsLogger
from load_balancer.wandb_logger import create_logger as create_wandb_logger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# NOTE: This is a placeholder for the actual router integration
# In the real implementation, this would import from artemis_final/router
def artemis_route(sample: dict) -> RouterOutput:
    """
    Placeholder for Artemis router integration.

    In the real implementation, this should import from:
        from artemis_final.router import artemis_route

    Args:
        sample: Sample data with task information

    Returns:
        RouterOutput with routing predictions
    """
    # This is a mock implementation
    # Real implementation should call the actual router
    return RouterOutput(
        sample_id=sample.get('sample_id', 'unknown'),
        task_type=sample.get('task_type', 'unknown'),
        router_probs={
            'small_vlm': 0.2,
            'medium_vlm': 0.5,
            'large_vlm': 0.3,
        },
        preferred_model='medium_vlm'
    )


# NOTE: This is a placeholder for the actual traffic generator
# In the real implementation, this would import from the traffic simulator module
def generate_traffic(
    load_profile_config,
    dataset: list,
    random_seed: Optional[int] = None
) -> Iterator[dict]:
    """
    Placeholder for traffic generator integration.

    In the real implementation, this should import from the traffic simulator:
        from artemis_final.traffic_simulator import generate_traffic

    Args:
        load_profile_config: Load profile configuration
        dataset: Dataset of samples
        random_seed: Random seed for reproducibility

    Yields:
        Request dictionaries with arrival timestamps
    """
    import random
    import numpy as np

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Simple Poisson process simulation
    qps = load_profile_config.qps
    duration_sec = load_profile_config.duration_sec
    inter_arrival_time_ms = 1000.0 / qps

    current_time_ms = 0.0
    num_requests = int(qps * duration_sec)

    for i in range(num_requests):
        # Sample from dataset
        sample = random.choice(dataset)

        # Add arrival timestamp
        request = sample.copy()
        request['arrival_ts_ms'] = current_time_ms
        request['request_id'] = i

        yield request

        # Sample next inter-arrival time (exponential distribution for Poisson process)
        current_time_ms += np.random.exponential(inter_arrival_time_ms)


def load_dataset_samples() -> list:
    """
    Load dataset samples for routing.

    This should load samples from the Ares dataset.
    For now, returns a mock dataset.

    Returns:
        List of sample dictionaries
    """
    # Mock dataset
    # In real implementation, load from artemis_final/ares/data/
    return [
        {'sample_id': f'sample_{i}', 'task_type': 'ocr'}
        for i in range(100)
    ] + [
        {'sample_id': f'sample_{i}', 'task_type': 'chart_vqa'}
        for i in range(100, 200)
    ]


def run_experiment(
    experiment_config: Optional[ExperimentConfig] = None,
    stats_path: Optional[Path] = None,
    capacity_config_path: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> dict:
    """
    Run a complete load balancer experiment.

    Args:
        experiment_config: Experiment configuration (uses default if None)
        stats_path: Path to stats JSON (uses default if None)
        capacity_config_path: Path to capacity config (uses default if None)
        output_dir: Output directory for logs (auto-generated if None)

    Returns:
        Dictionary with experiment results and metrics
    """
    # Use defaults if not provided
    if experiment_config is None:
        experiment_config = default_experiment_config()

    # Setup output directory
    if output_dir is None:
        output_dir = get_output_dir(experiment_config.name)

    logger.info(f"Starting experiment: {experiment_config.name}")
    logger.info(f"Output directory: {output_dir}")

    # Load configurations
    logger.info("Loading configurations...")
    model_configs = load_capacity_config(capacity_config_path)
    stats_dict = load_per_task_model_stats(stats_path)
    stats_registry = StatsRegistry(stats_dict)

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset_samples()

    # Initialize load balancer
    logger.info("Initializing load balancer...")
    load_balancer = ArtemisLoadBalancer(
        model_configs=model_configs,
        stats_registry=stats_registry,
        global_latency_sla_ms=experiment_config.global_latency_sla_ms,
        max_accuracy_drop=experiment_config.max_allowed_accuracy_drop,
        scheduling_mode=experiment_config.scheduling_mode,
        simulation_only=experiment_config.simulation_only,
    )

    # Initialize loggers
    csv_logger = None
    jsonl_logger = None
    wandb_logger = None

    if experiment_config.log_to_csv:
        csv_path = output_dir / "decisions.csv"
        csv_logger = CsvMetricsLogger(csv_path)
        logger.info(f"Logging to CSV: {csv_path}")

    if experiment_config.log_to_csv:  # Also log JSONL
        jsonl_path = output_dir / "decisions.jsonl"
        jsonl_logger = JsonlMetricsLogger(jsonl_path)
        logger.info(f"Logging to JSONL: {jsonl_path}")

    if experiment_config.log_to_wandb:
        wandb_config = {
            "experiment_name": experiment_config.name,
            "scheduling_mode": experiment_config.scheduling_mode,
            "global_latency_sla_ms": experiment_config.global_latency_sla_ms,
            "max_accuracy_drop": experiment_config.max_allowed_accuracy_drop,
            "simulation_only": experiment_config.simulation_only,
            "load_profiles": {
                name: {"qps": p.qps, "duration_sec": p.duration_sec}
                for name, p in experiment_config.load_profiles.items()
            },
            "model_configs": {
                name: {
                    "base_latency_ms": c.base_latency_ms,
                    "min_replicas": c.min_replicas,
                    "max_replicas": c.max_replicas,
                    "sla_ms": c.sla_ms,
                }
                for name, c in model_configs.items()
            }
        }

        run_name = f"{experiment_config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb_logger = create_wandb_logger(
            project=WANDB_PROJECT,
            run_name=run_name,
            config=wandb_config,
            entity=WANDB_ENTITY,
            tags=[experiment_config.scheduling_mode],
            enabled=True,
        )
        logger.info(f"Logging to W&B: {WANDB_PROJECT}/{run_name}")

    # Initialize SLA monitor
    sla_monitor = SlaMonitor(experiment_config.global_latency_sla_ms)

    # Run experiment across load profiles
    global_step = 0
    all_decisions = []

    try:
        for profile_name, profile_config in experiment_config.load_profiles.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Running load profile: {profile_name}")
            logger.info(f"  QPS: {profile_config.qps}")
            logger.info(f"  Duration: {profile_config.duration_sec}s")
            logger.info(f"{'='*60}\n")

            profile_start_time = time.time()
            profile_decisions = []

            # Generate and process traffic
            traffic_gen = generate_traffic(
                profile_config,
                dataset,
                random_seed=experiment_config.random_seed
            )

            for request in traffic_gen:
                # Route with Artemis
                router_output = artemis_route(request)

                # Create scheduling context
                context = SchedulingContext(
                    arrival_ts_ms=request['arrival_ts_ms'],
                    load_profile=profile_name,
                    metadata={'request_id': request.get('request_id')}
                )

                # Schedule with load balancer
                decision = load_balancer.schedule(router_output, context)

                # Log decision
                if csv_logger:
                    csv_logger.log(
                        experiment_config.name,
                        profile_name,
                        decision,
                        global_step=global_step
                    )

                if jsonl_logger:
                    jsonl_logger.log(
                        experiment_config.name,
                        profile_name,
                        decision,
                        global_step=global_step
                    )

                if wandb_logger:
                    wandb_logger.log_decision(decision, profile_name, step=global_step)

                # Update SLA monitor
                sla_monitor.update(decision, profile_name)
                profile_decisions.append(decision)
                all_decisions.append(decision)

                global_step += 1

                # Log progress periodically
                if global_step % 100 == 0:
                    logger.info(f"  Processed {global_step} requests...")

            # Profile completed
            profile_duration = time.time() - profile_start_time
            logger.info(f"\nCompleted {profile_name} in {profile_duration:.1f}s")
            logger.info(f"  Processed {len(profile_decisions)} requests")

            # Compute and log profile metrics
            profile_metrics = sla_monitor.snapshot()
            logger.info(f"  Violation rate: {profile_metrics.violation_rate:.2%}")
            logger.info(f"  Latency p95: {profile_metrics.latency_p95_ms:.1f}ms")
            logger.info(f"  Avg cost: ${profile_metrics.avg_cost_usd:.6f}")

            if wandb_logger:
                wandb_logger.log_sla_metrics(
                    profile_metrics,
                    profile_name,
                    stage="final"
                )

    finally:
        # Cleanup
        if csv_logger:
            csv_logger.close()
        if jsonl_logger:
            jsonl_logger.close()

    # Final metrics
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"{'='*60}")

    detailed_metrics = sla_monitor.detailed_snapshot()
    print_detailed_summary(detailed_metrics)

    # Log to W&B
    if wandb_logger:
        wandb_logger.log_detailed_metrics(detailed_metrics, stage="final")

        # Log summary
        summary = {
            "total_requests": len(all_decisions),
            "overall_violation_rate": detailed_metrics.overall.violation_rate,
            "overall_latency_p95_ms": detailed_metrics.overall.latency_p95_ms,
            "overall_avg_cost_usd": detailed_metrics.overall.avg_cost_usd,
            "total_cost_usd": detailed_metrics.overall.total_cost_usd,
        }
        wandb_logger.log_summary(summary)

        # Log artifacts
        if csv_logger:
            wandb_logger.log_artifact(
                output_dir / "decisions.csv",
                artifact_type="result",
                name=f"{experiment_config.name}_decisions"
            )

        wandb_logger.finish()

    # Return results
    return {
        "experiment_config": experiment_config,
        "detailed_metrics": detailed_metrics,
        "all_decisions": all_decisions,
        "output_dir": output_dir,
        "load_balancer_summary": load_balancer.get_summary(),
    }


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Run Artemis load balancer experiment"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="load_balancer_test",
        help="Experiment name"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["router_only", "capacity_aware", "cost_minimizing"],
        default="capacity_aware",
        help="Scheduling mode"
    )
    parser.add_argument(
        "--sla-ms",
        type=float,
        default=2000.0,
        help="Global latency SLA in milliseconds"
    )
    parser.add_argument(
        "--max-accuracy-drop",
        type=float,
        default=0.05,
        help="Maximum allowed accuracy drop"
    )
    parser.add_argument(
        "--simulation-only",
        action="store_true",
        help="Run in simulation-only mode (don't commit assignments)"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for logs"
    )

    args = parser.parse_args()

    # Build experiment config
    config = default_experiment_config()
    config.name = args.name
    config.scheduling_mode = args.mode
    config.global_latency_sla_ms = args.sla_ms
    config.max_allowed_accuracy_drop = args.max_accuracy_drop
    config.simulation_only = args.simulation_only
    config.log_to_wandb = not args.no_wandb
    config.random_seed = args.seed

    # Run experiment
    try:
        results = run_experiment(
            experiment_config=config,
            output_dir=args.output_dir
        )
        logger.info("\nExperiment completed successfully!")
        logger.info(f"Results saved to: {results['output_dir']}")
        return 0
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
