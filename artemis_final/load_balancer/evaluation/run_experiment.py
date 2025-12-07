"""
Main experiment orchestration script for load balancer evaluation.

This script:
1. Loads REAL profiling data from SQL database
2. Initializes router (using real probabilities), load balancer, and loggers
3. Runs traffic simulation across load profiles
4. Logs metrics to W&B and CSV
5. Computes and reports SLA metrics

Can be run from command line:
    python -m load_balancer.evaluation.run_experiment --mode capacity_aware
"""

import argparse
import logging
import sys
import time
import sqlite3
import random
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Iterator, Tuple, Dict, List
from datetime import datetime

# Add parent directory to path for imports
# artemis_final/load_balancer/evaluation/run_experiment.py -> artemis_final/
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


# === DATA LOADING FUNCTIONS ===

def load_dataset_from_sql(db_path: Path, limit: Optional[int] = None) -> pd.DataFrame:
    """Load profiling data from SQLite cache."""
    if not db_path.exists():
        logger.error(f"SQLite database not found at {db_path}")
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    logger.info(f"Loading data from {db_path}...")
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT 
        sample_id, source_dataset, router_task, data_split,
        model_name, latency_ms, cost_usd, confidence_score, glider_score
    FROM vlm_profiles
    WHERE router_task IS NOT NULL
    """
    if limit:
        query += f" LIMIT {limit}"
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    logger.info(f"Loaded {len(df)} records. Unique samples: {df['sample_id'].nunique()}")
    return df


def build_stats_from_df(df: pd.DataFrame) -> Dict:
    """Convert SQL profiling data to StatsRegistry format."""
    stats_dict = {}
    
    # Group by task and model
    for (task, model), group in df.groupby(['router_task', 'model_name']):
        if task not in stats_dict:
            stats_dict[task] = {}
        
        # Calculate average stats
        stats_dict[task][model] = {
            'accuracy': group['glider_score'].mean() if group['glider_score'].notna().any() else 0.8,
            'avg_latency_ms': group['latency_ms'].mean() if group['latency_ms'].notna().any() else 500,
            'avg_cost_usd': group['cost_usd'].mean() if group['cost_usd'].notna().any() else 0.0001,
        }
    
    logger.info(f"Built stats dict for {len(stats_dict)} tasks")
    return stats_dict


def get_model_scores_for_sample(df: pd.DataFrame, sample_id: str, model_names: List[str]) -> Dict[str, float]:
    """Get performance scores for a specific sample across all models."""
    sample_group = df[df['sample_id'] == sample_id]
    
    model_scores = {}
    for _, row in sample_group.iterrows():
        model_name = row['model_name']
        score = row['glider_score'] if pd.notna(row['glider_score']) else 0.5
        model_scores[model_name] = score
    
    # Ensure all models have scores (use 0.5 for missing)
    for model in model_names:
        if model not in model_scores:
            model_scores[model] = 0.5
            
    return model_scores


# === ROUTER AND TRAFFIC GENERATION ===

def artemis_route(sample_id: str, task_type: str, model_scores: Dict[str, float]) -> RouterOutput:
    """
    Simulate router probability generation based on actual model scores.
    Uses softmax to convert scores to probabilities.
    """
    model_names = list(model_scores.keys())
    scores = np.array([model_scores[m] for m in model_names])
    
    # Softmax
    exp_scores = np.exp(scores - scores.max())  # Numerical stability
    probs = exp_scores / exp_scores.sum()
    
    router_probs = {m: float(p) for m, p in zip(model_names, probs)}
    preferred_model = max(router_probs, key=router_probs.get)
    
    return RouterOutput(
        sample_id=str(sample_id),
        task_type=str(task_type),
        router_probs=router_probs,
        preferred_model=preferred_model
    )


def generate_traffic(
    load_profile_config,
    df: pd.DataFrame,
    model_names: List[str],
    random_seed: Optional[int] = None
) -> Iterator[Tuple[RouterOutput, float]]:
    """
    Generate traffic by sampling from real SQL data.
    
    Yields:
        (RouterOutput, arrival_ts_ms)
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Pre-compute unique sample IDs to sample from
    unique_sample_ids = df['sample_id'].unique()
    
    # Simple Poisson process simulation
    qps = load_profile_config.qps
    duration_sec = load_profile_config.duration_sec
    inter_arrival_time_ms = 1000.0 / qps

    current_time_ms = 0.0
    num_requests = int(qps * duration_sec)

    for i in range(num_requests):
        # 1. Pick a random sample ID from real data
        sample_id = random.choice(unique_sample_ids)
        
        # 2. Get its task type
        task_type = df[df['sample_id'] == sample_id]['router_task'].iloc[0]
        
        # 3. Get model scores for this sample
        model_scores = get_model_scores_for_sample(df, sample_id, model_names)
        
        # 4. Generate router output
        # Create a unique request ID based on the sample ID
        request_id = f"{sample_id}_{i}"
        router_output = artemis_route(request_id, task_type, model_scores)
        
        yield router_output, current_time_ms

        # Sample next inter-arrival time
        current_time_ms += np.random.exponential(inter_arrival_time_ms)


def run_experiment(
    experiment_config: Optional[ExperimentConfig] = None,
    stats_path: Optional[Path] = None,
    capacity_config_path: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> dict:
    """Run a complete load balancer experiment."""
    if experiment_config is None:
        experiment_config = default_experiment_config()

    if output_dir is None:
        output_dir = get_output_dir(experiment_config.name)

    logger.info(f"Starting experiment: {experiment_config.name}")
    logger.info(f"Output directory: {output_dir}")

    # Load configurations
    logger.info("Loading configurations...")
    model_configs = load_capacity_config(capacity_config_path)
    model_names = list(model_configs.keys())
    
    # Load Real Data
    db_path = Path(__file__).parents[2] / "router_train" / "data" / "vlm_router_cache.db"
    df_sql = load_dataset_from_sql(db_path)
    
    # Build Stats Registry from Data
    stats_dict = build_stats_from_df(df_sql)
    stats_registry = StatsRegistry(stats_dict)

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

    if experiment_config.log_to_csv:
        jsonl_path = output_dir / "decisions.jsonl"
        jsonl_logger = JsonlMetricsLogger(jsonl_path)

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
                df_sql,
                model_names,
                random_seed=experiment_config.random_seed
            )

            for router_output, arrival_ts_ms in traffic_gen:
                # Create scheduling context
                context = SchedulingContext(
                    arrival_ts_ms=arrival_ts_ms,
                    load_profile=profile_name,
                    metadata={'request_id': router_output.sample_id}
                )

                # Schedule with load balancer
                decision = load_balancer.schedule(router_output, context)

                # Log decision
                if csv_logger:
                    csv_logger.log(experiment_config.name, profile_name, decision, global_step=global_step)
                if jsonl_logger:
                    jsonl_logger.log(experiment_config.name, profile_name, decision, global_step=global_step)
                if wandb_logger:
                    wandb_logger.log_decision(decision, profile_name, step=global_step)

                # Update SLA monitor
                sla_monitor.update(decision, profile_name)
                profile_decisions.append(decision)
                all_decisions.append(decision)

                global_step += 1
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
                wandb_logger.log_sla_metrics(profile_metrics, profile_name, stage="final")

    finally:
        if csv_logger: csv_logger.close()
        if jsonl_logger: jsonl_logger.close()

    # Final metrics
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"{'='*60}")

    detailed_metrics = sla_monitor.detailed_snapshot()
    print_detailed_summary(detailed_metrics)

    if wandb_logger:
        wandb_logger.log_detailed_metrics(detailed_metrics, stage="final")
        
        summary = {
            "total_requests": len(all_decisions),
            "overall_violation_rate": detailed_metrics.overall.violation_rate,
            "overall_latency_p95_ms": detailed_metrics.overall.latency_p95_ms,
            "overall_avg_cost_usd": detailed_metrics.overall.avg_cost_usd,
            "total_cost_usd": detailed_metrics.overall.total_cost_usd,
        }
        wandb_logger.log_summary(summary)
        
        if csv_logger:
            wandb_logger.log_artifact(output_dir / "decisions.csv", artifact_type="result", name=f"{experiment_config.name}_decisions")
        
        wandb_logger.finish()

    return {
        "experiment_config": experiment_config,
        "detailed_metrics": detailed_metrics,
        "all_decisions": all_decisions,
        "output_dir": output_dir,
        "load_balancer_summary": load_balancer.get_summary(),
    }


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run Artemis load balancer experiment")
    parser.add_argument("--name", type=str, default="load_balancer_test", help="Experiment name")
    parser.add_argument("--mode", type=str, choices=["router_only", "capacity_aware", "cost_minimizing"], default="capacity_aware", help="Scheduling mode")
    parser.add_argument("--sla-ms", type=float, default=2000.0, help="Global latency SLA in milliseconds")
    parser.add_argument("--max-accuracy-drop", type=float, default=0.05, help="Maximum allowed accuracy drop")
    parser.add_argument("--simulation-only", action="store_true", help="Run in simulation-only mode")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for logs")

    args = parser.parse_args()

    config = default_experiment_config()
    config.name = args.name
    config.scheduling_mode = args.mode
    config.global_latency_sla_ms = args.sla_ms
    config.max_allowed_accuracy_drop = args.max_accuracy_drop
    config.simulation_only = args.simulation_only
    config.log_to_wandb = not args.no_wandb
    config.random_seed = args.seed

    try:
        results = run_experiment(experiment_config=config, output_dir=args.output_dir)
        logger.info("\nExperiment completed successfully!")
        logger.info(f"Results saved to: {results['output_dir']}")
        return 0
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
