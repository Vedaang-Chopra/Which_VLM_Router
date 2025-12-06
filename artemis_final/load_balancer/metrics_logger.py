"""
Metrics logging to CSV and JSONL formats.

This module provides utilities to log scheduling decisions and metrics
to local files for offline analysis.

Formats:
- CSV: Easy to load in pandas/Excel for quick analysis
- JSONL: Preserves full structure for detailed analysis
"""

import csv
import json
import logging
from pathlib import Path
from typing import Optional, List, TextIO
from datetime import datetime

from .types import SchedulingDecision, RequestMetrics

logger = logging.getLogger(__name__)


class CsvMetricsLogger:
    """
    Logger for scheduling decisions in CSV format.

    Creates a CSV file with one row per request, containing all
    relevant metrics for analysis.
    """

    # CSV column definitions
    COLUMNS = [
        "experiment_name",
        "load_profile",
        "global_step",
        "timestamp_iso",
        "sample_id",
        "task_type",
        "chosen_model",
        "preferred_model",
        "arrival_ts_ms",
        "queue_delay_ms",
        "service_time_ms",
        "total_latency_ms",
        "est_cost_usd",
        "est_accuracy",
        "model_queue_time_before_ms",
        "num_replicas",
        "sla_violated",
        "accuracy_drop",
        "missing_stats",
        "router_prob_chosen",
        "router_prob_preferred",
    ]

    def __init__(self, output_path: Path, append: bool = False):
        """
        Initialize CSV logger.

        Args:
            output_path: Path to output CSV file
            append: If True, append to existing file; if False, overwrite
        """
        self.output_path = output_path
        self.file: Optional[TextIO] = None
        self.writer: Optional[csv.DictWriter] = None
        self.row_count = 0

        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open file
        mode = 'a' if append else 'w'
        self.file = open(output_path, mode, newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.COLUMNS)

        # Write header if new file
        if not append or output_path.stat().st_size == 0:
            self.writer.writeheader()
            self.file.flush()

        logger.info(f"Initialized CSV logger: {output_path} (mode={mode})")

    def log(
        self,
        experiment_name: str,
        load_profile: str,
        decision: SchedulingDecision,
        global_step: Optional[int] = None,
        timestamp_iso: Optional[str] = None
    ):
        """
        Log a scheduling decision.

        Args:
            experiment_name: Name of the experiment
            load_profile: Load profile name
            decision: Scheduling decision to log
            global_step: Optional global step counter
            timestamp_iso: Optional ISO timestamp (will be generated if not provided)
        """
        if self.writer is None:
            raise RuntimeError("Logger not initialized")

        # Generate timestamp if not provided
        if timestamp_iso is None:
            timestamp_iso = datetime.fromtimestamp(
                decision.arrival_ts_ms / 1000.0
            ).isoformat()

        # Get router probabilities for chosen and preferred models
        router_prob_chosen = decision.router_probs.get(decision.chosen_model, 0.0)
        router_prob_preferred = decision.router_probs.get(decision.preferred_model, 0.0)

        # Build row
        row = {
            "experiment_name": experiment_name,
            "load_profile": load_profile,
            "global_step": global_step if global_step is not None else "",
            "timestamp_iso": timestamp_iso,
            "sample_id": decision.sample_id,
            "task_type": decision.task_type,
            "chosen_model": decision.chosen_model,
            "preferred_model": decision.preferred_model,
            "arrival_ts_ms": decision.arrival_ts_ms,
            "queue_delay_ms": decision.queue_delay_ms,
            "service_time_ms": decision.service_time_ms,
            "total_latency_ms": decision.total_latency_ms,
            "est_cost_usd": decision.est_cost_usd,
            "est_accuracy": decision.est_accuracy,
            "model_queue_time_before_ms": decision.model_queue_time_before_ms,
            "num_replicas": decision.num_replicas,
            "sla_violated": decision.sla_violated,
            "accuracy_drop": decision.accuracy_drop,
            "missing_stats": decision.missing_stats,
            "router_prob_chosen": router_prob_chosen,
            "router_prob_preferred": router_prob_preferred,
        }

        # Write row
        self.writer.writerow(row)
        self.row_count += 1

        # Flush periodically
        if self.row_count % 100 == 0:
            self.file.flush()

    def log_metrics(self, metrics: RequestMetrics):
        """
        Log a RequestMetrics object.

        Args:
            metrics: RequestMetrics to log
        """
        self.log(
            experiment_name=metrics.experiment_name,
            load_profile=metrics.load_profile,
            decision=metrics.decision,
            global_step=metrics.global_step,
            timestamp_iso=metrics.timestamp_iso,
        )

    def flush(self):
        """Flush the output file."""
        if self.file:
            self.file.flush()

    def close(self):
        """Close the logger."""
        if self.file:
            self.file.flush()
            self.file.close()
            self.file = None
            self.writer = None
            logger.info(f"Closed CSV logger: {self.output_path} ({self.row_count} rows)")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class JsonlMetricsLogger:
    """
    Logger for scheduling decisions in JSONL format.

    Each line is a JSON object with the complete decision structure.
    This format preserves all details including nested router_probs.
    """

    def __init__(self, output_path: Path, append: bool = False):
        """
        Initialize JSONL logger.

        Args:
            output_path: Path to output JSONL file
            append: If True, append to existing file; if False, overwrite
        """
        self.output_path = output_path
        self.file: Optional[TextIO] = None
        self.row_count = 0

        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open file
        mode = 'a' if append else 'w'
        self.file = open(output_path, mode)

        logger.info(f"Initialized JSONL logger: {output_path} (mode={mode})")

    def log(
        self,
        experiment_name: str,
        load_profile: str,
        decision: SchedulingDecision,
        global_step: Optional[int] = None,
        timestamp_iso: Optional[str] = None
    ):
        """
        Log a scheduling decision.

        Args:
            experiment_name: Name of the experiment
            load_profile: Load profile name
            decision: Scheduling decision to log
            global_step: Optional global step counter
            timestamp_iso: Optional ISO timestamp
        """
        if self.file is None:
            raise RuntimeError("Logger not initialized")

        # Generate timestamp if not provided
        if timestamp_iso is None:
            timestamp_iso = datetime.fromtimestamp(
                decision.arrival_ts_ms / 1000.0
            ).isoformat()

        # Build record
        record = {
            "experiment_name": experiment_name,
            "load_profile": load_profile,
            "global_step": global_step,
            "timestamp_iso": timestamp_iso,
            "decision": {
                "sample_id": decision.sample_id,
                "task_type": decision.task_type,
                "chosen_model": decision.chosen_model,
                "preferred_model": decision.preferred_model,
                "router_probs": decision.router_probs,
                "arrival_ts_ms": decision.arrival_ts_ms,
                "queue_delay_ms": decision.queue_delay_ms,
                "service_time_ms": decision.service_time_ms,
                "total_latency_ms": decision.total_latency_ms,
                "est_cost_usd": decision.est_cost_usd,
                "est_accuracy": decision.est_accuracy,
                "model_queue_time_before_ms": decision.model_queue_time_before_ms,
                "num_replicas": decision.num_replicas,
                "sla_violated": decision.sla_violated,
                "accuracy_drop": decision.accuracy_drop,
                "missing_stats": decision.missing_stats,
            }
        }

        # Write line
        self.file.write(json.dumps(record) + '\n')
        self.row_count += 1

        # Flush periodically
        if self.row_count % 100 == 0:
            self.file.flush()

    def log_metrics(self, metrics: RequestMetrics):
        """
        Log a RequestMetrics object.

        Args:
            metrics: RequestMetrics to log
        """
        self.log(
            experiment_name=metrics.experiment_name,
            load_profile=metrics.load_profile,
            decision=metrics.decision,
            global_step=metrics.global_step,
            timestamp_iso=metrics.timestamp_iso,
        )

    def flush(self):
        """Flush the output file."""
        if self.file:
            self.file.flush()

    def close(self):
        """Close the logger."""
        if self.file:
            self.file.flush()
            self.file.close()
            self.file = None
            logger.info(f"Closed JSONL logger: {self.output_path} ({self.row_count} rows)")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def load_decisions_from_csv(csv_path: Path) -> List[SchedulingDecision]:
    """
    Load scheduling decisions from a CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of SchedulingDecision objects
    """
    decisions = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Reconstruct router_probs (simplified - only has chosen and preferred)
            router_probs = {
                row['chosen_model']: float(row['router_prob_chosen']),
                row['preferred_model']: float(row['router_prob_preferred']),
            }

            decision = SchedulingDecision(
                sample_id=row['sample_id'],
                task_type=row['task_type'],
                chosen_model=row['chosen_model'],
                preferred_model=row['preferred_model'],
                router_probs=router_probs,
                arrival_ts_ms=float(row['arrival_ts_ms']),
                queue_delay_ms=float(row['queue_delay_ms']),
                service_time_ms=float(row['service_time_ms']),
                total_latency_ms=float(row['total_latency_ms']),
                est_cost_usd=float(row['est_cost_usd']),
                est_accuracy=float(row['est_accuracy']),
                model_queue_time_before_ms=float(row['model_queue_time_before_ms']),
                num_replicas=int(row['num_replicas']),
                sla_violated=row['sla_violated'].lower() == 'true',
                accuracy_drop=float(row['accuracy_drop']),
                missing_stats=row['missing_stats'].lower() == 'true',
            )
            decisions.append(decision)

    logger.info(f"Loaded {len(decisions)} decisions from {csv_path}")
    return decisions


def load_decisions_from_jsonl(jsonl_path: Path) -> List[SchedulingDecision]:
    """
    Load scheduling decisions from a JSONL file.

    Args:
        jsonl_path: Path to JSONL file

    Returns:
        List of SchedulingDecision objects
    """
    decisions = []

    with open(jsonl_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            dec_data = record['decision']

            decision = SchedulingDecision(
                sample_id=dec_data['sample_id'],
                task_type=dec_data['task_type'],
                chosen_model=dec_data['chosen_model'],
                preferred_model=dec_data['preferred_model'],
                router_probs=dec_data['router_probs'],
                arrival_ts_ms=dec_data['arrival_ts_ms'],
                queue_delay_ms=dec_data['queue_delay_ms'],
                service_time_ms=dec_data['service_time_ms'],
                total_latency_ms=dec_data['total_latency_ms'],
                est_cost_usd=dec_data['est_cost_usd'],
                est_accuracy=dec_data['est_accuracy'],
                model_queue_time_before_ms=dec_data['model_queue_time_before_ms'],
                num_replicas=dec_data['num_replicas'],
                sla_violated=dec_data['sla_violated'],
                accuracy_drop=dec_data['accuracy_drop'],
                missing_stats=dec_data['missing_stats'],
            )
            decisions.append(decision)

    logger.info(f"Loaded {len(decisions)} decisions from {jsonl_path}")
    return decisions
