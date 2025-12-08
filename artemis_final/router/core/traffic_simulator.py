"""
Traffic simulation utilities for Artemis Router.

Generates synthetic traffic and simulates various load patterns.
"""

import time
import random
from typing import Iterable, List, Optional, Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .schemas import Sample, InferenceResult
from .config import TrafficConfig


def make_synthetic_sample(idx: int, cfg: TrafficConfig) -> Sample:
    """
    Create a synthetic sample for testing.

    Args:
        idx: Sample index
        cfg: Traffic configuration

    Returns:
        Synthetic sample with random text and image
    """
    # Generate synthetic text
    text_words = []
    for _ in range(random.randint(10, cfg.synthetic_text_length)):
        # Random word length 3-10
        word_len = random.randint(3, 10)
        word = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=word_len))
        text_words.append(word)

    text = f"synthetic query {idx}: " + " ".join(text_words)

    # Generate synthetic image (random noise)
    h, w, c = cfg.synthetic_image_shape
    arr = (np.random.rand(h, w, c) * 255).astype("uint8")
    image = Image.fromarray(arr, mode="RGB")

    return Sample(
        sample_id=f"synthetic_{idx}",
        source="synthetic",
        text=text,
        image=image,
        image_uri=None,
        metadata={"split": "synthetic"},
        label=None,
    )


def generate_stream_synthetic(cfg: TrafficConfig) -> Iterable[Sample]:
    """
    Generate infinite stream of synthetic samples.

    Args:
        cfg: Traffic configuration

    Yields:
        Synthetic samples
    """
    idx = 0
    while True:
        yield make_synthetic_sample(idx, cfg)
        idx += 1


def generate_stream_from_samples(samples: List[Sample], repeat: bool = True) -> Iterable[Sample]:
    """
    Generate stream from a list of samples.

    Args:
        samples: List of samples to stream
        repeat: If True, cycle through samples infinitely

    Yields:
        Samples from the list
    """
    if not samples:
        raise ValueError("Cannot generate stream from empty sample list")

    if repeat:
        while True:
            # Shuffle each epoch for variety
            shuffled = samples.copy()
            random.shuffle(shuffled)
            for sample in shuffled:
                yield sample
    else:
        for sample in samples:
            yield sample


@dataclass
class TrafficStats:
    """Statistics from a traffic simulation run."""
    total_samples: int
    total_duration_sec: float
    actual_rps: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    model_distribution: dict
    errors: int


def run_traffic(
    route_fn: Callable[[Sample], InferenceResult],
    source: str,
    samples: Optional[List[Sample]] = None,
    traffic_cfg: Optional[TrafficConfig] = None,
    rps: float = 5.0,
    duration_sec: int = 60,
    batch_size: int = 1,
    verbose: bool = True,
) -> tuple[List[InferenceResult], TrafficStats]:
    """
    Simulate traffic at a specified rate.

    Args:
        route_fn: Function that routes a single sample
        source: "synthetic" or "db"
        samples: List of samples (required if source="db")
        traffic_cfg: Traffic config (required if source="synthetic")
        rps: Requests per second
        duration_sec: Simulation duration
        batch_size: Samples per request (for future batching support)
        verbose: Print progress updates

    Returns:
        Tuple of (results list, statistics)

    Note:
        Currently only supports batch_size=1. Batching to be added later.
    """
    if batch_size != 1:
        raise NotImplementedError("Batch size > 1 not yet implemented")

    # Setup sample generator
    if source == "synthetic":
        if traffic_cfg is None:
            raise ValueError("traffic_cfg required for synthetic source")
        stream = generate_stream_synthetic(traffic_cfg)
    elif source == "db":
        if samples is None or len(samples) == 0:
            raise ValueError("samples required for db source")
        stream = generate_stream_from_samples(samples, repeat=True)
    else:
        raise ValueError(f"Unknown source: {source}")

    # Simulation parameters
    interval = 1.0 / rps
    end_time = time.time() + duration_sec
    results = []
    errors = 0
    latencies = []

    start_time = time.time()
    request_count = 0

    if verbose:
        print(f"Starting traffic simulation: {rps} RPS for {duration_sec}s (source={source})")

    while time.time() < end_time:
        sample = next(stream)
        request_start = time.time()

        try:
            result = route_fn(sample)
            results.append(result)
            latencies.append(result.router_decision.inference_ms)
        except Exception as e:
            if verbose:
                print(f"[ERROR] Failed to route sample {sample.sample_id}: {e}")
            errors += 1

        request_count += 1

        # Sleep to maintain target RPS
        elapsed = time.time() - request_start
        sleep_time = max(0, interval - elapsed)
        time.sleep(sleep_time)

        # Progress update
        if verbose and request_count % 100 == 0:
            elapsed_total = time.time() - start_time
            actual_rps = request_count / elapsed_total
            print(f"  Progress: {request_count} requests, {actual_rps:.1f} RPS, {errors} errors")

    # Calculate statistics
    total_duration = time.time() - start_time
    actual_rps = len(results) / total_duration if total_duration > 0 else 0

    if latencies:
        avg_latency = np.mean(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
    else:
        avg_latency = p50 = p95 = p99 = 0.0

    # Model distribution
    model_counts = {}
    for result in results:
        model = result.router_decision.chosen_model
        model_counts[model] = model_counts.get(model, 0) + 1

    stats = TrafficStats(
        total_samples=len(results),
        total_duration_sec=total_duration,
        actual_rps=actual_rps,
        avg_latency_ms=avg_latency,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        model_distribution=model_counts,
        errors=errors,
    )

    if verbose:
        print(f"\nTraffic simulation complete:")
        print(f"  Total requests: {stats.total_samples}")
        print(f"  Duration: {stats.total_duration_sec:.1f}s")
        print(f"  Actual RPS: {stats.actual_rps:.2f}")
        print(f"  Avg latency: {stats.avg_latency_ms:.2f}ms")
        print(f"  P95 latency: {stats.p95_latency_ms:.2f}ms")
        print(f"  Errors: {stats.errors}")
        print(f"  Model distribution: {stats.model_distribution}")

    return results, stats


def run_traffic_pattern(
    route_fn: Callable[[Sample], InferenceResult],
    source: str,
    samples: Optional[List[Sample]] = None,
    traffic_cfg: Optional[TrafficConfig] = None,
    pattern: str = "constant",
    base_rps: float = 5.0,
    duration_sec: int = 120,
    verbose: bool = True,
) -> List[tuple[List[InferenceResult], TrafficStats]]:
    """
    Run traffic simulation with a time-varying pattern.

    Args:
        route_fn: Routing function
        source: "synthetic" or "db"
        samples: Sample list (for db source)
        traffic_cfg: Traffic config (for synthetic source)
        pattern: "constant", "ramp", "spike", or "wave"
        base_rps: Base RPS for pattern
        duration_sec: Total simulation duration
        verbose: Print progress

    Returns:
        List of (results, stats) tuples for each phase
    """
    if pattern == "constant":
        # Single constant rate
        results, stats = run_traffic(
            route_fn, source, samples, traffic_cfg,
            rps=base_rps,
            duration_sec=duration_sec,
            verbose=verbose,
        )
        return [(results, stats)]

    elif pattern == "ramp":
        # Ramp up: base -> 4x base over duration
        phases = []
        phase_duration = duration_sec // 4
        for multiplier in [1, 2, 3, 4]:
            rps = base_rps * multiplier
            if verbose:
                print(f"\n=== Ramp phase {multiplier}: {rps} RPS ===")
            results, stats = run_traffic(
                route_fn, source, samples, traffic_cfg,
                rps=rps,
                duration_sec=phase_duration,
                verbose=verbose,
            )
            phases.append((results, stats))
        return phases

    elif pattern == "spike":
        # Spike: base -> 10x -> base
        phases = []
        phase_duration = duration_sec // 3

        # Normal
        if verbose:
            print(f"\n=== Spike phase 1: {base_rps} RPS (normal) ===")
        results, stats = run_traffic(
            route_fn, source, samples, traffic_cfg,
            rps=base_rps,
            duration_sec=phase_duration,
            verbose=verbose,
        )
        phases.append((results, stats))

        # Spike
        spike_rps = base_rps * 10
        if verbose:
            print(f"\n=== Spike phase 2: {spike_rps} RPS (SPIKE) ===")
        results, stats = run_traffic(
            route_fn, source, samples, traffic_cfg,
            rps=spike_rps,
            duration_sec=phase_duration,
            verbose=verbose,
        )
        phases.append((results, stats))

        # Return to normal
        if verbose:
            print(f"\n=== Spike phase 3: {base_rps} RPS (normal) ===")
        results, stats = run_traffic(
            route_fn, source, samples, traffic_cfg,
            rps=base_rps,
            duration_sec=phase_duration,
            verbose=verbose,
        )
        phases.append((results, stats))

        return phases

    elif pattern == "wave":
        # Wave: oscillate between 0.5x and 2x base
        phases = []
        phase_duration = duration_sec // 6
        multipliers = [1.0, 1.5, 2.0, 1.5, 1.0, 0.5]

        for i, mult in enumerate(multipliers):
            rps = base_rps * mult
            if verbose:
                print(f"\n=== Wave phase {i+1}: {rps} RPS ===")
            results, stats = run_traffic(
                route_fn, source, samples, traffic_cfg,
                rps=rps,
                duration_sec=phase_duration,
                verbose=verbose,
            )
            phases.append((results, stats))

        return phases

    else:
        raise ValueError(f"Unknown pattern: {pattern}")
