
"""
CascadeFlow Experiment Script for VLM Router
===========================================

This script is a Python version of the CascadeFlow evaluation notebook.
It runs a rigorous experiment over the router test split, computes rich
evaluation metrics using the shared `Scorer`, saves results, and produces
a variety of diagnostic plots.

----------------------------------
KNOBS TO TUNE (MOST IMPORTANT ONES)
----------------------------------

1. DATA / OUTPUT
   - PROJECT_ROOT: base path of your repo.
   - TEST_DATASET_PATH: where your Parquet test split lives.
   - OUTPUT_DIR: where results + plots will be written.
   - MAX_SAMPLES: limit the number of samples (for quick sanity runs).

2. CASCADE / MODELS
   - MODEL_ENDPOINTS: mapping from logical model name -> vLLM base URL.
   - MODEL_PRICING: per-1K-token USD cost for each model (prompt/completion).
   - CASCADE_MODEL_ORDER: order of models in the cascade (small -> large).
   - QUALITY_THRESHOLD: confidence threshold per tier (for cascading).

3. GENERATION
   - MAX_TOKENS: max tokens per response.
   - TEMPERATURE: decoding temperature (0.0 for deterministic eval).

4. BUDGET / TELEMETRY
   - BUDGET_LIMIT: total USD budget for the run.
   - WARN_THRESHOLD: fraction of budget at which to start warning.

5. LOGGING
   - LOG_EVERY_SAMPLE: if True, log one line per sample.
   - LOG_DEBUG: if True, log detailed internal info per step.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import sys
import datetime as dt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# CascadeFlow imports (your env must have this installed)
from cascadeflow import CascadeAgent, ModelConfig
from cascadeflow.telemetry import CostTracker, MetricsCollector

# Scoring utilities from your project (must be importable in PYTHONPATH)
from cascadeflow.telemetry import cost_tracker
from evaluation import Scorer


# ============================================================
# LOGGING HELPER
# ============================================================

def log(msg: str, level: str = "INFO") -> None:
    """
    Simple timestamped logger with colored output.
    Levels: INFO, WARN, ERROR, DEBUG
    """
    timestamp = dt.datetime.now().strftime("%H:%M:%S")

    colors = {
        "INFO": "\033[94m",   # blue
        "WARN": "\033[93m",   # yellow
        "ERROR": "\033[91m",  # red
        "DEBUG": "\033[90m",  # gray
    }
    reset = "\033[0m"

    color = colors.get(level, "\033[94m")
    print(f"{color}[{timestamp}] [{level}] {msg}{reset}", file=sys.stdout)


# ============================================================
# CONFIGURATION
# ============================================================

# Paths
PROJECT_ROOT = Path.cwd().parent.parent  # adjust if needed
DATA_ROOT = PROJECT_ROOT / "dataset"

# Test split (align with router_lexico test trainer)
TEST_DATASET_PATH = DATA_ROOT / "final_dataset" / "router_lexico" / "router_test_trainer.parquet"

# Output directory for this experiment
OUTPUT_DIR = Path.cwd() / "outputs" / "cascadeflow_script_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Experiment settings
MAX_SAMPLES: Optional[int] = None  # None = full test set; e.g. 100 for 100 samples
RANDOM_SEED = 42

# Generation settings
MAX_TOKENS = 512
TEMPERATURE = 0.0  # deterministic for eval

# Budget / telemetry
BUDGET_LIMIT = 10.0       # USD
WARN_THRESHOLD = 0.8      # warn at 80% of budget

# vLLM/OpenAI-style endpoints
MODEL_ENDPOINTS: Dict[str, str] = {
    "Qwen/Qwen2.5-VL-3B-Instruct":   "http://localhost:8803/v1",
    "Qwen/Qwen2.5-VL-7B-Instruct":   "http://localhost:8802/v1",
    "google/gemma-3-27b-it":         "http://localhost:8800/v1",
    "Qwen/Qwen3-VL-8B-Thinking":     "http://localhost:8801/v1",
    "deepseek-ai/DeepSeek-OCR":      "http://localhost:8804/v1",
}

# USD pricing per 1K tokens (prompt + completion)
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "Qwen/Qwen2.5-VL-3B-Instruct":   {"prompt_per_1k": 0.0001, "completion_per_1k": 0.0001},
    "Qwen/Qwen2.5-VL-7B-Instruct":   {"prompt_per_1k": 0.0002, "completion_per_1k": 0.0002},
    "google/gemma-3-27b-it":         {"prompt_per_1k": 0.00009, "completion_per_1k": 0.00016},
    "Qwen/Qwen3-VL-8B-Thinking":     {"prompt_per_1k": 0.00018, "completion_per_1k": 0.0021},
    "deepseek-ai/DeepSeek-OCR":      {"prompt_per_1k": 0.00003, "completion_per_1k": 0.0001},
}

# Order of models in cascade (small -> large + specialist)
CASCADE_MODEL_ORDER: List[str] = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "google/gemma-3-27b-it",
    "Qwen/Qwen3-VL-8B-Thinking",
    "deepseek-ai/DeepSeek-OCR",
]

QUALITY_THRESHOLD = 0.7  # per-tier quality threshold inside CascadeFlow

# Logging / verbosity
LOG_EVERY_SAMPLE = False   # True = one INFO line per sample
LOG_DEBUG = False          # True = detailed DEBUG logs


# ============================================================
# UTILITIES
# ============================================================

def compute_true_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute true dollar cost from token counts using MODEL_PRICING."""
    pricing = MODEL_PRICING.get(model_name)
    if pricing is None:
        return 0.0
    return (
        (prompt_tokens / 1000.0) * pricing["prompt_per_1k"]
        + (completion_tokens / 1000.0) * pricing["completion_per_1k"]
    )


def check_model_health(model_name: str, base_url: str) -> Dict[str, Any]:
    """Ping /models endpoint to verify vLLM is up and model is loaded."""
    url = base_url.rstrip("/") + "/models"
    log(f"Checking model health: {model_name} @ {base_url}", "DEBUG" if LOG_DEBUG else "INFO")
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        model_ids = [m.get("id") for m in data.get("data", [])]
        healthy = len(model_ids) > 0
        return {
            "healthy": healthy,
            "model_ids": model_ids,
            "error": None,
            "base_url": base_url,
        }
    except Exception as e:
        return {
            "healthy": False,
            "model_ids": [],
            "error": str(e),
            "base_url": base_url,
        }


def build_cascade_agent(health_status: Dict[str, Dict[str, Any]]) -> CascadeAgent:
    """Build CascadeAgent from healthy models using CASCADE_MODEL_ORDER."""
    log("Building CascadeAgent from healthy endpoints...", "INFO")
    available_models = [m for m in CASCADE_MODEL_ORDER if health_status.get(m, {}).get("healthy")]
    if not available_models:
        raise RuntimeError("No healthy model endpoints found for cascade.")

    cascade_models: List[ModelConfig] = []
    for model_name in available_models:
        info = health_status[model_name]
        model_ids = info["model_ids"]
        if not model_ids:
            log(f"Skipping {model_name}: no model_ids returned by /models", "WARN")
            continue
        vllm_model_id = model_ids[0]
        if LOG_DEBUG:
            log(f"Adding tier: {model_name} (id={vllm_model_id})", "DEBUG")
        cfg = ModelConfig(
            name=vllm_model_id,
            provider="vllm",
            base_url=info["base_url"],
            cost=MODEL_PRICING[model_name]["prompt_per_1k"],
            quality_threshold=QUALITY_THRESHOLD,
            metadata={"logical_name": model_name},
        )
        cascade_models.append(cfg)

    if not cascade_models:
        raise RuntimeError("No valid models to add to cascade.")

    agent = CascadeAgent(models=cascade_models)
    log(f"CascadeAgent built with {len(cascade_models)} tiers.", "INFO")
    return agent


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class EvaluationResult:
    sample_id: str
    source_config: str
    router_task: str
    prompt: str
    ground_truth: str

    # Cascade outputs
    model_used: str
    response: str
    cascaded: bool
    draft_accepted: bool

    # Cost / latency
    latency_ms: float
    total_cost: float
    total_tokens: int

    # Accuracy-style metrics
    is_correct: bool = False
    exact_match: float = 0.0
    score_f1: float = 0.0
    score_exact_match_normalized: float = 0.0
    score_contains_gt: float = 0.0
    score_gt_in_response: float = 0.0
    score_numeric_match: Optional[float] = None
    score_mc_letter_match: Optional[float] = None
    pred_answer_letter: Optional[str] = None
    gt_answer_letter: Optional[str] = None

    # Extra metadata
    timestamp: str = ""
    error: Optional[str] = None


# ============================================================
# ASYNC EVALUATION
# ============================================================

async def evaluate_sample(
    row: pd.Series,
    agent: CascadeAgent,
    cascade_configs: List[ModelConfig],
    cost_tracker: CostTracker,
    metrics_collector: MetricsCollector,
) -> EvaluationResult:
    """Evaluate a single sample through the cascade."""
    sample_id = str(row.get("sample_id", ""))

    try:
        if LOG_EVERY_SAMPLE:
            log(f"Sample {sample_id} → evaluating...", "INFO")

        result = await agent.run(
            query=row["prompt_raw"],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        source_config = str(row.get("source_config", "unknown"))
        router_task = str(row.get("router_task", "unknown"))
        prompt_raw = str(row.get("prompt_raw", ""))
        ground_truth = str(row.get("ground_truth", ""))
        gt_type = str(row.get("ground_truth_type", "exact")) or "exact"

        response = getattr(result, "content", "") or ""
        latency_ms = float(getattr(result, "latency_ms", 0.0) or 0.0)
        cascaded = bool(getattr(result, "cascaded", False))
        draft_accepted = getattr(result, "draft_accepted", False)
        used_model = getattr(result, "model_used", "") or ""

        md = getattr(result, "metadata", {}) or {}
        prompt_tokens = int(md.get("prompt_tokens", md.get("input_tokens", 0)) or 0)
        completion_tokens = int(md.get("completion_tokens", md.get("output_tokens", 0)) or 0)
        total_tokens = prompt_tokens + completion_tokens

        logical_name = used_model
        for cfg in cascade_configs:
            if cfg.name == used_model:
                logical_name = (cfg.metadata or {}).get("logical_name", used_model)
                break

        total_cost = compute_true_cost(logical_name, prompt_tokens, completion_tokens)

        if LOG_DEBUG:
            log(
                f"Sample {sample_id}: model={logical_name}, cascaded={cascaded}, "
                f"latency={latency_ms:.1f}ms, tokens={total_tokens}, cost=${total_cost:.6f}",
                "DEBUG",
            )

        scores = Scorer.compute_all_scores(
            pred=response,
            gt=ground_truth,
            gt_type=gt_type,
        )

        # Telemetry
        # Telemetry: use full CostTracker API (model, provider, tokens, cost, ...)
        cost_tracker.add_cost(
            model=logical_name,          # logical model name (e.g. "Qwen/Qwen2.5-VL-3B-Instruct")
            provider="vllm",             # or whatever provider label you prefer
            tokens=total_tokens,
            cost=total_cost,
            query_id=sample_id,          # so you can trace back later
            metadata={
                "router_task": router_task,
                "source_config": source_config,
                "cascaded": cascaded,
                "draft_accepted": draft_accepted,
            },
        )

        metrics_collector.record(
            result,
            routing_strategy="cascade" if cascaded else "direct",
            complexity="complex" if cascaded else "simple",
        )



        return EvaluationResult(
            sample_id=sample_id,
            source_config=source_config,
            router_task=router_task,
            prompt=prompt_raw[:200],
            ground_truth=ground_truth,
            model_used=logical_name,
            response=response[:800],
            cascaded=cascaded,
            draft_accepted=bool(draft_accepted),
            latency_ms=latency_ms,
            total_cost=total_cost,
            total_tokens=total_tokens,
            is_correct=bool(scores.get("is_correct", False)),
            exact_match=float(scores.get("score_exact_match", 0.0) or 0.0),
            score_f1=float(scores.get("score_f1", 0.0) or 0.0),
            score_exact_match_normalized=float(scores.get("score_exact_match_normalized", 0.0) or 0.0),
            score_contains_gt=float(scores.get("score_contains_gt", 0.0) or 0.0),
            score_gt_in_response=float(scores.get("score_gt_in_response", 0.0) or 0.0),
            score_numeric_match=scores.get("score_numeric_match"),
            score_mc_letter_match=scores.get("score_mc_letter_match"),
            pred_answer_letter=scores.get("pred_answer_letter"),
            gt_answer_letter=scores.get("gt_answer_letter"),
            timestamp=datetime.now().isoformat(),
            error=None,
        )
    except Exception as e:
        log(f"ERROR while evaluating sample {sample_id}: {e}", "ERROR")
        return EvaluationResult(
            sample_id=sample_id,
            source_config=str(row.get("source_config", "unknown")),
            router_task=str(row.get("router_task", "unknown")),
            prompt=str(row.get("prompt_raw", ""))[:200],
            ground_truth=str(row.get("ground_truth", "")),
            model_used="error",
            response="",
            cascaded=False,
            draft_accepted=False,
            latency_ms=0.0,
            total_cost=0.0,
            total_tokens=0,
            is_correct=False,
            exact_match=0.0,
            timestamp=datetime.now().isoformat(),
            error=str(e),
        )


async def run_cascade_eval(
    test_df: pd.DataFrame,
    agent: CascadeAgent,
    cascade_configs: List[ModelConfig],
    cost_tracker: CostTracker,
    metrics_collector: MetricsCollector,
) -> List[EvaluationResult]:
    """Main evaluation loop over all test samples."""
    results: List[EvaluationResult] = []
    start_time = time.time()

    log(f"Starting evaluation over {len(test_df)} samples...", "INFO")

    for idx, (_, row) in enumerate(test_df.iterrows()):
        if idx % 20 == 0 and idx > 0:
            log(f"Progress: {idx}/{len(test_df)} samples processed", "INFO")
        res = await evaluate_sample(row, agent, cascade_configs, cost_tracker, metrics_collector)
        results.append(res)

    elapsed = time.time() - start_time
    log(f"Evaluation complete in {elapsed:.2f} seconds", "INFO")
    if len(results) > 0:
        log(f"Average time per sample: {elapsed / len(results):.2f} seconds", "INFO")

    failed = [r for r in results if r.error is not None]
    if failed:
        log(f"Failed samples: {len(failed)}", "WARN")
    else:
        log("No failed samples.", "INFO")

    return results


# ============================================================
# METRICS + SUMMARY
# ============================================================

def compute_summary(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute global + per-model metrics and return a summary dict."""
    success_df = results_df[results_df["error"].isna()].copy()

    overall_accuracy = success_df["is_correct"].mean() if len(success_df) > 0 else 0.0
    total_cost = success_df["total_cost"].sum()
    avg_cost = success_df["total_cost"].mean() if len(success_df) > 0 else 0.0
    avg_latency = success_df["latency_ms"].mean() if len(success_df) > 0 else 0.0

    model_stats = success_df.groupby("model_used").agg(
        n_samples=("sample_id", "count"),
        accuracy=("is_correct", "mean"),
        avg_cost=("total_cost", "mean"),
        total_cost=("total_cost", "sum"),
        avg_latency_ms=("latency_ms", "mean"),
    ).reset_index()

    task_stats = success_df.groupby("router_task").agg(
        n_samples=("sample_id", "count"),
        accuracy=("is_correct", "mean"),
        avg_cost=("total_cost", "mean"),
        total_cost=("total_cost", "sum"),
        avg_latency_ms=("latency_ms", "mean"),
    ).reset_index()

    summary = {
        "n_total": int(len(results_df)),
        "n_success": int(len(success_df)),
        "overall_accuracy": float(overall_accuracy),
        "total_cost": float(total_cost),
        "avg_cost": float(avg_cost),
        "avg_latency_ms": float(avg_latency),
        "per_model": model_stats.to_dict(orient="records"),
        "per_task": task_stats.to_dict(orient="records"),
    }
    return summary, model_stats, task_stats


# ============================================================
# PLOTTING HELPERS
# ============================================================

def plot_accuracy_vs_cost(model_stats: pd.DataFrame, outdir: Path) -> None:
    """Scatter plot of avg cost vs accuracy per model."""
    if model_stats.empty:
        log("Skipping accuracy vs cost plot (no model_stats).", "WARN")
        return
    log("Generating plot: accuracy vs cost per model", "INFO")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=model_stats,
        x="avg_cost",
        y="accuracy",
        s=120,
    )
    for _, row in model_stats.iterrows():
        plt.text(
            row["avg_cost"],
            row["accuracy"],
            str(row["model_used"]),
            fontsize=9,
            ha="left",
            va="bottom",
        )
    plt.xlabel("Average cost per sample (USD)")
    plt.ylabel("Accuracy")
    plt.title("CascadeFlow: Cost vs Accuracy per Model")
    plt.xscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    out_path = outdir / "cost_vs_accuracy_per_model.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    log(f"Saved: {out_path}", "INFO")


def plot_per_model_bars(model_stats: pd.DataFrame, outdir: Path) -> None:
    """Bar plots of per-model accuracy and total cost."""
    if model_stats.empty:
        log("Skipping per-model bar plots (no model_stats).", "WARN")
        return
    log("Generating plot: per-model accuracy and total cost", "INFO")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(data=model_stats, x="model_used", y="accuracy", ax=axes[0])
    axes[0].set_title("Accuracy per Model")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")

    sns.barplot(data=model_stats, x="model_used", y="total_cost", ax=axes[1])
    axes[1].set_title("Total Cost per Model (USD)")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    out_path = outdir / "per_model_accuracy_and_cost.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    log(f"Saved: {out_path}", "INFO")


def plot_task_accuracy(task_stats: pd.DataFrame, outdir: Path) -> None:
    """Bar plot of accuracy per router_task."""
    if task_stats.empty:
        log("Skipping per-task accuracy plot (no task_stats).", "WARN")
        return
    log("Generating plot: accuracy per task", "INFO")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=task_stats, x="router_task", y="accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Task (router_task)")
    plt.tight_layout()
    out_path = outdir / "accuracy_per_task.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    log(f"Saved: {out_path}", "INFO")


def plot_latency_hist(results_df: pd.DataFrame, outdir: Path) -> None:
    """Histogram of per-sample latency."""
    success_df = results_df[results_df["error"].isna()].copy()
    if success_df.empty:
        log("Skipping latency histogram (no successful samples).", "WARN")
        return
    log("Generating plot: latency histogram", "INFO")
    plt.figure(figsize=(8, 5))
    sns.histplot(success_df["latency_ms"], bins=30)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.title("Latency Distribution")
    plt.tight_layout()
    out_path = outdir / "latency_histogram.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    log(f"Saved: {out_path}", "INFO")


def plot_cumulative_cost(results_df: pd.DataFrame, outdir: Path) -> None:
    """Cumulative cost over samples (simulates spend over time)."""
    success_df = results_df[results_df["error"].isna()].copy().sort_values("timestamp")
    if success_df.empty:
        log("Skipping cumulative cost plot (no successful samples).", "WARN")
        return
    log("Generating plot: cumulative cost over evaluation", "INFO")
    success_df["cumulative_cost"] = success_df["total_cost"].cumsum()
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(success_df)), success_df["cumulative_cost"])
    plt.xlabel("Sample index (sorted by time)")
    plt.ylabel("Cumulative cost (USD)")
    plt.title("Cumulative Cost Over Evaluation")
    plt.tight_layout()
    out_path = outdir / "cumulative_cost.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    log(f"Saved: {out_path}", "INFO")


def plot_task_model_heatmap(results_df: pd.DataFrame, outdir: Path) -> None:
    """Heatmap of fraction of samples routed to each model per task."""
    success_df = results_df[results_df["error"].isna()].copy()
    if success_df.empty:
        log("Skipping task–model heatmap (no successful samples).", "WARN")
        return

    log("Generating plot: task–model routing heatmap", "INFO")
    ctab = (
        success_df.groupby(["router_task", "model_used"])
        .size()
        .reset_index(name="count")
    )
    # Normalize by task
    ctab["fraction"] = ctab.groupby("router_task")["count"].transform(
        lambda x: x / x.sum()
    )

    pivot = ctab.pivot(index="router_task", columns="model_used", values="fraction").fillna(0)

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Fraction of Samples per (Task, Model)")
    plt.tight_layout()
    out_path = outdir / "task_model_heatmap.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    log(f"Saved: {out_path}", "INFO")


def generate_all_plots(results_df: pd.DataFrame, model_stats: pd.DataFrame, task_stats: pd.DataFrame, outdir: Path) -> None:
    """Generate the full suite of diagnostic plots."""
    plot_accuracy_vs_cost(model_stats, outdir)
    plot_per_model_bars(model_stats, outdir)
    plot_task_accuracy(task_stats, outdir)
    plot_latency_hist(results_df, outdir)
    plot_cumulative_cost(results_df, outdir)
    plot_task_model_heatmap(results_df, outdir)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    log("=== CascadeFlow Experiment Script ===", "INFO")
    log(f"Project root: {PROJECT_ROOT}", "INFO")
    log(f"Dataset path: {TEST_DATASET_PATH}", "INFO")
    log(f"Output dir:   {OUTPUT_DIR}", "INFO")
    if MAX_SAMPLES is None:
        log("MAX_SAMPLES: running on FULL test set", "INFO")
    else:
        log(f"MAX_SAMPLES: running on {MAX_SAMPLES} samples", "INFO")
    log("------------------------------", "INFO")

    if not TEST_DATASET_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found at {TEST_DATASET_PATH}")

    # Load dataset
    test_df = pd.read_parquet(TEST_DATASET_PATH)
    if MAX_SAMPLES is not None:
        test_df = test_df.sample(n=MAX_SAMPLES, random_state=RANDOM_SEED)

    log(f"Loaded test split: {test_df.shape}", "INFO")
    log(str(test_df.head(3)[["sample_id", "router_task", "source_config", "ground_truth_type"]]), "DEBUG" if LOG_DEBUG else "INFO")

    # Health check
    log("Checking model endpoints...", "INFO")
    health_status: Dict[str, Dict[str, Any]] = {}
    for name, url in MODEL_ENDPOINTS.items():
        status = check_model_health(name, url)
        health_status[name] = status
        if status["healthy"]:
            log(f"{name} ONLINE → models={status['model_ids']}", "INFO")
        else:
            log(f"{name} OFFLINE → {status['error']}", "WARN")

    # Build cascade agent
    agent = build_cascade_agent(health_status)
    cascade_configs = agent.models
    log("Cascade tiers:", "INFO")
    for i, cfg in enumerate(cascade_configs, 1):
        logical = (cfg.metadata or {}).get("logical_name", cfg.name)
        log(f"  Tier {i}: logical={logical}, id={cfg.name}, url={cfg.base_url}", "INFO")

    # Initialize trackers
    cost_tracker = CostTracker(
        budget_limit=BUDGET_LIMIT,
        warn_threshold=WARN_THRESHOLD,
        verbose=True,
    )
    metrics_collector = MetricsCollector()

    # Run async evaluation
    results: List[EvaluationResult] = asyncio.run(
        run_cascade_eval(
            test_df=test_df,
            agent=agent,
            cascade_configs=cascade_configs,
            cost_tracker=cost_tracker,
            metrics_collector=metrics_collector,
        )
    )

    # Build DataFrame + save
    results_df = pd.DataFrame([asdict(r) for r in results])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_csv = OUTPUT_DIR / f"cascade_results_{timestamp}.csv"
    results_json = OUTPUT_DIR / f"cascade_results_{timestamp}.json"
    results_parquet = OUTPUT_DIR / f"cascade_results_{timestamp}.parquet"

    results_df.to_csv(results_csv, index=False)
    results_df.to_json(results_json, orient="records", indent=2)
    results_df.to_parquet(results_parquet, index=False)

    log(f"Saved results CSV:     {results_csv}", "INFO")
    log(f"Saved results JSON:    {results_json}", "INFO")
    log(f"Saved results Parquet: {results_parquet}", "INFO")

    # Compute summary + per-model/per-task stats
    summary, model_stats, task_stats = compute_summary(results_df)

    # Save summary JSON
    summary_path = OUTPUT_DIR / f"cascadeflow_summary_{timestamp}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log(f"Summary JSON saved: {summary_path}", "INFO")

    log("High-level summary:", "INFO")
    log(json.dumps(summary, indent=2), "INFO")

    # Generate plots
    log("Generating diagnostic plots...", "INFO")
    generate_all_plots(results_df, model_stats, task_stats, OUTPUT_DIR)
    log("All plots generated.", "INFO")

    log("=== Done. ===", "INFO")


if __name__ == "__main__":
    main()
