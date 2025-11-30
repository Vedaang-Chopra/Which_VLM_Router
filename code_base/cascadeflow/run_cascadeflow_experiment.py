#!/usr/bin/env python
# CascadeFlow VLM Experiment Script
# ---------------------------------
# Runs CascadeFlow on your VLM test dataset, evaluates answers, computes
# true cascade cost (summing all models used in the cascade), and writes:
#
#   - outputs/cascadeflow/cascadeflow_results.parquet
#   - outputs/cascadeflow/cascadeflow_summary.json
#   - outputs/cascadeflow/plot_cost_per_model.png
#   - outputs/cascadeflow/plot_accuracy_per_task.png
#   - outputs/cascadeflow/plot_cost_vs_accuracy_per_task.png
#
# Assumptions:
#   - You have a test parquet with columns:
#       sample_id, prompt_raw, ground_truth, ground_truth_type,
#       router_task, source_config, image_path
#   - evaluation.py is importable and exposes Scorer.compute_all_scores
#   - vLLM servers for each model are already running and reachable.
#
# Usage:
#   python run_cascadeflow_experiment.py \\
#     --dataset dataset/final_dataset/router_final/router_test_final.parquet

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
import pandas as pd
import requests

from cascadeflow import CascadeAgent, ModelConfig
from evaluation import Scorer  # make sure evaluation.py is on PYTHONPATH


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default paths (can be overridden via CLI)
DATA_ROOT = Path.cwd().parent.parent / "dataset"

DEFAULT_DATASET_PATH = DATA_ROOT  / "final_dataset" / "router_lexico" / "router_test_trainer.parquet"
print(f"Default dataset path: {DEFAULT_DATASET_PATH}")
OUTPUT_DIR = Path.cwd().parent.parent / "outputs" / "cascadeflow"

# Model pricing in USD per 1K tokens (edit for your setup)
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # "Qwen/Qwen2.5-VL-3B-Instruct":   {"prompt_per_1k": 0.0001, "completion_per_1k": 0.0001},
    # "Qwen/Qwen2.5-VL-7B-Instruct":   {"prompt_per_1k": 0.0002, "completion_per_1k": 0.0002},
    # "google/gemma-3-27b-it":         {"prompt_per_1k": 0.00007, "completion_per_1k": 0.00050},
    # "Qwen/Qwen3-VL-8B-Thinking":     {"prompt_per_1k": 0.00018, "completion_per_1k": 0.0021},
    # "deepseek-ai/DeepSeek-OCR":      {"prompt_per_1k": 0.00003, "completion_per_1k": 0.0001},
    
    "Qwen/Qwen2.5-VL-3B-Instruct":   {"prompt_per_1k": 10, "completion_per_1k": 100},
    "Qwen/Qwen2.5-VL-7B-Instruct":   {"prompt_per_1k": 20, "completion_per_1k": 200},
    "google/gemma-3-27b-it":         {"prompt_per_1k": 30, "completion_per_1k": 300},
    "Qwen/Qwen3-VL-8B-Thinking":     {"prompt_per_1k": 50, "completion_per_1k": 500},
    "deepseek-ai/DeepSeek-OCR":      {"prompt_per_1k": 15, "completion_per_1k": 150},
}

# vLLM endpoints (edit ports/hosts to match your deployment)
MODEL_ENDPOINTS: Dict[str, str] = {
    "Qwen/Qwen2.5-VL-3B-Instruct":   "http://localhost:8804/v1",
    "Qwen/Qwen2.5-VL-7B-Instruct":   "http://localhost:8803/v1",
    "google/gemma-3-27b-it":         "http://localhost:8800/v1",
    "Qwen/Qwen3-VL-8B-Thinking":     "http://localhost:8801/v1",
    "deepseek-ai/DeepSeek-OCR":      "http://localhost:8804/v1",
}


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------

def compute_true_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute dollar cost for a single model from token counts."""
    pricing = MODEL_PRICING.get(model_name)
    if pricing is None:
        return 0.0
    return (
        (prompt_tokens / 1000.0) * pricing["prompt_per_1k"]
        + (completion_tokens / 1000.0) * pricing["completion_per_1k"]
    )


def compute_cascade_cost_from_metadata(md: Dict[str, Any]) -> float:
    """
    Compute total cascade cost by summing costs of *all* models used.

    We look for per-step info in metadata in a robust way, then fall back
    to final model-only cost if detailed info isn't available.

    Expected metadata patterns (examples, adjust to your actual cascadeflow version):
      - md["steps"]: List[{"model": ..., "prompt_tokens": ..., "completion_tokens": ...}, ...]
      - or md["trace"]["steps"]
      - or md["cost_breakdown"]
    """
    total = 0.0

    steps = md.get("steps") or md.get("trace", {}).get("steps") or md.get("cost_breakdown", [])

    if isinstance(steps, list) and steps:
        for step in steps:
            if not isinstance(step, dict):
                continue
            # Try multiple keys for model name
            mname = (
                step.get("logical_name")
                or step.get("model_name")
                or step.get("model")
                or step.get("model_id")
            )
            if not mname:
                continue
            ptok = int(step.get("prompt_tokens", step.get("input_tokens", 0)) or 0)
            ctok = int(step.get("completion_tokens", step.get("output_tokens", 0)) or 0)
            total += compute_true_cost(mname, ptok, ctok)

    # Fallback: if we didn't find any per-step info, caller can fall back
    # to computing only on final model tokens.
    return total


# ---------------------------------------------------------------------------
# HTTP + CascadeFlow setup
# ---------------------------------------------------------------------------

def check_vllm_health(base_url: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {"base_url": base_url, "ok": False, "model_ids": [], "error": None}
    try:
        resp = requests.get(base_url.rstrip("/") + "/models", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        ids = [m.get("id") for m in data.get("data", [])]
        info["ok"] = True
        info["model_ids"] = ids
    except Exception as e:
        info["error"] = str(e)
    return info


def build_cascade_agent() -> CascadeAgent:
    """Build a CascadeAgent over available vLLM models, sorted by cost."""
    health_status: Dict[str, Dict[str, Any]] = {}
    for model_name, url in MODEL_ENDPOINTS.items():
        info = check_vllm_health(url)
        health_status[model_name] = info
        print(f"{model_name}: ok={info['ok']}, ids={info['model_ids']}, error={info['error']}")

    available_models = [
        m for m, info in health_status.items()
        if info["ok"] and m in MODEL_PRICING
    ]

    if not available_models:
        raise RuntimeError("No healthy models found. Check MODEL_ENDPOINTS or vLLM servers.")

    # Sort by cost (cheapest → most expensive)
    available_models = sorted(
        available_models,
        key=lambda m: MODEL_PRICING[m]["prompt_per_1k"],
    )

    print("\nCascade order (cheapest → most expensive):")
    for m in available_models:
        print("  -", m, MODEL_PRICING[m])

    cascade_models: List[ModelConfig] = []
    for m in available_models:
        info = health_status[m]
        base_url = info["base_url"]
        model_id = info["model_ids"][0] if info["model_ids"] else m

        cfg = ModelConfig(
            name=model_id,
            provider="vllm",
            base_url=base_url,
            cost=MODEL_PRICING[m]["prompt_per_1k"],
            quality_threshold=0.75,  # adjust if desired
            metadata={"logical_name": m},
        )
        cascade_models.append(cfg)

    agent = CascadeAgent(models=cascade_models)
    return agent


# ---------------------------------------------------------------------------
# Dataset + messages
# ---------------------------------------------------------------------------

def build_vlm_messages(prompt: str, image_path: Optional[str]) -> List[Dict[str, Any]]:
    """Build OpenAI-style messages with text + image (file:// URL)."""
    content: List[Dict[str, Any]] = []
    content.append({"type": "text", "text": prompt})

    if image_path is not None and isinstance(image_path, str) and image_path.strip():
        image_url = f"file://{image_path}"
        content.append({"type": "image", "image_url": image_url})

    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

@dataclass
class CascadeEvalResult:
    sample_id: str
    source_config: str
    router_task: str

    prompt_raw: str
    ground_truth: str
    ground_truth_type: str
    image_path: Optional[str]

    cascade_model: str
    cascade_logical_model: str
    cascaded: bool
    draft_accepted: Optional[bool]

    response_raw: str

    prompt_tokens: int
    completion_tokens: int
    cascade_cost: float
    cascade_latency_ms: float

    # Scoring
    is_correct: bool
    score_f1: float
    score_exact_match: float
    score_exact_match_normalized: float
    score_contains_gt: float
    score_gt_in_response: float
    score_numeric_match: Optional[float]
    score_mc_letter_match: Optional[float]
    pred_answer_letter: Optional[str]
    gt_answer_letter: Optional[str]

    # Error
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Main async experiment
# ---------------------------------------------------------------------------

async def run_experiment(dataset_path: Path, max_samples: Optional[int] = None) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    assert dataset_path.exists(), f"Dataset not found: {dataset_path}"
    df = pd.read_parquet(dataset_path)
    print(f"Loaded dataset: {dataset_path}  shape={df.shape}")

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"Subsampled to {len(df)} rows for this run")

    agent = build_cascade_agent()

    results: List[CascadeEvalResult] = []

    for idx, row in df.iterrows():
        if idx % 20 == 0:
            print(f"Evaluating {idx+1}/{len(df)} ...")

        sample_id = str(row.get("sample_id", ""))
        router_task = str(row.get("router_task", "unknown"))
        source_config = str(row.get("source_config", "unknown"))
        prompt_raw = str(row.get("prompt_raw", ""))
        ground_truth = str(row.get("ground_truth", ""))
        gt_type = str(row.get("ground_truth_type", "exact")) or "exact"
        image_path = row.get("image_path", None)

        # messages = build_vlm_messages(prompt_raw, image_path)

        try:
            t0 = time.time()
            result = await agent.run(query=prompt_raw, max_tokens=512, temperature=0.0)
            t1 = time.time()

            response = getattr(result, "content", "") or ""
            cascade_latency_ms = getattr(result, "latency_ms", (t1 - t0) * 1000.0)
            cascaded = getattr(result, "cascaded", False)
            draft_accepted = getattr(result, "draft_accepted", None)
            used_model = getattr(result, "model_used", "") or ""

            md = getattr(result, "metadata", {}) or {}
            prompt_tokens = int(md.get("prompt_tokens", md.get("input_tokens", 0)) or 0)
            completion_tokens = int(md.get("completion_tokens", md.get("output_tokens", 0)) or 0)

            # Figure out logical name for final model
            logical_name: str = used_model
            for cfg in agent.models:
                if cfg.name == used_model:
                    logical_name = (cfg.metadata or {}).get("logical_name", used_model)
                    break

            # Cost: first try per-step cascade cost, then fallback to final model only
            cascade_cost = compute_cascade_cost_from_metadata(md)
            if cascade_cost == 0.0:
                cascade_cost = compute_true_cost(logical_name, prompt_tokens, completion_tokens)

            scores = Scorer.compute_all_scores(pred=response, gt=ground_truth, gt_type=gt_type)

            rec = CascadeEvalResult(
                sample_id=sample_id,
                source_config=source_config,
                router_task=router_task,
                prompt_raw=prompt_raw,
                ground_truth=ground_truth,
                ground_truth_type=gt_type,
                image_path=image_path if isinstance(image_path, str) else None,
                cascade_model=used_model,
                cascade_logical_model=logical_name,
                cascaded=cascaded,
                draft_accepted=draft_accepted,
                response_raw=response,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cascade_cost=cascade_cost,
                cascade_latency_ms=cascade_latency_ms,
                is_correct=bool(scores.get("is_correct", False)),
                score_f1=float(scores.get("score_f1", 0.0) or 0.0),
                score_exact_match=float(scores.get("score_exact_match", 0.0) or 0.0),
                score_exact_match_normalized=float(scores.get("score_exact_match_normalized", 0.0) or 0.0),
                score_contains_gt=float(scores.get("score_contains_gt", 0.0) or 0.0),
                score_gt_in_response=float(scores.get("score_gt_in_response", 0.0) or 0.0),
                score_numeric_match=scores.get("score_numeric_match"),
                score_mc_letter_match=scores.get("score_mc_letter_match"),
                pred_answer_letter=scores.get("pred_answer_letter"),
                gt_answer_letter=scores.get("gt_answer_letter"),
                error=None,
            )
        except Exception as e:
            rec = CascadeEvalResult(
                sample_id=sample_id,
                source_config=source_config,
                router_task=router_task,
                prompt_raw=prompt_raw,
                ground_truth=ground_truth,
                ground_truth_type=gt_type,
                image_path=image_path if isinstance(image_path, str) else None,
                cascade_model="",
                cascade_logical_model="",
                cascaded=False,
                draft_accepted=None,
                response_raw="",
                prompt_tokens=0,
                completion_tokens=0,
                cascade_cost=0.0,
                cascade_latency_ms=0.0,
                is_correct=False,
                score_f1=0.0,
                score_exact_match=0.0,
                score_exact_match_normalized=0.0,
                score_contains_gt=0.0,
                score_gt_in_response=0.0,
                score_numeric_match=None,
                score_mc_letter_match=None,
                pred_answer_letter=None,
                gt_answer_letter=None,
                error=str(e),
            )

        results.append(rec)

    results_df = pd.DataFrame([asdict(r) for r in results])
    return results_df


# ---------------------------------------------------------------------------
# Plotting + summary helpers
# ---------------------------------------------------------------------------

def save_plots_and_summary(results_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success_df = results_df[results_df["error"].isna()] if "error" in results_df.columns else results_df.copy()

    n_total = int(len(results_df))
    n_success = int(len(success_df))
    overall_accuracy = float(success_df["is_correct"].mean() if len(success_df) > 0 else 0.0)
    total_cost = float(success_df["cascade_cost"].sum())
    avg_cost = float(success_df["cascade_cost"].mean() if len(success_df) > 0 else 0.0)
    avg_latency = float(success_df["cascade_latency_ms"].mean() if len(success_df) > 0 else 0.0)

    print(f"Total samples:     {n_total}")
    print(f"Successful samples:{n_success}")
    print(f"Overall accuracy:  {overall_accuracy:.4f}")
    print(f"Total cost:        {total_cost:.6f} USD")
    print(f"Avg cost / sample: {avg_cost:.6f} USD")
    print(f"Avg latency:       {avg_latency:.2f} ms")

    # Per-model stats
    if "cascade_logical_model" in success_df.columns:
        model_stats = success_df.groupby("cascade_logical_model").agg(
            n_samples=("sample_id", "count"),
            avg_cost=("cascade_cost", "mean"),
            total_cost=("cascade_cost", "sum"),
            avg_latency_ms=("cascade_latency_ms", "mean"),
            accuracy=("is_correct", "mean"),
        ).reset_index()
    else:
        model_stats = pd.DataFrame()

    # Per-task stats
    if "router_task" in success_df.columns:
        task_stats = success_df.groupby("router_task").agg(
            n_samples=("sample_id", "count"),
            accuracy=("is_correct", "mean"),
            avg_cost=("cascade_cost", "mean"),
            total_cost=("cascade_cost", "sum"),
            avg_latency_ms=("cascade_latency_ms", "mean"),
        ).reset_index()
    else:
        task_stats = pd.DataFrame()

    # Save summary JSON
    summary = {
        "n_total": n_total,
        "n_success": n_success,
        "overall_accuracy": overall_accuracy,
        "total_cost": total_cost,
        "avg_cost": avg_cost,
        "avg_latency_ms": avg_latency,
        "per_model": model_stats.to_dict(orient="records"),
        "per_task": task_stats.to_dict(orient="records"),
    }
    summary_path = OUTPUT_DIR / "cascadeflow_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON: {summary_path}")

    # Save results parquet
    parquet_path = OUTPUT_DIR / "cascadeflow_results.parquet"
    results_df.to_parquet(parquet_path, index=False)
    print(f"Saved per-sample results: {parquet_path}")

    # Plots
    if not model_stats.empty:
        plt.figure(figsize=(8, 4))
        plt.bar(model_stats["cascade_logical_model"], model_stats["avg_cost"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Avg cost per sample (USD)")
        plt.title("CascadeFlow: avg cost per model")
        plt.tight_layout()
        plot_path = OUTPUT_DIR / "plot_cost_per_model.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")

    if not task_stats.empty:
        plt.figure(figsize=(8, 4))
        plt.bar(task_stats["router_task"], task_stats["accuracy"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Accuracy")
        plt.title("CascadeFlow accuracy per task")
        plt.tight_layout()
        plot_path = OUTPUT_DIR / "plot_accuracy_per_task.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")

        # Cost vs accuracy per task
        plt.figure(figsize=(6, 4))
        plt.scatter(task_stats["avg_cost"], task_stats["accuracy"])
        for _, row in task_stats.iterrows():
            plt.text(row["avg_cost"], row["accuracy"], row["router_task"], fontsize=8)
        plt.xlabel("Avg cost per sample (USD)")
        plt.ylabel("Accuracy")
        plt.title("CascadeFlow: cost vs accuracy per task")
        plt.tight_layout()
        plot_path = OUTPUT_DIR / "plot_cost_vs_accuracy_per_task.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CascadeFlow VLM experiment on a dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET_PATH.resolve()),
        help="Path to test parquet dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional max number of samples to run (for quick debugging).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Arguments: {args}")
    dataset_path = Path(args.dataset)

    print(f"Using dataset: {dataset_path}")
    print(f"Output directory: {OUTPUT_DIR}")

    results_df = asyncio.run(run_experiment(DEFAULT_DATASET_PATH, max_samples=args.max_samples))
    save_plots_and_summary(results_df)


if __name__ == "__main__":
    main()
