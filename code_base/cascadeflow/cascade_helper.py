# 03 · CascadeFlow VLM Experiment (Sync Loop)

# This notebook runs a **CascadeFlow experiment** on your VLM test dataset, using a
# single cell with a synchronous-looking loop over `agent.run(...)` (wrapped once
# with `asyncio.run` to avoid Jupyter async/await issues).

# Steps:

# 1. Load the **test dataset** (`router_test_final.parquet` with images).
# 2. Configure **model pricing** and **vLLM endpoints**.
# 3. Build a **CascadeAgent** over your VLMs, sorted cheapest → most expensive.
# 4. Iterate over all rows in one cell, calling `agent.run(...)` per sample.
# 5. Evaluate with `Scorer` from `evaluation.py`.
# 6. Compute true **dollar cost** from tokens.
# 7. Save per-sample results to `outputs/cascadeflow/cascadeflow_results.parquet`.

import os
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

from cascadeflow import CascadeAgent, ModelConfig
from evaluation import Scorer  # ensure evaluation.py is importable

DATA_ROOT = Path.cwd().parent.parent / "dataset"

TEST_DATASET_PATH = DATA_ROOT  / "final_dataset" / "router_lexico" / "router_test_trainer.parquet"

OUTPUT_DIR = Path.cwd() / "outputs" / "cascadeflow"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_DATASET_PATH, OUTPUT_DIR

## 1 · Configure model pool, pricing, and endpoints

# Edit **MODEL_PRICING** and **MODEL_ENDPOINTS** to match your cluster.
# Costs are in USD per 1k tokens.

# --- Model pricing (USD per 1K tokens) ---
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "Qwen/Qwen2.5-VL-3B-Instruct":   {"prompt_per_1k": 0.0002, "completion_per_1k": 0.0002},
    "Qwen/Qwen2.5-VL-7B-Instruct":   {"prompt_per_1k": 0.0003, "completion_per_1k": 0.0003},
    "google/gemma-3-27b-it":         {"prompt_per_1k": 0.0008, "completion_per_1k": 0.0008},
    "Qwen/Qwen3-VL-8B-Thinking":     {"prompt_per_1k": 0.0010, "completion_per_1k": 0.0010},
    "deepseek-ai/DeepSeek-OCR":      {"prompt_per_1k": 0.0004, "completion_per_1k": 0.0004},
}

# --- vLLM endpoints (EDIT ports / hosts as needed) ---
MODEL_ENDPOINTS: Dict[str, str] = {
    "Qwen/Qwen2.5-VL-3B-Instruct":   "http://localhost:8801/v1",
    "Qwen/Qwen2.5-VL-7B-Instruct":   "http://localhost:8802/v1",
    "google/gemma-3-27b-it":         "http://localhost:8803/v1",
    "Qwen/Qwen3-VL-8B-Thinking":     "http://localhost:8804/v1",
    "deepseek-ai/DeepSeek-OCR":      "http://localhost:8805/v1",
}

def compute_true_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model_name)
    if pricing is None:
        return 0.0
    return (
        (prompt_tokens / 1000.0) * pricing["prompt_per_1k"]
        + (completion_tokens / 1000.0) * pricing["completion_per_1k"]
    )

### 1.1 · Health check vLLM endpoints

# Ping `/models` on each endpoint to verify availability.

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

health_status: Dict[str, Dict[str, Any]] = {}
for model_name, url in MODEL_ENDPOINTS.items():
    info = check_vllm_health(url)
    health_status[model_name] = info
    print(f"{model_name}: ok={info['ok']}, ids={info['model_ids']}, error={info['error']}")

### 1.2 · Build CascadeFlow model configs (cheap → expensive)

# Keep only healthy models, sort by cost, and create `ModelConfig`s.

available_models = [
    m for m, info in health_status.items()
    if info["ok"] and m in MODEL_PRICING
]

if not available_models:
    raise RuntimeError("No healthy models found. Check MODEL_ENDPOINTS or vLLM servers.")

available_models = sorted(
    available_models,
    key=lambda m: MODEL_PRICING[m]["prompt_per_1k"],
)

print("Cascade order (cheapest → most expensive):")
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
        cost=MODEL_PRICING[m]["prompt_per_1k"],  # relative cost for CascadeFlow
        quality_threshold=0.75,                  # EDIT thresholds if desired
        metadata={"logical_name": m},
    )
    cascade_models.append(cfg)

cascade_models

agent = CascadeAgent(models=cascade_models)
agent

## 2 · Load test dataset

# We load your **final test dataset**. It must contain:

# - `sample_id`
# - `prompt_raw`
# - `ground_truth`
# - `ground_truth_type`
# - `router_task`
# - `source_config`
# - `image_path`

assert TEST_DATASET_PATH.exists(), f"Test dataset not found: {TEST_DATASET_PATH}"
test_df = pd.read_parquet(TEST_DATASET_PATH)

print("Test shape:", test_df.shape)
print("Columns:", test_df.columns.tolist())
test_df[["sample_id", "router_task", "source_config"]].head()

# Optional: subsample for quicker debugging
MAX_SAMPLES = 500  # e.g., 500; set to None for full test
RANDOM_SEED = 42

if MAX_SAMPLES is not None and len(test_df) > MAX_SAMPLES:
    test_df = test_df.sample(n=MAX_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)

len(test_df)

## 3 · Helper: build VLM messages (prompt + image)

# We use an OpenAI-style message format with `text` + `image_url` content.
# Adjust `image_url` if your vLLM server expects something else.

from typing import Any

def build_vlm_messages(prompt: str, image_path: Optional[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    content.append({"type": "text", "text": prompt})

    if image_path is not None and isinstance(image_path, str) and image_path.strip():
        image_url = f"file://{image_path}"
        content.append({"type": "image", "image_url": image_url})

    return [{"role": "user", "content": content}]

## 4 · Result schema

# We use a dataclass for clear per-sample records.

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

## 5 · Main experiment loop (single cell with asyncio.run)

# This cell contains the **only loop** that calls `agent.run(...)`. We wrap that loop
# in a small async function `_run_all()` and then call `asyncio.run(_run_all())` once.

# All logic and the for-loop live in this cell to avoid scattered async usage.

import asyncio

results: List[CascadeEvalResult] = []

async def _run_all():
    for idx, row in test_df.iterrows():
        if idx % 20 == 0:
            print(f"Evaluating {idx+1}/{len(test_df)} ...")

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

            logical_name: str = used_model
            for cfg in agent.models:
                if cfg.name == used_model:
                    logical_name = (cfg.metadata or {}).get("logical_name", used_model)
                    break

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

    return results

# Run the async loop ONCE
results = asyncio.run(_run_all())

len(results)

## 6 · Save per-sample results

results_df = pd.DataFrame([asdict(r) for r in results])
print("Results shape:", results_df.shape)
results_df.head()

parquet_path = OUTPUT_DIR / "cascadeflow_results.parquet"
results_df.to_parquet(parquet_path, index=False)
print("Saved per-sample results to:", parquet_path)

## 7 · Summary metrics and quick plots

success_df = results_df[results_df["error"].isna()] if "error" in results_df.columns else results_df.copy()

print("Total samples:", len(results_df))
print("Successful samples:", len(success_df))

overall_accuracy = success_df["is_correct"].mean() if len(success_df) > 0 else 0.0
total_cost = success_df["cascade_cost"].sum()
avg_cost = success_df["cascade_cost"].mean() if len(success_df) > 0 else 0.0
avg_latency = success_df["cascade_latency_ms"].mean() if len(success_df) > 0 else 0.0

print("Overall accuracy:", overall_accuracy)
print("Total cost:", total_cost)
print("Avg cost per sample:", avg_cost)
print("Avg latency (ms):", avg_latency)

model_stats = success_df.groupby("cascade_logical_model").agg(
    n_samples=("sample_id", "count"),
    avg_cost=("cascade_cost", "mean"),
    total_cost=("cascade_cost", "sum"),
    avg_latency_ms=("cascade_latency_ms", "mean"),
    accuracy=("is_correct", "mean"),
).reset_index()

model_stats

plt.figure(figsize=(8, 4))
plt.bar(model_stats["cascade_logical_model"], model_stats["avg_cost"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Avg cost per sample (USD)")
plt.title("CascadeFlow: avg cost per model")
plt.tight_layout()
plt.show()

task_stats = success_df.groupby("router_task").agg(
    n_samples=("sample_id", "count"),
    accuracy=("is_correct", "mean"),
    avg_cost=("cascade_cost", "mean"),
    total_cost=("cascade_cost", "sum"),
    avg_latency_ms=("cascade_latency_ms", "mean"),
).reset_index()

task_stats

plt.figure(figsize=(8, 4))
plt.bar(task_stats["router_task"], task_stats["accuracy"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Accuracy")
plt.title("CascadeFlow accuracy per task")
plt.tight_layout()
plt.show()

## 8 · Save summary JSON

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

summary_path = OUTPUT_DIR / "cascadeflow_summary_sync.json"
with summary_path.open("w") as f:
    import json
    json.dump(summary, f, indent=2)

summary_path
