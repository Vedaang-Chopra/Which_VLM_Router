"""
CascadeFlow vs Artemis Router Comparison Experiment
===================================================

This script runs a rigorous side-by-side comparison of two routing strategies:
1. CascadeFlow: A confidence-based sequential chain (Small -> Medium -> Large).
2. Artemis: A predictive router trained to maximize reward (utility).

It executes both strategies on the same test set samples and logs detailed metrics
for accuracy, cost, and latency.

Usage:
    python cascade_vs_artemis.py [--max_samples 100] [--output_dir ./outputs]
"""

import asyncio
import json
import time
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import pandas as pd
import numpy as np
import httpx
import requests

# -----------------------------------------------------------------------------
# Path Setup
# -----------------------------------------------------------------------------
# Add project root to path to allow imports from code_base
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "code_base"))  # For cascadeflow
sys.path.append(str(PROJECT_ROOT / "code_base" / "cascadeflow")) # For evaluation.py
sys.path.append(str(PROJECT_ROOT / "code_base" / "cascadeflow" / "cascadeflow")) # For cascadeflow package inside the repo

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
# CascadeFlow
try:
    from cascadeflow import CascadeAgent, ModelConfig
    from cascadeflow.telemetry import CostTracker, MetricsCollector
    from evaluation import Scorer  # From code_base/cascadeflow/evaluation.py
except ImportError as e:
    print(f"Error importing CascadeFlow modules: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Default Paths
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "outputs"
DEFAULT_TEST_DATASET = PROJECT_ROOT / "dataset_old_copy" / "final_dataset" / "router_lexico" / "router_test_trainer.parquet"
ARTEMIS_API_URL = "http://localhost:8000/v1/chat/completions"

# Model Configuration (Reusable from cf_exp_2.py)
MODEL_ENDPOINTS = {
    "Qwen/Qwen2.5-VL-3B-Instruct":   "http://localhost:8803/v1",
    "Qwen/Qwen2.5-VL-7B-Instruct":   "http://localhost:8802/v1",
    "google/gemma-3-27b-it":         "http://localhost:8800/v1",
    "Qwen/Qwen3-VL-8B-Thinking":     "http://localhost:8801/v1",
    "deepseek-ai/DeepSeek-OCR":      "http://localhost:8804/v1",
}

MODEL_PRICING = {
    "Qwen/Qwen2.5-VL-3B-Instruct":   {"prompt_per_1k": 0.0001, "completion_per_1k": 0.0001},
    "Qwen/Qwen2.5-VL-7B-Instruct":   {"prompt_per_1k": 0.0002, "completion_per_1k": 0.0002},
    "google/gemma-3-27b-it":         {"prompt_per_1k": 0.00009, "completion_per_1k": 0.00016},
    "Qwen/Qwen3-VL-8B-Thinking":     {"prompt_per_1k": 0.00018, "completion_per_1k": 0.0021},
    "deepseek-ai/DeepSeek-OCR":      {"prompt_per_1k": 0.00003, "completion_per_1k": 0.0001},
}

CASCADE_MODEL_ORDER = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "google/gemma-3-27b-it",
    "Qwen/Qwen3-VL-8B-Thinking",
    "deepseek-ai/DeepSeek-OCR",
]

QUALITY_THRESHOLD = 0.7
MAX_TOKENS = 512
TEMPERATURE = 0.0

# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    sample_id: str
    router_task: str
    ground_truth_type: str
    
    # Cascade Results
    c_model: str
    c_latency: float
    c_cost: float
    c_correct: bool
    c_cascaded: bool
    
    # Artemis Results
    a_model: str
    a_latency: float
    a_cost: float
    a_correct: bool
    
    # Metadata
    prompt_tokens: int = 0
    completion_tokens: int = 0  # Approx, can vary between runs
    
# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def compute_cost(model_name: str, p_tok: int, c_tok: int) -> float:
    pricing = MODEL_PRICING.get(model_name)
    if not pricing:
        # Fallback pricing if unknown
        return (p_tok/1000)*0.0002 + (c_tok/1000)*0.0002
    return (p_tok/1000)*pricing["prompt_per_1k"] + (c_tok/1000)*pricing["completion_per_1k"]

def check_endpoints(url_msg="Checking vLLM endpoints..."):
    print(url_msg)
    healthy = {}
    for name, url in MODEL_ENDPOINTS.items():
        try:
            r = requests.get(f"{url.rstrip('/')}/models", timeout=2)
            if r.status_code == 200:
                healthy[name] = True
                print(f"  [OK] {name}")
            else:
                healthy[name] = False
                print(f"  [FAIL] {name} (Status {r.status_code})")
        except Exception:
            healthy[name] = False
            print(f"  [FAIL] {name} (Connection Error)")
    return healthy

def check_artemis_health():
    print(f"Checking Artemis API @ {ARTEMIS_API_URL}...")
    try:
        # Check health endpoint which is at /health usually
        base = ARTEMIS_API_URL.replace("/v1/chat/completions", "/health")
        r = requests.get(base, timeout=5)
        if r.status_code == 200:
            print("  [OK] Artemis API is online.")
            return True
        else:
            print(f"  [FAIL] Artemis API status {r.status_code}")
            return False
    except Exception as e:
        print(f"  [FAIL] Artemis API Connection Error: {e}")
        return False

def build_cascade_agent(healthy_models: Dict[str, bool]) -> "CascadeAgent":
    available = [m for m in CASCADE_MODEL_ORDER if healthy_models.get(m)]
    if not available:
        raise RuntimeError("No healthy models for Cascade!")
    
    configs = []
    for m in available:
        cfg = ModelConfig(
            name=m, 
            provider="vllm",
            base_url=MODEL_ENDPOINTS[m],
            cost=MODEL_PRICING[m]["prompt_per_1k"], 
            quality_threshold=QUALITY_THRESHOLD,
            metadata={"logical_name": m}
        )
        configs.append(cfg)
    
    return CascadeAgent(models=configs)

# -----------------------------------------------------------------------------
# Artemis Client
# -----------------------------------------------------------------------------

class ArtemisClient:
    def __init__(self, url):
        self.url = url
        self.client = httpx.AsyncClient(timeout=120.0) # Long timeout for full chain
        
    async def route_and_infer(self, prompt: str, mode: str = "balanced"):
        payload = {
            "model": "router-auto", # Ignored by router usually, but good practice
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "router_mode": mode
        }
        start = time.time()
        try:
            resp = await self.client.post(self.url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            
            latency = (time.time() - start) * 1000.0
            content = data["choices"][0]["message"]["content"]
            model = data["model"]
            usage = data.get("usage", {})
            router_meta = data.get("router_metadata", {})
            
            # Use total latency from client side, or from metadata if we want just routing?
            # User wants "Artemis" vs "Cascade".
            # Cascade run() includes inference. So Client Side Latency is correct.
            
            return {
                "model": model,
                "response": content,
                "latency": latency,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0)
            }
        except Exception as e:
            print(f"Artemis API Call Failed: {e}")
            return {
                "model": "error",
                "response": "",
                "latency": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0
            }

# -----------------------------------------------------------------------------
# Comparison Logic
# -----------------------------------------------------------------------------

async def evaluate_sample(
    row: pd.Series,
    cascade_agent: "CascadeAgent",
    artemis_client: ArtemisClient,
    semaphore: asyncio.Semaphore
) -> ComparisonResult:
    async with semaphore:
        sample_id = row["sample_id"]
        prompt = row["prompt_raw"]
        gt = row["ground_truth"]
        gt_type = row.get("ground_truth_type", "exact")
        
        # --- Run Cascade ---
        c_start = time.time()
        try:
            c_res = await cascade_agent.run(query=prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
            c_latency = (time.time() - c_start) * 1000.0
            
            c_model = c_res.model_used
            c_response = getattr(c_res, "content", "")
            c_cascaded = getattr(c_res, "cascaded", False)
            
            c_meta = getattr(c_res, "metadata", {}) or {}
            p_tok = c_meta.get("prompt_tokens", 0)
            c_tok = c_meta.get("completion_tokens", 0)
            c_cost = compute_cost(c_model, p_tok, c_tok)
        except Exception as e:
            print(f"Cascade Failed Sample {sample_id}: {e}")
            c_model = "error"
            c_response = ""
            c_cascaded = False
            c_latency = 0.0
            c_cost = 0.0
        
        c_scores = Scorer.compute_all_scores(c_response, gt, gt_type)
        c_correct = bool(c_scores.get("is_correct", False))
        
        # --- Run Artemis ---
        a_res = await artemis_client.route_and_infer(prompt, mode="balanced")
        a_model = a_res["model"]
        a_response = a_res["response"]
        a_latency = a_res["latency"]
        a_p_tok = a_res["prompt_tokens"]
        a_c_tok = a_res["completion_tokens"]
        a_cost = compute_cost(a_model, a_p_tok, a_c_tok)
        
        a_scores = Scorer.compute_all_scores(a_response, gt, gt_type)
        a_correct = bool(a_scores.get("is_correct", False))
        
        return ComparisonResult(
            sample_id=sample_id,
            router_task=row["router_task"],
            ground_truth_type=gt_type,
            c_model=c_model,
            c_latency=c_latency,
            c_cost=c_cost,
            c_correct=c_correct,
            c_cascaded=c_cascaded,
            a_model=a_model,
            a_latency=a_latency,
            a_cost=a_cost,
            a_correct=a_correct,
            prompt_tokens=a_p_tok, # Using Artemis tokens as ref
            completion_tokens=a_c_tok
        )

# -----------------------------------------------------------------------------
# Mock Implementations
# -----------------------------------------------------------------------------

@dataclass
class MockCascadeResult:
    model_used: str = "mock_model"
    content: str = "This is a mock response from Cascade."
    cascaded: bool = False
    metadata: Dict = None
    cost: float = 0.0001
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {"prompt_tokens": 10, "completion_tokens": 10}

class MockAgent:
    async def run(self, query: str, max_tokens: int = 512, temperature: float = 0.0):
        # Simulate slight delay
        await asyncio.sleep(0.1)
        return MockCascadeResult()

class MockArtemisClient:
    async def route_and_infer(self, prompt: str, mode: str = "balanced"):
        await asyncio.sleep(0.1)
        return {
            "model": "mock_artemis_model",
            "response": "This is a mock response from Artemis.",
            "latency": 100.0,
            "prompt_tokens": 15,
            "completion_tokens": 15
        }

# -----------------------------------------------------------------------------
# Main Runner
# -----------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--offset", type=int, default=0, help="Start index")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--mock", action="store_true", help="Run with mock models (no vLLM/API needed)")
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Starting Comparison Experiment (Mock={args.mock}) ===")
    print(f"Output Dir: {out_dir}")
    
    if args.mock:
        print("WARNING: Running in MOCK mode. Results are simulated.")
        cascade_agent = MockAgent()
        artemis_client = MockArtemisClient()
    else:
        # 1. Check Health (vLLM)
        healthy = check_endpoints()
        
        # 2. Check Health (Artemis API)
        if not check_artemis_health():
            print("ERROR: Artemis API is not reachable. Please start 'artemis_final.system_api.main' in another terminal.")
            print("Required to run comparison.")
            sys.exit(1)
                
        # 3. Init Cascade
        try:
            cascade_agent = build_cascade_agent(healthy)
            print("Cascade Agent Initialized.")
        except Exception as e:
            print(f"Failed to init Cascade Agent: {e}")
            sys.exit(1)
            
        artemis_client = ArtemisClient(ARTEMIS_API_URL)
    
    # 4. Load Data
    print(f"Loading dataset from {DEFAULT_TEST_DATASET}...")
    try:
        df = pd.read_parquet(DEFAULT_TEST_DATASET)
        if args.max_samples:
            df = df.iloc[args.offset : args.offset + args.max_samples]
        print(f"Running on {len(df)} samples...")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)
    
    # 5. Run Loop
    semaphore = asyncio.Semaphore(5) # Concurrency limit (lower to be safe)
    tasks = []
    
    for _, row in df.iterrows():
        # Pass mock or real agents
        tasks.append(evaluate_sample(row, cascade_agent, artemis_client, semaphore))
    
    results = []
    # Use tqdm if possible
    try:
        from tqdm.asyncio import tqdm
        for f in tqdm.as_completed(tasks):
            results.append(await f)
    except ImportError:
        for i, f in enumerate(asyncio.as_completed(tasks)):
            if i % 10 == 0: print(f"Processing {i}/{len(tasks)}...")
            results.append(await f)
            
    # 6. Save Results
    res_df = pd.DataFrame([asdict(r) for r in results])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    csv_path = out_dir / f"comparison_results_{timestamp}.csv"
    res_df.to_csv(csv_path, index=False)
    
    # 7. Print Summary
    print("\n=== Final Summary ===")
    print(f"Total Samples: {len(res_df)}")
    
    if len(res_df) > 0:
        # Cascade Stats
        c_acc = res_df["c_correct"].mean() * 100
        c_cost = res_df["c_cost"].sum()
        c_lat = res_df["c_latency"].mean()
        print(f"[Cascade] Accuracy: {c_acc:.2f}% | Total Cost: ${c_cost:.4f} | Avg Latency: {c_lat:.1f}ms")
        
        # Artemis Stats
        a_acc = res_df["a_correct"].mean() * 100
        a_cost = res_df["a_cost"].sum()
        a_lat = res_df["a_latency"].mean()
        print(f"[Artemis] Accuracy: {a_acc:.2f}% | Total Cost: ${a_cost:.4f} | Avg Latency: {a_lat:.1f}ms")
        
        # Delta
        print(f"--- Delta (Artemis - Cascade) ---")
        print(f"Accuracy: {a_acc - c_acc:.2f}%")
        print(f"Cost: ${a_cost - c_cost:.4f}")
        print(f"Latency: {a_lat - c_lat:.1f}ms")
    else:
        print("No results generated.")

if __name__ == "__main__":
    asyncio.run(main())
