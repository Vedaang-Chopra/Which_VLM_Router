#!/usr/bin/env python3
"""
Demo Full Pipeline Script

Demonstrates the complete Artemis VLM Router system:
1. Load samples from database (or use synthetic)
2. Run router predictions
3. Apply load balancer scheduling
4. Show metrics and analysis

Usage:
    python scripts/demo_full_pipeline.py
    python scripts/demo_full_pipeline.py --num-samples 100 --mode balanced
"""

import sys
import os
import time
import argparse
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("demo")


@dataclass
class DemoConfig:
    """Configuration for the demo."""
    num_samples: int = 50
    router_mode: str = "balanced"
    device: str = "cpu"
    use_database: bool = True
    checkpoint_name: str = "best_reward_router.pt"
    show_details: bool = True


@dataclass
class DemoResult:
    """Result from a single routing decision."""
    sample_id: str
    prompt_snippet: str
    task_type: str
    chosen_model: str
    rewards: Dict[str, float]
    confidence: float
    fallback_triggered: bool
    router_latency_ms: float


def load_samples_from_db(limit: int = 50) -> List[Dict[str, Any]]:
    """Load samples from PostgreSQL database."""
    try:
        from artemis_final.ares.configs.db_config import DB_URL, TABLES
        from sqlalchemy import create_engine, text
        
        engine = create_engine(DB_URL)
        
        query = f"""
            SELECT 
                sample_id, prompt_text, router_task, source_dataset
            FROM {TABLES['samples']}
            WHERE prompt_text IS NOT NULL
            ORDER BY RANDOM()
            LIMIT :limit
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(query), {"limit": limit})
            samples = []
            for row in result.fetchall():
                samples.append({
                    "sample_id": row[0],
                    "prompt": row[1],
                    "router_task": row[2] or "vqa",
                    "source_dataset": row[3] or "unknown"
                })
            return samples
    except Exception as e:
        logger.warning(f"Failed to load from database: {e}")
        return []


def generate_synthetic_samples(num: int = 50) -> List[Dict[str, Any]]:
    """Generate synthetic samples for demo."""
    tasks = ["vqa", "ocr", "chartqa", "diagram_reasoning", "document_qa"]
    datasets = ["ai2d", "docvqa", "chartqa", "textvqa", "infographicvqa"]
    
    prompts = [
        "What is the main subject of this image?",
        "Extract all visible text from this document.",
        "What is the value shown in the bar chart for 2023?",
        "Explain the relationship shown in this diagram.",
        "What is the total amount on this receipt?",
        "Describe the process illustrated in this flowchart.",
        "What color is the object in the center?",
        "How many items are listed in this table?",
        "What does the legend in this chart represent?",
        "Summarize the key information in this infographic.",
    ]
    
    samples = []
    for i in range(num):
        samples.append({
            "sample_id": f"synthetic_{i:04d}",
            "prompt": prompts[i % len(prompts)],
            "router_task": tasks[i % len(tasks)],
            "source_dataset": datasets[i % len(datasets)],
        })
    return samples


def compute_confidence(rewards: Dict[str, float]) -> float:
    """Compute confidence from rewards."""
    if not rewards:
        return 0.0
    
    sorted_rewards = sorted(rewards.values(), reverse=True)
    if len(sorted_rewards) < 2:
        return 1.0
    
    gap = sorted_rewards[0] - sorted_rewards[1]
    return min(gap / 0.2, 1.0)


def run_demo(config: DemoConfig):
    """Run the full demo pipeline."""
    
    print("\n" + "="*70)
    print("🚀 ARTEMIS VLM ROUTER - FULL PIPELINE DEMO")
    print("="*70)
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Load/Generate Samples
    # ─────────────────────────────────────────────────────────────────────
    print("\n📦 Step 1: Loading samples...")
    
    if config.use_database:
        samples = load_samples_from_db(config.num_samples)
        if samples:
            print(f"   ✓ Loaded {len(samples)} samples from PostgreSQL")
        else:
            print("   ⚠ Database unavailable, using synthetic samples")
            samples = generate_synthetic_samples(config.num_samples)
    else:
        samples = generate_synthetic_samples(config.num_samples)
        print(f"   ✓ Generated {len(samples)} synthetic samples")
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Initialize Router
    # ─────────────────────────────────────────────────────────────────────
    print("\n🧠 Step 2: Initializing router...")
    
    checkpoint_path = PROJECT_ROOT / "checkpoints" / config.checkpoint_name
    
    if not checkpoint_path.exists():
        print(f"   ⚠ Checkpoint not found: {checkpoint_path}")
        print("   Using mock router for demo")
        router = None
    else:
        try:
            from artemis_final.router.artemis_router import RewardRouterInference
            router = RewardRouterInference(
                checkpoint_path=str(checkpoint_path),
                device=config.device,
                verbose=False
            )
            print(f"   ✓ Loaded router from {config.checkpoint_name}")
            print(f"   ✓ Device: {config.device}")
            print(f"   ✓ Models: {router.model_names}")
        except Exception as e:
            print(f"   ⚠ Failed to load router: {e}")
            router = None
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 3: Initialize Fallback Handler
    # ─────────────────────────────────────────────────────────────────────
    print("\n🔄 Step 3: Initializing fallback handler...")
    
    try:
        from artemis_final.router.artemis_router.fallback import create_fallback_handler
        fallback = create_fallback_handler(
            confidence_threshold=0.3,
            top_k=2,
            prefer_larger=True
        )
        print("   ✓ Fallback handler ready (threshold=0.3)")
    except Exception as e:
        print(f"   ⚠ Fallback handler unavailable: {e}")
        fallback = None
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 4: Run Routing
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n🎯 Step 4: Running router on {len(samples)} samples...")
    print(f"   Mode: {config.router_mode}")
    print()
    
    results: List[DemoResult] = []
    model_counts = defaultdict(int)
    fallback_count = 0
    total_latency = 0.0
    
    for i, sample in enumerate(samples):
        start_time = time.time()
        
        if router is not None:
            # Real router
            try:
                router_result = router.route(
                    prompt=sample["prompt"],
                    mode=config.router_mode,
                    metadata={
                        "router_task": sample["router_task"],
                        "source_dataset": sample["source_dataset"]
                    }
                )
                chosen_model = router_result["chosen_model"]
                rewards = router_result["rewards"]
            except Exception as e:
                logger.warning(f"Router failed on sample {i}: {e}")
                chosen_model = "qwen2_5_vl_7b"
                rewards = {"qwen2_5_vl_7b": 0.5}
        else:
            # Mock router
            models = ["deepseek_ocr", "qwen2_5_vl_3b", "qwen2_5_vl_7b", 
                      "qwen3_vl_8b_thinking", "gemma_3_27b"]
            rewards = {m: np.random.uniform(0.3, 0.9) for m in models}
            chosen_model = max(rewards, key=rewards.get)
        
        latency_ms = (time.time() - start_time) * 1000
        total_latency += latency_ms
        
        # Apply fallback logic
        confidence = compute_confidence(rewards)
        fallback_triggered = False
        
        if fallback is not None and confidence < 0.3:
            fallback_result = fallback.apply(
                rewards=rewards,
                original_choice=chosen_model,
                task_type=sample["router_task"]
            )
            if fallback_result.fallback_triggered:
                chosen_model = fallback_result.chosen_model
                fallback_triggered = True
                fallback_count += 1
        
        model_counts[chosen_model] += 1
        
        result = DemoResult(
            sample_id=sample["sample_id"],
            prompt_snippet=sample["prompt"][:50] + "...",
            task_type=sample["router_task"],
            chosen_model=chosen_model,
            rewards=rewards,
            confidence=confidence,
            fallback_triggered=fallback_triggered,
            router_latency_ms=latency_ms
        )
        results.append(result)
        
        # Show progress
        if config.show_details and (i < 5 or i == len(samples) - 1):
            fb_marker = " [FALLBACK]" if fallback_triggered else ""
            print(f"   [{i+1:3d}] {sample['router_task']:20s} → {chosen_model:25s} "
                  f"(conf={confidence:.2f}, {latency_ms:.1f}ms){fb_marker}")
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 5: Print Summary
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("📊 DEMO RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n📈 Routing Statistics:")
    print(f"   • Total samples:      {len(results)}")
    print(f"   • Mode:               {config.router_mode}")
    print(f"   • Fallbacks triggered: {fallback_count} ({100*fallback_count/len(results):.1f}%)")
    print(f"   • Avg router latency: {total_latency/len(results):.1f}ms")
    
    print(f"\n🔧 Model Selection Distribution:")
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(results)
        bar = "█" * int(pct / 2)
        print(f"   {model:25s} {count:4d} ({pct:5.1f}%) {bar}")
    
    # Confidence distribution
    confidences = [r.confidence for r in results]
    print(f"\n📊 Confidence Distribution:")
    print(f"   • Min:    {min(confidences):.3f}")
    print(f"   • Mean:   {np.mean(confidences):.3f}")
    print(f"   • Median: {np.median(confidences):.3f}")
    print(f"   • Max:    {max(confidences):.3f}")
    
    # Low confidence samples
    low_conf = [r for r in results if r.confidence < 0.3]
    if low_conf:
        print(f"\n⚠ Low Confidence Decisions ({len(low_conf)} samples):")
        for r in low_conf[:5]:
            print(f"   • {r.sample_id}: {r.task_type} → {r.chosen_model} (conf={r.confidence:.2f})")
    
    print("\n" + "="*70)
    print("✅ Demo complete!")
    print("="*70 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Artemis VLM Router Demo")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of samples to process")
    parser.add_argument("--mode", type=str, default="balanced",
                        choices=["accuracy", "cheap", "fast", "balanced"],
                        help="Routing mode")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for router (cpu, cuda, mps)")
    parser.add_argument("--no-db", action="store_true",
                        help="Use synthetic samples instead of database")
    parser.add_argument("--checkpoint", type=str, default="best_reward_router.pt",
                        help="Checkpoint filename")
    parser.add_argument("--quiet", action="store_true",
                        help="Hide individual sample details")
    
    args = parser.parse_args()
    
    config = DemoConfig(
        num_samples=args.num_samples,
        router_mode=args.mode,
        device=args.device,
        use_database=not args.no_db,
        checkpoint_name=args.checkpoint,
        show_details=not args.quiet,
    )
    
    run_demo(config)


if __name__ == "__main__":
    main()
