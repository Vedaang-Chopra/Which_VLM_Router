# ARTEMIS — Overview

## What It Does

ARTEMIS is a cost-aware Vision-Language Model (VLM) router that dynamically selects the optimal VLM for each user request. Rather than routing every query to the most capable (and expensive) model, ARTEMIS uses a trained MLP classifier on top of frozen CLIP + DistilBERT encoders to predict reward scores for each of five candidate VLMs, then selects the cheapest model that meets the user's accuracy constraints.

The system routes across five VLMs: **deepseek_ocr** (specialized OCR), **qwen2_5_vl_3b** (lightweight), **qwen2_5_vl_7b** (balanced), **qwen3_vl_8b_thinking** (reasoning), and **gemma_3_27b** (maximum accuracy). It supports four routing modes — accuracy, cheap, fast, and balanced — allowing operators to trade cost against quality per request.

## Architecture Summary

The system is composed of four primary modules:

1. **Router** (`artemis_final/router/`) — Loads a trained checkpoint (Reward, Pairwise, or Classical architecture) and predicts reward scores for all five VLMs given a text prompt. Takes ~5–50ms depending on hardware.

2. **Load Balancer** (`artemis_final/load_balancer/`) — Receives the router's decision and applies SLA constraints (latency targets, queue capacity) before dispatching. Can override the router's choice if the preferred model is overloaded.

3. **Inference Engine** (`artemis_final/inference_engine/`) — Unified OpenAI-compatible client for calling all five VLM backends. Tracks latency, cost, and token usage for each call.

4. **ARES** (`artemis_final/ares/`) — Evaluation and data pipeline. Runs VLM Judge (Molmo) and Glider evaluators to score responses, writes results to PostgreSQL, and feeds data back into the retraining loop.

Supporting modules: **Router Training** (`router_train/`) for training new checkpoints from PostgreSQL data; **Data Loop** (`data_loop/`) for online logging and periodic retraining; **System API** (`system_api/`) for the FastAPI OpenAI-compatible endpoint.

## Key Design Decisions

1. **Frozen Encoders + Lightweight MLP.** Rather than fine-tuning large vision-language backbones, ARTEMIS freezes DistilBERT (66M params) and CLIP image embeddings, training only a small MLP head (~2 layers, 512-dim). This makes routing inference fast and cheap enough to be a pre-dispatch step.

2. **Reward-Based Formulation.** The router predicts a scalar reward for each VLM in each routing mode, trained with MSE loss against multi-objective reward functions. This decouples the training objective from the specific dispatch policy and allows a single checkpoint to serve all four routing modes.

3. **SLA-Aware Load Balancing.** The load balancer is not just a passthrough — it maintains per-model queue state and SLA monitors, and can redirect traffic away from overloaded models even when the router prefers them. This decouples routing accuracy from system availability.

4. **PostgreSQL-Backed Training Loop.** All samples, responses, and evaluations are stored in PostgreSQL, enabling periodic retraining from accumulated data. The loop supports hot-swapping router checkpoints without downtime.

5. **Three Router Architectures.** Reward (MSE prediction), Pairwise (margin ranking), and Classical (cross-entropy classification) — allowing operators to pick the approach best suited to their training data quality and latency requirements.

## Current State

**Working:** Router inference (all three architectures), load balancer scheduling with SLA monitoring, inference engine client, ARES evaluation pipeline (scorer + judge + glider), router training notebooks, FastAPI system API, PostgreSQL schema and DB operations.

**Partial:** Load balancer capacity config loading (has TODOs about config override respect), ARES data collection (returns None for some error paths), data loop (mostly stubs).

**Not yet implemented:** Traffic simulation (`traffic_simulator.py` has NotImplementedError at line 142), cascadeflow domain routing strategy (incomplete), full online retraining trigger from the system API.

See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for the full component table.

## Quick Start

```python
from artemis_final.router.public_api import init_router, route_request

# Initialize router (loads checkpoint)
init_router()

# Route a single request
result = route_request(
    prompt="What is shown in this diagram?",
    mode="balanced",
    metadata={"task": "diagram_reasoning"}
)
# result["chosen_model"] -> e.g. "qwen2_5_vl_7b"
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full data flow diagrams.
