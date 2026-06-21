# ARTEMIS — Overview

## What It Is

ARTEMIS is an intelligent routing system for Vision-Language Models. Given a user query with optional image, it decides which of five available VLMs to use — balancing accuracy, cost, and latency. It is not a model itself; it sits in front of existing VLM deployments and chooses between them automatically.

The core mechanism is a small neural network (frozen DistilBERT text encoder + 2-layer MLP) trained to predict reward scores for each of the five VLMs. The VLM with the highest predicted reward for the given mode is selected. Four modes are available: **accuracy** (best quality, uses larger models), **cheap** (lowest cost, uses the smallest model), **fast** (lowest latency), and **balanced** (reasonable trade-off).

The five candidate VLMs are: `deepseek_ocr` (specialized for text extraction), `qwen2_5_vl_3b` (fastest, cheapest), `qwen2_5_vl_7b` (balanced), `qwen3_vl_8b_thinking` (reasoning-capable), and `gemma_3_27b` (largest, most accurate). All must expose an OpenAI-compatible `/v1/chat/completions` endpoint.

ARTEMIS is built for continuous improvement: every request and its outcome are stored in PostgreSQL, periodically evaluated against ground truth and by VLM judges, and used to retrain the router. This means the router adapts to the actual distribution of queries in production, rather than only what was seen during initial training.

## Architecture in One Paragraph

A request arrives at the FastAPI endpoint (`/v1/chat/completions`), which passes it through three stages. The **Router** encodes the text with DistilBERT, adds learned embeddings for each VLM and the routing mode, runs the MLP to produce reward scores, and returns the highest-scoring model. The **Load Balancer** then checks whether that model's queue and latency SLA would be violated — if so, it redirects to the next-best feasible model. The **Inference Engine** calls the selected VLM's API endpoint and returns the response. Asynchronously, the **Evaluation Pipeline** scores the response against ground truth and VLM judges, writing results back to PostgreSQL for periodic **Router Retraining**. See [ARCHITECTURE.md](ARCHITECTURE.md) for diagrams and [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for the full component breakdown.

## Current State

**Working end-to-end:** Router inference (all 3 architectures), load balancer scheduling with SLA monitoring, FastAPI endpoints, PostgreSQL schema and DB operations, training notebooks, and the clean `artemis_core/src/` reference implementation.

**Not yet end-to-end:** The inference engine has stub methods returning `False` — it cannot make actual VLM calls yet. The data loop's `retrain()` body is empty — automated retraining does not run. These are the two main blockers for a fully automatic production pipeline.

**Research variants:** CascadeFlow (cascade-based routing), FrugalGPT (model-chain optimization), and LOVM (orchestration benchmarks) exist as separate codebases. They are not integrated with the main ARTEMIS pipeline.

**What this means in practice:** You can run the router + load balancer in isolation. You can train new router checkpoints via notebooks. You cannot yet make end-to-end requests through the FastAPI endpoint until the inference engine is completed.

See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for the full honest breakdown.

## How to Run It

```bash
# Start the FastAPI server (requires inference_engine to be completed first)
cd artemis_final
python -m uvicorn system_api.main:app --reload --port 8000

# Or use Docker (includes PostgreSQL):
docker-compose up -d

# Train a new router checkpoint:
jupyter notebook artemis_final/router_train/notebooks/02_reward_router_sql_to_training.ipynb

# Run a routing-only demo (no VLM calls needed):
python -c "
from artemis_final.router.public_api import init_router, route_request
init_router()
result = route_request('What is shown in this diagram?', mode='balanced')
print(result['chosen_model'])
"
```

## Where to Go Next

- **Understand the system:** Start with [ARCHITECTURE.md](ARCHITECTURE.md) (diagrams + component descriptions), then [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) (what works, what doesn't)
- **Start coding:** Read [docs/ai_context/INDEX.md](../ai_context/INDEX.md) for the module registry and data flow, then open the relevant module doc in `docs/ai_context/modules/`
- **Train the router:** Follow `router_train/notebooks/02_reward_router_sql_to_training.ipynb` — this is fully working via notebooks even though the `service.py` entry point is incomplete
- **Understand what's safe to build on:** See the "Safe to Build On" and "Do Not Build On Yet" sections in [docs/ai_context/SYSTEM_STATE.md](../ai_context/SYSTEM_STATE.md)
