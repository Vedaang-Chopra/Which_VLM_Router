# Module: Router

## What It Does

Routes each incoming request to the optimal VLM by scoring all five candidates with a trained MLP classifier on top of frozen DistilBERT embeddings.

## How It Fits In

First stage of the ARTEMIS pipeline. Receives a text prompt (and optional image URI) from the System API and returns a ranked list of VLM rewards.

## Architecture

```mermaid
graph TD
    Prompt["Text prompt<br/>+ metadata"] --> Format[Format text:<br/>[ROUTER] Task: {task}<br/>Dataset: {dataset}<br/>Question: {text}]
    Format --> Enc[DistilBERT Encoder<br/>frozen, 66M params]
    Enc --> Vec[768-dim embedding]
    Vec --> Concat[Concatenate embeddings]
    Concat --> ME[Model Embeddings<br/>5 × 32-dim]
    Concat --> MLE[Mode Embedding<br/>4-dim one-hot]
    ME --> MLP[MLP Head<br/>768+190 → 512 → 5]
    MLE --> MLP
    MLP --> Rewards["Rewards: {deepseek: 0.2, qwen3b: 0.7, qwen7b: 0.85, qwen8b: 0.76, gemma27b: 0.92}"]
    Rewards --> Argmax[argmax]
    Argmax --> Out["{chosen_model, rewards, inference_ms}"]
```

## Key Files

| File | What It Does |
|---|---|
| `public_api.py` | Entry point: `init_router()`, `route_request()`, `route_batch()` |
| `router_service.py` | RouterService class, orchestrates prediction |
| `core/inference_reward_router.py` | RewardRouterInference — main inference class, loads `.pt` checkpoint |
| `core/legacy/inference_classical_router.py` | ClassicalRouterInference — CE/KL loss variant |
| `core/legacy/inference_pairwise_router.py` | PairwiseRouterInference — margin ranking variant |
| `core/schemas.py` | Sample, RouterDecision dataclasses |
| `core/config.yaml` | Model list, embedding dimensions, MLP architecture |
| `checkpoints/` | Trained `.pt` files: `best_reward_router.pt`, `best_pairwise_router.pt`, `best_classical_router.pt` |

## Routing Modes

| Mode | Typical Choice | Objective |
|---|---|---|
| `accuracy` | gemma_3_27b | Maximize quality |
| `cheap` | qwen2_5_vl_3b | Minimize cost |
| `fast` | qwen2_5_vl_3b | Minimize latency |
| `balanced` | qwen2_5_vl_7b | Multi-objective |

## Status

**PARTIAL.** Router inference works end-to-end (all three architectures). Load the checkpoint, encode text, run MLP, get rewards. See `artemis_final/router/public_api.py`.

**Broken:** `core/traffic_simulator.py` line 142 — `TrafficSimulator.run()` raises `NotImplementedError`. The simulation harness is not functional.

**Incomplete:** Some methods in `router_service.py` return `None` for unimplemented paths.
