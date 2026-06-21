# Module Map: router
>
> Directory: artemis_final/router/
> Entry point: public_api.py::init_router()
> Status: PARTIAL

## Responsibility

Predicts reward scores for all five VLMs given a text prompt, using frozen DistilBERT + MLP classifier. Returns the model with the highest predicted reward for the given routing mode.

## File Index

| File | Layer | Purpose |
|---|---|---|
| `public_api.py` | Runner | Global singleton router; init_router, route_request, route_batch |
| `router_service.py` | Runner | RouterService class; orchestrates checkpoint loading and prediction |
| `setup_router.py` | Runner | CLI router setup script |
| `core/inference_reward_router.py` | Core | RewardRouterInference — main MLP inference class |
| `core/legacy/inference_classical_router.py` | Core | ClassicalRouterInference — CE/KL variant |
| `core/legacy/inference_pairwise_router.py` | Core | PairwiseRouterInference — margin ranking variant |
| `core/schemas.py` | Schema | Sample, RouterDecision dataclasses |
| `core/config.py` | Schema | RouterConfig dataclass |
| `core/api_io.py` | Core | HTTP/DB sample loading for inference |
| `core/fallback.py` | Core | Confidence-based fallback routing |
| `core/lb_interface.py` | Orchestration | Kafka producer for LB (TODO: implement) |
| `core/logging_wandb.py` | Utility | W&B experiment logging |
| `core/traffic_simulator.py` | Utility | Traffic simulation (BROKEN: NotImplementedError) |
| `router_config_reward.yaml` | Config | MLP dims, model list, embedding sizes, float32 |

## Public API

| Function | File | Signature | Purpose |
|---|---|---|---|
| `init_router` | public_api.py | `init_router(config_path?)` | Initialize global RouterService from checkpoint |
| `route_request` | public_api.py | `route_request(prompt, mode, metadata?) -> Dict` | Route single query |
| `route_batch` | public_api.py | `route_batch(prompts, modes?, metadata_list?) -> List[Dict]` | Route batch |
| `load_router_from_checkpoint` | public_api.py | `load_router_from_checkpoint(router_type, path?, device?)` | Load specific router architecture |

## Internal Call Graph

```
route_request(prompt, mode, metadata)
  → RouterService.predict(prompt, mode, metadata)
    → RewardRouterInference.forward() or Classical/Pairwise variant
      → DistilBERT encode (frozen)
      → Add model embeddings (5 × 32-dim)
      → Add mode embedding (4-dim one-hot)
      → MLP forward → 5 reward scores
      → argmax → chosen_model
  → {chosen_model, rewards, mode, inference_ms}
```

## Dependencies

Internal: `common.config_loader`, `router.core.schemas`
External: `torch`, `transformers` (DistilBERT), `PIL`, `numpy`, `logging`

## Change Guide

- **To change routing logic**: edit `core/inference_reward_router.py` (or the classical/pairwise variants)
- **To add a new router type**: add to `public_api.py::load_router_from_checkpoint()` and create a new inference class
- **To change config**: edit `router_config_reward.yaml` (model list, MLP dimensions)
- **To enable W&B logging**: use `core/logging_wandb.py` logger
- **To simulate traffic**: use `load_balancer::simulate_traffic()` instead of `core/traffic_simulator.py` (broken)
