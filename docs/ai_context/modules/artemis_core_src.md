# Module: artemis_core_src
>
> Status: COMPLETE
> Directory: artemis_core/src/artemis/
> Entry point: router/router.py (ArtemisRouter), inference/client.py, load_balancer/balancer.py
> Last updated: 2026-06-20

## Purpose

Minimal, self-contained reference implementation of the ARTEMIS router + load balancer + inference client. ~859 lines, zero findings. Used as clean reference for understanding the core design.

## Public API

| Function | File | Purpose |
|---|---|---|
| `ArtemisRouter` | `router/router.py` | DistilBERT + MLP router; `route(query) -> chosen_model` |
| `ArtemisLoadBalancer` | `load_balancer/balancer.py` | Capacity-aware load balancer |
| `VLMClient` | `inference/client.py` | OpenAI-compatible VLM inference client |
| `load_global_config` | `common/config_loader.py` | Config loading |

## Internal Structure

| File | Responsibility |
|---|---|
| `common/config_loader.py` | GlobalConfig, load_global_config() |
| `common/utils.py` | Shared utilities |
| `inference/client.py` | OpenAI-compatible client |
| `inference/messages.py` | Message formatting |
| `inference/models.py` | Model metadata |
| `load_balancer/balancer.py` | Capacity-aware scheduling |
| `load_balancer/types.py` | CapacityConfig, LoadBalancingResult |
| `router/model.py` | RouterModel dataclass |
| `router/router.py` | ArtemisRouter — DistilBERT + MLP inference |

## Dependencies

External: `torch`, `transformers` (DistilBERT), `openai` or compatible client

## Known Issues

None. This module is clean, complete, and functional.

## What an Agent Must Know

- This is the "reference implementation" — the cleanest code in the project. Use it to understand the core architecture before reading the fuller (and messier) artemis_final/ implementations.
- All components are minimal but functional. Config loading, inference, routing, and load balancing all work.
