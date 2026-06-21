# Scan Log — Which_VLM_Router / Artemis
**Date:** 2026-06-20
**Agent:** pi-cascade
**Batches completed:** 18 of ?

## Coverage
- Files scanned: 376
- Python files: 233
- Markdown files: 42
- Notebooks: 71
- Config files: 30

## Modules (42)

| Module | Status | Files | Notable |
|---|---|---|---|
| `cascadeflow` | PLACEHOLDER | 59 | 85 |
| `ares` | PLACEHOLDER | 52 | 41 |
| `which_vlm/artemis` | PLACEHOLDER | 31 | 9 |
| `lovm` | COMPLETE | 27 | 1 |
| `router` | PLACEHOLDER | 20 | 13 |
| `frugal_gpt` | PARTIAL | 19 | 7 |
| `router_train` | PLACEHOLDER | 19 | 5 |
| `root` | PLACEHOLDER | 17 | 3 |
| `which_vlm/experiments` | COMPLETE | 16 | 2 |
| `artemis_core/src` | COMPLETE | 14 | 0 |
| `load_balancer` | PARTIAL | 14 | 6 |
| `which_vlm/ares` | PLACEHOLDER | 12 | 11 |
| `inference_engine` | PLACEHOLDER | 9 | 5 |
| `common` | COMPLETE | 7 | 1 |
| `examples/load_balancer` | COMPLETE | 7 | 0 |
| `which_vlm/inference_api_call` | PLACEHOLDER | 7 | 3 |
| `data_loop` | PLACEHOLDER | 5 | 4 |
| `which_vlm/configs` | COMPLETE | 5 | 0 |
| `system_api` | COMPLETE | 4 | 0 |
| `examples/router` | COMPLETE | 3 | 0 |
| `README.md` | COMPLETE | 2 | 0 |
| `aurelio` | COMPLETE | 2 | 0 |
| `examples/ops` | COMPLETE | 2 | 0 |
| `helpers` | COMPLETE | 2 | 1 |
| `main.py` | COMPLETE | 2 | 2 |
| `requirements.txt` | COMPLETE | 2 | 0 |
| `tests` | COMPLETE | 2 | 0 |
| `01_router_single_and_batch_modes.ipynb` | COMPLETE | 1 | 0 |
| `02_router_experiments_and_modes.ipynb` | COMPLETE | 1 | 0 |
| `COMPLETE_SYSTEM_OVERVIEW.md` | PARTIAL | 1 | 2 |
| `IMPLEMENTATION_WALKTHROUGH.md` | COMPLETE | 1 | 0 |
| `REFACTOR_NOTES.md` | COMPLETE | 1 | 0 |
| `artemis_core` | COMPLETE | 1 | 0 |
| `artemis_final` | COMPLETE | 1 | 0 |
| `cascade_experiments` | COMPLETE | 1 | 2 |
| `config` | COMPLETE | 1 | 0 |
| `docker-compose.yml` | COMPLETE | 1 | 0 |
| `examples` | COMPLETE | 1 | 1 |
| `examples/README.md` | COMPLETE | 1 | 0 |
| `test_vllm_model.ipynb` | COMPLETE | 1 | 0 |
| `utils` | COMPLETE | 1 | 0 |
| `which_vlm/__init__.py` | COMPLETE | 1 | 0 |

## Notable Findings

Total: 204 patterns across 376 files.

### NotImplementedError (8 instances)
- `artemis_final/router/core/traffic_simulator.py`: raise NotImplementedError (line 142)
- `code_base/cascadeflow/cascadeflow/cascadeflow/providers/base.py`: raise NotImplementedError (line 551)
- `code_base/cascadeflow/cascadeflow/cascadeflow/providers/base.py`: raise NotImplementedError (line 611)
- `code_base/frugal_gpt/FrugalGPT/src/FrugalGPT/llmchain.py`: raise NotImplementedError (line 19)
- `code_base/frugal_gpt/FrugalGPT/src/service/modelservice.py`: raise NotImplementedError (line 59)
- `code_base/frugal_gpt/FrugalGPT/src/service/modelservice.py`: raise NotImplementedError (line 117)
- `code_base/frugal_gpt/FrugalGPT/src/service/modelservice.py`: raise NotImplementedError (line 122)
- `AGENTS.MD`: raise NotImplementedError (line 112)

### TODOs / FIXMEs (8 instances)
- `artemis_final/COMPLETE_SYSTEM_OVERVIEW.md`: TODO: ** Update notebooks for reward router (line 405)
- `artemis_final/COMPLETE_SYSTEM_OVERVIEW.md`: TODO: ** Create FastAPI server wrapper (line 406)
- `artemis_final/router/core/lb_interface.py`: TODO: Implement Kafka producer (line 98)
- `artemis_final/load_balancer/load_balancer_service.py`: TODO: If we want to fully respect 'cfg' overrides (like specific model configs passed in memory), (line 45)
- `code_base/cascadeflow/cascadeflow/cascadeflow/agent.py`: TODO: Convert to DomainCascadeStrategy when implemented (line 238)
- `code_base/cascadeflow/cascadeflow/cascadeflow/agent.py`: TODO: Re-implement ResponseCache in v0.2.1 (line 249)
- `code_base/frugal_gpt/FrugalGPT/src/service/modelservice.py`: TODO: change this to real token count (line 189)
- `code_base/frugal_gpt/FrugalGPT/src/service/modelservice.py`: TODO: change this to real token count (line 433)

### Key Findings

**7 NotImplementedError stubs:**
- `artemis_final/router/core/traffic_simulator.py` line 142
- `code_base/cascadeflow/.../providers/base.py` lines 551, 611
- `code_base/frugal_gpt/.../llmchain.py` line 19
- `code_base/frugal_gpt/.../modelservice.py` lines 59, 117, 122

**144 Placeholder returns** scattered across cascadeflow (55), ares (38), and others.
Many are short-circuit `return None` for unimplemented conditional paths.

**New discovery:** `artemis_core/src/artemis/` — a minimal 859-line implementation
of config_loader, inference client, load balancer, and router. Clean (0 findings).
Different from both `artemis_core/main.py` and `artemis_final/`.

## Status Summary
- COMPLETE: 29 modules
- PARTIAL: 3 modules
- PLACEHOLDER: 10 modules
