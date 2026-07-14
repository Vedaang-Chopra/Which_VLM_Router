# Phase 1 Discovery Report — ARTEMIS Full System Audit

**Date:** 2026-07-12  
**Agent:** Claude Code  
**Scan Source:** `SCAN_MANIFEST.json` (413 Python files across 4 roots)  

---

## 1. Scan Summary

| Root | Python Files | Total Files | Status |
|------|--------------|-------------|--------|
| `artemis_final/` | 111 | 111 | **Primary production codebase** |
| `artemis_core/` | 14 | 14 | **Clean reference implementation** |
| `code_base/` | 288 | 4459 | **Research baselines & external copies** |
| *(other)* | 0 | 0 | — |

**Total discovered modules (packages with `__init__.py`):** 59  
**Note:** 28 modules belong to `code_base/cascadeflow/` (external research baseline copied in); 4 to `code_base/vllm_semantic_router/` (external). The **ARTEMIS-native modules** = 27.

---

## 2. Module Inventory Cross-Check

### 2.1 ARTEMIS-Native Modules (artemis_final + artemis_core)

| Discovered Module | Py Files | Known List Entry | Status |
|-------------------|----------|------------------|--------|
| `artemis_final/ares` | 37 | `ares` (52) | ✅ Match (count diff: scan=37 vs doc=52) |
| `artemis_final/router` | 17 | `router` (20) | ✅ Match (diff: 17 vs 20) |
| `artemis_final/router_train` | 18 | `router_train` (19) | ✅ Match |
| `artemis_final/load_balancer` | 14 | `load_balancer` (14) | ✅ Exact match |
| `artemis_final/inference_engine` | 8 | `inference_engine` (9) | ✅ Match |
| `artemis_final/common` | 5 | `common` (7) | ✅ Match |
| `artemis_final/data_loop` | 5 | `data_loop` (5) | ✅ Exact match |
| `artemis_final/system_api` | 4 | `system_api` (4) | ✅ Exact match |
| `artemis_final/router/core` | 13 | — | 🔶 **Submodule not separately listed** |
| `artemis_final/router/core/legacy` | 3 | — | 🔶 **Submodule not separately listed** |
| `artemis_final/router_train/training` | 8 | — | 🔶 Submodule |
| `artemis_final/router_train/models` | 4 | — | 🔶 Submodule |
| `artemis_final/ares/evaluation` | 10 | — | 🔶 Submodule |
| `artemis_final/ares/parallel` | 6 | — | 🔶 Submodule |
| `artemis_final/ares/db` | 4 | — | 🔶 Submodule |
| `artemis_final/ares/metrics` | 3 | — | 🔶 Submodule |
| `artemis_final/ares/utils` | 3 | — | 🔶 Submodule |
| `artemis_final/ares/configs` | 3 | — | 🔶 Submodule |
| `artemis_final/ares/data` | 3 | — | 🔶 Submodule |
| `artemis_final/load_balancer/core` | 9 | — | 🔶 Submodule |
| `artemis_final/load_balancer/evaluation` | 2 | — | 🔶 Submodule |
| `artemis_core/src/artemis` | 14 | `artemis_core/src` (14) | ✅ Match |
| `artemis_core/src/artemis/router` | 3 | — | 🔶 Submodule |
| `artemis_core/src/artemis/load_balancer` | 3 | — | 🔶 Submodule |
| `artemis_core/src/artemis/common` | 3 | — | 🔶 Submodule |
| `artemis_core/src/artemis/inference` | 4 | — | 🔶 Submodule |

**New modules discovered (not in known list):**

- `artemis_final/router/core` (13 files) — Core router implementations
- `artemis_final/router/core/legacy` (3 files) — Legacy pairwise/classical routers
- `artemis_final/router_train/training` (8 files) — Training loops
- `artemis_final/router_train/models` (4 files) — Model definitions
- `artemis_final/ares/evaluation` (10 files) — Evaluation pipelines
- `artemis_final/ares/parallel` (6 files) — Parallel evaluation
- `artemis_final/ares/db` (4 files) — DB operations
- `artemis_final/ares/metrics` (3 files) — GPU metrics
- `artemis_final/ares/utils` (3 files) — Utilities
- `artemis_final/ares/configs` (3 files) — Configs
- `artemis_final/ares/data` (3 files) — Data loading
- `artemis_final/load_balancer/core` (9 files) — Scheduler, SLA monitor, stats
- `artemis_final/load_balancer/evaluation` (2 files) — LB evaluation
- `artemis_core/src/artemis/router` (3 files) — Clean router
- `artemis_core/src/artemis/load_balancer` (3 files) — Clean LB
- `artemis_core/src/artemis/common` (3 files) — Clean config
- `artemis_core/src/artemis/inference` (4 files) — Clean inference client

**Modules in known list NOT found by discovery:**

- `which_vlm/artemis` (31 files) — **EXISTS in code_base/which_vlm/artemis/** but has no `__init__.py` at package root
- `which_vlm/ares` (12 files) — **EXISTS in code_base/which_vlm/ares/**
- `which_vlm/experiments` (16 files) — **EXISTS in code_base/which_vlm/experiments/**
- `which_vlm/inference_api_call` (7 files) — **EXISTS in code_base/which_vlm/inference_api_call/**
- `which_vlm/configs` (5 files) — **EXISTS in code_base/which_vlm/configs/**
- `cascadeflow` (59 files) — **EXISTS in code_base/cascadeflow/** (external copy)
- `frugal_gpt` (19 files) — **EXISTS in code_base/frugal_gpt/**
- `lovm` (27 files) — **EXISTS in code_base/lovm/**
- `examples/load_balancer` (7 files) — **EXISTS in examples/load_balancer/**
- `examples/router` (3 files) — **EXISTS in examples/router/**
- `helpers` (2 files) — **EXISTS in code_base/helpers/**
- `aurelio` (2 files) — **EXISTS in code_base/aurelio/**
- `tests` (2 files) — **EXISTS in artemis_final/tests/**
- `cascade_experiments` (1 file) — **EXISTS in artemis_final/cascade_experiments/**

**Key Finding:** The "known list" mixes **ARTEMIS-native modules** (artemis_final/*, artemis_core/*) with **external research baselines copied into code_base/**. The scan correctly finds all — the discrepancy is classification, not missing code.

---

### 2.2 External Baselines in code_base/ (Research References)

| Module | Location | Py Files | Status in Known List |
|--------|----------|----------|---------------------|
| `cascadeflow` | `code_base/cascadeflow/cascadeflow/cascadeflow` | 94 | `cascadeflow` (59) — count diff |
| `vllm_semantic_router` | `code_base/vllm_semantic_router/semantic-router/bench/vllm_semantic_router_bench` | 21 | Not listed |
| `which_vlm` | `code_base/which_vlm` | 36 | Split into 5 entries in known list |
| `lovm` | `code_base/lovm/LOVM/modelGPT` + `LOVM/LOVM` | 16 | `lovm` (27) — count diff |
| `frugal_gpt` | `code_base/frugal_gpt/FrugalGPT/src/FrugalGPT` | 11 | `frugal_gpt` (19) — count diff |

---

## 3. Codebase Identity Resolution (Phase 0)

### Which codebase produced `multitask_eval_summary.csv`?

**Answer: `artemis_final/` — definitively.**

**Evidence:**

1. **Notebook path:** `artemis_final/router_train/notebooks/06_eval_multitask_reward_router.ipynb` loads checkpoint from `artemis_final/checkpoints/best_multitask_router_v1.pt` and data from `dataset/data/router_profiles_with_utility.parquet`
2. **Checkpoint location:** `artemis_final/checkpoints/best_multitask_router_v1.pt` exists (270MB); `artemis_core/` has no checkpoints
3. **Git history:** `artemis_final/` has 200+ commits; `artemis_core/` has ~5 commits (minimal reference)
4. **Config system:** `artemis_final/common/config_loader.py` loads `artemis.yaml` with all production paths; `artemis_core/` has no config loader
5. **Router architectures:** `artemis_final/router_train/models/` has all 3 (Reward, Pairwise, Classical); `artemis_core/src/artemis/router/model.py` has only RewardRouter
6. **Evaluation output:** `multitask_eval_summary.csv` written to `artemis_final/router_train/notebooks/results/`

**Conclusion:** `artemis_final/` is the **production codebase** that produced all paper results. `artemis_core/` is a clean, minimal reference implementation (~859 lines) created later for documentation/teaching. `code_base/` contains frozen research baselines (CascadeFlow, FrugalGPT, LOVM, WhichVLM-v1).

---

## 4. Verification Checklist for Phase 2

Every component below must be verified with execution evidence (not code reading alone).

| # | Component | Claim to Verify | Verification Method | Blockers |
|---|-----------|-----------------|---------------------|----------|
| 1 | **Router Architecture** | DistilBERT (768) + model_emb(32) + mode_emb(16) → MLP(256→128→1) | Load checkpoint, inspect `state_dict` shapes | None |
| 2 | **Hidden Dim = 256** | Config claims 512; checkpoint shows 256 | `torch.load()` checkpoint, print layer shapes | None |
| 3 | **CLIP Absence** | Images enter as width/height/AR tokens only | Trace `format_sample_text()` in inference — no image encoder called | None |
| 4 | **H = 1.0 Hardcoded** | Hallucination term is no-op in all reward formulas | Read `reward_definitions.py`: `df["H"] = 1.0` | None |
| 5 | **Pairwise = Reward at Inference** | Same forward pass, different loss only | Run both `inference_reward_router.py` and `inference_pairwise_router.py` on same input, diff outputs | None |
| 6 | **Startup Bug #1** | `GlobalConfig` missing `inference` field | Import `GlobalConfig`, check fields | None |
| 7 | **Startup Bug #2** | Retrainer imports `SystemConfig` (doesn't exist) | Import `retrainer.py`, catch `ImportError` | None |
| 8 | **Startup Bug #3** | Retrainer checkpoint loading incompatible | Attempt `load_state_dict` with v1 checkpoint | None |
| 9 | **Data Split** | 339,056 rows, 67,935 unique samples, 5 VLMs, train/val/test | Query PostgreSQL or parquet directly | Need DB/parquet access |
| 10 | **Balanced Recovery = 90.3%** | `multitask_eval_summary.csv` shows 0.9026 | Re-run eval notebook Cell 5-6, diff against CSV | GPU for DistilBERT |
| 11 | **Load Balancer Scheduling** | Capacity-aware + SLA monitoring works | Run `simulate_traffic()` in `public_api.py` | None |
| 12 | **ARES Evaluation** | Scorer + VLMJudge + Glider pipeline runs | Execute `RouterEvalPipeline` on test subset | GPU + Molmo/Glider models |
| 13 | **Inference Engine** | `WhichVLMClient` actually calls VLM endpoints | Start vLLM servers, send request | vLLM servers + GPU |
| 14 | **Retrainer** | `retrain()` is empty stub | Read `retrainer.py` line-by-line | None (code inspection sufficient) |
| 15 | **Traffic Simulator** | `NotImplementedError` in `run()` | Import and call `TrafficSimulator.run()` | None |

---

## 5. Decisions Needed from Vedaang

| # | Decision | Context |
|---|----------|---------|
| 1 | **Drop CascadeFlow/FrugalGPT/LOVM from audit scope?** | These are external research baselines in `code_base/` — not ARTEMIS. They inflate file counts and have known `NotImplementedError`s. |
| 2 | **Require live PostgreSQL for data verification?** | 339K rows in parquet; DB may have same or different state. Parquet is sufficient for row counts. |
| 3 | **GPU access for router inference verification?** | DistilBERT runs on CPU but slow. MPS (Apple Silicon) works. Need to confirm execution environment. |
| 4 | **vLLM endpoints for Inference Engine test?** | `WhichVLMClient` requires live model servers. Mock or skip? |
| 5 | **Molmo/Glider models for ARES eval?** | Heavy models (7B+). Can use `Scorer` (ground-truth only) without them. |
| 6 | **Fix startup bugs or document as known?** | Bugs #1-3 are in `data_loop/retrainer.py` and `common/config_loader.py`. Fix now or flag? |
| 7 | **Treat `artemis_core/` as separate deliverable?** | It's a clean reference impl. Should it have its own audit or be noted as "verified clean"? |

---

## 6. Next Steps

1. **Vedaang reviews this report** — resolves Decisions Needed above
2. **Phase 2 TODO list finalized** — I will produce the executable task list
3. **Phase 3 Execution** — Run verifications, capture evidence, update docs

---

*End of Phase 1 Report*
