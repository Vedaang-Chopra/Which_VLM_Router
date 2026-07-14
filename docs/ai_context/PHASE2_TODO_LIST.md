# Phase 2 — Audit TODO List (Executable Verification Tasks)

**Source:** Phase 1 Discovery Report (`docs/ai_context/PHASE1_DISCOVERY_REPORT.md`)  
**Status:** Awaiting Vedaang approval before execution  
**Execution Model:** One task per session, with evidence capture. `VERIFIED-WORKING` / `VERIFIED-BROKEN` / `UNVERIFIED` only.

---

## Task Inventory

| ID | Module | Claim to Verify | Verification Command / Method | Evidence Required | Blocker / Status |
|----|--------|-----------------|-------------------------------|-------------------|------------------|
| **T01** | `router` (arch) | Hidden dim = 256 (not 512). Architecture: DistilBERT(768) + model_emb(32) + mode_emb(16) → MLP(256→128→1) × 4 heads | `python3 -c "import torch; ckpt=torch.load('artemis_final/checkpoints/best_multitask_router_v1.pt', map_location='cpu'); [print(k, v.shape) for k,v in ckpt.items() if 'routing' in k or 'task' in k]"` | Terminal output showing `routing_heads.*.0.weight: [256, 816]` etc. | **NONE** — local CPU only |
| **T02** | `router` (arch) | CLIP is NOT used. Image info enters as `ImgWidth`, `ImgHeight`, `ImgAR` tokens in text only | 1. Read `artemis_final/router/core/inference_reward_router.py::format_sample_text()`<br>2. Run inference on CPU with dummy PIL Image, capture input_ids, confirm no image tensor | Code excerpt + inference log showing text-only tokenization | **NONE** |
| **T03** | `router_train` (reward) | Hallucination term `H` hardcoded to 1.0 in all 4 reward formulas | 1. Read `artemis_final/router_train/reward_definitions.py` lines 190-210<br>2. Grep for `df\["H"\] = 1.0` | Screenshot of code lines + grep output | **NONE** |
| **T04** | `router` (inference) | Pairwise router inference = Reward router inference (same forward, different loss only) | 1. Load both `inference_reward_router.py` and `inference_pairwise_router.py`<br>2. Run on identical input (CPU)<br>3. `diff <(python reward.py) <(python pairwise.py)` | Diff output showing identical `chosen_model` and `rewards` dict | **NONE** (CPU only) |
| **T05** | `common` (config) | **Startup Bug #1:** `GlobalConfig` (actually `GlobalConfig` in `config_loader.py`) missing `inference` field referenced by `system_api/pipeline.py` | `python3 -c "from artemis_final.common.config_loader import load_global_config; cfg=load_global_config(); print(hasattr(cfg, 'inference'), hasattr(cfg.router, 'inference'))"` | Terminal output showing `False False` | **NONE** |
| **T06** | `data_loop` (retrainer) | **Startup Bug #2:** `retrainer.py:6` imports `SystemConfig` from `common.config_loader` — class doesn't exist (it's `GlobalConfig`) | `python3 -c "from artemis_final.data_loop.retrainer import Retrainer"` | ImportError traceback showing `cannot import name 'SystemConfig'` | **NONE** |
| **T07** | `data_loop` (retrainer) | **Startup Bug #3:** Checkpoint loading incompatible — v1 checkpoint has `routing_heads` (4 heads), model expects single `routing_mlp` | 1. Load v1 checkpoint<br>2. Instantiate `RewardRouterModel` from `router_train/models/reward_router.py`<br>3. Try `model.load_state_dict(ckpt)` | Error message or success log | **NONE** |
| **T08** | `data` (dataset) | Parquet stats: 339,056 rows, 67,935 unique samples, 5 VLMs, train/val/test split | `python3 -c "import pandas as pd; df=pd.read_parquet('dataset/data/router_profiles_with_utility.parquet'); print('rows:',len(df)); print('samples:',df.sample_id.nunique()); print('models:',df.model_name.nunique()); print(df.data_split.value_counts())"` | Terminal output matching claimed numbers | **NONE** |
| **T09** | `router_train` (eval) | **Balanced mode recovery = 90.3% utility** — re-run eval notebook Cells 5-6, diff against `multitask_eval_summary.csv` | Execute `artemis_final/router_train/notebooks/06_eval_multitask_reward_router.ipynb` Cells 1-7 (skip CascadeFlow), capture `recovery` for `balanced` | Notebook output + diff vs CSV (0.902624...) | **GPU/MPS recommended** (DistilBERT on 200K rows ~5 min on MPS) |
| **T10** | `load_balancer` | `simulate_traffic()` runs end-to-end and produces `TrafficSimulationResult` with latency/cost/SLA metrics | `python3 -c "from artemis_final.load_balancer.public_api import simulate_traffic; import asyncio; r=asyncio.run(simulate_traffic(num_requests=100)); print(r.avg_latency_ms, r.sla_violation_rate, r.avg_cost_usd)"` | Terminal output with numeric results | **NONE** |
| **T11** | `ares` (eval) | `Scorer` (ground-truth accuracy/F1) works without heavy models | `python3 -c "from artemis_final.ares.evaluation.evaluation import Scorer; s=Scorer(); print('Scorer init ok')"` | Successful import + instantiation | **NONE** |
| **T12** | `ares` (eval) | `VLMJudge` (Molmo) and `GliderEvaluator` load heavy models — confirm they are optional/guarded | Read `judge_molmo.py` and `evaluation.py` for lazy loading or try/except | Code excerpts showing guard | **NONE** |
| **T13** | `inference_engine` | `WhichVLMClient.run_batch()` returns `False` (stub) — confirm non-functional | `python3 -c "from artemis_final.inference_engine.runners import run_batch; import asyncio; print(asyncio.run(run_batch([])))"` | Output `False` | **NONE** |
| **T14** | `router` (traffic sim) | `TrafficSimulator.run()` raises `NotImplementedError` | `python3 -c "from artemis_final.router.core.traffic_simulator import TrafficSimulator; ts=TrafficSimulator(); ts.run()"` | Traceback with `NotImplementedError` | **NONE** |
| **T15** | `data_loop` (retrainer) | `retrain()` body is empty (pass/no-op) | Read `artemis_final/data_loop/retrainer.py::run_retraining()` — confirm no training loop | Code excerpt showing empty/stub implementation | **NONE** |

---

## Execution Rules

1. **One task per session** — complete, verify, log, then stop.
2. **Evidence = terminal output or code excerpt** — no "I read the code and it looks right."
3. **Status values only:** `VERIFIED-WORKING` | `VERIFIED-BROKEN` | `UNVERIFIED` (with blocker named).
4. **Log every task** to `docs/ai_context/AGENT_EXECUTION_LOG.md` using the template.
5. **Update `SYSTEM_STATE.md`** after each task with corrected status.
6. **If blocker = GPU/MPS/vLLM** — mark `UNVERIFIED` with exact requirement, do not guess.

---

## Parallelization Groups (for planning)

| Group | Tasks | Dependency | Can Run Concurrently? |
|-------|-------|------------|----------------------|
| **G1: Static Code Checks** | T01, T02, T03, T04, T05, T06, T07, T13, T14, T15 | None | ✅ Yes (all CPU, no external deps) |
| **G2: Data Verification** | T08 | None | ✅ Yes |
| **G3: Eval Pipeline (Light)** | T10, T11, T12 | G1 (config must load) | ✅ Yes |
| **G4: Heavy Eval** | T09 | G1, G2, GPU/MPS | ❌ Sequential (needs model load) |

**Recommended order:** G1 → G2 → G3 → G4

---

## Decisions Needed Before Execution

| # | Decision | Options | Recommendation |
|---|----------|---------|----------------|
| D1 | **Scope: Drop external baselines?** | A) Audit only `artemis_final/` + `artemis_core/`<br>B) Include `code_base/` baselines | **A** — they're frozen research copies, not ARTEMIS |
| D2 | **DB vs Parquet for T08?** | A) Query PostgreSQL (if running)<br>B) Use parquet file (static, known good) | **B** — parquet is the source of truth for training |
| D3 | **GPU for T09?** | A) Run on MPS (Apple Silicon)<br>B) Skip if no GPU, mark UNVERIFIED<br>C) Use CPU (slow ~15 min) | **A** if MPS available, else **C** |
| D4 | **Fix bugs or document?** | A) Fix T05-T07 in this audit<br>B) Document as known issues, fix later | **B** — audit is verification, not repair |
| D5 | **vLLM for T13?** | A) Start local vLLM servers (needs GPU)<br>B) Mock client, verify structure only<br>C) Skip, mark UNVERIFIED | **B** — structure check sufficient for audit |

---

## Approval

**Vedaang: Review the above. Reply with:**

- "APPROVED" to begin Phase 3 execution (I'll start with G1)
- "MODIFY: [specific changes]" to adjust tasks/blockers
- Answers to D1-D5 above

*No execution begins until explicit approval.*
