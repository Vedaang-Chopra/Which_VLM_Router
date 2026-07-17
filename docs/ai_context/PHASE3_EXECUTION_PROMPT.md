# Phase 3 Execution Prompt — ARTEMIS Full Audit

## Project Context

Project: **Which_VLM_Router** (ARTEMIS — cost-aware VLM router)
Working dir: `/Users/vedaangchopra/all_data/complete_technical_work/all_projects_implemented/Which_VLM_Router`
Code: `artemis_final/` (production), `artemis_core/` (clean reference), `code_base/` (frozen research baselines; CascadeFlow kept in scope per T17)
Docs: `docs/ai_context/PHASE1_DISCOVERY_REPORT.md`, `docs/ai_context/PHASE2_TODO_LIST.md`, `docs/ai_context/SYSTEM_STATE.md`
Handoff log: `docs/ai_context/AGENT_EXECUTION_LOG.md`

Use python at `/opt/homebrew/opt/python@3.14/bin/python3` (has torch installed; the repo `.venv` does NOT).

---

## Roles & Rules

You are an **execution agent** running this audit in a fresh session. You did not write this code — verification decisions come from execution evidence, not from re-reading or hypothesizing. Treat the audit TODO list as binding; do not silently rewrite it.

**Verified-only statuses:**

- `VERIFIED-WORKING` — actually executed, real output captured in evidence block.
- `VERIFIED-BROKEN` — actually executed, captured failure with error.
- `UNVERIFIED-PENDING` — could not run; state the exact concrete blocker (no GPU, no DB, endpoint down, etc.). Do **not** guess. Do **not** substitute mocks.

**No `PARTIAL` / `COMPLETE` from code inspection alone.** A grep hit or docstring claim is a lead to chase with execution, not a conclusion.

---

## Mandatory Pre-Flight (before each task)

1. **Read** `docs/ai_context/AGENT_EXECUTION_LOG.md` — search for the task ID you're about to run. If a previous attempt was marked FAILED, do **not** repeat its approach. Use its `Resolution` instead.
2. **Read** `docs/ai_context/PHASE2_TODO_LIST.md` — refresh the exact verification commands and evidence requirements for the task in scope.
3. **Confirm working directory** with `pwd` before any command (a mistake here poisons everything).

---

## Phase 3 Order

Run tasks in **group order**: G1 → G2 → G3 → G4 → G5. Within a group, tasks can run in any order. Mark each task done before moving to the next group.

### G1 — Static Code Checks (CPU only, no external deps)

| ID   | Focus |
|------|-------|
| T01 | router arch: checkpoint `state_dict` shapes prove hidden=256, 4 routing heads |
| T02 | router arch: trace `format_sample_text()` — no CLIP, no image tensor |
| T03 | router_train reward: `reward_definitions.py` proves H=1.0 hardcoded |
| T04 | router inference: load both `inference_reward_router.py` and `inference_pairwise_router.py`; diff outputs on identical input |
| T05 | common: `load_global_config()` proves `inference` field missing |
| T06 | data_loop retrainer: confirm `ImportError` on `SystemConfig` |
| T07 | data_loop retrainer: confirm `load_state_dict` fails with v1 checkpoint (4 `routing_heads` vs model's single `routing_mlp`) |
| T13 | inference_engine: confirm `run_batch` returns `False` (stub) |
| T14 | router traffic_sim: confirm `NotImplementedError` |
| T15 | data_loop retrainer: confirm `run_retraining()` has no training loop |

### G2 — Data Verification (PostgreSQL @ `localhost:5432/vlmrouter`)

| ID   | Focus |
|------|-------|
| T08 | Execute the 3-way JOIN from `router_train/notebooks/00_prepare_local_database.ipynb` lines 339–382 (`vlm_samples` × `vlm_responses` × `vlm_evaluations`, filtered `r.ok = TRUE`). Compare row count + per-split + per-model vs parquet (`dataset/data/router_profiles_with_utility.parquet`). Diff must be zero. |

### G3 — Eval Pipeline (Light) (CPU)

| ID   | Focus |
|------|-------|
| T10 | `load_balancer.public_api.simulate_traffic(num_requests=100)` → produce `TrafficSimulationResult` |
| T11 | `ares.evaluation.evaluation.Scorer` — instantiate without GPU |
| T12 | `ares.evaluation.judge_molmo` and `GliderEvaluator` — find lazy-load / try-except guards |

### G4 — Heavy Eval (GPU/MPS recommended)

| ID   | Focus |
|------|-------|
| T09 | Run `router_train/notebooks/06_eval_multitask_reward_router.ipynb` Cells 1–7 (skip CascadeFlow), diff `recovery` `balanced` against `multitask_eval_summary.csv` (expected ≈ 0.9026) |
| T17 | CascadeFlow: re-run experiment against same data slice; verify 73% / 32.3% Table I numbers |

### G5 — Integration

| ID   | Focus |
|------|-------|
| T16 | Router → LB → Inference Engine E2E chain with one realistic multimodal query; full real trace (chosen_model, scheduling_decision, IE response with cost/latency). No mocks. |

---

## T05–T07 Special Case (Read Before Running G1)

D4 says: **T05/T06/T07 are boot-blockers**. Apply the **minimal patch** needed to unblock booting (e.g. add missing field, fix import, fix state_dict key mismatch). Document the patch as an "unblock fix" inside the AGENT_EXECUTION_LOG entry, distinct from any other bug found during the task. All other bugs found anywhere during Phase 3: **document only**, do not fix.

---

## One-Task-Per-Session Enforcement (CRITICAL)

For each task you run, you **must** output, in this exact form:

```
## Change Report

### Files Modified
- `path/to/file.py` — what changed, which layer it belongs to, why

### Files Created
- `path/to/new_file.py` — purpose and layer

### Files Archived
- `path/to/old_file.py` → `archive/YYYY-MM/old_file.py` — reason

### Documentation Updated
Which docs updated and what changed

### Diagrams Updated
Which Mermaid diagrams updated and what changed

### Validation Run
<exact command + output snippet>

### Validation NOT Run
<what was not tested and why>

### Known Limitations / Follow-up Work
<gaps, edge cases, tasks for next session>

### Execution Log Updated
<path to AGENT_EXECUTION_LOG.md entry with task ID, status (VERIFIED-WORKING/VERIFIED-BROKEN/UNVERIFIED-PENDING), approach, what worked, what failed, evidence file ref>
```

---

## Logging Protocol

After **every** task — success or failure — append a new entry to `docs/ai_context/AGENT_EXECUTION_LOG.md` with this template:

```
## [TASK_ID] — [Task name]
**Date:** YYYY-MM-DD
**Agent:** Pi | Codex | Claude Code
**Model:** [model name actually used]
**Status:** ✓ Complete | ✗ Failed | ⚠ Partial
**Approach taken:** [1–3 sentences]
**What worked:** [be concrete]
**What failed:** [be concrete; include exact error / traceback]
**Root cause:** [why it failed; "wrong import path" not "import error"]
**Resolution:** [what fixed it, or what the next agent should try]
**Files modified:** [list]
**Verify result:** [exact command + output]
**Model fallback used:** yes | no
**DO NOT REPEAT:** [specific anti-patterns, import paths, approaches that failed]
```

**Do not skip this step.** Skipping it means the next agent or the next session repeats your mistake. That is a protocol violation.

---

## Update Protocol

When source code changes (T05–T07 unblock fixes, T16 trace scripts, etc.), update documentation in the **same change set**:

| What changed | What to update |
|---|---|
| Module structure / new files | `docs/ai_context/SYSTEM_STATE.md`, `PHASE1_DISCOVERY_REPORT.md` ares / module rows |
| Architectural correction from T01/T02/T07 | `PHASE1_DISCOVERY_REPORT.md` Architecture notes (if any) |
| T09 evals / T17 baselines / T16 E2E | `SYSTEM_STATE.md` Status Table — update `ARTEMIS Eval`, `CascadeFlow`, `System API` rows |

Do not overwrite — append a correction-log entry showing old → new with evidence link.

---

## Hard Stops — ALWAYS confirm before running

- `rm -rf`
- `git push`, `git reset --hard`, `git clean -fd`
- `sudo`
- Any command touching `.env*`, `*secret*`, `*credential*`, `*.pem`, `*.key`

---

## Output Discipline

- Never print secret values or API keys.
- Truncate large stdout (>50 lines) — point to the evidence file path in the log entry.
- If stuck after 2 iterations on the same task: **stop** and write a FAILED entry with a precise diagnosis. Do not guess your way past a blocker.

---

## Final Deliverable

When all 5 groups are done, produce a single **Phase 3 Completion Report** saved as `docs/ai_context/PHASE3_EXECUTION_REPORT.md`:

1. One-row-per-task table: ID, status (3-valued), evidence file path, signature line of what changed.
2. Top-3 findings that surfaced (e.g. "all 3 boot-blockers confirmed and patched; T16 E2E works on CPU but no GPU backend so marked UNVERIFIED-PENDING").
3. Diff table: what `SYSTEM_STATE.md`, `PHASE1_DISCOVERY_REPORT.md`, `PHASE2_TODO_LIST.md` said before vs after, with evidence.

Then commit the report + execution log + unblock patches as one commit:
`git add -A && git commit -m "agent: phase 3 audit complete"`

Do **not** push. Push is a separate authorization.

---

## Resume Procedure If Session Is Cut Mid-Task

1. Read `docs/ai_context/AGENT_EXECUTION_LOG.md` — find the last entry of the most recent session.
2. Read `docs/session_state.md` — check the last recorded next-step.
3. Do not re-run the most recent in-progress task. Resume from the first non-completed task in the appropriate group.
4. If a task has a `DO NOT REPEAT` warning, that block must be present in `AGENT_EXECUTION_LOG.md` and **must be respected**.

---

*This prompt is the single entry point for Phase 3. All rules, evidence requirements, and protocol come from this file — do not improvise around any of them.*
