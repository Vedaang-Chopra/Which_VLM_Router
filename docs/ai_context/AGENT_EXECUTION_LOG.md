# Agent Execution Log

# Location: docs/ai_context/AGENT_EXECUTION_LOG.md

# Updated by: every agent, after every task attempt (success or failure)

# Read by:    every agent, before starting any task

#

# PURPOSE: Prevent repeated mistakes across sessions and across models

# If an approach is marked FAILED, do not repeat it without explicit human override

---

## HOW TO USE THIS LOG

### Before starting any task

1. Open this file.
2. Search for entries matching the files, modules, or approach you plan to use.
3. If a FAILED entry exists for your planned approach: use the Resolution instead.
4. If no entry exists: proceed, but write one after completing.

### After completing or failing any task

Add an entry at the top of the Entries section using the template below.
Do not skip this step. It is a required output of every task.

---

## Entry Template

Copy this block and fill it in. Add at the TOP of the Entries section.

```markdown
## [TASK_ID] — [Task name]
**Date:** YYYY-MM-DD
**Agent:** Pi | Codex | Claude Code
**Model:** [model name actually used]
**Status:** ✓ Complete | ✗ Failed | ⚠ Partial

**Approach taken:**
[1–3 sentences describing what was tried]

**What worked:**
[Specific things that succeeded — be concrete]

**What failed:**
[Specific things that failed — include exact error messages or symptoms]

**Root cause:**
[Why it failed — be specific. "Wrong import path" not just "import error"]

**Resolution:**
[What fixed it, or what the next agent should try instead]

**Files modified:**
- `path/to/file.py`

**Verify result:**
[Exact command and output, or "Verification not run — reason"]

**Model fallback used:** yes | no
[If yes: which model was tried first, why it failed, what fallback was used]

**DO NOT REPEAT:**
[Specific anti-patterns, import paths, approaches that failed for this task/module]
```

---

## Entries

## SYNC-003 — Force sync local branch to official

**Date:** 2026-07-08
**Agent:** Pi
**Model:** local
**Status:** ✓ Complete

**Approach taken:**
Fetched the latest remote state, confirmed local `main` was the authoritative branch, and force-pushed `main` to `official/main`.

**What worked:**
`git push --force-with-lease official main` updated the GitHub remote successfully, and the remote now points at local HEAD.

**What failed:**
Nothing failed. The branch was divergent, so a normal push would not have been sufficient.

**Root cause:**
The remote had newer sync/documentation commits that were lower priority than the local branch history.

**Resolution:**
Use a force push from local `main` to `official/main` when local history should win.

**Files modified:**

- `docs/ai_context/AGENT_EXECUTION_LOG.md`
- `docs/session_state.md`

**Verify result:**
`git rev-parse HEAD` matched `git rev-parse official/main` after the force push.

**Model fallback used:** no

**DO NOT REPEAT:**

- Do not use a normal push when local and remote histories have diverged and local should win.
- Do not forget the post-sync handoff docs after forcing the remote.

## AUDIT-001 — Phase 1 discovery and Phase 2 TODO list

**Date:** 2026-07-12
**Agent:** Pi
**Model:** local
**Status:** ✓ Complete

**Approach taken:**
Scanned the repository structure, reconstructed the module inventory from the code itself, resolved the authoritative codebase via git history, and saved the audit deliverables in `docs/ai_context/`.

**What worked:**
`git log`, file timestamp checks, checkpoint inspection, and direct code reads confirmed that `artemis_final/` produced the trained checkpoint and `multitask_eval_summary.csv`.

**What failed:**
Nothing failed during discovery. The main issue was stale documentation: several module counts/statuses in `SYSTEM_STATE.md` did not match the actual tree.

**Root cause:**
The existing docs were derived from an older scan and mixed ARTEMIS-native modules with copied baseline projects.

**Resolution:**
Saved a fresh Phase 1 discovery report and a Phase 2 executable TODO list under `docs/ai_context/` for review before execution.

**Files modified:**

- `docs/ai_context/PHASE1_DISCOVERY_REPORT.md`
- `docs/ai_context/PHASE2_TODO_LIST.md`
- `docs/session_state.md`

**Verify result:**
`git status --short` showed the new docs before commit.

**Model fallback used:** no

**DO NOT REPEAT:**

- Do not trust the old module inventory without re-scanning the filesystem.
- Do not assume `artemis_core/` produced the paper results; it was added later.
- Do not skip saving the audit output in `docs/ai_context/`.

<!-- New entries go HERE, at the top -->

## SYNC-002 — Sync gitignore changes

**Date:** 2026-07-08
**Agent:** Pi
**Model:** local
**Status:** ✓ Complete

**Approach taken:**
Committed the local `.gitignore` edits, pushed the branch to `official/main`, and then prepared the handoff docs so the sync state is recorded for the next agent.

**What worked:**
The `.gitignore` diff was small and committed cleanly. `git push official main` updated the remote branch successfully.

**What failed:**
Nothing failed in the commit/push flow. The only necessary follow-up was updating the execution log and session state.

**Root cause:**
The repo had a local `.gitignore` modification that had not been committed yet.

**Resolution:**
Commit the `.gitignore` update as `pi: Sync gitignore updates`, push to `official/main`, and record the sync in the docs.

**Files modified:**

- `.gitignore`
- `docs/ai_context/AGENT_EXECUTION_LOG.md`
- `docs/session_state.md`

**Verify result:**
`git rev-parse HEAD` matched `git rev-parse official/main` after the push.

**Model fallback used:** no

**DO NOT REPEAT:**

- Do not leave local `.gitignore` edits uncommitted when the request is to sync git changes.
- Do not skip the handoff docs after a successful git sync.

## SYNC-001 — Push branch with large-file cleanup

**Date:** 2026-07-08
**Agent:** Pi
**Model:** local
**Status:** ✓ Complete

**Approach taken:**
Rewrote branch history with `git-filter-repo` to remove `dataset/data/vlm_router_cache.db`, restored the remote after `git-filter-repo` deleted `origin`, cleaned `.gitignore`, then committed the validation notebook and force-pushed the rewritten branch.

**What worked:**
`git-filter-repo` successfully removed the oversized database file from all commits. After fetching the remote-tracking ref, `git push --force-with-lease official main` completed successfully and `HEAD` now matches `official/main`.

**What failed:**
A normal `git push` was rejected by GitHub with `GH001: Large files detected` because `dataset/data/vlm_router_cache.db` was still in history. An initial `--force-with-lease` push failed with `stale info` before fetching the updated remote-tracking ref.

**Root cause:**
The oversized DB file was committed in repo history, not just present in the working tree. `git-filter-repo` also removed the `origin` remote, so the local remote-tracking ref had to be refreshed before force-pushing.

**Resolution:**
Use `git-filter-repo --path dataset/data/vlm_router_cache.db --invert-paths --force`, then `git fetch official` and `git push --force-with-lease official main`.

**Files modified:**

- `examples/load_balancer/06_real_vlm_validation.ipynb`
- `.gitignore`
- `docs/ai_context/AGENT_EXECUTION_LOG.md`
- `docs/session_state.md`

**Verify result:**
`git status --short` returned clean, and `git rev-parse HEAD` matched `git rev-parse official/main`.

**Model fallback used:** no

**DO NOT REPEAT:**

- Do not rely on `git rm --cached` alone for a file already committed in history.
- Do not push before fetching after `git-filter-repo` removes the remote-tracking ref.
- Do not assume the remote is named `origin`; confirm with `git remote -v`.

<!-- Example entry — delete when real entries are added:

## EXAMPLE-1 — Build repair loop core
**Date:** 2026-06-01
**Agent:** Pi
**Model:** qwen2.5-coder:7b (local)
**Status:** ⚠ Partial

**Approach taken:**
Implemented the repair loop using a recursive call structure within `repair_loop.py`.

**What worked:**
The incremental B-Rep trajectory analysis logic was correct.
Logging with `getLogger(__name__)` worked cleanly.

**What failed:**
Import of `BRepUtils` from `src/cad_design/core/brep_utils.py` raised ModuleNotFoundError.
The CODEBASE_MAP.md had a stale path — `brep_utils` had moved to `src/utils/brep/`.

**Root cause:**
CODEBASE_MAP.md was not updated when brep_utils was refactored last session.

**Resolution:**
Corrected import to `from src.utils.brep.brep_utils import BRepUtils`.
Updated CODEBASE_MAP.md to reflect current location.

**Files modified:**
- `src/repair/repair_loop.py`
- `docs/ai_context/CODEBASE_MAP.md`

**Verify result:**
`pytest tests/test_repair_loop.py -v` → 8/8 passing

**Model fallback used:** no

**DO NOT REPEAT:**
- Do not import BRepUtils from `src/cad_design/core/`. It lives in `src/utils/brep/`.
- Do not trust CODEBASE_MAP.md without cross-checking against actual file tree for brep utilities.

-->
