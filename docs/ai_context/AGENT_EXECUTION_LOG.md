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

<!-- New entries go HERE, at the top -->

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
