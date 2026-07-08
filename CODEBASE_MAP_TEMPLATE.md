# CODEBASE_MAP.md

> AI-facing codebase map. This file is the single source of truth for module ownership.
> **Keep this file updated whenever module structure, APIs, or ownership changes.**
> Location: `docs/ai_context/CODEBASE_MAP.md`

---

## Project Overview

> One paragraph: what this project does, what problem it solves, and what the top-level pipeline looks like.

---

## Top-Level Directory Structure

```
project_root/
  AGENTS.md                  # Mandatory agent rules
  CONVENTIONS.md             # Code style standards
  SKILLS.md                  # Procedural workflows
  requirements.txt           # Python dependencies
  .env                       # Secrets and env config (not committed)
  .gitignore

  docs/
    specs/                   # Feature specification documents
    ai_context/              # This file and other agent-facing docs
    human_docs/              # Human-facing system docs

  notebooks/                 # Inspection and debugging notebooks

  utils/                     # Shared utilities across all modules
    llm_clients/
    data_loading/
    logging/
    visualization/
    config/

  module_a/                  # <describe responsibility>
  module_b/                  # <describe responsibility>

  tests/                     # Runnable tests
  archive/                   # Deprecated code (do not import from here)
```

---

## Module Ownership

> For each major module, describe what it owns and what it does NOT own.
> Update this table whenever a module is added, removed, or restructured.

| Module | Owns | Does NOT own |
|---|---|---|
| `module_a/` | Description of responsibility | What belongs elsewhere |
| `module_b/` | Description of responsibility | What belongs elsewhere |
| `utils/llm_clients/` | LLM API wrappers, retry logic, token counting | Prompt construction, output parsing |
| `utils/logging/` | Logger setup, log formatting | Business logic |

---

## Public Entry Points

> Where external code (notebooks, other modules, CLI) should import from.
> Anything not listed here is an internal file and should not be imported directly.

| Module | Public Interface | Description |
|---|---|---|
| `module_a` | `module_a/interfaces.py` | Public functions exposed to external consumers |
| `module_a` | `module_a/runners.py` | Workflow orchestration entry points |
| `module_b` | `module_b/api.py` | ... |
| `utils` | `utils/llm_clients/client.py` | ... |

---

## Shared Utilities

> Reusable components that are used across more than one module.
> If you need this functionality, import from here — do not reimplement.

| Utility | Location | What it provides |
|---|---|---|
| LLM client | `utils/llm_clients/` | Unified API for calling LLMs |
| Logger | `utils/logging/` | Standardized logging setup |
| Config loader | `utils/config/` | Loads `.env` and YAML configs |
| Visualization | `utils/visualization/` | Common plotting functions |

---

## Schemas and Data Contracts

> Canonical locations for shared schemas and typed data structures.
> Do not redefine these elsewhere.

| Schema | Location | Used by |
|---|---|---|
| `ExampleSchema` | `module_a/schemas/example.py` | module_a, module_b |

---

## What NOT to Duplicate

> Things that already exist and must be reused, not reimplemented.

- LLM calls → use `utils/llm_clients/`
- Logging setup → use `utils/logging/`
- Config loading → use `utils/config/`
- _(Add entries here as the project grows)_

---

## Dependency Direction

Dependencies must only flow in one direction. Lower layers cannot import from higher layers.

```
notebooks
    ↓ imports from
interfaces.py / runners.py / api.py
    ↓ imports from
orchestration / core
    ↓ imports from
utils / schemas
```

**Circular imports are forbidden.** If a circular dependency appears, the design is wrong.

---

## Known Gaps / Technical Debt

> Document known issues with the current structure. Update as debt is addressed.

- [ ] Gap or debt item 1
- [ ] Gap or debt item 2

---

## Module Inventory

> Public functions and classes exposed by each module through its interface layer.
> Update this table when public APIs are added, changed, or removed.
> This is what agents should check before implementing new functionality.

| Module | Public Symbol | Type | Description |
|---|---|---|---|
| `module_a.interfaces` | `run_feature()` | function | Runs the main feature pipeline end-to-end |
| `module_a.interfaces` | `FeaturePipeline` | class | Stateful pipeline with shared config |
| `module_b.api` | `get_result()` | function | Retrieves a stored result by ID |

---

## Changelog

> Brief log of structural changes to the codebase. Not feature changes — structural changes.

| Date | Change |
|---|---|
| YYYY-MM-DD | Initial structure created |
