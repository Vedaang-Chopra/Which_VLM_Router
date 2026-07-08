# Which_VLM_Router — Project AGENTS.md
# Global rules: ~/agent-governance/AGENTS.md — read that first.
# This file contains ONLY project-specific overrides.
# Do not repeat any rule from the global file.

---

## Global Rules

All global rules and skills apply unchanged.
Global rules: `~/agent-governance/AGENTS.md`
Skills root:  `~/agent-governance/skills/core/`

---

## Project Overview

**Project:** Which_VLM_Router — VLM Router Evaluation and Selection
**Language:** Python
**Environment:** (fill in conda env or venv path)
**Entry point:** `code_base/` — see README.md
**Test command:** (fill in)
**Conventions:** `CONVENTIONS.md` at project root

---

## Model Registry

| Alias | Models (priority order) | Rate tier | Use case |
|---|---|---|---|
| `planning` | (fill in) | paid | Spec, plan, decompose only |
| `complex-reasoning` | (fill in) | paid | Complex logic, debugging |
| `fast-code` | (fill in) | free (rate limited) | Boilerplate, simple functions |
| `local` | (fill in) | local, unlimited | Simple edits, formatting |

---

## Project Constraints

(No project-specific constraints identified in audit. Add constraints here as they are discovered.)
