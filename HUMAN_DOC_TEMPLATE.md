# <Document Title>

> Human-facing documentation. Written for someone who understands the domain but has not read the code.
> Location: `docs/human_docs/<doc_name>.md`
> **Last updated:** YYYY-MM-DD

---

## What this document covers

> One sentence.

---

## Overview

> 2–3 sentences. What this system/module does, what problem it solves, and what a human needs to understand before reading further. No implementation details.

---

## Prerequisites

> What does someone need to have set up or know before using this?

- Python environment: `pip install -r requirements.txt`
- Environment variables: copy `.env.example` → `.env` and fill in values.
- _(other setup steps)_

---

## How to Run

> Step-by-step. Be specific. Include exact commands.

```bash
# Step 1: ...
python -m module_name.runners

# Step 2: check output
ls outputs/
```

Expected output: _(describe what success looks like)_

---

## How to Debug

> Common failure modes and where to look.

| Symptom | Likely cause | Where to look |
|---|---|---|
| `KeyError: 'model_name'` | Missing `.env` variable | `.env` file, `utils/config/` |
| Empty output file | LLM call failed silently | `logs/<run_id>.log`, stage 2 |
| Notebook cell fails on import | Module not installed | `requirements.txt`, `pip install` |

For deeper inspection, open `notebooks/<module>_inspection.ipynb` and run section by section.

---

## Key Concepts

> Explain 2–4 concepts a human must understand to work with this system.
> Use plain language. No code. Link to specs for deep detail.

**Concept A:** ...

**Concept B:** ...

---

## Common Mistakes

> What trips people up. Keep this list short and specific.

- Do not edit files in `archive/` — they are retired code.
- Do not hardcode config values directly in Python files. Use `.env`.
- _(add as patterns emerge)_

---

## How to Verify It Is Working

> Concrete, runnable checks.

```bash
pytest tests/test_module_name.py -v
```

Or open `notebooks/<feature>_inspection.ipynb` and confirm:
- Section 2 output matches expected shape: ...
- Section 4 artifact is written to `outputs/`.

---

## Related Documents

- Spec: `docs/specs/<NNN>_feature.md`
- AI context: `docs/ai_context/CODEBASE_MAP.md`
- Workflow map: `docs/ai_context/SYSTEM_WORKFLOW_MAP.md`
