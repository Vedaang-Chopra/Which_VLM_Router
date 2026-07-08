# SYSTEM_WORKFLOW_MAP.md

> AI-facing document. Describes how data and control flow through the system end-to-end.
> This is NOT the same as the codebase map (which covers ownership).
> This answers: **how does the system actually work, step by step?**
> Location: `docs/ai_context/SYSTEM_WORKFLOW_MAP.md`
> **Update this file whenever the end-to-end pipeline changes.**

---

## System Purpose

> One paragraph: what this system does from the perspective of inputs and outputs.
> What goes in? What comes out? What is the system solving?

---

## End-to-End Pipeline

> Describe the full flow from input to output as a numbered sequence.
> For each stage: what triggers it, what it consumes, what it produces, which module owns it.

```
INPUT: <describe the input — file, API call, database query, user request, etc.>
  │
  ▼
[Stage 1: <Name>]
  Owner:    module_a/runners.py → run_stage_a()
  Consumes: <input type>
  Produces: <output type>
  Notes:    <any important behavior, failure mode, or side effect>
  │
  ▼
[Stage 2: <Name>]
  Owner:    module_b/core/processor.py → ProcessorClass.run()
  Consumes: <stage 1 output>
  Produces: <output type>
  Notes:
  │
  ▼
[Stage 3: <Name>]
  Owner:    ...
  │
  ▼
OUTPUT: <describe the final output — file, API response, database write, artifact, etc.>
```

---

## Data Flow Diagram

> Optional but recommended for complex systems.
> Use ASCII or reference an external diagram file.

```
raw_input
  → load_inputs()        [module_a]
  → preprocess()         [utils/data_loading]
  → run_model()          [module_b]
  → parse_output()       [module_b/parsing]
  → evaluate()           [module_b/evaluation]
  → write_artifacts()    [module_a/runners]
```

---

## Key Decision Points

> Where does the system branch, retry, or make conditional choices?

| Decision Point | Location | Condition | Outcome |
|---|---|---|---|
| Retry on LLM failure | `utils/llm_clients/client.py` | HTTP error or timeout | Retry up to `MAX_RETRIES`, then raise |
| Skip evaluation | `module_b/runners.py` | `config.skip_eval = True` | Evaluation stage skipped |

---

## Side Effects and Artifacts

> What does the system write, create, or modify as a result of running?

| Artifact | Location | When created |
|---|---|---|
| Output file | `outputs/<run_id>/result.json` | End of pipeline |
| Log file | `logs/<run_id>.log` | Throughout pipeline |
| Trace record | `traces/<run_id>/` | If tracing enabled |

---

## Error Propagation

> How do errors move through the system? What is caught, what is re-raised, what produces a fallback?

- Stage 1 errors: raised immediately, pipeline halts. No partial output written.
- Stage 2 errors: caught and logged; fallback value `None` returned to stage 3.
- _(Fill in per system behavior)_

---

## Configuration Points

> What config values control system behavior? Where are they loaded?

| Config Key | Default | Effect |
|---|---|---|
| `MODEL_NAME` | `gpt-4o` | Which LLM is called in stage 2 |
| `MAX_RETRIES` | `3` | Retry budget for LLM calls |
| `SKIP_EVAL` | `false` | Skip evaluation stage |
| _(add more)_ | | |

Config is loaded from: `.env` via `utils/config/loader.py`

---

## Known Bottlenecks / Fragile Points

> Where does the system break under load, bad input, or unexpected state?

- [ ] Stage 2 LLM call has no timeout — hangs indefinitely on network issues.
- [ ] Output artifact naming is not collision-safe under concurrent runs.
- _(Add as discovered)_
