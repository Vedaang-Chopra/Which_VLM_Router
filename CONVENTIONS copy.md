# CONVENTIONS.md

Code style and structural standards for this codebase.
Apply to all Python code unless a spec explicitly overrides them.

---

## 1. Type Hints and Docstrings

All **public** functions and classes must have both.

```python
def load_dataset(path: str, split: str = "train") -> pd.DataFrame:
    """Load a dataset split from disk and return as a DataFrame."""
    ...

class EvaluationPipeline:
    """Runs evaluation over model outputs and returns scored results."""
    def __init__(self, config: EvalConfig) -> None: ...
```

- Docstring minimum: one sentence describing what the function/class does.
- Return type must always be annotated. Use `None` explicitly when there is no return.
- Use `Optional[X]` or `X | None` for nullable arguments.
- Private functions (`_name`) require type hints but may omit docstrings.

---

## 2. Import Organization

Within every file, use this order with a blank line between groups:

```python
# 1. Standard library
import os
import json
from pathlib import Path

# 2. Third-party
import numpy as np
import torch
from pydantic import BaseModel

# 3. Internal
from module_name.interfaces import run_pipeline
from utils.logging import get_logger
```

- No wildcard imports (`from module import *`).
- No unused imports.
- No circular imports.
- Always import from a submodule's public interface layer, not from deep internal files.

---

## 3. Classes vs Functions

**Use a class when:**
- Logic has shared state or shared configuration across methods.
- Multiple methods operate on the same concept or data.
- There is a lifecycle (init → run → teardown).
- The same arguments would otherwise be passed to many related functions repeatedly.

**Use a plain function when:**
- Logic is stateless.
- It is a single transformation, conversion, or utility.
- It is reusable across modules without needing shared context.
- It would be a one-method class with no state.

---

## 4. Function Design

- **One function = one responsibility.** If you are writing "and" in the function name, split it.
- **Target length: under 40 lines.** If longer, decompose into sub-functions.
- **Avoid deep nesting.** Flatten with early returns and guard clauses.
- **Validate inputs at boundaries.** Do not silently trust callers.

```python
# Bad
def process_and_evaluate_and_save(data, model, out_path):
    ...

# Good
def preprocess(data: RawData) -> ProcessedData: ...
def evaluate(model: Model, data: ProcessedData) -> Results: ...
def save_results(results: Results, out_path: str) -> None: ...
```

---

## 5. Runner / Orchestration Design

Runners compose; they do not implement.

```python
# Bad — one function owns the entire pipeline
def run_pipeline(inputs):
    # 150 lines of mixed logic

# Good — runner composes small, named stage functions
def load_inputs(path: str) -> Inputs: ...
def run_generation(inputs: Inputs) -> Outputs: ...
def run_evaluation(outputs: Outputs) -> Results: ...

def run_full_pipeline(path: str) -> Results:
    inputs = load_inputs(path)
    outputs = run_generation(inputs)
    return run_evaluation(outputs)
```

- Each stage function must be independently callable and testable.
- `run_full_pipeline()` (or equivalent) contains no business logic — only composition.

---

## 6. Directory Structure

Organize by responsibility, not flat by file count:

```
module_name/
  README.md              # module purpose and entry points (short)
  interfaces.py          # public API — what external code may import
  runners.py             # workflow orchestration
  schemas/               # dataclasses, Pydantic models, typed contracts
  core/                  # main algorithms and feature logic
  orchestration/         # pipeline composition (if separate from runners)
  data_loading/          # loaders, parsers, dataset access
  evaluation/            # metrics, scoring, validation logic
  visualization/         # plots, rendering, inspection helpers
  parsing/               # string/output parsing utilities
  utils/                 # local-only helpers (not shared across modules)
```

- Do not keep many unrelated files flat at the module root.
- If a directory contains more than ~5 unrelated Python files, group them.
- Every directory that acts as a module must have a `README.md`.

---

## 7. Shared Utilities

- If a utility is used by more than one module, it does not belong inside a single module.
- Move it to the project-level `utils/` or `common/` directory.
- `utils/` must not import from high-level feature modules. Dependency direction is one-way: utilities do not depend on features.

Common candidates for shared utilities:
- LLM client wrappers
- Data loading helpers
- Logging setup
- Config loading
- Visualization primitives
- Common evaluation helpers
- Schema validation helpers

---

## 8. Schemas and Data Contracts

- Define each schema once. Do not redefine the same dataclass or Pydantic model across multiple files.
- Shared schemas live in a top-level `schemas/` or in the owning module's `schemas/` directory.
- Public functions must document their return structure. If the return is complex, use a typed dataclass or Pydantic model.
- When a schema changes shape, update it in one place and version it if it affects downstream consumers.

---

## 9. Configuration and Secrets

- No hardcoded values in implementation files: model names, file paths, thresholds, table names, machine-specific settings.
- Configurable values go in a config file (YAML, JSON, or `.env`).
- Secrets and credentials go in `.env`. Add `.env` to `.gitignore`.
- Use `python-dotenv` or equivalent to load env variables.
- Load config once at startup; do not scatter `os.getenv()` calls throughout the codebase.

---

## 10. Error Handling and Logging

No silent failures. No fake success states.

```python
# Bad
try:
    result = call_llm(prompt)
except Exception:
    pass  # silent failure

# Good
try:
    result = call_llm(prompt)
except LLMCallError as e:
    logger.error(f"LLM call failed | prompt_id={prompt_id} | error={e}")
    raise
```

Log around meaningful operations:
- LLM calls (input summary, model, latency)
- Pipeline stage entry and exit
- File reads and writes
- Artifact creation
- Retries (attempt number, reason)
- Validation failures

Use log levels correctly: `DEBUG` for internals, `INFO` for stage progress, `WARNING` for recoverable issues, `ERROR` for failures.

---

## 11. Dependency Management

- Every new package must be added to `requirements.txt` or `pyproject.toml` immediately.
- Do not install packages inside notebooks without recording them in the dependency file.
- Pin versions for reproducibility in production modules.
- Do not import packages at the module level if they are only needed in one function — import inside the function and note the optional dependency.

---

## 12. Notebook Standards

- Notebooks are **inspection and debugging surfaces** — not implementation owners.
- Call production Python functions from notebook cells. Do not reimplement logic inside cells.
- Structure cells in sections matching the feature's logical stages.
- Include: inputs, intermediate states, outputs, artifact previews, edge case checks.
- Cells must run sequentially top-to-bottom with no undocumented hidden state.
- **Clear all cell outputs before committing a notebook.**
- If logic written in a notebook becomes reusable, move it into a Python module.
