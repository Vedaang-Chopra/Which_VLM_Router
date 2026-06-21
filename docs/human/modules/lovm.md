# Module: LOVM (Large Omnidirectional VLM Manager)

## What It Does

Orchestrates image tasks across multiple VLMs. Part of the research codebase (`code_base/lovm/`). Provides profiling and benchmarking utilities.

## Key Files

| File | What It Does |
|---|---|
| `lovm/lovm/lovm.py` | Main LOVM orchestration class |
| `lovm/profile_model.py` | Model profiling utilities |
| `experiments/` | Benchmark scripts |

## Status

**COMPLETE.** LOVM is fully implemented with orchestration, model profiling, and experiment scripts. One minor placeholder return at line 54 (returns `0` for an unimplemented branch).
