# Module: lovm
>
> Status: COMPLETE
> Directory: code_base/lovm/
> Entry point: lovm/lovm/lovm.py
> Last updated: 2026-06-20

## Purpose

Large Omnidirectional VLM Manager — orchestrates image tasks across multiple VLMs, provides profiling and benchmarking utilities. Research codebase.

## Public API

| Class | File | Purpose |
|---|---|---|
| `LOVM` | `lovm/lovm/lovm.py` | Main orchestration class |
| `profile_model` | `lovm/profile_model.py` | Model profiling utilities |

## Internal Structure

| File | Responsibility |
|---|---|
| `lovm/lovm/lovm.py` | LOVM — orchestration + task management |
| `lovm/profile_model.py` | Profiling functions for model comparison |
| `experiments/` | Benchmark scripts and experiment runners |

## Known Issues

- `lovm/lovm/lovm.py:54` — minor `return 0` placeholder for an unimplemented branch.

## What an Agent Must Know

- LOVM is a complete, self-contained research implementation. Use for benchmarking and profiling.
- It is not integrated with the main ARTEMIS router pipeline.
