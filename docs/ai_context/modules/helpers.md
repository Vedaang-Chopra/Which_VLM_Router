# Module: helpers
>
> Status: COMPLETE
> Directory: code_base/helpers/
> Last updated: 2026-06-21

## Purpose

Shared utility scripts for GPU monitoring and VLM response evaluation.

## Public API

| Function | File | What it does |
|---|---|---|
| `FastAPI /metrics` | `gpu_metrics.py` | Returns full system + GPU snapshot as JSON |
| `FastAPI /metrics/flat` | `gpu_metrics.py` | Flattened numeric key->float view of metrics |
| `FastAPI /health` | `gpu_metrics.py` | Liveness check |

## Internal Structure

| File | Layer | What it does |
|---|---|---|
| `gpu_metrics.py` | Core | FastAPI server exposing /metrics endpoint; uses psutil + pynvml; optional GPU via NVML; graceful degradation if libs unavailable |
| `llava-critic.py` | Utility | LLaVA-based critic for evaluating VLM responses |

## External Dependencies

`fastapi`, `uvicorn`, `psutil` (optional), `pynvml` (optional, for GPU metrics)

## Known Issues

None.

## What an agent must know before editing this module

- Run `python -m uvicorn gpu_metrics:app --host 0.0.0.0 --port 9000` on the same machine as the vLLM server.
- Gracefully degrades if psutil/pynvml are unavailable — returns empty dicts for unavailable metrics.
- The `/metrics` endpoint returns CPU, memory, disk, network, process, and GPU data — attach this JSON to vLLM requests for logging.
