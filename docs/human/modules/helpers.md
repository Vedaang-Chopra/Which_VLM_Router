# Module: helpers

## What It Does

Two standalone utility scripts for monitoring and evaluation. `gpu_metrics.py` is a FastAPI server that exposes system and GPU telemetry; `llava-critic.py` evaluates VLM responses using a LLaVA-based critic.

## How It Works

`gpu_metrics.py` uses `psutil` for system metrics (CPU, memory, disk, network) and `pynvml` for NVIDIA GPU telemetry. Both are optional — the server starts and degrades gracefully if they're absent.

## Key Files

| File | What it does |
|---|---|
| `gpu_metrics.py` | FastAPI app at port 9000. GET `/metrics` returns full JSON snapshot; GET `/metrics/flat` returns flattened floats; GET `/health` for liveness. |
| `llava-critic.py` | LLaVA-based response critic. Evaluates VLM outputs for quality. |

## Run

```bash
# On the vLLM server machine
pip install fastapi uvicorn psutil nvidia-ml-py3
python -m uvicorn gpu_metrics:app --host 0.0.0.0 --port 9000

# Then call from client
curl http://VLLM_HOST:9000/metrics
```

## Status

**COMPLETE.** Both scripts are functional. gpu_metrics.py gracefully handles missing optional dependencies.
