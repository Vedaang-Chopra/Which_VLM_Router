# Artemis VLM Router

**Artemis** is a modular, high-performance routing system for Vision-Language Models (VLMs). It intelligently dispatches user requests (image + text) to the most appropriate VLM backend based on the query's complexity, cost constraints, latency requirements, and model strengths.

The system consists of three core components working in unison:
1.  **Router**: A lightweight neural or classical module that analyzes the input prompt/image and predicts the "utility" or score of each available VLM.
2.  **Load Balancer**: A system-aware scheduler that takes the router's preferences and makes the final routing decision, enforcing global SLAs (cost/latency) and managing model queue capacities.
3.  **Inference Layer**: A unified interface to execute requests against various VLM backends (e.g., vLLM endpoints).

## Key Features

-   **Multi-Modal Routing**: Optimized for Visual Question Answering (VQA), OCR, Captioning, and Reasoning tasks.
-   **SLA-Aware Scheduling**: Balances accuracy against cost and latency budgets in real-time.
-   **Multiple Routing Modes**:
    -   `accuracy`: Prioritizes the most capable model regardless of cost.
    -   `cheap`: Minimizes cost while maintaining acceptable quality.
    -   `fast`: Optimizes for lowest latency.
    -   `balanced`: A weighted combination of all factors.
-   **Model Agnostic**: Supports any VLM backend (Gemma 3, Qwen-VL, DeepSeek-OCR, Llama-4, etc.).
-   **Extensible**: Modular design allowing plug-and-play replacement of routing logic or load balancing strategies.

## Repository Layout

The codebase is organized as follows:

-   `artemis_final/router/`: The core routing logic (neural, reward-based, classical routers) and training code.
-   `artemis_final/load_balancer/`: Capacity-aware scheduling, SLA monitoring, and system metrics.
-   `artemis_final/inference_engine/`: Unified client for calling VLM APIs (vLLM, OpenAI, etc.).
-   `artemis_final/common/`: Shared utilities and the **centralized configuration** loader.
-   `artemis_final/examples/`: **Start here.** Jupyter notebooks demonstrating end-to-end usage.
-   `artemis_final/01_end_to_end_image_inference.ipynb`: The primary "Hello World" notebook.

## Installation & Setup

### 1. Environment
Activate the project virtual environment (assuming standard deployment):

```bash
source /home/hice1/vchopra37/scratch/projects/vlm_router/vlm_router_env/bin/activate
```

### 2. Dependencies
Install the package in editable mode to ensure all imports work correctly:

```bash
pip install -e .
```

### 3. Model Serving (vLLM)
Artemis expects VLM backends to be available. Below are standard commands to serve supported models using vLLM. Ensure you allocate appropriate GPUs.

**Gemma 3 27B Instruct (Port 8000/8001)**
```bash
CUDA_VISIBLE_DEVICES=0 nohup vllm serve google/gemma-3-27b-it --trust-remote-code --dtype bfloat16 --host 0.0.0.0 --port 8000 --max-model-len 32768 > gemma3_27b_it_1.log 2>&1 &
```

**Qwen3-VL 8B Thinking (Port 8002/8003)**
```bash
CUDA_VISIBLE_DEVICES=0 nohup vllm serve Qwen/Qwen3-VL-8B-Thinking --trust-remote-code --dtype bfloat16 --host 0.0.0.0 --port 8002 --max-model-len 32768 --gpu-memory-utilization 0.60 --reasoning-parser qwen3 > qwen3_vl_8b_thinking_1.log 2>&1 &
```

**Qwen2.5-VL 7B Instruct (Port 8004/8005)**
```bash
CUDA_VISIBLE_DEVICES=0 nohup vllm serve Qwen/Qwen2.5-VL-7B-Instruct --trust-remote-code --dtype bfloat16 --host 0.0.0.0 --port 8004 --max-model-len 32768 > qwen_vlm_7b_1.log 2>&1 &
```

**DeepSeek OCR (Port 8008/8009)**
```bash
CUDA_VISIBLE_DEVICES=0 nohup vllm serve deepseek-ai/DeepSeek-OCR --trust-remote-code --dtype bfloat16 --host 0.0.0.0 --port 8008 > deepseek_ocr1.log 2>&1 &
```

**Glider (Port 8010/8011)**
```bash
CUDA_VISIBLE_DEVICES=0 nohup vllm serve PatronusAI/glider --trust-remote-code --dtype bfloat16 --host 0.0.0.0 --port 8010 --max-model-len 32768 --max-num-seqs 32 > glider1.log 2>&1 &
```

**Llama-4 Scout 17B (Port 8012/8013)**
```bash
CUDA_VISIBLE_DEVICES=0 nohup vllm serve nvidia/Llama-4-Scout-17B-16E-Instruct-FP8 --trust-remote-code --kv-cache-dtype fp8 --tensor-parallel-size 1 --max-model-len 65536 --gpu-memory-utilization 0.85 --host 0.0.0.0 --port 8012 > llama4_scout_judge_fp8_64k_gpu1.log 2>&1 &
```

---

## Configuration

Configuration is the **single source of truth** for Artemis. It ensures all components (Router, Load Balancer, Inference) view the system consistently.

**Main Config File**: `artemis_final/common/artemis.yaml`

This file is critical because it defines:
1.  **Database**: Connection string for logging and stats.
2.  **Router**: Path to the trained checkpoint (`checkpoints/best_multitask_router_v1.pt`) and model architecture.
3.  **Load Balancer SLAs**: Global budgets (`total_cost_budget_usd`), latency targets (`default_latency_ms`), and task-specific constraints (e.g., stricter accuracy for OCR).
4.  **Models Registry**: The most important section. It lists every available model backend, its URL, pricing, throughput limits (`max_qps_per_replica`), and expected latency.

**Example `artemis.yaml` segment:**
```yaml
load_balancer:
  default_scheduling_mode: "capacity_aware"
  task_slas:
    ocr: { max_latency_ms: 1000, min_accuracy: 0.92 }

models:
  - name: qwen2_5_vl_7b
    base_url: http://localhost:8804/v1
    pricing: { prompt_per_1k: 0.0002, completion_per_1k: 0.0002 }
    max_qps_per_replica: 1.2
```

To modify system behavior (e.g., add a new model, change costs), edit this file.

---

## Quickstart (End-to-End Execution)

The best way to run Artemis is to use the provided Jupyter notebook which initializes the full pipeline.

**File**: `artemis_final/01_end_to_end_image_inference.ipynb`

### Minimal Python Workflow
If you prefer a script, here is how the components interact programmatically:

```python
import time
from common.config_loader import load_global_config
from router.router_service import RouterService
from load_balancer.load_balancer_service import LoadBalancerService
from inference_engine.inference_service import InferenceService

# 1. Initialize System
config = load_global_config("configs/artemis.yaml")
router = RouterService(config)
lb = LoadBalancerService(config)
engine = InferenceService(config)

# 2. Define Request
prompt = "Describe this image."
image = "path/to/image.jpg"
mode = "balanced" # Options: accuracy, cheap, fast
sample_id = "req_001"

# 3. Router Step: Get model probabilities/scores
router_output = router.predict(prompt, mode=mode)
# Returns: {'chosen_model': 'qwen...', 'rewards': {...}}

# 4. Load Balancer Step: Make final scheduling decision based on load/SLA
# This step might override the router if the preferred model is overloaded.
decision = lb.schedule(
    sample_id=sample_id,
    task_type="vqa",
    router_probs=router_output['rewards'],
    preferred_model=router_output['chosen_model']
)
final_model = decision['chosen_model']

# 5. Inference Step: Execute
result = engine.call_model(final_model, prompt, image)
print(f"Response from {final_model}: {result['text']}")
```

---

## Router Module – Public API

The Router analyzes the input to determine which model *should* ideally handle the request.

**Import**: `from artemis_final.router.public_api import ...`

### Key Functions

*   **`init_router(config_path: Optional[str])`**
    Initializes the global router instance using the specified config.

*   **`route_request(prompt: str, mode: str, metadata: dict) -> Dict`**
    Main entrypoint.
    *   `prompt`: User query.
    *   `mode`: 'accuracy', 'cheap', 'fast', 'balanced'.
    *   **Returns**: Dictionary with `chosen_model` and `rewards` (scores for all models).

*   **`load_router_from_checkpoint(router_type, checkpoint_path, ...)`**
    Advanced usage for research/notebooks to load a specific trained model (e.g., 'reward', 'classical') directly from a `.pt` file without the full service wrapper.

**Example**:
```python
from artemis_final.router.public_api import load_router_from_checkpoint

router = load_router_from_checkpoint("reward", "checkpoints/best_router.pt")
calc = router.route("What does the text say?", mode="accuracy")
print(f"Recommended Model: {calc['chosen_model']}")
```

---

## Load Balancer Module – Public API

The Load Balancer ensures the system remains stable and cost-effective. It takes the router's "wish" and realities of system load to make a binding decision.

**Import**: `from artemis_final.load_balancer.public_api import ...`

### Key Functions

*   **`init_load_balancer(config_path: str)`**
    Initializes the global scheduler, loading model capacity constraints from `artemis.yaml`.

*   **`schedule_request(sample_id, task_type, router_probs, preferred_model) -> Dict`**
    Decides where to send the request.
    *   `router_probs`: The output scores from the Router.
    *   **Returns**: A decision dict containing:
        *   `chosen_model`: The final model to use.
        *   `is_overloaded`: Boolean flag if system is under stress.
        *   `estimated_latency_ms`: Predicted latency.

*   **`get_metrics() -> Dict`**
    Returns current system health stats (e.g., avg latency, SLA violation rates).

*   **`simulate_traffic(arrival_rate, duration_s, ...)`**
    Runs a synthetic load test to validate configuration/SLAs without real GPU backends.

**Example**:
```python
from artemis_final.load_balancer.public_api import init_load_balancer, schedule_request

init_load_balancer() # Loads default config
decision = schedule_request(
    sample_id="test_1",
    task_type="ocr",
    router_probs={"deepseek_ocr": 0.9, "gemma": 0.1},
    preferred_model="deepseek_ocr"
)
print(f"Routing to: {decision['chosen_model']}")
```

---

## FAQ / Troubleshooting

**Q: "Router checkpoint not found" error during initialization?**
A: Check `artemis.yaml`. The `checkpoint_path` must point to a valid `.pt` file relative to the repo root. Ensure you have downloaded the weights to `artemis_final/checkpoints/`.

**Q: The Load Balancer keeps selecting "cheap" models even in "accuracy" mode.**
A: Check your budgets in `artemis.yaml`. If `total_cost_budget_usd` is exhausted or if the cheap model's score is significantly higher due to calibration, the LB forces a downgrade. Try increasing the budget or adjusting `RouterConfig`.

**Q: Inference fails with connection errors.**
A: Ensure your `vllm serve` commands are running and the ports match those defined in `artemis.yaml`. You can test connectivity with `curl http://localhost:8000/v1/models`.

**Q: How do I add a new model?**
A: Add a new entry to the `models` list in `artemis_final/common/artemis.yaml`. You must specify its `base_url`, `pricing`, and `base_latency_ms` for the load balancer to schedule it correctly.
