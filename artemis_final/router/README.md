# Artemis Router

**Text-Only Reward Router for VLM Model Selection**

The Artemis Router module provides intelligent model routing for Vision-Language tasks. It uses a trained reward model to predict which VLM (Vision-Language Model) will provide the best response for a given prompt and image, optimizing for Accuracy, Cost, or Latency.

## 🚀 Quick Start

### Python API (Recommended)

The easiest way to use the router is via the high-level API:

```python
from artemis_router.router_api import route_text
from PIL import Image

# Route a simple query
# (Router automatically loads model singleton on first call)
result = route_text(
    "What is shown in this diagram?",
    image=Image.open("diagram.jpg"),
    mode="balanced"
)

print(f"Chosen Model: {result['chosen_model']}")
# Output: 'qwen2_5_vl_7b'
```

### Installation

```bash
cd artemis_final/router
pip install -r requirements.txt
```

---

## 📂 Directory Layout

The router module is organized as follows:

- **`artemis_router/`**: The core Python package.
  - **`router_api.py`**: **Main Entry Point**. High-level functions (`route_text`, `route_sample`).
  - **`inference_reward_router.py`**: The core router engine (loads checkpoint, runs inference).
  - **`router_service.py`**: Service wrapper class (for dependency injection / apps).
  - **`schemas.py`**: Data classes (`Sample`, `RouterDecision`).
  - **`legacy/`**: Old router implementations (Classical, Pairwise) - kept for reference.

- **`notebooks/`**:
  - `00_reward_router_setup_and_test.ipynb`: Main walkthrough and test notebook.
  - `01_router_unit_test.ipynb`: Detailed unit tests.
  - `02_traffic_simulation.ipynb`: Performance stress testing.

- **`router_config_reward.yaml`**: Configuration file.

---

## 🧠 Core Concepts

### Reward Router
The current production router is a **Reward-based Router**. It predicts a scalar "reward" (quality score) for each of the 5 available VLMs.
- It is **Text-Only**: It uses the text prompt and **image metadata** (resolution, aspect ratio) but does not encode the image pixels itself (speed optimization).
- It takes a **Routing Mode** input to adjust its behavior.

### Routing Modes
| Mode | Objective | Best For |
|------|-----------|----------|
| **accuracy** | Maximize quality | Complex tasks, research, high-stakes |
| **cheap** | Minimize cost | High volume, simple tasks |
| **fast** | Minimize latency | Real-time apps, simple checks |
| **balanced** | Trade-off | General purpose default |

### Supported VLMs
Routes between 5 models:
- `deepseek_ocr`
- `qwen2_5_vl_3b`
- `qwen2_5_vl_7b`
- `qwen3_vl_8b_thinking`
- `gemma_3_27b`

---

## 🛠️ Detailed Usage

### Using `RouterService` (Class-based)

For applications where you want to manage the lifecycle (loading once) explicitly:

```python
from common.config_loader import GlobalConfig
from artemis_router.router_api import RouterService

# Initialize
config = GlobalConfig() # Loads from router_config_reward.yaml
router = RouterService(config)

# Predict
result = router.predict(
    prompt="Extract text",
    mode="fast"
)
print(result)
```

### Simulation / Stress Test

You can easily run a traffic simulation to check RPS (Requests Per Second) on your hardware:

```python
from artemis_router.router_api import run_traffic_simulation

# Generate synthetic traffic at 100 RPS for 10 seconds
results, stats = run_traffic_simulation(rps=100.0, duration_sec=10)

print(f"Avg Latency: {stats.avg_latency_ms:.2f}ms")
```

---

## 🧪 Testing

1. **Jupyter Notebooks**: Open `notebooks/00_reward_router_setup_and_test.ipynb` for a complete visual guide.
2. **Unit Tests**: Run `notebooks/01_router_unit_test.ipynb`.
3. **Manual Check**:
   ```bash
   python setup_router.py
   ```
   This script verifies dependencies, checks for checkpoints, and ensures the environment is ready.

---

## 📊 Logging & Retraining

The router is designed to improve over time.
- **Inference Logs**: Can be saved to SQL (`vlm_router_logs`) via the Load Balancer integration.
- **W&B**: Supports Weights & Biases logging for experiments.
- **Retraining**: Use the logs to fine-tune the reward predictor (see `router_train` module documentation).

## ⚠️ Troubleshooting

- **Checkpoint Not Found**: Ensure you have `best_reward_router.pt` in `artemis_final/checkpoints/`.
- **Import Errors**: Make sure your `PYTHONPATH` acts from `artemis_final` or install the package in editable mode.
- **Device Support**: Automatically selects CUDA > MPS > CPU. pass `device='cpu'` explicitly if needed.

