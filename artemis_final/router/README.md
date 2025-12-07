# Artemis Router - Reward Router Inference Module

**Text-Only Router for VLM Model Selection**

This module provides inference capabilities for the trained **Reward Router**, which predicts the best VLM model for each query based on text prompts and routing modes.

---

## 📖 Table of Contents

1. [Quick Start (30 seconds)](#quick-start)
2. [Architecture Overview](#architecture)
3. [Notebooks](#notebooks)
4. [Python API Usage](#python-api)
5. [Routing Modes](#routing-modes)
6. [Configuration](#configuration)
7. [Load Balancer Integration](#load-balancer-integration)
8. [Performance](#performance)
9. [Troubleshooting](#troubleshooting)
10. [File Structure](#file-structure)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd artemis_final/router
pip install -r requirements.txt
```

### 2. Test the Router

Open and run the setup notebook:

```bash
jupyter notebook notebooks/00_reward_router_setup_and_test.ipynb
```

Or test via Python:

```python
from artemis_router.inference_reward_router import RewardRouterInference

router = RewardRouterInference(
    checkpoint_path='../checkpoints/best_reward_router.pt',
    device='cpu'  # or 'cuda:0' or 'mps'
)

result = router.route(
    prompt="What is shown in this diagram?",
    mode="balanced",
    metadata={'router_task': 'diagram_reasoning', 'source_dataset': 'ai2d'}
)

print(f"Route to: {result['chosen_model']}")
print(f"Rewards: {result['rewards']}")
```

**✓ If this works, you're ready to go!**

---

## 🏗️ Architecture

### What is the Reward Router?

The Reward Router is a **text-only** neural network that predicts which VLM model to use for a given query.

```
┌──────────────────────────────────────────────────────────┐
│  Text Prompt + Metadata                                  │
│  "What is shown in this diagram?"                        │
│  Task: diagram_reasoning, Mode: accuracy                 │
└─────────────────┬────────────────────────────────────────┘
                  ↓
         ┌────────────────┐
         │  DistilBERT    │  Text Encoder (768-dim)
         │   Encoder      │
         └────────┬───────┘
                  ↓
         ┌────────────────┐
         │ Model + Mode   │  Add learned embeddings
         │  Embeddings    │  (5 models × 32-dim + 4 modes × 16-dim)
         └────────┬───────┘
                  ↓
         ┌────────────────┐
         │  MLP Head      │  2 hidden layers (512-dim)
         │  (Reward Net)  │
         └────────┬───────┘
                  ↓
┌──────────────────────────────────────────────────────────┐
│  Predicted Rewards for Each Model                        │
│  deepseek_ocr:         0.23                              │
│  qwen2_5_vl_3b:        0.68                              │
│  qwen2_5_vl_7b:        0.85                              │
│  qwen3_vl_8b_thinking: 0.76                              │
│  gemma_3_27b:          0.92  ← CHOSEN (highest reward)  │
└──────────────────────────────────────────────────────────┘
```

### Key Points

- **No vision encoder** - Text-only for speed
- **Predicts rewards** - Not probabilities, but quality scores
- **5 VLM models** - Routes between 5 different VLMs
- **4 routing modes** - accuracy, cheap, fast, balanced
- **Trained on real data** - Profiling data from actual VLM responses

---

## 📓 Notebooks

Three notebooks are provided for testing and understanding the router:

### 1. [00_reward_router_setup_and_test.ipynb](notebooks/00_reward_router_setup_and_test.ipynb)

**Purpose:** Complete walkthrough of router usage

**Contents:**
- Load trained router checkpoint
- Test basic routing with sample prompts
- Compare all 4 routing modes
- Test multiple task types
- Performance analysis (latency, throughput)
- Visualizations

**Use when:** First-time setup, learning how router works

### 2. [01_router_unit_test.ipynb](notebooks/01_router_unit_test.ipynb)

**Purpose:** Unit tests for router functionality

**Contents:**
- Router initialization tests
- Basic routing tests
- Mode switching tests
- Batch processing tests
- Edge case handling
- Performance validation
- Determinism checks

**Use when:** Validating router works correctly, debugging issues

### 3. [02_traffic_simulation.ipynb](notebooks/02_traffic_simulation.ipynb)

**Purpose:** Traffic pattern simulation and performance testing

**Contents:**
- Constant rate traffic simulation
- Burst traffic patterns
- Mixed mode analysis
- Latency distribution analysis
- Model selection distribution

**Use when:** Performance testing, capacity planning

---

## 💻 Python API Usage

### Basic Usage

```python
from artemis_router.inference_reward_router import RewardRouterInference

# Initialize router
router = RewardRouterInference(
    checkpoint_path='../checkpoints/best_reward_router.pt',
    device='cpu',  # 'cuda:0' for GPU, 'mps' for Apple Silicon
    verbose=True
)

# Route a single request
result = router.route(
    prompt="What is the capital of France?",
    mode="balanced",
    metadata={'router_task': 'qa', 'source_dataset': 'test'}
)

# Access results
print(result['chosen_model'])  # e.g., 'qwen2_5_vl_3b'
print(result['rewards'])       # {model: reward} for all 5 models
print(result['mode'])          # 'balanced'
print(result['inference_ms'])  # e.g., 12.4
```

### Multiple Prompts

```python
prompts = [
    {
        "prompt": "Extract text from this document.",
        "mode": "fast",
        "metadata": {'router_task': 'ocr', 'source_dataset': 'docvqa'}
    },
    {
        "prompt": "Analyze this complex chart.",
        "mode": "accuracy",
        "metadata": {'router_task': 'chartqa', 'source_dataset': 'chart2text'}
    },
]

for p in prompts:
    result = router.route(**p)
    print(f"{p['prompt'][:30]}... → {result['chosen_model']}")
```

### Get Router Stats

```python
stats = router.get_stats()
print(f"Device: {stats['device']}")
print(f"Models: {stats['model_names']}")
print(f"Modes: {stats['mode_names']}")
```

---

## 🎯 Routing Modes

The router supports 4 different routing modes, each optimized for different objectives:

| Mode | Objective | Typical Choice | Best For |
|------|-----------|----------------|----------|
| **accuracy** | Maximize quality | `gemma_3_27b` (largest) | Research, critical tasks, when quality matters most |
| **cheap** | Balance quality/cost | `qwen2_5_vl_3b` (smallest) | High-volume processing, budget constraints |
| **fast** | Balance quality/latency | `qwen2_5_vl_3b` (smallest) | Real-time applications, low-latency requirements |
| **balanced** | Multi-objective | `qwen2_5_vl_7b` (medium) | General-purpose, default choice |

### Mode Selection Guide

```python
# Critical research task - want best quality
result = router.route(prompt, mode="accuracy", metadata=...)

# Processing millions of samples - need to save costs
result = router.route(prompt, mode="cheap", metadata=...)

# Real-time chatbot - need instant responses
result = router.route(prompt, mode="fast", metadata=...)

# Not sure - want good balance
result = router.route(prompt, mode="balanced", metadata=...)
```

### Example: Same Prompt, Different Modes

```python
prompt = "Analyze this complex scientific diagram."
metadata = {'router_task': 'diagram_reasoning', 'source_dataset': 'ai2d'}

for mode in ["accuracy", "cheap", "fast", "balanced"]:
    result = router.route(prompt, mode=mode, metadata=metadata)
    print(f"{mode:12s} → {result['chosen_model']}")

# Output:
# accuracy     → gemma_3_27b
# cheap        → qwen2_5_vl_3b
# fast         → qwen2_5_vl_3b
# balanced     → qwen2_5_vl_7b
```

---

## ⚙️ Configuration

### Checkpoint Path

The router loads from trained checkpoint files. Available checkpoints:

```
artemis_final/checkpoints/
├── best_reward_router.pt        ← RECOMMENDED (180MB, text-only)
├── best_pairwise_router.pt      (254MB)
└── best_classical_router.pt     (254MB)
```

### Device Selection

```python
# CPU (slowest, most compatible)
router = RewardRouterInference(checkpoint_path=..., device='cpu')

# NVIDIA GPU (fastest)
router = RewardRouterInference(checkpoint_path=..., device='cuda:0')

# Apple Silicon (M1/M2/M3)
router = RewardRouterInference(checkpoint_path=..., device='mps')
```

### Metadata Format

Metadata provides context to the router. Fields:

- `router_task` (str): Task type (e.g., 'vqa', 'ocr', 'chartqa', 'diagram_reasoning')
- `source_dataset` (str): Dataset name (e.g., 'ai2d', 'docvqa', 'test')

Example:

```python
metadata = {
    'router_task': 'ocr',
    'source_dataset': 'docvqa'
}
```

**Note:** Metadata is optional but recommended for better routing decisions.

---

## 🔗 Load Balancer Integration

The router is designed to work with a load balancer that dispatches requests to VLM backends.

### Integration Flow

```
User Request
     ↓
[Router Inference]
     ├─ Predict rewards for all models
     └─ Choose model with highest reward
     ↓
[Routing Decision]
     ├─ chosen_model: "qwen2_5_vl_7b"
     ├─ rewards: {...}
     └─ mode: "balanced"
     ↓
[Load Balancer]
     ├─ Receive routing decision
     ├─ Dispatch to VLM backend
     └─ Get VLM response
     ↓
[Return to User]
```

### Example Integration Code

```python
import requests
import time

# Route the request
result = router.route(
    prompt="What is shown?",
    mode="balanced",
    metadata={'router_task': 'vqa', 'source_dataset': 'test'}
)

# Send to load balancer
lb_message = {
    "sample_id": "req_001",
    "chosen_model": result['chosen_model'],
    "rewards": result['rewards'],
    "mode": "balanced",
    "timestamp": time.time(),
    "prompt": "What is shown?",
}

# POST to load balancer endpoint
response = requests.post(
    "http://localhost:8000/route",
    json=lb_message
)

vlm_response = response.json()
```

### Load Balancer Interface (Future)

The `lb_interface.py` module provides helper functions for LB integration:

```python
from artemis_router.lb_interface import send_to_lb, format_lb_message

# Format message
lb_msg = format_lb_message(
    sample_id="req_001",
    router_result=result,
    prompt="What is shown?",
)

# Send to LB
vlm_response = send_to_lb(lb_msg, lb_url="http://localhost:8000/route")
```

---

## ⚡ Performance

### Router Inference Latency

| Configuration | Latency (P50) | Latency (P95) | Throughput |
|---------------|---------------|---------------|------------|
| CPU | 20-50ms | 50-100ms | 20-50 RPS |
| GPU (CUDA) | 5-15ms | 15-30ms | 100-200 RPS |
| Apple Silicon (MPS) | 10-25ms | 25-50ms | 50-100 RPS |

### End-to-End Latency (Router + VLM)

| Component | Latency | Notes |
|-----------|---------|-------|
| Router inference | 5-50ms | Depends on device |
| Load balancer | 1-5ms | Network + dispatch |
| VLM inference | 100-5000ms | Depends on model size |
| **Total** | **106-5055ms** | VLM dominates |

**Key Insight:** Router adds <5% overhead on GPU

### Performance Tips

1. **Use GPU** - 5-10x faster than CPU
2. **Batch requests** - Process multiple at once (future feature)
3. **Cache results** - Same prompt → same routing decision
4. **Monitor latency** - Use P95/P99 for capacity planning

---

## 🔧 Troubleshooting

### Router returns random/incorrect results

**Cause:** Text formatting mismatch with training

**Solution:** Ensure metadata is provided. The router formats text as:
```
[ROUTER] Task: {task}. Dataset: {dataset}. Question: {prompt}
```

This is handled automatically by `inference_reward_router.py`.

### "Checkpoint not found" error

**Cause:** Incorrect path to checkpoint file

**Solution:** Update checkpoint path:
```python
checkpoint_path = '../checkpoints/best_reward_router.pt'  # Relative
# OR
checkpoint_path = '/absolute/path/to/best_reward_router.pt'  # Absolute
```

### Slow inference (>100ms on GPU)

**Possible causes:**
1. Using CPU instead of GPU
2. GPU not properly configured
3. First inference (model loading overhead)

**Solutions:**
```python
# Check device
print(router.device)  # Should be 'cuda:0' or 'mps'

# Warm up the model
for _ in range(5):
    router.route("test", mode="balanced", metadata={})
```

### ImportError: cannot import 'RewardRouterInference'

**Cause:** Module path not set correctly

**Solution:**
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))  # If in notebooks/
# OR
sys.path.insert(0, '/path/to/artemis_final/router')  # Absolute path
```

---

## 📂 File Structure

```
artemis_final/router/
├── README.md                              ← This file
├── requirements.txt                        ← Python dependencies
│
├── artemis_router/                         ← Main module
│   ├── __init__.py
│   ├── inference_reward_router.py          ⭐ Main inference API
│   ├── api_io.py                           Helper for API I/O
│   ├── lb_interface.py                     Load balancer interface
│   ├── logging_wandb.py                    W&B logging (optional)
│   ├── schemas.py                          Data schemas
│   └── traffic_simulator.py                Traffic simulation utils
│
├── notebooks/                              ← Jupyter notebooks
│   ├── 00_reward_router_setup_and_test.ipynb  ⭐ Main walkthrough
│   ├── 01_router_unit_test.ipynb            Unit tests
│   └── 02_traffic_simulation.ipynb          Traffic simulations
│
└── router_config_reward.yaml               Configuration file (optional)
```

### Key Files

- **`inference_reward_router.py`** - Main inference wrapper, use this!
- **`00_reward_router_setup_and_test.ipynb`** - Start here to learn
- **`01_router_unit_test.ipynb`** - Validate functionality
- **`README.md`** - This documentation

### Deleted Files (Cleaned Up)

The following old multimodal router files have been removed:
- `router_engine.py` (old multimodal inference)
- `router_model.py` (old CLIP-based model)
- `feature_extractor.py` (old vision features)
- `db_io.py` (old database utils)
- `config.py` (old config loader)
- `router_config_example.yaml` (incompatible config)

---

## 📚 Additional Documentation

For more detailed information, see:

- **[COMPLETE_SYSTEM_OVERVIEW.md](../COMPLETE_SYSTEM_OVERVIEW.md)** - End-to-end architecture (training → inference → load balancing)
- **[router_train/README.md](../router_train/README.md)** - Training pipeline documentation
- **[router_train/ENHANCED_TRAINING_GUIDE.md](../router_train/ENHANCED_TRAINING_GUIDE.md)** - Training best practices

---

## ✅ Validation Checklist

Before using in production:

- [ ] Router loads checkpoint successfully
- [ ] Test routing on sample prompts works
- [ ] All 4 routing modes functional
- [ ] Performance meets requirements (latency, throughput)
- [ ] Load balancer endpoint configured
- [ ] Monitoring/logging set up

---

## 🎓 Key Takeaways

1. **Use `best_reward_router.pt`** - Most flexible, recommended checkpoint
2. **Use `inference_reward_router.py`** - Correct API for text-only router
3. **Start with notebooks** - Best way to learn and test
4. **Choose the right mode** - accuracy/cheap/fast/balanced based on needs
5. **Use GPU for production** - Much faster than CPU (~10x)
6. **Monitor in production** - Track latency and routing decisions

**You're ready to use the trained router! 🚀**

For questions or issues, refer to the notebooks and this documentation.
