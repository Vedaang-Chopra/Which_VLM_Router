# 🎯 Artemis VLM Router

**Intelligent Vision-Language Model Routing System**

Artemis is a complete, production-ready pipeline for dynamically routing requests to the optimal Vision-Language Model (VLM) based on task requirements, latency constraints, and cost considerations.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modules](#modules)
  - [Router](#router-artemis_finalrouter)
  - [Router Training](#router-training-artemis_finalrouter_train)
  - [Load Balancer](#load-balancer-artemis_finalload_balancer)
  - [Inference Engine](#inference-engine-artemis_finalinference_engine)
  - [ARES (Data & Evaluation)](#ares-data--evaluation-artemis_finalares)
  - [Data Loop](#data-loop-artemis_finaldata_loop)
  - [System API](#system-api-artemis_finalsystem_api)
  - [Common](#common-utilities-artemis_finalcommon)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Notebooks Guide](#notebooks-guide)
- [Docker Deployment](#docker-deployment)
- [Advanced Usage](#advanced-usage)

---

## Overview

Artemis routes incoming VLM requests to one of 5 supported models based on trained neural network predictions:

| Model | Description | Best For |
|-------|-------------|----------|
| `deepseek_ocr` | Specialized OCR model | Document extraction, text recognition |
| `qwen2_5_vl_3b` | Lightweight VLM (3B params) | Fast responses, cost-sensitive workloads |
| `qwen2_5_vl_7b` | Mid-size VLM (7B params) | Balanced performance/cost |
| `qwen3_vl_8b_thinking` | Reasoning-focused VLM (8B) | Complex reasoning, multi-step tasks |
| `gemma_3_27b` | Large VLM (27B params) | Maximum accuracy, research tasks |

### Key Features

✅ **Multi-strategy Routing** – Three router architectures (Reward, Pairwise, Classical)  
✅ **Multi-mode Support** – Accuracy, cheap, fast, or balanced optimization  
✅ **SLA-Aware Load Balancing** – Queue management with latency guarantees  
✅ **End-to-end Pipeline** – From training to inference to monitoring  
✅ **OpenAI-Compatible API** – Drop-in replacement for `/v1/chat/completions`  
✅ **Hot Reloading** – Retrain and swap models without downtime  

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ARTEMIS VLM ROUTER SYSTEM                        │
└─────────────────────────────────────────────────────────────────────────┘

                           User Request (prompt + image)
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  1. ROUTER                                                               │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ • Text Encoder (DistilBERT)                                        │  │
│  │ • Model & Mode Embeddings                                          │  │
│  │ • MLP Head → Predict rewards for all 5 models                      │  │
│  │ • Choose: argmax(rewards)                                          │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│  Output: {chosen_model, rewards, mode, inference_ms}                     │
└──────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  2. LOAD BALANCER                                                        │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ • SLA verification (latency targets)                               │  │
│  │ • Queue capacity checks                                            │  │
│  │ • Accuracy constraint enforcement                                  │  │
│  │ • May override router decision if constraints violated             │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│  Output: {final_model, queue_wait_ms, total_latency_ms}                  │
└──────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  3. INFERENCE ENGINE                                                     │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ • OpenAI-compatible client                                         │  │
│  │ • Multi-model support (LLM & VLM)                                  │  │
│  │ • Latency & cost tracking                                          │  │
│  │ • Confidence score extraction                                      │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│  Output: {response_text, usage, latency_ms, cost}                        │
└──────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  4. DATA COLLECTION & RETRAINING LOOP                                    │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ • Log all requests, decisions, responses to PostgreSQL             │  │
│  │ • Collect user feedback                                            │  │
│  │ • Trigger periodic retraining                                      │  │
│  │ • Hot-swap router checkpoints                                      │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                            Response to User
```

---

## Directory Structure

```
artemis_final/
├── README.md                          ← This file
├── COMPLETE_SYSTEM_OVERVIEW.md        ← Detailed architecture docs
├── requirements.txt                   ← Python dependencies
├── main.py                            ← FastAPI application entry point
├── Dockerfile                         ← Container definition
├── docker-compose.yml                 ← Full stack deployment
├── __init__.py                        ← Package init
│
├── configs/                           ← Configuration files
│   └── artemis.yaml                   ← Master configuration
│
├── checkpoints/                       ← Trained model checkpoints
│   ├── best_reward_router.pt          ← Recommended router
│   ├── best_pairwise_router.pt        ← Pairwise ranking router
│   └── best_classical_router.pt       ← CE/KL loss router
│
├── notebooks/                         ← 📓 ALL NOTEBOOKS (centralized)
│   ├── README.md                      ← Notebooks guide
│   ├── ares/                          ← Data & evaluation notebooks
│   ├── load_balancer/                 ← Load balancing tutorials
│   ├── router/                        ← Router inference notebooks
│   └── router_train/                  ← Training workflows
│
├── scripts/                           ← Demo & utility scripts
│   ├── run_demo.sh                    ← One-command demo
│   ├── demo_full_pipeline.py          ← Full pipeline demo
│   └── demo_retrain_improvement.py    ← Retraining showcase
│
├── router/                            ← Router inference module
│   ├── artemis_router/                ← Core router package
│   │   ├── inference_reward_router.py ⭐ Main inference API
│   │   ├── inference_pairwise_router.py
│   │   ├── inference_classical_router.py
│   │   ├── fallback.py                ← Confidence-based fallback
│   │   └── schemas.py
│   ├── router_config_reward.yaml
│   └── README.md
│
├── router_train/                      ← Router training module
│   ├── config.py, db_utils.py, reward_definitions.py
│   ├── models/                        ← Model architectures
│   ├── training/                      ← Training loops
│   └── README.md
│
├── load_balancer/                     ← SLA-aware load balancing
│   ├── scheduler.py                   ← ArtemisLoadBalancer core
│   ├── model_state.py, sla_monitor.py, stats_registry.py
│   └── README.md
│
├── inference_engine/                  ← VLM inference client
│   ├── client.py                      ← WhichVLMClient
│   ├── runners.py, config.py, messages.py
│   └── readme.md
│
├── ares/                              ← Data pipeline & evaluation
│   ├── configs/                       ← DB & model configs
│   ├── db/                            ← Database layer + migrations
│   ├── data/                          ← Dataset loading
│   ├── evaluation/                    ← Evaluation pipeline
│   └── utils/
│
├── data_loop/                         ← Online learning loop
│   ├── collector.py                   ← Data collection to DB
│   ├── error_tracker.py               ← Routing error tracking
│   └── retrainer.py                   ← Automated retraining
│
├── system_api/                        ← FastAPI application
│   ├── main.py, pipeline.py, schemas.py
│
└── common/                            ← Shared utilities
    ├── config_loader.py, types.py, db.py
```

---

## Installation

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ (for training data)
- CUDA-capable GPU (optional, recommended for production)

### Basic Installation

```bash
cd artemis_final

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### GPU Support

For NVIDIA GPU acceleration:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For Apple Silicon (M1/M2/M3):

```bash
pip install torch torchvision  # MPS backend is included
```

---

## Quick Start

### 1. Basic Router Usage

```python
from artemis_final.router.artemis_router import RewardRouterInference

# Initialize router
router = RewardRouterInference(
    checkpoint_path='checkpoints/best_reward_router.pt',
    device='cpu'  # or 'cuda' or 'mps'
)

# Route a request
result = router.route(
    prompt="What is shown in this diagram?",
    mode="balanced",  # accuracy, cheap, fast, or balanced
    metadata={
        'router_task': 'diagram_reasoning',
        'source_dataset': 'ai2d'
    }
)

print(f"Route to: {result['chosen_model']}")
print(f"Rewards: {result['rewards']}")
print(f"Inference time: {result['inference_ms']:.1f}ms")
```

### 2. Full Pipeline Usage

```python
from artemis_final.router.artemis_router import ClassicalRouterInference
from artemis_final.load_balancer import ArtemisLoadBalancer, RouterOutput, SchedulingContext
from artemis_final.inference_engine.client import WhichVLMClient

# 1. Initialize components
router = ClassicalRouterInference('checkpoints/best_classical_router.pt')
lb = ArtemisLoadBalancer(...)  # See load_balancer docs
client = WhichVLMClient.from_yaml('ares/configs/models.yaml')

# 2. Route
router_result = router.route(prompt, mode="balanced", metadata={...})

# 3. Load balance
router_output = RouterOutput(
    sample_id="sample_123",
    task_type="vqa",
    router_probs=router_result['rewards'],
    preferred_model=router_result['chosen_model']
)
lb_decision = lb.schedule(router_output, SchedulingContext(...))

# 4. Execute inference
result = client.vlm.run_image(
    image=image_data,
    text=prompt,
    models=[lb_decision.chosen_model]
)
```

### 3. Start the API Server

```bash
# Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Call the API

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "messages": [
            {"role": "user", "content": "What is in this image?"}
        ],
        "router_mode": "balanced"
    }
)

print(response.json())
```

---

## Modules

### Router (`artemis_final/router/`)

The router predicts the optimal VLM for each request using a trained neural network.

**Three Router Strategies:**

| Router | Training Method | Best For |
|--------|-----------------|----------|
| `RewardRouterInference` | Reward prediction (MSE loss) | General use, most flexible |
| `PairwiseRouterInference` | Pairwise ranking (margin loss) | High routing accuracy |
| `ClassicalRouterInference` | Classification (CE + KL loss) | Fast convergence |

**Usage:**

```python
from artemis_final.router.artemis_router import RewardRouterInference

router = RewardRouterInference(
    checkpoint_path='checkpoints/best_reward_router.pt',
    device='cuda'
)

# Get routing decision
result = router.route(
    prompt="Extract text from this receipt",
    mode="fast",
    metadata={'router_task': 'ocr', 'source_dataset': 'docvqa'}
)
```

**Routing Modes:**

| Mode | Objective | Typical Selection |
|------|-----------|-------------------|
| `accuracy` | Maximize quality | Largest model (gemma_3_27b) |
| `cheap` | Minimize cost | Smallest model (qwen2_5_vl_3b) |
| `fast` | Minimize latency | Fastest model (qwen2_5_vl_3b) |
| `balanced` | Multi-objective | Mid-size model (qwen2_5_vl_7b) |

📖 See [router/README.md](router/README.md) for detailed documentation.

---

### Router Training (`artemis_final/router_train/`)

Train new routers using profiling data from PostgreSQL.

**Training Pipeline:**

1. Load profiling data (samples, responses, evaluations)
2. Compute multi-objective rewards per (sample, model, mode)
3. Train router with chosen loss function
4. Evaluate against oracle and baselines
5. Export checkpoint

**Quick Training:**

```bash
cd router_train
python scripts/run_train_router.py --epochs 10 --batch-size 64 --device cuda
```

**Or use notebooks:**

```bash
jupyter notebook notebooks/02_reward_router_sql_to_training.ipynb
```

📖 See [router_train/README.md](router_train/README.md) for detailed training guide.

---

### Load Balancer (`artemis_final/load_balancer/`)

Post-router scheduling with SLA constraints and queue management.

**Features:**

- **SLA Verification** – Ensure latency targets are met
- **Queue Management** – Track per-model request queues
- **Autoscaling Simulation** – Scale replicas based on load
- **Constraint Enforcement** – Override router if model busy

**Scheduling Modes:**

| Mode | Description |
|------|-------------|
| `router_only` | Always use router's choice (no LB logic) |
| `capacity_aware` | Consider SLA + queue capacity |
| `cost_minimizing` | Pick cheapest model meeting constraints |

**Usage:**

```python
from artemis_final.load_balancer import (
    ArtemisLoadBalancer,
    RouterOutput,
    SchedulingContext,
    load_capacity_config,
    StatsRegistry,
    load_per_task_model_stats
)

# Setup
stats_registry = StatsRegistry(load_per_task_model_stats())
model_configs = load_capacity_config()

lb = ArtemisLoadBalancer(
    model_configs=model_configs,
    stats_registry=stats_registry,
    global_latency_sla_ms=2000.0,
    max_accuracy_drop=0.05,
    scheduling_mode="capacity_aware"
)

# Schedule
decision = lb.schedule(router_output, scheduling_context)
```

📖 See [load_balancer/README.md](load_balancer/README.md) for full documentation.

---

### Inference Engine (`artemis_final/inference_engine/`)

Unified client for calling multiple VLM backends via OpenAI-compatible APIs.

**Usage:**

```python
from artemis_final.inference_engine.client import WhichVLMClient

# Initialize from config
client = WhichVLMClient.from_yaml('ares/configs/models.yaml')

# LLM (text only)
llm_result = client.llm.run_single(
    prompt="Explain what a router is.",
    models="all"
)

# VLM (image + text)
vlm_result = client.vlm.run_image(
    image="photo.jpg",
    text="What is in this image?",
    models=["qwen2_5_vl_7b"]
)
```

📖 See [inference_engine/readme.md](inference_engine/readme.md) for API details.

---

### ARES (Data & Evaluation) (`artemis_final/ares/`)

Data pipeline for collecting, storing, and evaluating VLM responses.

**Components:**

- **Database Layer** – PostgreSQL schema and operations
- **Dataset Loading** – Load samples with images from DB
- **Evaluation Pipeline** – Score responses with judge models
- **Parallel Processing** – High-throughput inference

**Database Tables:**

| Table | Purpose |
|-------|---------|
| `vlm_samples` | Input samples (prompts, metadata) |
| `vlm_images` | Image data (blobs, dimensions) |
| `vlm_responses` | Model outputs (text, tokens, cost) |
| `vlm_evaluations` | Quality scores (accuracy, F1, judge scores) |

📖 See notebooks in `ares/notebooks/` for detailed workflows.

---

### Data Loop (`artemis_final/data_loop/`)

Online learning infrastructure for continuous improvement.

**Components:**

- **DataCollector** – Log live requests to PostgreSQL
- **Retrainer** – Periodic model retraining
- **Traffic Simulator** – Generate synthetic traffic for testing

**Usage:**

```python
from artemis_final.data_loop.collector import DataCollector

collector = DataCollector(config)

# Log a request
sample_id = collector.log_sample_start(
    request_id="req_001",
    router_mode="balanced",
    input_messages=[...],
    router_decision={...},
    lb_decision={...}
)

# Log response
collector.log_model_response(
    sample_id=sample_id,
    model_name="qwen2_5_vl_7b",
    raw_response={...},
    latency_ms=250
)
```

---

### System API (`artemis_final/system_api/`)

FastAPI application providing OpenAI-compatible endpoints.

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/chat/completions` | POST | Main inference endpoint |
| `/feedback` | POST | Submit user feedback |
| `/admin/retrain` | POST | Trigger retraining |

---

### Common Utilities (`artemis_final/common/`)

Shared configuration and type definitions.

**GlobalConfig:**

```python
from artemis_final.common.config_loader import load_global_config

config = load_global_config()

print(config.db.url)
print(config.router.checkpoint_path)
print(config.load_balancer.global_sla_ms)
```

---

## Configuration

### Master Config (`configs/artemis.yaml`)

```yaml
db:
  url: "postgresql+psycopg2://artemis:artemis@postgres:5432/artemis"

router:
  checkpoint_path: "checkpoints/best_reward_router.pt"
  config_file: "router/router_config_reward.yaml"
  device: "cpu"

load_balancer:
  config_file: "load_balancer/load_balancer_config.yaml"

inference_engine:
  models_file: "ares/configs/models.yaml"

retraining:
  epochs: 1
  batch_size: 8
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | From config |
| `CONFIG_PATH` | Path to artemis.yaml | `configs/artemis.yaml` |
| `ROUTER_CONFIG_PATH` | Override router config | From artemis.yaml |

---

## API Reference

### POST `/v1/chat/completions`

OpenAI-compatible chat completion endpoint.

**Request:**

```json
{
  "messages": [
    {"role": "user", "content": "What is in this image?"}
  ],
  "model": "router-auto",
  "router_mode": "balanced",
  "temperature": 0.7,
  "max_tokens": 512
}
```

**Response:**

```json
{
  "id": "req-uuid",
  "object": "chat.completion",
  "model": "qwen2_5_vl_7b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "This image shows..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 100,
    "total_tokens": 150
  }
}
```

---

## Notebooks Guide

All notebooks are now in the centralized `notebooks/` directory:

```bash
cd notebooks
jupyter notebook
```

### Router Notebooks (`notebooks/router/`)

| Notebook | Purpose |
|----------|---------|
| `01_understanding_router_architectures.ipynb` | Compare all 3 router strategies |
| `02_router_unit_tests.ipynb` | Validate router functionality |
| `03_experiments_and_load_testing.ipynb` | Performance benchmarking |

### Training Notebooks (`notebooks/router_train/`)

| Notebook | Purpose |
|----------|---------|
| `00_prepare_local_database.ipynb` | Cache PostgreSQL data locally |
| `02_reward_router_sql_to_training.ipynb` | **Main training workflow** ⭐ |
| `03_pairwise_ranking_router.ipynb` | Train pairwise router |
| `04_classical_ce_kl_router.ipynb` | Train CE/KL router |

### Load Balancer Notebooks (`notebooks/load_balancer/`)

| Notebook | Purpose |
|----------|---------|
| `00_pipeline_tutorial.ipynb` | **Full pipeline demo** ⭐ |
| `02_load_balancer_stress_test.ipynb` | Performance under load |

### Data & Evaluation Notebooks (`notebooks/ares/`)

| Notebook | Purpose |
|----------|---------|
| `01_parallel_inference_to_db.ipynb` | Run VLM inference at scale |
| `02_eval_scoring.ipynb` | Score responses with judges |
| `03_cost_utility_computation.ipynb` | Compute rewards |
| `04_eda_evaluations.ipynb` | Analyze evaluation data |

---

## Docker Deployment

### Quick Start

```bash
# Start full stack (PostgreSQL + Artemis API)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f artemis-router
```

### Production Deployment

1. **Build image:**
   ```bash
   docker build -t artemis-router:latest .
   ```

2. **Configure volumes:** Mount checkpoints and config:
   ```yaml
   volumes:
     - ./checkpoints:/app/checkpoints
     - ./configs:/app/configs:ro
   ```

3. **Set environment:**
   ```bash
   export DATABASE_URL=postgresql://...
   export CONFIG_PATH=configs/artemis.yaml
   ```

---

## Advanced Usage

### Custom Reward Functions

Edit `router_train/reward_definitions.py`:

```python
def compute_reward_custom(A, H, custom_metric, weights):
    """Custom reward combining accuracy and your metric."""
    return A * H + weights.custom_weight * custom_metric
```

### Adding New Models

1. Add to `ares/configs/models.yaml`:
   ```yaml
   - name: new_model
     base_url: http://localhost:9000/v1
     api_key: EMPTY
     model_id: org/new-model
     model_type: vlm
   ```

2. Update router config to include the new model in `model_name_order`.

3. Retrain the router with data from the new model.

### Hot-Swapping Routers

```python
# Via API
requests.post("http://localhost:8000/admin/retrain")

# Or programmatically
from artemis_final.data_loop.retrainer import Retrainer

retrainer = Retrainer(config, collector)
new_checkpoint = retrainer.retrain_once()
router_service.reload_model(new_checkpoint)
```

---

## Performance

### Router Latency

| Device | P50 Latency | P95 Latency | Throughput |
|--------|-------------|-------------|------------|
| CPU | 20-50ms | 50-100ms | 20-50 RPS |
| CUDA | 5-15ms | 15-30ms | 100-200 RPS |
| Apple MPS | 10-25ms | 25-50ms | 50-100 RPS |

### End-to-End Latency

| Component | Latency |
|-----------|---------|
| Router | 5-50ms |
| Load Balancer | 1-5ms |
| VLM Inference | 100-5000ms |
| **Total** | **~106-5055ms** |

**Key Insight:** Router overhead is <5% of total latency on GPU.

---

## License

Part of the Artemis VLM Router project.

---

## Citation

```bibtex
@software{artemis_router_2024,
  title={Artemis: Multi-Objective VLM Router},
  author={Vedaang Chopra},
  year={2024},
  url={https://github.com/Vedaang-Chopra/Which_VLM_Router}
}
```

---

## Support

- 📖 **Documentation:** See individual module READMEs
- 📓 **Notebooks:** Best way to learn the system
- 🐛 **Issues:** Check troubleshooting sections in docs
- 📊 **Monitoring:** Use W&B integration for production

**You're ready to route VLM requests intelligently! 🚀**
