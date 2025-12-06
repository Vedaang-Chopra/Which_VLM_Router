# Artemis Router – VLM Router Inference & Traffic Simulation

**Fast, modular router inference service for production VLM routing.**

---

## Overview

The Artemis Router module provides a complete inference system for the trained VLM router. It supports:

- **Low-latency inference**: Optimized single and batch routing
- **Multiple input sources**: Database samples, HTTP requests, synthetic traffic
- **Comprehensive logging**: SQL and Weights & Biases integration
- **Load balancer integration**: Real-time routing decision dispatch
- **Traffic simulation**: Test router under various load patterns

This phase focuses on router inference and traffic simulation. Dynamic scaling and full HTTP API integration are planned for future phases.

---

## Directory Structure

```
router/
├── artemis_router/              # Main Python package
│   ├── __init__.py             # Package initialization
│   ├── config.py               # Configuration system (YAML → typed objects)
│   ├── schemas.py              # Core data structures (Sample, RouterDecision, etc.)
│   ├── router_model.py         # Router architecture and checkpoint loading
│   ├── feature_extractor.py    # Sample → tensor conversion
│   ├── router_engine.py        # High-level routing API
│   ├── db_io.py                # Database read/write operations
│   ├── logging_wandb.py        # Weights & Biases logging
│   ├── lb_interface.py         # Load balancer communication
│   ├── api_io.py               # HTTP request/response types
│   └── traffic_simulator.py    # Traffic generation and simulation
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_router_unit_test.ipynb      # Router functionality tests
│   └── 02_traffic_simulation.ipynb    # Traffic pattern simulation
│
├── sql/                         # SQL schemas
│   └── router_logs_schema.sql  # Router logs table definition
│
├── router_config_example.yaml   # Example configuration file
└── README.md                    # This file
```

---

## Quick Start

### 1. Prerequisites

- **Trained router checkpoint**: From training phase (see `../ares/`)
- **PostgreSQL database**: With sample data loaded
- **Python environment**: PyTorch, transformers, sqlalchemy, etc.

```bash
# Install dependencies (if not already installed)
pip install torch transformers sqlalchemy pandas pillow pyyaml wandb
```

### 2. Configuration

Copy and customize the example configuration:

```bash
cp router_config_example.yaml router_config.yaml
```

Update the following in `router_config.yaml`:

- `router.checkpoint_path`: Path to your trained router `.pt` file
- `router.device`: `"cuda:0"`, `"cpu"`, or `"mps"`
- `data.db_url`: Your PostgreSQL connection string
- `data.image_root_dir`: Root directory for images
- `logging.wandb_enabled`: Set to `false` if not using W&B

### 3. Setup Database

Create the router logs table:

```bash
psql -U vlmrouter -d vlmrouter -f sql/router_logs_schema.sql
```

### 4. Run Unit Tests

Open and run the unit test notebook:

```bash
jupyter notebook notebooks/01_router_unit_test.ipynb
```

This will:
- Load the router and configuration
- Test with synthetic samples
- Test with database samples
- Verify logging and model selection

### 5. Run Traffic Simulation

Open the traffic simulation notebook:

```bash
jupyter notebook notebooks/02_traffic_simulation.ipynb
```

Configure simulation parameters and run various traffic patterns.

---

## Core Components

### RouterEngine

The main inference service class. Provides high-level APIs for routing.

**Example usage:**

```python
from artemis_router import load_config, RouterEngine

# Load configuration
cfg = load_config("router_config.yaml")

# Initialize engine
engine = RouterEngine(cfg)

# Route a single sample
result = engine.route_by_id("sample_123", split="test")

# Route a batch
results = engine.route_split("test", limit=100)

# Get statistics
stats = engine.get_stats()
```

### Sample

Unified data structure for all input sources:

```python
from artemis_router import Sample
from PIL import Image

sample = Sample(
    sample_id="example_001",
    source="http",
    text="What is shown in this image?",
    image=Image.open("image.jpg"),
    image_uri="http://example.com/image.jpg",
    metadata={"split": "test"},
    label=None,
)

result = engine.route_sample(sample)
```

### RouterDecision

Output of routing with probabilities and chosen model:

```python
print(result.router_decision.chosen_model)
# => "qwen2_5_vl_7b"

print(result.router_decision.probs)
# => {"deepseek_ocr": 0.05, "qwen2_5_vl_3b": 0.15, ...}

print(result.router_decision.inference_ms)
# => 12.3
```

### Traffic Simulation

Simulate various load patterns:

```python
from artemis_router.traffic_simulator import run_traffic

results, stats = run_traffic(
    route_fn=engine.route_sample,
    source="synthetic",
    traffic_cfg=cfg.traffic,
    rps=10.0,           # 10 requests/second
    duration_sec=60,    # 60 seconds
    verbose=True,
)

print(stats.actual_rps)        # Achieved RPS
print(stats.avg_latency_ms)    # Average latency
print(stats.p95_latency_ms)    # P95 latency
print(stats.model_distribution) # Model selection counts
```

**Traffic patterns:**

- `"constant"`: Steady rate
- `"ramp"`: Gradual increase (1x → 4x)
- `"spike"`: Sudden burst (1x → 10x → 1x)
- `"wave"`: Oscillating (1x → 1.5x → 2x → ...)

---

## Configuration Reference

### Router Section

```yaml
router:
  checkpoint_path: "/path/to/router.pt"
  device: "cuda:0"
  model_name_order: ["model1", "model2", ...]
  dtype: "float16"
  num_threads: 4
  warmup: true
```

**Key settings:**

- `checkpoint_path`: Trained router weights
- `device`: Inference device
- `model_name_order`: **MUST** match training order exactly
- `dtype`: `"float32"` or `"float16"` (FP16 recommended for GPU)
- `warmup`: Run warmup inference on startup

### Data Section

```yaml
data:
  db_url: "postgresql://user:pass@localhost/db"
  samples_table: "cauldron_samples"
  logs_table: "router_live_logs"
  id_column: "sample_id"
  text_column: "prompt_raw"
  image_path_column: "image_path"
  label_column: "router_best_model_name"
  split_column: "split"
  image_root_dir: "/path/to/images"
```

### Logging Section

```yaml
logging:
  sql_enabled: true
  wandb_enabled: true
  wandb_project: "artemis-router"
  wandb_run_name: "production-v1"
  wandb_entity: null
  log_router_probs: true
```

---

## Database Schema

The router logs are stored in the `router_live_logs` table:

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL | Primary key |
| `timestamp` | FLOAT | Unix timestamp |
| `sample_id` | VARCHAR | Sample identifier |
| `source` | VARCHAR | `"db"`, `"http"`, or `"synthetic"` |
| `split` | VARCHAR | Dataset split (if from DB) |
| `text` | TEXT | Question/prompt |
| `image_uri` | TEXT | Image path or URL |
| `label` | TEXT | Ground truth (if available) |
| `router_chosen_model` | VARCHAR | Selected model |
| `router_probs` | JSONB | Probability distribution |
| `router_inference_ms` | FLOAT | Latency in milliseconds |
| `extra_metadata` | JSONB | Additional metadata |

**Example queries:**

```sql
-- Model usage distribution
SELECT router_chosen_model, COUNT(*) as count
FROM router_live_logs
GROUP BY router_chosen_model;

-- Latency statistics
SELECT
    AVG(router_inference_ms) as avg_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY router_inference_ms) as p95_ms
FROM router_live_logs;

-- Accuracy (for samples with labels)
SELECT
    COUNT(*) FILTER (WHERE router_chosen_model = label) * 100.0 / COUNT(*) as accuracy_pct
FROM router_live_logs
WHERE label IS NOT NULL;
```

---

## Integration with ARES

The router module reuses components from the `ares` data and training module:

- **Dataset schema**: Sample structure matches ARES Cauldron format
- **Feature extraction**: Same text/image preprocessing as training
- **Model architecture**: Exact same `MultimodalRouter` as in training
- **Database**: Connects to same PostgreSQL instance

**Data flow:**

```
ares/              → Training data, evaluation, metrics
  ├── data/        → Dataset building, loading
  ├── inference/   → VLM inference runners
  └── notebooks/   → Training experiments

router/            → Router inference service
  ├── artemis_router/  → Production routing engine
  └── notebooks/       → Testing and simulation
```

---

## Performance Optimization

### Warmup

The router runs warmup inference on startup to allocate GPU kernels:

```python
engine = RouterEngine(cfg)  # Warmup happens automatically if enabled
```

### Batching

For higher throughput, use batch routing:

```python
# Load samples
samples = load_samples_from_db(engine.db_engine, cfg.data, "test", limit=1000)

# Route in batch (single forward pass)
results = engine.route_batch(samples)
```

**Benefits:**
- Single model forward pass for all samples
- Better GPU utilization
- Lower per-sample latency

### FP16 Inference

Use `dtype: "float16"` for ~2x speedup on GPU:

```yaml
router:
  dtype: "float16"
  device: "cuda:0"
```

### Thread Tuning

For CPU inference, adjust thread count:

```yaml
router:
  num_threads: 8  # Adjust based on CPU cores
  device: "cpu"
```

---

## Troubleshooting

### Checkpoint Loading Fails

**Error:** `FileNotFoundError` or `RuntimeError: state dict mismatch`

**Solutions:**
- Verify `checkpoint_path` is correct
- Ensure `model_name_order` matches training
- Check that encoder names match training config

### Database Connection Fails

**Error:** `OperationalError: could not connect to server`

**Solutions:**
- Verify PostgreSQL is running
- Check `db_url` connection string
- Ensure database and tables exist

### Image Loading Fails

**Error:** `FileNotFoundError` or `Image.open` errors

**Solutions:**
- Verify `image_root_dir` path
- Check that `image_path` column values are relative to root
- Ensure images exist on disk

### Slow Inference

**Symptoms:** High latency, low throughput

**Solutions:**
- Enable FP16: `dtype: "float16"`
- Use GPU: `device: "cuda:0"`
- Enable warmup: `warmup: true`
- Use batching for multiple samples
- Check CPU/GPU utilization

---

## Future Enhancements

Planned for upcoming phases:

1. **FastAPI HTTP Server**
   - REST API for router inference
   - Async request handling
   - Multipart image uploads

2. **Dynamic Load Balancing**
   - Automatic VLM backend scaling
   - Request queue management
   - Health checks and failover

3. **Advanced Monitoring**
   - Prometheus metrics
   - Grafana dashboards
   - Alert rules

4. **Caching Layer**
   - Redis-based result caching
   - Deduplication
   - Cache warming

---

## Citation

If you use this router system in your research, please cite:

```bibtex
@software{artemis_router_2025,
  title = {Artemis Router: Production VLM Routing System},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/artemis-router}
}
```

---

## License

[Specify your license here]

---

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: [your-email@example.com]

---

**Related Documentation:**
- [ARES Module](../ares/README.md) - Data and training
- [Router Training](../ares/notebooks/08_training_router_with_images.ipynb)
- [Router Analysis](../ares/notebooks/09_inference_router_analysis.ipynb)
