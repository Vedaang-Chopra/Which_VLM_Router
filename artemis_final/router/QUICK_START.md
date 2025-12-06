# Artemis Router - Quick Start Guide

Get up and running with the Artemis Router in 5 minutes.

---

## Step 1: Install Dependencies

```bash
pip install torch transformers sqlalchemy pandas pillow pyyaml numpy wandb requests
```

---

## Step 2: Configure the Router

Copy and edit the example configuration:

```bash
cd artemis_final/router
cp router_config_example.yaml router_config.yaml
```

**Edit `router_config.yaml` and update:**

```yaml
router:
  checkpoint_path: "/path/to/your/router_checkpoint.pt"  # ← Update this
  device: "cuda:0"  # or "cpu" or "mps"

data:
  db_url: "postgresql://user:password@localhost/dbname"  # ← Update this
  image_root_dir: "/path/to/your/images"  # ← Update this
```

---

## Step 3: Setup Database

Create the router logs table:

```bash
psql -U vlmrouter -d vlmrouter -f sql/router_logs_schema.sql
```

---

## Step 4: Run Setup Script

Test everything is working:

```bash
python setup_router.py --config router_config.yaml
```

This will:
- ✅ Check all dependencies
- ✅ Validate configuration
- ✅ Test database connectivity
- ✅ Run a smoke test

---

## Step 5: Start Using the Router

### Python API

```python
from artemis_router import load_config, RouterEngine

# Initialize
cfg = load_config("router_config.yaml")
engine = RouterEngine(cfg)

# Route a sample from database
result = engine.route_by_id("sample_123", split="test")

print(f"Chosen: {result.router_decision.chosen_model}")
print(f"Latency: {result.router_decision.inference_ms:.2f}ms")
```

### Notebooks

Open and run the interactive notebooks:

```bash
# Unit tests and validation
jupyter notebook notebooks/01_router_unit_test.ipynb

# Traffic simulation and performance testing
jupyter notebook notebooks/02_traffic_simulation.ipynb
```

---

## Common Tasks

### Route a Single Sample

```python
# From database
result = engine.route_by_id("sample_id", split="test")

# From custom sample
from artemis_router import Sample
from PIL import Image

sample = Sample(
    sample_id="custom_001",
    source="http",
    text="What is in this image?",
    image=Image.open("image.jpg"),
    metadata={},
)

result = engine.route_sample(sample)
```

### Route Multiple Samples

```python
# Load batch from database
results = engine.route_split("test", limit=100)

# Process custom batch
samples = [...]  # List of Sample objects
results = engine.route_batch(samples)
```

### Run Traffic Simulation

```python
from artemis_router.traffic_simulator import run_traffic

results, stats = run_traffic(
    route_fn=engine.route_sample,
    source="synthetic",  # or "db"
    traffic_cfg=cfg.traffic,
    rps=10.0,
    duration_sec=60,
    verbose=True,
)

print(f"Avg latency: {stats.avg_latency_ms:.2f}ms")
print(f"P95 latency: {stats.p95_latency_ms:.2f}ms")
```

### Analyze Logs

```sql
-- Model usage
SELECT router_chosen_model, COUNT(*)
FROM router_live_logs
GROUP BY router_chosen_model;

-- Latency stats
SELECT
    AVG(router_inference_ms) as avg_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY router_inference_ms) as p95_ms
FROM router_live_logs;
```

---

## Troubleshooting

### "Checkpoint not found"
→ Update `router.checkpoint_path` in config

### "Database connection failed"
→ Check `data.db_url` and ensure PostgreSQL is running

### "Image not found"
→ Verify `data.image_root_dir` path

### Slow inference
→ Try `dtype: "float16"` and `device: "cuda:0"`

---

## Next Steps

1. ✅ Run `01_router_unit_test.ipynb` for comprehensive testing
2. ✅ Run `02_traffic_simulation.ipynb` for performance analysis
3. 📖 Read [README.md](README.md) for full documentation
4. 🚀 Build your application!

---

## Help

- 📖 Full docs: [README.md](README.md)
- 🐛 Issues: [GitHub Issues](https://github.com/yourrepo/issues)
- 💬 Questions: your-email@example.com

**Happy routing! 🚀**
