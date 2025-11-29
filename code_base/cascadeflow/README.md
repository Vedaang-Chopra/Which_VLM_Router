# CascadeFlow VLM Routing Experiment

This directory contains code for running CascadeFlow routing experiments on the VLM router dataset and comparing results with the trained neural router.

## Files

- **`cascadeflow_experiment.py`** - Core experiment module with async helpers
- **`cascadeflow_vlm_experiment.ipynb`** - Interactive notebook for running experiments
- **`README.md`** - This file

## Overview

CascadeFlow is a cascading routing framework that can dynamically select between multiple VLM models based on query complexity, cost, and quality thresholds. This experiment compares CascadeFlow's routing decisions against the trained neural router.

## Quick Start

### 1. Install Dependencies

```bash
pip install cascadeflow pandas pillow matplotlib seaborn tqdm nest_asyncio
```

### 2. Start vLLM Servers

You need to have vLLM servers running for the models you want to test. Example:

```bash
# Terminal 1 - Small model (e.g., Qwen2-VL-2B)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --port 8000 \
  --served-model-name qwen2-vl-2b-instruct

# Terminal 2 - Medium model (e.g., Qwen2.5-VL-3B)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --port 8001 \
  --served-model-name qwen2.5-vl-3b-instruct

# Terminal 3 - Larger model (e.g., Qwen2.5-VL-7B)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --port 8002 \
  --served-model-name qwen2.5-vl-7b-instruct
```

### 3. Run Notebook

```bash
cd code_base/cascadeflow
jupyter notebook cascadeflow_vlm_experiment.ipynb
```

Or use the Python API directly:

```python
import cascadeflow_experiment as cf_exp

# Use default configuration
config = cf_exp.create_default_config()

# Or customize
config.max_samples = 100
config.cascade_models[0].base_url = "http://your-server:8000/v1"

# Run experiment
results_df, summary, paths = cf_exp.run_and_save(config)

# View results
print(summary)
```

## Notebook Workflow

The notebook guides you through:

1. **Setup & Paths** - Configure dataset and image locations
2. **Load Dataset** - Load router training/test data
3. **Resolve Images** - Map Cauldron dataset IDs to local image files
4. **Display Samples** - Visualize sample images and prompts
5. **Configure Experiment** - Set up CascadeFlow models and parameters
6. **Run Experiment** - Execute routing on samples
7. **Analyze Results** - View routing decisions, costs, latency
8. **Visualize** - Charts for model usage, cost distribution, task patterns
9. **Export** - Save results for comparison with trained router

## Experiment Configuration

Key parameters in `ExperimentConfig`:

```python
@dataclass
class ExperimentConfig:
    dataset_path: Path              # Router dataset (train/val/test)
    cauldron_lookup_path: Path      # Image metadata
    image_root: Path                # Local image cache
    output_dir: Path                # Where to save results
    cascade_models: list[ModelSpec] # Models in cascade
    max_samples: int = 100          # Limit for testing
    seed: int = 42                  # Reproducibility
    generation_max_tokens: int = 300
    generation_temperature: float = 0.1
    verbose_agent: bool = False
    experiment_name: str = "cascadeflow_vlm_router"
```

Model specification:

```python
ModelSpec(
    name="qwen2.5-vl-3b-instruct",
    base_url="http://localhost:8001/v1",
    cost=0.00004,           # Cost per request
    temperature=0.1,        # Sampling temperature
    speed_ms=400,           # Expected latency
    keywords=["vision"],    # Model capabilities
    domains=["vlm"],        # Application domains
)
```

## Output Files

After running, you'll get:

```
dataset/which_vlm_data/results/
├── cascadeflow_vlm_router_20250129-143022.parquet
├── cascadeflow_vlm_router_20250129-143022.csv
└── cascadeflow_routing_comparison.parquet  # For router comparison
```

### Output Schema

| Column | Description |
|--------|-------------|
| `sample_id` | Unique sample identifier |
| `router_task` | Task type (OCR, reasoning, etc.) |
| `model_used` | Which model CascadeFlow selected |
| `total_cost` | Cost of this request |
| `latency_ms` | Time taken (milliseconds) |
| `cascaded` | Whether cascading occurred |
| `routing_strategy` | CascadeFlow's routing method |
| `routing_reason` | Why this model was chosen |
| `raw_response` | Model's generated response |
| `error` | Any errors encountered |

## Metrics Tracked

### Cost Metrics
- Mean/median cost per sample
- Total experiment cost
- Cost distribution across models
- Cost by task type

### Latency Metrics
- Mean/median/P95 latency
- Latency by model
- Latency by task type

### Routing Metrics
- Model usage distribution
- Cascading frequency
- Routing strategies used
- Task → Model patterns

### Quality Metrics (when ground truth available)
- Accuracy by model
- Accuracy by task
- Quality vs cost tradeoff

## Comparing with Neural Router

To compare CascadeFlow with the trained router:

1. **Run CascadeFlow** on test set:
   ```python
   config.dataset_path = "dataset/final_dataset/router_pivot_dataset_test.parquet"
   config.max_samples = None  # Use full test set
   cf_results, _, _ = cf_exp.run_and_save(config)
   ```

2. **Run trained router** on same samples:
   ```python
   # Using your router inference code
   router_results = run_router_inference(test_samples)
   ```

3. **Compare metrics**:
   ```python
   comparison = pd.merge(
       cf_results[['sample_id', 'model_used', 'total_cost']],
       router_results[['sample_id', 'model_used', 'total_cost']],
       on='sample_id',
       suffixes=('_cascadeflow', '_router')
   )

   # Agreement rate
   agreement = (comparison['model_used_cascadeflow'] ==
                comparison['model_used_router']).mean()
   print(f"Routing agreement: {agreement:.1%}")

   # Cost comparison
   print(f"CascadeFlow mean cost: ${comparison['total_cost_cascadeflow'].mean():.6f}")
   print(f"Router mean cost: ${comparison['total_cost_router'].mean():.6f}")
   ```

## Expected Results

Based on the router training setup, you should see:

### Model Distribution
- ~43% using mid-tier model (best cost-performance)
- ~28% using cheapest model (simple OCR tasks)
- ~26% using larger model (complex reasoning)
- ~3% using largest model (edge cases)

### Cost
- Mean cost: ~$0.000037 per sample
- Significantly lower than always using largest model
- Comparable to or better than heuristic routing

### Routing Patterns
- OCR tasks → cheaper models
- Reasoning tasks → larger models
- Charts/diagrams → mid-tier models
- Task-specific routing learned from data

## Troubleshooting

### vLLM Connection Errors
- Verify servers are running: `curl http://localhost:8000/v1/models`
- Check firewall/port settings
- Update `base_url` in model specs

### Out of Memory
- Reduce `max_samples`
- Use smaller models
- Reduce batch size in vLLM servers

### Missing Images
- Images are fetched from Cauldron on-demand
- Check `IMAGE_ROOT` path is writable
- Verify network access to Cauldron dataset

### Slow Execution
- Reduce `max_samples` for testing
- Enable `verbose_agent=True` to see progress
- Check vLLM server GPU utilization

## Architecture Notes

### Why CascadeFlow?

CascadeFlow provides an interesting baseline for comparison because it uses:
- **Rule-based cascading** instead of learned routing
- **Quality thresholds** to decide when to cascade
- **Explicit cost-quality tradeoffs** in configuration

This contrasts with the neural router which:
- **Learns routing** from data
- **Implicitly captures** task-quality-cost relationships
- **Adapts to patterns** in the training distribution

### Async Implementation

The experiment module uses async/await to efficiently handle multiple API calls:
- `run_experiment_async()` - Core async logic
- `run_experiment()` - Sync wrapper with `nest_asyncio` support
- Compatible with both scripts and Jupyter notebooks

## Development

### Adding New Models

```python
model_specs.append(
    cf_exp.ModelSpec(
        name="your-model-name",
        base_url="http://localhost:PORT/v1",
        cost=0.0001,  # Estimate from pricing
        temperature=0.1,
        speed_ms=500,  # Benchmark on your hardware
        keywords=["vision", "your", "tags"],
        domains=["vlm"],
    )
)
```

### Custom Routing Strategies

CascadeFlow supports custom routing strategies. See [CascadeFlow docs](https://github.com/cascadeflow/cascadeflow) for:
- Quality-based routing
- Cost-constrained routing
- Latency-optimized routing
- Custom routing functions

## References

- **CascadeFlow**: [GitHub](https://github.com/cascadeflow/cascadeflow)
- **vLLM**: [Documentation](https://docs.vllm.ai/)
- **Router Training**: See `code_base/which_vlm/artemis/TRAINING_GUIDE.md`
- **Dataset**: See `code_base/which_vlm/dataset_builder/`

## License

Same as parent project.
