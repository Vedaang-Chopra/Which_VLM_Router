# Which VLM Router

A research-grade toolkit for benchmarking **Vision-Language Models (VLMs)** across dozens of multimodal tasks, deriving router-friendly supervision, and standing up a unified inference layer that can talk to any OpenAI-compatible model host. The project pairs a highly parallelized data-collection pipeline with Glider- and semantic-F1-based judges so you can quickly discover *which* model you should route a request to.

---

## Table of Contents
1. [Why This Project Exists](#why-this-project-exists)
2. [Key Capabilities](#key-capabilities)
3. [Repository Layout](#repository-layout)
4. [Environment & Dependencies](#environment--dependencies)
5. [Quick Start](#quick-start)
6. [Evaluation Pipeline](#evaluation-pipeline)
7. [Results, Metrics & Router Labels](#results-metrics--router-labels)
8. [Inference API Layer](#inference-api-layer)
9. [Configuration](#configuration)
10. [Troubleshooting](#troubleshooting)
11. [Documentation Map](#documentation-map)
12. [Contributing & Next Steps](#contributing--next-steps)

---

## Why This Project Exists

Routing work across heterogeneous VLMs is hard because no single open benchmark covers:

- The full diversity of visual reasoning tasks (document OCR, chart QA, diagram comprehension, etc.).
- Fine-grained answers that mix free-form generation, multiple choice, numeric reasoning, and extraction.
- Fast iteration when expensive models and evaluators (e.g., Glider) are involved.

**Which VLM Router** fills that gap by:

1. Pulling thousands of samples from *HuggingFaceM4/the_cauldron* (50+ configs) with detailed task labels.
2. Hitting every configured VLM simultaneously via multi-level parallelism (configs × batches × models).
3. Logging exact match, contains, numeric/MC correctness, latency, token usage, and estimated cost.
4. Adding optional semantic-F1 and Glider rubric scoring post-hoc without repeating inference.
5. Producing router-ready supervision so you can train policy models to dispatch future requests.

---

## Key Capabilities

- **Parallelized evaluation** — Up to 200 concurrent inference requests (10 configs × 4 batches × 5 models).
- **Dataset awareness** — Built-in taxonomy that maps every Cauldron config → router task → ground-truth type.
- **Extensible scoring** — Exact match, contains, token F1, numeric tolerance, MC letter, semantic F1, and Glider judgments.
- **Unified inference layer** — `which_vlm.inference_api_call` can fan out prompts across any OpenAI-compatible endpoint.
- **Automatic bookkeeping** — Structured Parquet outputs, summary JSONs, checkpoints, and resume capabilities.
- **Fast verification** — CLI utility detects missing configs, run completeness, and aggregated metrics in seconds.

---

## Repository Layout

| Path | Description |
|------|-------------|
| `code_base/which_vlm/dataset_builder/` | Core evaluation notebooks, utilities, feature extractors, scorers, and run artifacts. |
| `code_base/which_vlm/inference_api_call/` | Lightweight client, config loader, suites, and runner for OpenAI-style APIs. |
| `code_base/which_vlm/configs/` | YAML templates for model endpoints and dataset subsets. |
| `code_base/frugal_gpt/`, `code_base/cascadeflow/` | Additional experiments/environments (not directly touched by the main pipeline). |
| `dataset/` | Example media plus any cached intermediate data. |
| `EVALUATION_WORKFLOW_SUMMARY.md`, `GLIDER_TIMEOUT_FIXES.md`, `QUICK_START.md` | Ops-focused docs referenced throughout this README. |
| `vlm_router/` | Python virtual environment used by some notebooks/scripts. |

> **Tip:** All evaluation outputs are stored under `code_base/which_vlm/dataset_builder/experiment_data/runs/exp_YYYYMMDD_HHMMSS/`. Semantic post-hoc artifacts live inside a `semantic_evaluation/` subfolder.

---

## Environment & Dependencies

The repo does not pin dependencies via `requirements.txt` yet, but the following Python packages are required:

```
python >= 3.10
pip install datasets pillow pandas pyarrow numpy tqdm requests pyyaml openai
```

Optional (semantic evaluation & plotting):

```
pip install matplotlib seaborn rich
```

System requirements:

- Access to vLLM (or any OpenAI-compatible server) for each VLM you want to benchmark.
- A dedicated vLLM instance for the **PatronusAI/glider** evaluator (defaults to `localhost:8805`).
- Sufficient GPU RAM to host the selected models. See [`GLIDER_TIMEOUT_FIXES.md`](./GLIDER_TIMEOUT_FIXES.md) for recommended launches.

Environment setup example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r <(printf "datasets\npillow\npandas\npyarrow\nnumpy\ntqdm\nrequests\npyyaml\nopenai\n")
export PYTHONPATH=$PYTHONPATH:$(pwd)/code_base
```

---

## Quick Start

1. **Fix / launch Glider (critical).**
   ```bash
   pkill -f "vllm serve PatronusAI/glider"
   CUDA_VISIBLE_DEVICES=0 nohup vllm serve PatronusAI/glider \
     --trust-remote-code \
     --dtype bfloat16 \
     --host 0.0.0.0 \
     --port 8805 \
     --max-model-len 8192 \
     --gpu-memory-utilization 0.85 \
     --max-num-seqs 32 \
     --disable-log-requests \
     > glider_8805.log 2>&1 &
   curl http://localhost:8805/v1/models
   ```

2. **Fast evaluation run (10–15 min).**
   - Open `code_base/which_vlm/dataset_builder/fast_parallel_evaluation.ipynb`.
   - Set `ENABLE_SEMANTIC_F1 = False`, `ENABLE_GLIDER_EVAL = False`.
   - Choose configs (default: `ALL_CAULDRON_CONFIGS`) and `N_SAMPLES_PER_CONFIG`.
   - Run the notebook; outputs land in a fresh `experiment_data/runs/exp_*` directory.

3. **Verify the run.**
   ```bash
   cd code_base/which_vlm/dataset_builder
   python verify_results.py --latest
   ```

4. **Optional semantic post-hoc (30–60 min).**
   - Open `semantic_evaluation_posthoc.ipynb`.
   - Set `USE_LATEST_RUN = True`, pick a sampling strategy (e.g., `"per_model"`), and enable semantic / Glider flags.
   - Execute all cells to annotate subsets with semantic-F1 and Glider rubric scores.

For a condensed view of the ops workflow, see [`QUICK_START.md`](./QUICK_START.md).

---

## Evaluation Pipeline

### 1. Dataset ingestion

- Uses `datasets` streaming to pull samples from *HuggingFaceM4/the_cauldron*.
- `config.py` defines `ALL_CAULDRON_CONFIGS`, task buckets, and ground-truth types.
- `CauldronLoader` (`dataset_loader.py`) extracts image(s), prompt, answer, MC options, and router task labels per sample.

### 2. Feature extraction & scoring

- `FeatureExtractor` (`modules.py`) computes image stats (width, height, aspect ratio) and prompt descriptors (length, question type, MC presence).
- `Scorer` (`evaluation.py`) calculates:
  - `score_exact_match`, `score_exact_match_normalized`, `score_contains_gt`, `score_gt_in_response`
  - Token-level F1, numeric tolerance matches, multiple-choice letter accuracy
  - Refusal detection, boolean `is_correct`
- Optional semantic-F1 and Glider rubric scoring live in the same module (`GliderEvaluator`).

### 3. Multi-level parallel execution

Implemented in `fast_parallel_evaluation_utils.py` (see `PARALLELIZATION_ARCHITECTURE.md` for diagrams):

| Level | Executor | Purpose | Key knobs |
|-------|----------|---------|-----------|
| Config | `ProcessPoolExecutor` | Different Cauldron configs simultaneously | `MAX_WORKERS_CONFIGS`, `parallel_configs` |
| Batch | `ProcessPoolExecutor` | Split each config into batches | `BATCH_SIZE`, `MAX_WORKERS_BATCHES` |
| Model | `ThreadPoolExecutor` | Hit all VLM endpoints in parallel per batch | `len(models)` |

`run_parallel_evaluation` orchestrates everything and writes one Parquet per config plus `all_results.parquet`.

### 4. Notebooks

- **`fast_parallel_evaluation.ipynb`**
  - Primary driver for inference.
  - Controls sample counts, selected configs, toggles for semantic / Glider scoring, resume mode, worker counts, and checkpointing.
  - Saves summary statistics under `summary.json` and drops a `COMPLETED.txt` sentinel when finished.

- **`semantic_evaluation_posthoc.ipynb`**
  - Loads existing `all_results.parquet`.
  - Supports sampling strategies: `"all"`, `"random"`, `"per_model"`, `"per_config"`.
  - Filters by ground-truth type, models, or configs.
  - Generates `semantic_results_*.parquet`, `semantic_scores_*.parquet`, and `summary_semantic_*.json`.

### 5. Utility scripts

- `verify_results.py` — Inspects the latest (or specific) run, counts samples/models per config, confirms combined file integrity, and optionally lists missing configs so you can resume only what failed.
- `fast_parallel_evaluation_utils.configure(...)` — Adjusts request timeouts, evaluator port, and evaluation toggles programmatically (useful for scripting or headless runs).

---

## Results, Metrics & Router Labels

```
experiment_data/runs/exp_20250127_123456/
├── docvqa.parquet
├── chartqa.parquet
├── ...
├── all_results.parquet
├── summary.json
├── COMPLETED.txt
└── semantic_evaluation/
    ├── semantic_results_semantic_*.parquet
    ├── semantic_scores_semantic_*.parquet
    └── summary_semantic_*.json
```

Each `SampleRecord` row (see `config.py`) contains:

- **Identity** — `sample_id`, `source_config`, router task, timestamps, Cauldron indices.
- **Input features** — Image size/aspect/file size, prompt lengths, detected question type, MC options.
- **Model metadata** — `model_name`, `model_id`, inference parameters.
- **Scores** — Exact/contains match, numeric tolerance, MC letter, token F1, semantic precision/recall/F1, Glider score + reasoning/highlights.
- **Cost metrics** — Token usage, latency, estimated USD cost.

Router supervision helpers live in `modules.py`:

- `compute_routing_labels(df)` — Chooses the fastest correct model per sample, or the highest F1 if none are correct.
- `analyze_model_strengths(df)` — Produces per-task accuracy tables and identifies the best model per router task.

These tables can be fed into downstream router training or reporting notebooks.

---

## Inference API Layer

`which_vlm/inference_api_call` is a stand-alone mini-framework for firing OpenAI-style chat completions at many endpoints.

### Highlights

- Single config file (YAML/JSON) declares every endpoint (`base_url`, `model_id`, `api_key`, pricing, defaults).
- `WhichVLMClient` exposes `client.llm` and `client.vlm` suites for text-only or multimodal prompts.
- Built atop the official `openai` Python SDK, so payloads and responses stay compatible with vLLM, LM Studio, OpenAI, etc.
- Returns structured dicts containing response text, latency, usage, estimated cost, and raw payloads for debugging.

### Usage Snippet

```python
from which_vlm.inference_api_call.client import WhichVLMClient

client = WhichVLMClient.from_yaml("code_base/which_vlm/configs/models_vlm.yaml")
print("VLM models:", client.list_vlm_models())

image_example = "dataset/cartoon_describe.png"
question = "Describe the main character and the setting."

results = client.vlm.run_image(
    image=image_example,
    text=question,
    models="all",
    max_tokens=256,
)

for name, out in results.items():
    print(name, "->", out["response_text"], "(latency:", int(out["latency_ms"]), "ms)")
```

Need bare-metal access to the APIs only once; after that you can reuse the same client across notebooks, experiments, or unit tests.

---

## Configuration

- **Model endpoints** — `code_base/which_vlm/configs/models_vlm.yaml` (example ports 8010–8023). Each entry declares pricing and optional `extra_params`. Avoid duplicating `temperature`, `top_p`, etc. between `extra_params` and runtime kwargs.
- **Dataset subsets** — `code_base/which_vlm/configs/datasets_cauldron.yaml` shows how to limit runs to specific Cauldron configs with per-config metadata.
- **Runtime knobs** — `mac_config.yaml`, `test_config.yaml`, and notebook cells define more constrained settings for laptops or dry runs.
- **ExperimentConfig dataclass** — centralizes run IDs, sampling temperature, output directories, and serialization via `to_dict()`.

When running on new hardware:

1. Update the YAML ports to match the vLLM servers you have running.
2. Point notebooks/scripts to the refreshed config.
3. If needed, reduce concurrency by lowering `MAX_WORKERS_CONFIGS`, `MAX_WORKERS_BATCHES`, and `BATCH_SIZE`.

---

## Troubleshooting

| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| Glider timeouts or 404s | Server still on port 8005 or default timeout too low | Restart Glider on **8805**, bump `REQUEST_TIMEOUT` to 180–300s. |
| `dict() got multiple values for keyword 'temperature'` | `extra_params` already defines `temperature` | Remove the duplicate from the YAML and set it only when calling `.run_*`. |
| OOM / CUDA errors | Too many concurrent requests | Drop `MAX_WORKERS_CONFIGS/BATCHES`, shrink `BATCH_SIZE`, or split models across GPUs. |
| Missing configs in run | Notebook interruption midway | `python verify_results.py --latest --check-missing` shows unprocessed configs—rerun notebook with `configs_to_process = [...]` and `RESUME_MODE = True`. |
| Semantic notebook finds zero samples | Filters too strict | Relax `FILTER_GT_TYPES`, `FILTER_MODELS`, or use `SAMPLE_STRATEGY="random"` with a higher count. |
| Image paths cannot be read by inference suite | Relative-path mismatch | Ensure `sys.path` includes repo root and that notebook working dir matches expected image locations. |

More operational tips are in [`GLIDER_TIMEOUT_FIXES.md`](./GLIDER_TIMEOUT_FIXES.md) and the in-notebook troubleshooting cells.

---

## Documentation Map

- [`QUICK_START.md`](./QUICK_START.md) — 5-minute setup checklist.
- [`EVALUATION_WORKFLOW_SUMMARY.md`](./EVALUATION_WORKFLOW_SUMMARY.md) — Detailed narrative of the collection → analysis pipeline.
- [`GLIDER_TIMEOUT_FIXES.md`](./GLIDER_TIMEOUT_FIXES.md) — Commands and parameters for stable Glider evaluations.
- [`code_base/which_vlm/dataset_builder/PARALLELIZATION_ARCHITECTURE.md`](code_base/which_vlm/dataset_builder/PARALLELIZATION_ARCHITECTURE.md) — Visual explanation of the 3-level executor stack.
- [`code_base/which_vlm/dataset_builder/SEMANTIC_EVALUATION_GUIDE.md`](code_base/which_vlm/dataset_builder/SEMANTIC_EVALUATION_GUIDE.md) — How to configure post-hoc Glider scoring and sampling strategies.
- [`code_base/which_vlm/inference_api_call/readme.md`](code_base/which_vlm/inference_api_call/readme.md) — Deep dive into the inference client package.

---

## Contributing & Next Steps

1. **Fork + branch** — Standard GitHub workflow (`feature/<slug>`). Keep Python files formatted (Black) and notebooks cleanly executed.
2. **Extend configs** — Add new VLM endpoints or Cauldron subsets via YAML; ensure they’re documented in this README if generally useful.
3. **Add metrics** — `evaluation.py` is the home for new scorers, judges, or heuristics; populate `SampleRecord` fields accordingly.
4. **Router training** — Use `compute_routing_labels` outputs to train your own dispatcher (e.g., via CascadeFlow or FrugalGPT, both included under `code_base/`).
5. **Share findings** — Summaries, plots, or instructions that help others reproduce your results belong in `EVALUATION_WORKFLOW_SUMMARY.md` or a new doc in the repo root.

For questions or context, refer to the in-repo docs linked above or open an issue in your downstream Git hosting environment.

Happy routing! 🚀
