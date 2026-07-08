# ARTEMIS Deep Architecture Audit
> Date: 2026-06-22
> Method: Direct code reads + PyTorch checkpoint inspection + memory corpus retrieval
> Rule: No inference without evidence — every claim is traced to a file read or checkpoint inspection.

---

## 1. Router Architectures

### 1.1 Reward Router (Multitask Variant — Production)

**File:** `artemis_final/router_train/models/reward_router.py` (237 lines)
**Inference wrapper:** `artemis_final/router/core/inference_reward_router.py`

#### Class
```python
class RewardRouterModel(nn.Module)
```

#### `__init__` parameters
| Param | Type | Default |
|---|---|---|
| `config` | `RouterModelConfig` | required |
| `num_models` | `int` | required |
| `num_modes` | `int` | required |
| `num_tasks` | `int` | required |
| `text_encoder_hidden_size` | `Optional[int]` | `None` (auto-detected) |

#### Architecture
```
Text encoder: AutoModel.from_pretrained(config.text_encoder_name)
  CLS token: last_hidden_state[:, 0, :]   # 768-dim for DistilBERT

model_embedding: nn.Embedding(num_models, config.model_emb_dim)   # 5 x 32
mode_embedding:  nn.Embedding(num_modes,  config.mode_emb_dim)    # 4 x 16

Concatenation: [text_CLS (768) | model_emb (32) | mode_emb (16)] -> 816-dim

Routing MLP (utility head) -- one per mode, "routing_heads" in checkpoint:
  Linear(816 -> 256) -> ReLU -> Dropout(0.1)
  Linear(256 -> 128) -> ReLU
  Linear(128 ->   1) -> scalar utility_hat

Task classification head -- "task_head" in checkpoint:
  Linear(768 -> 256) -> ReLU -> Dropout(0.1)
  Linear(256 ->  30) -> task_logits   (text features only, no model/mode embeddings)
```

**Confirmed from checkpoint inspection of `best_multitask_router_v1.pt`:**
- `routing_heads[0..3]`: shapes `(256, 816) -> (128, 256) -> (1, 128)` — one head per mode
- `task_head`: shapes `(256, 768) -> (30, 256)` — 30-class task classifier

#### `forward()` returns
```python
{
    'utility_hat': tensor([batch_size]),    # squeezed from [batch_size, 1]
    'task_logits': tensor([batch_size, 30])
}
```

#### Training
- **Loss:** `nn.MSELoss()` between `utility_hat` and target reward scalar
- **Optimizer:** AdamW, `lr=3e-5`, `weight_decay=0.01` (excluded from bias/LayerNorm params)
- **Scheduler:** `CosineAnnealingLR(T_max=total_steps - warmup_steps)`, stepped every batch
- **Gradient clipping:** `clip_grad_norm_(model.parameters(), 1.0)`
- **Validation metric:** MSE (primary) + Pearson correlation (secondary)
- **Checkpoint save:** `model.save(path)` only when `val_loss < best_val_mse`
- **Early stopping:** Disabled by default (`patience=None`)

#### Inference wrapper (`RewardRouterInference`)

Input text formatting:
```
Primary (70%):     "[ROUTER] PromptLenWords: X. ImgWidth: W. ImgHeight: H. ImgAR: A. Question: {prompt}"
Full metadata (10%): "[ROUTER] Task: {task}. Dataset: {dataset}. SourceConfig: {config}. Split: {split}. PromptLenWords: X. ImgWidth: W. ImgHeight: H. ImgAR: A. Question: {prompt}"
No image:          "[ROUTER] Question: {prompt}"
```

**Images are NOT processed as pixels.** Width/height/aspect ratio injected as text tokens.

```python
route(prompt: str, image: Optional[Image.Image]=None, mode: str="accuracy",
      metadata: Optional[Dict]=None) -> Dict
```

Returns:
```python
{
    'chosen_model':    str,
    'chosen_model_id': int,
    'rewards':         Dict[str, float],   # all 5 models -> predicted utility
    'mode':            str,
    'inference_ms':    float
}
```

`route_batch()` is a simple Python loop over `route()` — no true batch optimization.

Special handling: DeBERTa fast-tokenizer bug -> fallback to `DebertaV2Tokenizer.from_pretrained` (slow tokenizer). Old checkpoints use `_ConfigRemappingUnpickler` to remap pickled `config` module paths.

---

### 1.2 Pairwise Router (Legacy)

**File:** `artemis_final/router_train/models/pairwise_router.py`
**Checkpoint:** `old_checkpoints/best_pairwise_router.pt` (254 MB, Dec 6 2025)

```python
class PairwiseRouterModel(nn.Module)
# __init__ params:
# num_models, num_modes=4, text_encoder_name="distilbert-base-uncased",
# model_embed_dim=32, mode_embed_dim=16, hidden_dim=256, dropout=0.1
```

#### Architecture
```
Concatenation: [CLS (768) | model_emb (32) | mode_emb (16)] -> 816-dim

MLP (self.scorer):
  Linear(816 -> 256) -> ReLU -> Dropout(0.1)
  Linear(256 -> 128) -> ReLU -> Dropout(0.1)
  Linear(128 ->   1) -> scalar score
```

Tokenizer instantiated inside `__init__`, called inside `forward()` (unlike Reward router which pre-tokenizes externally).

#### Training loss
```python
mean(max(0, margin - (score_i - score_j)))   # default margin=1.0
# score_i = preferred model, score_j = less preferred model per (sample, mode) pair
```

#### Inference
**"Pairwise" = training objective only, NOT inference procedure.**
Inference: scores all (N_samples x M_models) triples directly, argsorts descending per sample.
No C(M,2) pair enumeration or vote aggregation at inference time.

Checkpoint metadata:
```python
model_to_id: {'deepseek_ocr': 0, 'gemma_3_27b': 1, 'qwen2_5_vl_3b': 2,
              'qwen2_5_vl_7b': 3, 'qwen3_vl_8b_thinking': 4}
mode_to_id:  {'accuracy': 0, 'balanced': 1, 'cheap': 2, 'fast': 3}
config:      {num_models:5, num_modes:4, text_encoder_name:'distilbert-base-uncased',
              model_embed_dim:32, mode_embed_dim:16, hidden_dim:256, dropout:0.1}
```

---

### 1.3 Classical Router (Legacy)

**File:** `artemis_final/router_train/models/classical_router.py`
**Checkpoint:** `old_checkpoints/best_classical_router.pt` (254 MB, Dec 6 2025)

```python
class ClassicalRouterModel(nn.Module)
# __init__ params:
# num_models, num_modes=4, text_encoder_name="distilbert-base-uncased",
# mode_embed_dim=16, hidden_dim=256, dropout=0.1
# NOTE: NO model_embed_dim -- no per-model embeddings
```

#### Architecture
```
Concatenation: [CLS (768) | mode_emb (16)] -> 784-dim   (no model embedding)

MLP (self.classifier):
  Linear(784 -> 256) -> ReLU -> Dropout(0.1)
  Linear(256 -> 128) -> ReLU -> Dropout(0.1)
  Linear(128 ->   5) -> logits over all 5 models simultaneously
```

Single forward pass predicts all 5 model logits — fastest inference of the three routers.

#### Training loss
```python
total_loss = alpha * CE(logits, hard_labels) + (1-alpha) * KL(log_softmax(logits/T), soft_labels)
# default: alpha=0.5, temperature=2.0, KL reduction="batchmean"
```

Soft labels from `create_soft_labels_from_metrics()`: softmax over per-model metric values per (sample_id, mode_id) group.

`predict()` -> `(argmax_predictions, softmax_probabilities)`
`predict_top_k()` -> top-k model IDs and probabilities

---

### 1.4 CLIP Status

**ABSENT.** No CLIP, `CLIPModel`, `CLIPProcessor`, or `openai/clip` in `artemis_final/router/`. Images are not processed as visual features at any routing stage.

---

### 1.5 Architectural Comparison

| Property | Reward Router (Multitask) | Pairwise Router | Classical Router |
|---|---|---|---|
| Class | `RewardRouterModel` | `PairwiseRouterModel` | `ClassicalRouterModel` |
| Text encoder | DistilBERT (frozen) | DistilBERT (frozen) | DistilBERT (frozen) |
| Model embeddings | Yes (32-dim) | Yes (32-dim) | **No** |
| Mode embeddings | Yes (16-dim) | Yes (16-dim) | Yes (16-dim) |
| MLP input dim | **816** | 816 | 784 |
| MLP hidden dims | 256 -> 128 | 256 -> 128 | 256 -> 128 |
| MLP output | 1 (scalar per model per pass) | 1 (scalar) | **5 (all models, one pass)** |
| Auxiliary head | Task head (768->256->30) | None | None |
| Training loss | MSE | Margin ranking | CE + KL (soft labels) |
| Forward passes per query | 5 (one per model) | 5 | **1** |
| Checkpoint size | 255-258 MB | 254 MB | 254 MB |
| Status | **Production** | Legacy | Legacy |

---

## 2. Reward Functions

### 2.1 Exact Formulas per Mode

From `artemis_final/router_train/reward_definitions.py`:

**Accuracy mode:**
```
reward = A^2.0 * H
# accuracy_exp = 2.0
```

**Cheap mode:**
```
reward = A * H - 0.7 * cost_norm^1.2
# cheap_cost_weight=0.7, cheap_cost_exp=1.2
```

**Fast mode:**
```
reward = A * H - 0.7 * lat_norm^1.2
# fast_lat_weight=0.7, fast_lat_exp=1.2
```

**Balanced mode:**
```
reward = A^2.0 * H  +  0.3 * C^0.5  -  0.3 * cost_norm^1.1  -  0.3 * lat_norm^1.1
# balanced_acc_exp=2.0, conf_weight=0.3, conf_exp=0.5,
# cost_weight=0.3, cost_exp=1.1, lat_weight=0.3, lat_exp=1.1
```

### 2.2 Helpfulness (H) Definition

**H = 1.0 universally.** No hallucination detector is integrated. H is a no-op multiplicative placeholder in all formulas.

### 2.3 Variable Definitions

| Variable | Source | Notes |
|---|---|---|
| `A` (accuracy) | `(glider_score / 5.0).clip(0, 1)` | Glider is 0-5 LLM judge score |
| `H` (helpfulness) | `1.0` hardcoded | No hallucination signal available |
| `C` (confidence proxy) | `confidence_score` col if exists, else falls back to `A` | |
| `cost_norm` | `estimated_cost_usd / 95th-pct`, clipped [0,1] | DB col: `cost_norm_new` |
| `lat_norm` | `latency_ms / 95th-pct`, clipped [0,1] | DB col: `lat_norm` |

**Important:** If columns `utility_accuracy`, `utility_cheap`, `utility_fast`, `utility_balanced` already exist in the DB query result, pre-computed values are used directly — formula computation is bypassed. Reward formulas are a one-time dataset construction step, not repeated at training time.

---

## 3. Training Pipeline

### 3.1 Data Loading — SQL Schema

**Database:** `postgresql+psycopg2://vlmrouter:vlmrouter@localhost:5432/vlmrouter`

**3 tables:**

| Table | Key columns |
|---|---|
| `vlm_samples` (21 cols) | `sample_id`, `source_config`, `source_dataset`, `router_task`, `prompt_text`, `ground_truth`, `img_width`, `img_height`, `img_aspect_ratio`, `txt_prompt_length_words` |
| `vlm_responses` (55 cols) | `model_name`, `is_correct`, `estimated_cost_usd`, `latency_ms`, `input_tokens`, `output_tokens`, `utility_accuracy`, `utility_cheap`, `utility_fast`, `utility_balanced`, `cost_norm_new`, `lat_norm`, `glider_score`, `judge_molmo_score`, `judge_molmo_rank_group`, `data_split` |
| `vlm_evaluations` (20 cols) | `glider_score`, `glider_reasoning`, `judge_molmo_score`, `judge_molmo_rank_group` |

**Dataset totals:** 339,056 rows, 67,935 unique samples, 5 VLMs (exactly 5 rows per sample).

**Train/val/test split:** Pre-assigned in `data_split` column — not computed at training time.

| Split | Rows | Unique samples |
|---|---|---|
| train | 237,978 (70.2%) | 47,682 |
| val | 51,113 (15.1%) | 10,242 |
| test | 49,965 (14.7%) | 10,011 |

**Local cache:** SQLite at `router_train/data/vlm_router_cache.db` (230.52 MB). Parquet mirror at `router_train/data/router_profiles_with_utility.parquet`. Notebooks load from SQLite to avoid network latency.

### 3.2 Training Loop (Reward Router)

```
Loss:      nn.MSELoss()
Optimizer: AdamW(lr=3e-5, weight_decay=0.01)
           -- bias/LayerNorm excluded from weight decay
Scheduler: CosineAnnealingLR(T_max=total_steps - warmup_steps)
           -- stepped every batch, not every epoch
Grad clip: clip_grad_norm_(model.parameters(), 1.0)
Save:      model.save(path) when val_mse improves
```

Training history (per epoch): `{train_loss, train_corr, val_loss, val_corr, lr}`.

Retraining (from live traffic, `data_loop/retrainer.py`): `LR=1e-5`, `epochs=1`, `batch_size=8`.

### 3.3 Validation / Metrics

- **Primary:** MSE between predicted utility and target reward
- **Secondary:** Pearson correlation

**Observed results on test set (`multitask_eval_summary.csv`, 199,862 rows):**

| Mode | Routing Acc | Oracle Utility | Router Utility | Gap | Recovery |
|---|---|---|---|---|---|
| accuracy | 35.9% | 0.772 | 0.618 | 0.154 | 80.1% |
| cheap | 29.3% | 0.844 | 0.691 | 0.153 | 81.8% |
| fast | 30.7% | 0.844 | 0.697 | 0.146 | 82.7% |
| **balanced** | **35.2%** | **0.865** | **0.780** | **0.084** | **90.3%** |

Task classification accuracy: **1.0** on all modes. Utility correlation: **0.686**.

### 3.4 Checkpoint Format

```python
# New multitask format (v0, v1):
{
    'config':           RouterModelConfig,
    'num_models':       int,
    'num_modes':        int,
    'num_tasks':        int,
    'text_hidden_size': int,
    'state_dict':       OrderedDict
}

# Old format (pairwise, classical):
{
    'model_state_dict': OrderedDict,
    'model_to_id':      Dict[str, int],
    'mode_to_id':       Dict[str, int],
    'config':           dict
}
```

---

## 4. Dataset

### 4.1 Index Mappings (from `router_train/data/*.json`)

**model_index.json** (index = model ID):
```json
["deepseek_ocr", "gemma_3_27b", "qwen2_5_vl_3b", "qwen2_5_vl_7b", "qwen3_vl_8b_thinking"]
```

**mode_index.json** (index = mode ID):
```json
["accuracy", "cheap", "fast", "balanced"]
```

**task_index.json** — 30 tasks (index = task ID):
```json
["abstract_reasoning", "chart_captioning", "chart_reasoning", "code_generation",
 "counting", "dense_captioning", "diagram_captioning", "diagram_reasoning",
 "difference_detection", "document_ocr", "general_vqa", "geometry_reasoning",
 "handwriting_ocr", "icon_reasoning", "image_captioning", "knowledge_vqa",
 "map_reasoning", "medical_report", "medical_vqa", "meme_classification",
 "rendered_text_ocr", "scene_text_ocr", "science_reasoning", "spatial_reasoning",
 "table_math", "table_reasoning", "textbook_qa", "ui_captioning",
 "visual_mrc", "web_understanding"]
```

### 4.2 Cauldron Profiling Pipeline

Source: `HuggingFaceM4/the_cauldron` via HuggingFace Datasets streaming.

```python
CAULDRON_REPO = "HuggingFaceM4/the_cauldron"
load_dataset(CAULDRON_REPO, config_name, streaming=True)['train'].take(n_samples)
```

Each Cauldron sample: `images` (list of PIL.Image) + `texts` (list of dicts with `user`, `assistant`, `source`).

`extract_qa_from_sample()` -> `{'image', 'prompt', 'ground_truth', 'source'}`.

Random sampling uses 5x buffer multiplier (up to 1000) then `random.sample()`.

Cauldron configs used: `aokvqa`, `docvqa`, `hateful_memes`, `ai2d`, and others.

### 4.3 Aurelio Dataset (Wide-Format Parquet)

**File:** `code_base/aurelio/router_pivot_dataset_train.parquet`
**Shape:** (37,504, 223) — one row per Cauldron sample

Per-model column groups (repeated for each of 5 VLMs):
```
{model}__response_raw           {model}__response_parsed
{model}__score_exact_match      {model}__score_f1
{model}__score_mc_letter_match  {model}__is_correct
{model}__input_tokens           {model}__output_tokens
{model}__latency_ms             {model}__estimated_cost_usd
{model}__glider_score           {model}__glider_reasoning
```

Sample-level columns:
```
sample_id, source_dataset, source_config, router_task, ground_truth,
ground_truth_type, img_width, img_height, txt_prompt_length_chars,
txt_question_type, txt_has_mc_options, best_model, aurelio_best_model_pred, subset_split
```

Sample ID format: `{source_config}_{index}_{image_hash}` (e.g., `ai2d_00000_45f9e7163ea99b4c`).

`best_model` = oracle label (used for classical router training).
`aurelio_best_model_pred` = Aurelio baseline prediction.
`subset_split` = train/val/test assignment at build time.

Simplified version: `code_base/aurelio/df_subset.parquet` (37,504 x 37) — strips raw responses and semantic F1.

**This dataset was produced by actually running all 5 VLMs on 37,504 Cauldron prompts.**

---

## 5. Broken Components

### 5.1 Inference Engine

**Status: IMPLEMENTED (not stubbed).** `WhichVLMClient` / `OpenAIStyleRunner` in `inference_engine/client.py`:
- One `openai.OpenAI` client per `ModelEndpoint`; multiple endpoints per model -> `random.choice` load balancing
- `chat()`: retries `max_retries=3`, exponential backoff `sleep(0.5 * (attempt+1))`
- DeepSeek-OCR: uses `/v1/completions` not `/v1/chat/completions`; multimodal content converted to `"\n<image>\n"` text placeholder
- `fanout()`: calls all models in parallel via `ThreadPoolExecutor(max_workers=4)`
- Cost formula: `(prompt_tokens/1000)*pricing['prompt_per_1k'] + (completion_tokens/1000)*pricing['completion_per_1k']`

**CRITICAL BUG — `InferenceService` (`inference_engine/inference_service.py` line 35):**
```python
self.models_file = self.base_dir / cfg.inference.models_file
# AttributeError: 'GlobalConfig' object has no attribute 'inference'
```
`GlobalConfig` has: `db`, `router`, `load_balancer`, `data_collection`, `retraining`, `models`, `_base_dir` — no `inference` field. Crashes at `init_system()` startup.

**Fix needed:** Add `InferenceConfig` dataclass and `inference: InferenceConfig` field to `GlobalConfig`, plus a corresponding `inference:` section in the YAML config.

### 5.2 Retrainer (`data_loop/retrainer.py` — 197 lines)

**Status: Substantially implemented, two critical bugs.**

Functionality:
```python
# Fetch query:
SELECT FROM vlm_samples_collected JOIN vlm_responses_collected
WHERE feedback_score IS NOT NULL LIMIT {limit}

# Reward normalization:
df['reward'] = df['reward_signal'] / 5.0

# Missing image fields:
img_width=0, img_height=0, img_aspect_ratio=0.0

# Train/val split: sequential 80/20
train = df.iloc[:split_idx]
val   = df.iloc[split_idx:]

# Retraining config:
LR=1e-5, epochs=1, batch_size=8, weight_decay=0.01, warmup_ratio=0.1, scheduler='cosine'

# Output:
checkpoints/retrain/best.pt
```

**Bug 1 — Wrong import:**
```python
from config import SystemConfig   # SystemConfig does not exist
# Should be: from artemis_final.common.config_loader import GlobalConfig
```

**Bug 2 — Checkpoint loading incompatibility:**
```python
torch.load(ckpt_path)   # plain load without _ConfigRemappingUnpickler
# Will fail on new multitask checkpoint format (config module path remapping required)
```

### 5.3 ARES `public_api.py`

Known from prior sessions: early `return None` paths at lines 79 and 107. Exact conditions not audited in this pass — see Section 9 open questions.

---

## 6. Existing Checkpoints

### 6.1 Files

| File | Size | Date | Status |
|---|---|---|---|
| `checkpoints/best_multitask_router_v0.pt` | 255 MB | Dec 8 2025 | **Loadable — production** |
| `checkpoints/best_multitask_router_v1.pt` | 258 MB | Dec 8 2025 | **Loadable — production** |
| `old_checkpoints/best_classical_router.pt` | 254 MB | Dec 6 2025 | Loadable — legacy |
| `old_checkpoints/best_pairwise_router.pt` | 254 MB | Dec 6 2025 | Loadable — legacy |
| `old_checkpoints/best_reward_router.pt` | **705 MB** | Dec 6 2025 | **BROKEN** — `ModuleNotFoundError: No module named 'config'` |
| `router_train/models/checkpoints/best_reward_router.pt` | 256 MB | Dec 6 2025 | Not inspected |
| `code_base/which_vlm/artemis/saved_output/router_checkpoints_final/router_best.pt` | Unknown | — | Not inspected |

The 705 MB broken reward router is 2.8x larger than classical/pairwise. Architecture unknown because it fails to load.

### 6.2 Confirmed Weight Shapes (from `best_multitask_router_v1.pt`)

```
routing_heads.{0-3}.0.weight: (256, 816)   <- confirms input dim 768+32+16=816
routing_heads.{0-3}.0.bias:   (256,)
routing_heads.{0-3}.2.weight: (128, 256)
routing_heads.{0-3}.2.bias:   (128,)
routing_heads.{0-3}.4.weight: (1, 128)
routing_heads.{0-3}.4.bias:   (1,)
task_head.0.weight:            (256, 768)   <- text-only input
task_head.0.bias:              (256,)
task_head.2.weight:            (30, 256)    <- 30 task classes
task_head.2.bias:              (30,)
```

---

## 7. ARES Evaluation

### 7.1 Scorer (Static Metrics)

`Scorer.compute_all_scores()` applies rule-based metrics per response row: exact match, F1, MC letter match, `is_correct` boolean. No model calls.

### 7.2 VLMJudge (Molmo) — Listwise Ranking

**File:** `artemis_final/ares/evaluation/judge_molmo.py` (262 lines)

Type: **listwise multimodal ranking** with actual image included as base64.

```python
# Message format:
[{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": image_url}},   # base64 PNG
    {"type": "text",      "text": prompt}
]}]
```

Prompt (`VLM_JUDGE_PROMPT`): asks judge to score each answer A/B/C/D on 0-10 scale and produce ranking groups.

Processing:
1. Shuffle model names -> assign letters A/B/C/D (ordering-bias mitigation)
2. Call judge via thread-safe round-robin load balancing
3. Parse: `json.loads()` -> fallback regex `r'["\'](A-Z)["\']:\s*([\d.]+)'` for malformed responses
4. Remap letter -> original model name

Output:
```python
{
    'per_model': {model_name: {'score': float, 'rank_group': int}, ...},
    'raw_json':  str
}
```

Requires >= 2 answers; returns error dict if < 2 responses.

### 7.3 Evaluation Pipeline Orchestration

**File:** `artemis_final/ares/evaluation/router_eval_pipeline.py`

Data load — 4-table JOIN:
```sql
vlm_responses r
JOIN vlm_samples s
LEFT JOIN vlm_images i
LEFT JOIN vlm_evaluations e
```

Three parallel evaluation stages per batch:
1. **Static metrics** (`run_static_metrics()`) — `Scorer.compute_all_scores()` per row
2. **Glider** (`run_glider()`) — `GliderEvaluator`, `ThreadPoolExecutor(max_workers=min(64, N))`
3. **VLM Judge** (`run_vlm_judge()`) — groups by `sample_id`, calls `evaluate_listwise()` with all 5 model answers, `ThreadPoolExecutor(max_workers=min(64, N))`

Glider and VLM Judge run **in parallel** at top level: `ThreadPoolExecutor(max_workers=2)`.

Resumability: `ProgressTracker` writes `eval_progress.json` tracking `completed_samples` per `source_config`.

Write-back: `insert_evaluations()` records `glider_score`, `glider_reasoning`, `judge_molmo_score`, `judge_molmo_rank_group` to `vlm_evaluations` table.

Image delivery: `_bytes_to_data_url()` converts `image_bytes` -> PNG base64 data URL.

---

## 8. Integration Contracts

### 8.1 Router Config (`router_config_reward.yaml`)

```yaml
text_encoder: distilbert-base-uncased
max_seq_length: 256
model_emb_dim: 32
mode_emb_dim: 16
hidden_dim: 512        # NOTE: config says 512, checkpoint shows 256 -- checkpoint is ground truth
dropout: 0.1
dtype: float32
device: mps            # Apple Silicon Metal Performance Shaders
models: [deepseek_ocr, qwen2_5_vl_3b, qwen2_5_vl_7b, qwen3_vl_8b_thinking, gemma_3_27b]
modes: [accuracy, cheap, fast, balanced]
default_mode: balanced
include_metadata: true
```

### 8.2 System API Pipeline (`system_api/pipeline.py`)

`init_system()` instantiates in order:
1. `DataCollector(cfg)`
2. `RouterService(cfg)`
3. `LoadBalancerService(cfg)`
4. `InferenceService(cfg)` **<-- CRASHES HERE** (`cfg.inference` AttributeError)
5. `Retrainer(cfg)`

`handle_chat_completion()` — 5-step flow with graceful degradation:
```
1. Extract prompt: concatenate all user-role messages
2. router_svc.predict(prompt, mode=req.router_mode or "balanced")
   FALLBACK: model="qwen2_5_vl_7b", rewards={}
3. lb_svc.schedule(sample_id, task_type="vlm", router_probs, preferred_model)
   MAY OVERRIDE router choice; sets was_overridden=True in RouterMetadata
4. inf_svc.call_model(model_name, prompt, temperature, max_tokens)
   FAILURE: returns error string as content, does not raise
5. collector.log_sample_start() -> collector.log_model_response()
   FAILURE: caught and logged silently
```

Response: OpenAI-compatible `ChatCompletionResponse` with `router_metadata` extension field.

`RouterDecision.model_probs` populated from `router_result["rewards"]` — reward scores used as probability proxies.

`lb_svc.record_outcome(lb_decision, {"latency_ms": ..., "success": ...})` feeds latency back to load balancer.

### 8.3 GlobalConfig Fields

```python
GlobalConfig:
  db:              DBConfig
  router:          RouterConfig
  load_balancer:   LBConfig
  data_collection: DataCollectionConfig
  retraining:      RetrainingConfig
  models:          ...
  _base_dir:       Path
  # MISSING: inference: InferenceConfig  <-- causes InferenceService crash
```

Config file `artemis_final/configs/artemis.yaml` **does not exist** at that path. Actual config location unresolved (see Section 9).

---

## 9. Open Questions

1. **`artemis_final/configs/artemis.yaml`** does not exist at the expected path. GlobalConfig must load from elsewhere. Need `grep -r "GlobalConfig\|load_config\|from_yaml" artemis_final/common/` to find actual load path.

2. **`ares/public_api.py` lines 79 and 107** — early `return None` conditions not read in this pass. Need targeted read to identify function names, conditions, and intended behavior.

3. **`old_checkpoints/best_reward_router.pt` (705 MB)** — 2.8x larger than classical/pairwise. Architecture unknown; file fails to load due to missing `config` module pickle reference. May represent a pre-multitask reward router with larger hidden dims.

4. **`ares/configs/models.yaml`** — not read. VLM endpoint URLs, API keys pattern, capacity/SLA targets not confirmed.

5. **Load balancer scheduling algorithm** — `lb_svc.schedule()` signature confirmed but internal algorithm (round-robin? least-connections? SLA-weighted?) not read from `load_balancer/core/`.

6. **`hidden_dim` discrepancy** — `router_config_reward.yaml` says `512`; checkpoint weight shapes show `256`. Checkpoint is authoritative. Config was likely not updated after hyperparameter reduction.

7. **`router_train/models/checkpoints/best_reward_router.pt` (256 MB)** — likely an intermediate checkpoint from a different serialization step. Not inspected.

8. **`data_loop/collector.py`** — present (178 lines), called from `pipeline.py`, but schema of `vlm_samples_collected` and `vlm_responses_collected` live-traffic tables not audited.

9. **Pairwise router training data construction** — how (preferred_model_i, less_preferred_model_j) pairs are generated from raw utility scores was not read.

---

*Audit completed 2026-06-22. Sources: PyTorch checkpoint inspection, direct file reads, memory corpus (39 observations from prior sessions). All findings traceable to specific file or checkpoint inspection.*
