# VLM Router Database Schema

This document describes all tables and columns in the VLM Router PostgreSQL database.

## Tables Overview

| Table | Purpose | Primary Key |
|-------|---------|-------------|
| `vlm_samples` | Input samples (prompts, ground truth) | `sample_id` |
| `vlm_images` | Image binary data | `image_id` |
| `vlm_responses` | Model outputs and scores | `response_id` |
| `vlm_evaluations` | Glider & Semantic F1 scores | `evaluation_id` |

---

## Table: `vlm_samples`

Stores input samples from Cauldron datasets.

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | VARCHAR(255) PK | Unique ID: `{config}_{idx}_{hash8}` |
| `run_id` | VARCHAR(100) | Experiment run identifier |
| `source_config` | VARCHAR(100) | Cauldron config name (e.g., `ai2d`, `chartqa`) |
| `source_dataset` | VARCHAR(255) | Original dataset name |
| `source_index` | INTEGER | Index within source dataset |
| `router_task` | VARCHAR(100) | Task type: `vqa`, `ocr`, `chart`, `diagram`, etc. |
| `ground_truth_type` | VARCHAR(50) | `multiple_choice`, `freeform`, `numeric` |
| `data_split` | VARCHAR(20) | `train` (70%), `val` (15%), `test` (15%) |
| `prompt_text` | TEXT | Raw question text |
| `prompt_formatted` | TEXT | Formatted prompt sent to model |
| `system_prompt` | TEXT | System prompt used |
| `mc_options` | TEXT | Multiple choice options (JSON array) |
| `ground_truth` | TEXT | Correct answer |
| `gt_answer_letter` | VARCHAR(10) | For MC: correct option letter (A/B/C/D) |
| `txt_prompt_length_chars` | INTEGER | Prompt length in characters |
| `txt_prompt_length_words` | INTEGER | Prompt length in words |
| `txt_question_type` | VARCHAR(50) | Detected question type |
| `txt_has_mc_options` | BOOLEAN | Whether MC options are present |
| `image_id` | VARCHAR(255) FK | Reference to `vlm_images.image_id` |
| `created_at` | TIMESTAMPTZ | Row creation time |
| `updated_at` | TIMESTAMPTZ | Last update time |

---

## Table: `vlm_images`

Stores image data as binary blobs.

| Column | Type | Description |
|--------|------|-------------|
| `image_id` | VARCHAR(255) PK | Same as `sample_id` (1:1 relationship) |
| `image_bytes` | BYTEA | Raw image bytes |
| `image_hash` | VARCHAR(64) | SHA256 hash for deduplication |
| `img_width` | INTEGER | Image width in pixels |
| `img_height` | INTEGER | Image height in pixels |
| `img_aspect_ratio` | FLOAT | Width / Height |
| `img_file_size_bytes` | INTEGER | Size of image in bytes |
| `created_at` | TIMESTAMPTZ | Row creation time |

---

## Table: `vlm_responses`

Stores model responses and computed scores. One row per (sample, model) pair.

### Identification

| Column | Type | Description |
|--------|------|-------------|
| `response_id` | SERIAL PK | Auto-increment ID |
| `sample_id` | VARCHAR(255) FK | Reference to `vlm_samples` |
| `model_name` | VARCHAR(100) | Model name: `deepseek_ocr`, `qwen2_5_vl_3b`, etc. |
| `model_prefix` | VARCHAR(10) | Short prefix: `m1`, `m2`, `m3`, `m4`, `m5` |
| `model_id` | VARCHAR(255) | Full model ID from vLLM |

### Response Data

| Column | Type | Description |
|--------|------|-------------|
| `response_raw` | TEXT | Raw model output |
| `response_parsed` | TEXT | Cleaned/parsed response |
| `response_length_chars` | INTEGER | Response length in characters |
| `response_length_tokens` | INTEGER | Response length in tokens |

### Token & Timing

| Column | Type | Description |
|--------|------|-------------|
| `input_tokens` | INTEGER | Prompt tokens |
| `output_tokens` | INTEGER | Completion tokens |
| `total_tokens` | INTEGER | `input_tokens + output_tokens` |
| `latency_ms` | FLOAT | End-to-end latency in milliseconds |

### Status

| Column | Type | Description |
|--------|------|-------------|
| `ok` | BOOLEAN | True if response succeeded |
| `error_message` | TEXT | Error message if failed |
| `stop_reason` | TEXT | Reason for stopping (e.g., `stop`, `length`) |
| `is_refusal` | BOOLEAN | True if model refused to answer |

### Confidence

| Column | Type | Description | Formula |
|--------|------|-------------|---------|
| `confidence_score` | FLOAT | Confidence score [0, 1] | See below |
| `confidence_source` | VARCHAR(50) | `logprobs` or `heuristic` | - |
| `confidence_reason` | TEXT | Explanation of confidence | - |

**Confidence Formula (logprobs method):**
```
confidence = mean(exp(logprob) for each token)
```

**Confidence Formula (heuristic fallback):**
- MC question: 0.8 if answer matches option letter
- Contains "I don't know": 0.2
- Response too short: 0.3
- Default: 0.5

### Scoring

| Column | Type | Description | Formula |
|--------|------|-------------|---------|
| `score_exact_match` | FLOAT | 1.0 if response == ground_truth | `1.0 if match else 0.0` |
| `score_exact_match_normalized` | FLOAT | After normalization (lowercase, strip) | Same with normalization |
| `score_f1` | FLOAT | Token-level F1 score | `2 * P * R / (P + R)` |
| `score_contains_gt` | FLOAT | 1.0 if GT substring in response | Contains check |
| `score_gt_in_response` | FLOAT | 1.0 if GT found in response | Same |
| `score_numeric_match` | FLOAT | 1.0 if numbers match | For numeric answers |
| `score_mc_letter_match` | FLOAT | 1.0 if MC letter correct | `pred_letter == gt_letter` |
| `is_correct` | BOOLEAN | Overall correctness flag | Combined logic |
| `pred_answer_letter` | VARCHAR(10) | Predicted MC letter (A/B/C/D) | Extracted from response |

### Cost

| Column | Type | Description | Formula |
|--------|------|-------------|---------|
| `estimated_cost_usd` | FLOAT | Estimated API cost | `(input_tokens/1000 * input_price) + (output_tokens/1000 * output_price)` |

### GPU Metrics

Captured at inference time from vLLM server.

| Column | Type | Description |
|--------|------|-------------|
| `gpu_name` | VARCHAR(100) | GPU model (e.g., `NVIDIA H200`) |
| `gpu_index` | INTEGER | GPU index (0, 1, 2, ...) |
| `gpu_util_percent` | FLOAT | GPU compute utilization % |
| `gpu_mem_used_mb` | FLOAT | Memory used in MB |
| `gpu_mem_total_mb` | FLOAT | Total GPU memory in MB |
| `gpu_mem_free_mb` | FLOAT | Free memory in MB |
| `gpu_temp_celsius` | FLOAT | GPU temperature |
| `gpu_power_watts` | FLOAT | Current power draw |
| `gpu_power_limit_watts` | FLOAT | Power limit |
| `gpu_memory_util_percent` | FLOAT | Memory utilization % |

### Inference Config

| Column | Type | Description |
|--------|------|-------------|
| `inference_temperature` | FLOAT | Temperature used (default: 0.0) |
| `inference_max_tokens` | INTEGER | Max tokens setting (default: 512) |
| `inference_top_p` | FLOAT | Top-p sampling (default: 1.0) |

### Computed Scores (from Notebook 02)

| Column | Type | Description | Formula |
|--------|------|-------------|---------|
| `sample_score` | FLOAT | Per-sample quality score | `0.7*is_correct + 0.1*score_f1 + 0.2*glider_norm` |
| `perf_hier` | FLOAT | Hierarchical performance | `0.7*sample + 0.2*task_prior + 0.1*global_prior` |
| `cost_norm` | FLOAT | Normalized cost [0, 1] | `(cost - c_min) / (c_max - c_min)` |
| `utility` | FLOAT | Final utility score | `perf_hier - λ * cost_norm` (λ=10000) |

### Timestamps

| Column | Type | Description |
|--------|------|-------------|
| `created_at` | TIMESTAMPTZ | Row creation time |
| `updated_at` | TIMESTAMPTZ | Last update time |

---

## Table: `vlm_evaluations`

Stores LLM-as-judge evaluations. One row per (sample, model) pair.

### Identification

| Column | Type | Description |
|--------|------|-------------|
| `evaluation_id` | SERIAL PK | Auto-increment ID |
| `sample_id` | VARCHAR(255) FK | Reference to `vlm_samples` |
| `model_name` | VARCHAR(100) | Model being evaluated |
| `response_id` | INTEGER FK | Reference to `vlm_responses` |

### Glider Evaluation

LLM-as-judge scoring using PatronusAI/glider or similar.

| Column | Type | Description |
|--------|------|-------------|
| `glider_score` | FLOAT | Score 0-5 (5 = perfect) |
| `glider_reasoning` | TEXT | Explanation of score |
| `glider_highlight` | TEXT | Key phrases highlighted |
| `glider_raw_output` | TEXT | Full evaluator response |

**Glider Score Rubric:**
- **5**: Perfect, complete, correct
- **4**: Mostly correct, minor issues
- **3**: Partially correct
- **2**: Significant errors
- **1**: Mostly incorrect
- **0**: Completely wrong or irrelevant

### Semantic F1

Fine-grained factual accuracy evaluation.

| Column | Type | Description | Formula |
|--------|------|-------------|---------|
| `semantic_f1_precision` | FLOAT | Precision [0, 1] | `matched_gen / total_gen` |
| `semantic_f1_recall` | FLOAT | Recall [0, 1] | `matched_gt / total_gt` |
| `semantic_f1_f1` | FLOAT | F1 score [0, 1] | `2 * P * R / (P + R)` |
| `semantic_f1_gen_statements` | JSONB | Atomic statements from response |
| `semantic_f1_gt_statements` | JSONB | Atomic statements from ground truth |
| `semantic_f1_matches` | JSONB | Statement match pairs |
| `semantic_f1_labels` | JSONB | Match labels for each pair |

### Timestamps

| Column | Type | Description |
|--------|------|-------------|
| `created_at` | TIMESTAMPTZ | Row creation time |
| `updated_at` | TIMESTAMPTZ | Last update time |

---

## Indexes

```sql
CREATE INDEX idx_samples_source_config ON vlm_samples(source_config);
CREATE INDEX idx_samples_router_task ON vlm_samples(router_task);
CREATE INDEX idx_samples_data_split ON vlm_samples(data_split);
CREATE INDEX idx_responses_sample_id ON vlm_responses(sample_id);
CREATE INDEX idx_responses_model_name ON vlm_responses(model_name);
CREATE INDEX idx_responses_is_correct ON vlm_responses(is_correct);
CREATE INDEX idx_evaluations_sample_id ON vlm_evaluations(sample_id);
CREATE INDEX idx_evaluations_model_name ON vlm_evaluations(model_name);
```

---

## Model Mapping

| Prefix | Model Name | Description |
|--------|------------|-------------|
| `m1` | `deepseek_ocr` | DeepSeek VL2 (OCR optimized) |
| `m2` | `qwen2_5_vl_3b` | Qwen2.5-VL 3B |
| `m3` | `qwen2_5_vl_7b` | Qwen2.5-VL 7B |
| `m4` | `qwen3_vl_8b_thinking` | Qwen3-VL 8B (thinking) |
| `m5` | `gemma_3_27b` | Gemma 3 27B |

---

## Data Split Ratios

| Split | Ratio | Purpose |
|-------|-------|---------|
| `train` | 70% | Training data for router model |
| `val` | 15% | Validation / hyperparameter tuning |
| `test` | 15% | Final evaluation (held out) |
