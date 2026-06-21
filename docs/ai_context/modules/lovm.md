# Module: lovm
>
> Status: COMPLETE
> Entry point: LOVM/LOVM/lovm.py::LOVM
> Last updated: 2026-06-21

## Purpose

Benchmarks Large Vision-Language Models (LVMs) using language alone. Predicts how different models will rank on a dataset, then evaluates predictions against ground truth using accuracy, Kendall's tau, and L1 loss.

## Public API

| Class / Function | Signature | What it does |
|---|---|---|
| `LOVM` | `LOVM(pred_target?, num_rank?, return_mean?)` | Main benchmark class; evaluate model/dataset rankings |
| `evaluate_dataset_rank` | `evaluate_dataset_rank(pred_df: pd.DataFrame) -> pd.DataFrame` | Evaluate predicted dataset rankings vs ground truth |
| `evaluate_model_rank` | `evaluate_model_rank(pred_df: pd.DataFrame) -> pd.DataFrame` | Evaluate predicted model rankings vs ground truth |
| `evaluate_model_pred` | `evaluate_model_pred(pred_df: pd.DataFrame) -> pd.DataFrame` | Evaluate predicted model performance scores |
| `get_imagenet_model_rank` | `get_imagenet_model_rank() -> pd.DataFrame` | Return ImageNet-1k as baseline model ranking |
| `eval_lovm` | `eval_lovm(pred, gt, type?)` | Standalone eval function |

## Internal Structure

| File | Layer | What it does |
|---|---|---|
| `LOVM/LOVM/lovm.py` | Core | LOVM class; evaluation metrics (acc, k_tau, l1); ground truth pivot tables |
| `LOVM/LOVM/constants/constants.py` | Schema | GROUND_TRUTH_CSV, NUM_RANK=5, column names |
| `LOVM/LOVM/modelGPT/model_gpt_predictor.py` | Core | ModelGPTPredictor — GPT-based dataset/model ranking |
| `LOVM/LOVM/modelGPT/encode_dataset.py` | Utility | Encode dataset into features for ModelGPT |
| `LOVM/LOVM/modelGPT/gen_caption_dataset.py` | Utility | Generate caption datasets for synthetic data |
| `LOVM/LOVM/modelGPT/calc_text_features.py` | Utility | Extract text features for model prediction |
| `LOVM/generate_results.py` | Runner | Generate results from LOVM benchmark |
| `LOVM/generate_ablation.py` | Runner | Ablation study generator |
| `LOVM/LOVM/latex_util.py` | Utility | LaTeX formatting for paper-ready tables |

## External Dependencies

- `pandas` — dataframe manipulation and pivot tables
- `numpy` — numerical computation
- `scipy.stats.kendalltau` — Kendall's tau correlation
- `openai` / `anthropic` — ModelGPT predictor (GPT-based ranking)

## Known Issues

- `LOVM/LOVM/lovm.py:54` — `get_k_tau()` returns `0` when `len(true) <= 2` (minor: short sequences give zero correlation rather than NaN). This is the one `return 0` placeholder in the module.

## What an Agent Must Know Before Editing

- Ground truth is loaded from `LOVM/LOVM/eval_table.csv` (columns: model_fullname, dataset). To add new models or datasets, update this CSV.
- `NUM_RANK = 5` means only the top-5 are evaluated. Set to a higher number for deeper ranking analysis.
- The `pred_target` argument (default: "acc1") selects which column of eval_table.csv to use as the ground truth metric.
- The ModelGPT predictor uses GPT to predict dataset rank from text features — this requires an OpenAI API key.
- Evaluation results DataFrames always include a `mean` row when `return_mean=True`.
