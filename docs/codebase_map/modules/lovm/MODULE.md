# Codebase Map: lovm
>
> Directory: code_base/lovm/LOVM/LOVM/
> Entry point: lovm.py::LOVM
> Status: COMPLETE

## Responsibility

Benchmarks LVM performance using language-based prediction. Given predicted model/dataset rankings, computes accuracy, Kendall's tau, and L1 loss against ground truth. Also provides a ModelGPT-based predictor for zero-shot ranking.

## File Index

| File | Layer | Purpose |
|---|---|---|
| `lovm.py` | Core | LOVM class; metric dict (acc/k_tau/l1); ground truth pivot tables; evaluate_* methods |
| `constants/constants.py` | Schema | GROUND_TRUTH_CSV, NUM_RANK=5, MODEL_NAME_COL, DATASET_COL |
| `modelGPT/model_gpt_predictor.py` | Core | ModelGPTPredictor: LOO model rank prediction via GPT |
| `modelGPT/create_models.py` | Core | Create ModelGPT model instances |
| `modelGPT/encode_dataset.py` | Utility | Encode dataset metadata into text features |
| `modelGPT/gen_caption_dataset.py` | Utility | Generate caption datasets |
| `modelGPT/gen_syn_dataset.py` | Utility | Generate synthetic datasets |
| `modelGPT/calc_text_features.py` | Utility | Text feature extraction |
| `modelGPT/utils.py` | Utility | Shared modelGPT utilities |
| `latex_util.py` | Utility | LaTeX table generation |
| `generate_results.py` | Runner | Main results generation script |
| `generate_ablation.py` | Runner | Ablation study runner |
| `dataset_tasks.json`, `dataset_domains.json` | Config | Dataset metadata |
| `models.yml` | Config | Model configuration |

## Change Guide

- **To change evaluation metrics**: edit `lovm.py::METRIC_DICT`; add new functions matching `get_acc` signature
- **To change ground truth**: update `constants/constants.py::GROUND_TRUTH_CSV`; retrain ModelGPT if it changes
- **To add a new dataset**: add rows to `eval_table.csv` (model_fullname, dataset, acc1, ...)
- **To add new ModelGPT features**: edit `modelGPT/encode_dataset.py`

## Call Chain

```
# Direct evaluation (no ModelGPT)
LOVM(pred_target, num_rank, return_mean)
  → __init__: load GROUND_TRUTH_CSV → pivot to model_rank_gt_df, dataset_rank_gt_df
  → evaluate_model_rank(pred_df)
    → _evaluate(pred_df, model_rank_gt_df, ['acc', 'k_tau'])
      → for each column: sort, compute METRIC_DICT[metric]
      → add mean row if return_mean
  → evaluate_dataset_rank(pred_df) [same pattern with dataset_rank_gt_df]
  → evaluate_model_pred(pred_df) [uses 'l1' metric]

# With ModelGPT predictor
ModelGPTPredictor(features_df)
  → loo_model_rank() → per-model leave-one-out prediction
  → evaluate_model_rank(predicted_rank) → LOVM.evaluate_model_rank()
```

## Dependencies

Internal: None (standalone research module)
External: `pandas`, `numpy`, `scipy.stats`, `openai` (for ModelGPT predictor)
