# Aurelio (Dataset Utilities)

## What It Does

Dataset utilities for preparing pivot datasets used in router training and evaluation. Provides the `router_pivot_dataset_train.parquet` and `router_pivot_dataset_test.parquet` files used to train and evaluate the router model.

## Key Files

| File | What It Does |
|---|---|
| `router_pivot_dataset_train.parquet` | Training set: (sample_id, features, ground_truth_model, reward_labels) |
| `router_pivot_dataset_test.parquet` | Test set: same structure as train |
| Dataset loading utilities | Load parquet, split train/test, prepare for PyTorch DataLoader |

## Current Status

**COMPLETE.** Dataset files are present and usable. No known issues.
