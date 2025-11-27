# Multi-Level Parallelization Architecture

## Overview

The evaluation system now implements a **3-level parallelization architecture** that efficiently distributes workload across multiple configs, data batches, and models simultaneously.

## Architecture Levels

### Level 1: Config-Level Parallelism
- **Executor**: `ProcessPoolExecutor`
- **Scope**: Multiple Cauldron configs processed in parallel
- **Configuration**: `MAX_WORKERS_CONFIGS` (default: 10)
- **Toggle**: `PARALLEL_CONFIGS` boolean flag
- **Use Case**: Process different datasets/configs simultaneously

```python
# Example: 10 configs can be processed in parallel
with ProcessPoolExecutor(max_workers=10) as executor:
    for config in configs:
        executor.submit(process_config, config, ...)
```

### Level 2: Batch-Level Parallelism
- **Executor**: `ProcessPoolExecutor` (within each config)
- **Scope**: Data batches within a single config
- **Configuration**:
  - `BATCH_SIZE`: Number of samples per batch (default: 8)
  - `MAX_WORKERS_BATCHES`: Parallel batch workers (default: 4)
- **Use Case**: Split large datasets into manageable chunks processed in parallel

```python
# Example: 2000 samples split into 250 batches of 8 samples each
# 4 batches processed in parallel at any time
batches = split_into_batches(samples, batch_size=8)
with ProcessPoolExecutor(max_workers=4) as executor:
    for batch in batches:
        executor.submit(process_data_batch, batch, ...)
```

### Level 3: Model-Level Parallelism
- **Executor**: `ThreadPoolExecutor` (within each batch)
- **Scope**: All models process the same batch concurrently
- **Configuration**: Automatically uses `len(MODELS)` workers
- **Use Case**: Maximize GPU utilization by hitting all model endpoints simultaneously

```python
# Example: 5 models process the same batch in parallel
with ThreadPoolExecutor(max_workers=len(models)) as executor:
    for model in models:
        executor.submit(process_model_batch, batch, model, ...)
```

## Data Flow Example

```
Input: 20 configs × 2000 samples × 5 models = 200,000 total samples

Level 1 (Config Parallelism):
├── Config 1 (Worker 1) ──┐
├── Config 2 (Worker 2) ──┤
├── Config 3 (Worker 3) ──┤── 10 configs in parallel
├── ...                   │
└── Config 10 (Worker 10)─┘

Level 2 (Batch Parallelism - within Config 1):
├── Batch 1: samples 0-7 (Worker 1)   ──┐
├── Batch 2: samples 8-15 (Worker 2)  ──┤
├── Batch 3: samples 16-23 (Worker 3) ──┤── 4 batches in parallel
└── Batch 4: samples 24-31 (Worker 4) ──┘
    └── (Next batch waits for a worker to free up)

Level 3 (Model Parallelism - within Batch 1):
├── Model 1: gemma-3-27b (Thread 1)      ──┐
├── Model 2: qwen3-vl-8b (Thread 2)      ──┤
├── Model 3: qwen2.5-vl-7b (Thread 3)    ──┤── 5 models in parallel
├── Model 4: qwen2.5-vl-3b (Thread 4)    ──┤
└── Model 5: deepseek-ocr (Thread 5)     ──┘
```

## Performance Characteristics

### Parallelism Calculation
- **Total parallel tasks** = `MAX_WORKERS_CONFIGS × MAX_WORKERS_BATCHES × len(MODELS)`
- **Example**: 10 × 4 × 5 = 200 concurrent inference requests (theoretical max)
- **Actual throughput** depends on GPU availability and network I/O

### Resource Usage
- **CPU**: Primarily for orchestration and data processing
- **GPU**: Model inference (limited by number of GPUs/vLLM instances)
- **Network**: HTTP requests to model endpoints
- **Memory**: Batching limits memory usage per worker

## Configuration Tuning Guide

### For Maximum Throughput
```python
MAX_WORKERS_CONFIGS = 20      # High config parallelism
MAX_WORKERS_BATCHES = 8       # High batch parallelism
BATCH_SIZE = 4                # Smaller batches for faster turnaround
PARALLEL_CONFIGS = True       # Enable config parallelism
```

### For Resource-Constrained Systems
```python
MAX_WORKERS_CONFIGS = 2       # Low config parallelism
MAX_WORKERS_BATCHES = 2       # Low batch parallelism
BATCH_SIZE = 16               # Larger batches to reduce overhead
PARALLEL_CONFIGS = False      # Sequential config processing
```

### For Debugging
```python
MAX_WORKERS_CONFIGS = 1       # One config at a time
MAX_WORKERS_BATCHES = 1       # One batch at a time
BATCH_SIZE = 1                # One sample at a time
PARALLEL_CONFIGS = False      # Sequential processing
```

## Key Benefits

1. **Scalability**: Handles large-scale evaluations efficiently
2. **Flexibility**: Tune each parallelism level independently
3. **Resource Control**: Prevent system overload with configurable limits
4. **GPU Utilization**: All models process data simultaneously
5. **Fault Isolation**: Failures in one batch/config don't affect others
6. **Progress Tracking**: Per-config and per-batch progress bars

## Implementation Details

### Function Hierarchy
```
run_parallel_evaluation()
├── ProcessPoolExecutor (configs) if parallel_configs=True
│   └── process_config()
│       └── ProcessPoolExecutor (batches)
│           └── process_data_batch()
│               └── ThreadPoolExecutor (models)
│                   └── process_model_batch()
│                       └── process_sample()
```

### Key Changes from Original
1. **Added `process_data_batch()`**: New function for batch-level processing
2. **Updated `process_model_batch()`**: Added `start_idx` parameter for correct sample indexing
3. **Enhanced `process_config()`**: Implements batch-level parallelism with ProcessPoolExecutor
4. **Upgraded `run_parallel_evaluation()`**: Supports both parallel and sequential config processing

## Usage Example

```python
results_df = fast_eval_utils.run_parallel_evaluation(
    configs=ALL_CAULDRON_CONFIGS,
    models=MODELS,
    n_samples=2000,
    max_workers=10,              # Level 1: Config parallelism
    run_id=RUN_ID,
    output_dir=OUTPUT_DIR,
    batch_size=8,                # Level 2: Batch size
    max_workers_batches=4,       # Level 2: Batch parallelism
    parallel_configs=True,       # Level 1: Enable/disable
)
```

## Performance Monitoring

The system provides detailed progress tracking:
- **Config-level**: `tqdm` progress bar showing config completion
- **Batch-level**: Per-config `tqdm` showing batch progress
- **Output logs**: Individual batch/model completion messages
- **Final stats**: Total time, records, throughput

## Troubleshooting

### Issue: Too many parallel processes
**Solution**: Reduce `MAX_WORKERS_CONFIGS` and `MAX_WORKERS_BATCHES`

### Issue: Out of memory errors
**Solution**: Increase `BATCH_SIZE` to reduce number of concurrent batches

### Issue: GPU timeout errors
**Solution**:
- Reduce total parallelism (fewer concurrent requests)
- Increase `REQUEST_TIMEOUT`
- Check vLLM instance capacity

### Issue: Debugging failures
**Solution**: Set `PARALLEL_CONFIGS=False` and reduce workers to 1 for sequential processing
