"""
Configuration examples for different parallelization scenarios.

Use these configurations as starting points and adjust based on your hardware.
"""

# ==============================================================================
# MAXIMUM THROUGHPUT - For systems with many GPUs and high CPU count
# ==============================================================================
MAX_THROUGHPUT = {
    "MAX_WORKERS_CONFIGS": 20,      # Process 20 configs simultaneously
    "MAX_WORKERS_BATCHES": 8,       # 8 batches per config in parallel
    "BATCH_SIZE": 4,                # Small batches for quick turnaround
    "PARALLEL_CONFIGS": True,       # Enable config-level parallelism
    "REQUEST_TIMEOUT": 120,         # Higher timeout for safety
}
# Expected parallelism: 20 × 8 × 5 models = 800 concurrent requests (theoretical)
# Recommended for: 8+ GPUs, 32+ CPU cores, 128GB+ RAM


# ==============================================================================
# BALANCED - Good balance between speed and resource usage
# ==============================================================================
BALANCED = {
    "MAX_WORKERS_CONFIGS": 10,      # Process 10 configs simultaneously
    "MAX_WORKERS_BATCHES": 4,       # 4 batches per config in parallel
    "BATCH_SIZE": 8,                # Moderate batch size
    "PARALLEL_CONFIGS": True,       # Enable config-level parallelism
    "REQUEST_TIMEOUT": 60,          # Standard timeout
}
# Expected parallelism: 10 × 4 × 5 models = 200 concurrent requests (theoretical)
# Recommended for: 4-8 GPUs, 16-32 CPU cores, 64GB+ RAM


# ==============================================================================
# CONSERVATIVE - For resource-constrained systems
# ==============================================================================
CONSERVATIVE = {
    "MAX_WORKERS_CONFIGS": 4,       # Process 4 configs simultaneously
    "MAX_WORKERS_BATCHES": 2,       # 2 batches per config in parallel
    "BATCH_SIZE": 16,               # Larger batches to reduce overhead
    "PARALLEL_CONFIGS": True,       # Enable config-level parallelism
    "REQUEST_TIMEOUT": 60,          # Standard timeout
}
# Expected parallelism: 4 × 2 × 5 models = 40 concurrent requests (theoretical)
# Recommended for: 2-4 GPUs, 8-16 CPU cores, 32GB+ RAM


# ==============================================================================
# MINIMAL - For testing or very limited resources
# ==============================================================================
MINIMAL = {
    "MAX_WORKERS_CONFIGS": 2,       # Process 2 configs simultaneously
    "MAX_WORKERS_BATCHES": 1,       # 1 batch per config at a time
    "BATCH_SIZE": 32,               # Large batches to minimize overhead
    "PARALLEL_CONFIGS": False,      # Sequential config processing
    "REQUEST_TIMEOUT": 60,          # Standard timeout
}
# Expected parallelism: 1 × 1 × 5 models = 5 concurrent requests
# Recommended for: 1-2 GPUs, 4-8 CPU cores, 16GB+ RAM


# ==============================================================================
# DEBUG MODE - For troubleshooting and development
# ==============================================================================
DEBUG = {
    "MAX_WORKERS_CONFIGS": 1,       # One config at a time
    "MAX_WORKERS_BATCHES": 1,       # One batch at a time
    "BATCH_SIZE": 1,                # One sample at a time
    "PARALLEL_CONFIGS": False,      # Sequential processing
    "REQUEST_TIMEOUT": 300,         # Long timeout for debugging
}
# Expected parallelism: 1 × 1 × 5 models = 5 concurrent requests
# Recommended for: Debugging, testing, development


# ==============================================================================
# SINGLE GPU - Optimized for single GPU inference
# ==============================================================================
SINGLE_GPU = {
    "MAX_WORKERS_CONFIGS": 1,       # One config at a time
    "MAX_WORKERS_BATCHES": 1,       # One batch at a time
    "BATCH_SIZE": 8,                # Moderate batch size
    "PARALLEL_CONFIGS": False,      # Sequential config processing
    "REQUEST_TIMEOUT": 60,          # Standard timeout
}
# Expected parallelism: 1 × 1 × 1 model = 1 request at a time
# Note: When using a single GPU, process one model at a time sequentially
# Recommended for: 1 GPU, 4-8 CPU cores, 16GB+ RAM


# ==============================================================================
# FAST ITERATION - For quick experiments on subset of data
# ==============================================================================
FAST_ITERATION = {
    "MAX_WORKERS_CONFIGS": 5,       # Few configs in parallel
    "MAX_WORKERS_BATCHES": 2,       # 2 batches per config
    "BATCH_SIZE": 4,                # Small batches
    "PARALLEL_CONFIGS": True,       # Enable parallelism
    "REQUEST_TIMEOUT": 60,          # Standard timeout
    "N_SAMPLES_PER_CONFIG": 100,    # Limited samples for testing
}
# Expected parallelism: 5 × 2 × 5 models = 50 concurrent requests
# Recommended for: Quick testing, hyperparameter tuning


# ==============================================================================
# Usage Example
# ==============================================================================
"""
# In your notebook, select a configuration:

from parallel_config_examples import BALANCED

# Apply the configuration
MAX_WORKERS_CONFIGS = BALANCED["MAX_WORKERS_CONFIGS"]
MAX_WORKERS_BATCHES = BALANCED["MAX_WORKERS_BATCHES"]
BATCH_SIZE = BALANCED["BATCH_SIZE"]
PARALLEL_CONFIGS = BALANCED["PARALLEL_CONFIGS"]
REQUEST_TIMEOUT = BALANCED["REQUEST_TIMEOUT"]

# Or use it directly in the function call:
results_df = fast_eval_utils.run_parallel_evaluation(
    configs=configs_to_process,
    models=MODELS,
    n_samples=N_SAMPLES_PER_CONFIG,
    max_workers=BALANCED["MAX_WORKERS_CONFIGS"],
    run_id=RUN_ID,
    output_dir=OUTPUT_DIR,
    batch_size=BALANCED["BATCH_SIZE"],
    max_workers_batches=BALANCED["MAX_WORKERS_BATCHES"],
    parallel_configs=BALANCED["PARALLEL_CONFIGS"],
)
"""


# ==============================================================================
# Hardware-Based Recommendations
# ==============================================================================
HARDWARE_RECOMMENDATIONS = {
    "1 GPU, 4-8 cores, 16GB RAM": MINIMAL,
    "2-4 GPUs, 8-16 cores, 32GB RAM": CONSERVATIVE,
    "4-8 GPUs, 16-32 cores, 64GB RAM": BALANCED,
    "8+ GPUs, 32+ cores, 128GB RAM": MAX_THROUGHPUT,
    "Single GPU testing": SINGLE_GPU,
    "Development/Debugging": DEBUG,
    "Quick experiments": FAST_ITERATION,
}


def get_recommended_config(gpus: int, cpu_cores: int, ram_gb: int) -> dict:
    """
    Get recommended configuration based on hardware specs.

    Args:
        gpus: Number of GPUs available
        cpu_cores: Number of CPU cores
        ram_gb: RAM in GB

    Returns:
        Configuration dictionary
    """
    if gpus >= 8 and cpu_cores >= 32 and ram_gb >= 128:
        return MAX_THROUGHPUT
    elif gpus >= 4 and cpu_cores >= 16 and ram_gb >= 64:
        return BALANCED
    elif gpus >= 2 and cpu_cores >= 8 and ram_gb >= 32:
        return CONSERVATIVE
    elif gpus == 1:
        return SINGLE_GPU
    else:
        return MINIMAL


def print_config_info(config: dict, name: str = "Selected"):
    """Print configuration details."""
    print(f"\n{'='*80}")
    print(f"{name} Configuration")
    print(f"{'='*80}")
    print(f"Config-level workers:  {config['MAX_WORKERS_CONFIGS']}")
    print(f"Batch-level workers:   {config['MAX_WORKERS_BATCHES']}")
    print(f"Batch size:            {config['BATCH_SIZE']}")
    print(f"Parallel configs:      {config['PARALLEL_CONFIGS']}")
    print(f"Request timeout:       {config['REQUEST_TIMEOUT']}s")

    # Assuming 5 models
    n_models = 5
    max_concurrent = (
        config['MAX_WORKERS_CONFIGS'] *
        config['MAX_WORKERS_BATCHES'] *
        n_models
    ) if config['PARALLEL_CONFIGS'] else (
        config['MAX_WORKERS_BATCHES'] *
        n_models
    )
    print(f"\nTheoretical max concurrent requests: {max_concurrent}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Example: Print all configurations
    configs = {
        "MAX_THROUGHPUT": MAX_THROUGHPUT,
        "BALANCED": BALANCED,
        "CONSERVATIVE": CONSERVATIVE,
        "MINIMAL": MINIMAL,
        "DEBUG": DEBUG,
        "SINGLE_GPU": SINGLE_GPU,
    }

    for name, config in configs.items():
        print_config_info(config, name)
