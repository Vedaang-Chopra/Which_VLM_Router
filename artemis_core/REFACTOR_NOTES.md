# Refactor Notes

## Overview
This refactor audited `artemis_core` to meet strict engineering standards. The primary focus was on **correctness, reproducibility, and clarity**. We moved away from "catch-all" error handling to specific exception types, enforced strict configuration validation, and added explicit controls for random seeds.

## Key Modifications

### 1. Reproducibility
- **Added `set_seed` utility**: Located in `src/artemis/common/utils.py`. This function sets seeds for `random`, `numpy`, and `torch` (including CuDNN determinism) to ensure consistent results across runs.
- **CLI Seed Argument**: `main.py` now accepts a `--seed` argument (defaulting to 42) which is passed to `set_seed` immediately upon startup.

### 2. Configuration Validation
- **Strict `config_loader.py`**: The `load_config` function now strictly validates that *essential* keys (`db`, `router`, `load_balancer`) exist. It raises `ValueError` or `FileNotFoundError` explicitly if config is invalid, preventing silent failures or messy stack traces later.

### 3. Error Handling
- **Refactored `main.py`**:
    - Removed broad `try...except Exception` blocks where possible.
    - Explicitly catches `FileNotFoundError` (for missing config/models) and `ValueError` (for invalid args).
    - Logs unexpectedly unhandled exceptions with full tracebacks for debugging (`exc_info=True`).
    - Exits with non-zero status codes on failure, essential for CI/CD pipelines.

### 4. Code Structure
- **Removed "Mock" Fallbacks**: The system now fails fast if critical resources (like the router checkpoint) are missing, rather than substituting a Mock router silently. This ensures that a successful run *actually* means the system is working as intended.

## Validation
- **Reproduction Script**: Added `reproduce_experiment.sh` to run a standardized experiment with a fixed seed and logging enabled.
- **Verification**: Run `sh reproduce_experiment.sh 42 balanced` to verify end-to-end flow.

## Limitations
- **Image handling**: `main.py` assumes local image paths.
- **Dependencies**: Requires `torch` and `transformers` installed.
