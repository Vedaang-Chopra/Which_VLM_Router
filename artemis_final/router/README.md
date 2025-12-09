# Artemis Router

The Router module selects the optimal Vision-Language Model (VLM) for a given query based on utility scores derived from cost, latency, and accuracy predictions.

## Overview

The router operates by analyzing the input prompt and image (if applicable) to predict a "utility score" for each available backend model. It supports multiple routing modes to align with different system goals (e.g., minimizing cost vs. maximizing accuracy).

## Architecture

- **Public API** (`public_api.py`): The external interface for initializing the router and making routing requests.
- **Router Service** (`router_service.py`): Manages the lifecycle of the router, including configuration loading and model initialization.
- **Inference Core** (`core/inference_reward_router.py`): Contains the neural network logic (DeBERTa-based) for predicting model rewards.

## Usage

### Initialization

The router must be initialized with a configuration file or dictionary before use.

```python
from artemis_final.router.public_api import init_router

# Initialize with default configuration
init_router()
```

### Routing Requests

Use `route_request` to get a model recommendation.

```python
from artemis_final.router.public_api import route_request

# Text-only query
decision = route_request(
    prompt="Explain the theory of relativity",
    mode="balanced" # Options: 'accuracy', 'fast', 'cheap', 'balanced'
)
print(f"Selected Model: {decision['decision']}")

# Image query (if supported by the specific router implementation)
# Image data is typically passed via metadata or specific arguments depending on the router version.
```

## Directory Structure

- `public_api.py`: Main entry point for consumers.
- `router_service.py`: Service-level abstraction.
- `core/`: Implementation of routing algorithms and neural models.
- `router_config_reward.yaml`: Configuration for the reward-based router.
