# Artemis Router Module

The **Router Module** is responsible for selecting the optimal VLM (Vision-Language Model) for a given query based on performance, cost, and latency trade-offs.

## Architecture

The router uses a **Reward-Based Routing** approach (and legacy strategies) to predict the "utility" of each model for a specific input.

### Core Components
- **Inference Engine** (`router.core.inference_reward_router`): Uses a trained neural router (DeBERTa + Heads) to predict model rewards.
- **Router Service** (`router.router_service`): Wraps the core inference logic, handles checkpoints, and integrates with the Artemis system.
- **Public API** (`router.public_api`): The single entry point for external consumers.

## Usage

### Initialization
```python
from router import init_router, route_request

# Initialize the global router instance
# (Optional) pass config_dict to override defaults
init_router()
```

### Routing a Request
```python
from PIL import Image

# Route a text-only query
decision = route_request("What is the capital of France?", mode="fast")
print(f"Selected Model: {decision['model']}")

# Route an image query
img = Image.open("chart.png")
decision = route_request(
    prompt="Analyze this chart",
    image=img,
    mode="accuracy"  # 'accuracy', 'fast', 'cheap', 'balanced'
)
```

## Directory Structure
```
router/
├── public_api.py         # Main entry point (facade)
├── router_service.py     # Service integration
├── core/                 # Core implementation
│   ├── inference_reward_router.py
│   ├── legacy/           # Older router implementations
│   ├── config.py
│   └── ...
└── README.md
```
