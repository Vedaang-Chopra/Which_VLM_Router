# Artemis: Intelligent VLM Router

Artemis is a high-performance routing system for Vision-Language Models (VLMs). It optimizes inference by dynamically routing queries to the most appropriate model based on:

- **Accuracy**: Routing complex queries to larger, more capable models.
- **Cost**: Offloading simpler tasks to smaller, cheaper models.
- **Latency**: ensuring SLA compliance through smart load balancing.

## Key Features

- 🧠 **Reward-Based Routing**: Learned router predicting utility scores for each model.
- ⚖️ **SLA-Aware Load Balancer**: Dynamically distributes load to maintain latency targets.
- ⚡ **Unified Inference Client**: Simple, standard interface for multiple VLM backends (OpenAI-compatible).
- 🛠 **Modular Architecture**: Clean separation of Router, Load Balancer, and Inference Engine.

## Project Structure

```
artemis_core/
├── config/             # Configuration files
├── examples/           # Demo scripts and usage examples
├── src/
│   └── artemis/        # Core source code
│       ├── common/     # Shared utilities
│       ├── router/     # Neural routing logic
│       ├── load_balancer/ # Capacity & SLA scheduling
│       └── inference/  # Unified VLM client
└── main.py             # CLI Entrypoint
```

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Configuration

Edit `config/artemis.yaml` to define your available models and endpoints.

### 3. Usage

Run the main CLI to process a single request:

```bash
python main.py --prompt "What is in this image?" --image "/path/to/image.jpg" --mode balanced
```

Run the demo pipeline script:

```bash
python examples/demo_pipeline.py
```

## Configuration

The system is configured via `config/artemis.yaml`. See the file for detailed comments on:
- Database connections
- Model endpoints (URL, API Key)
- Router model paths
- Load Balancer SLAs

## License

[MIT License](LICENSE)
