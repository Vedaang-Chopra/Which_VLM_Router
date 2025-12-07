"""
Artemis Final - VLM Router System

Complete VLM routing pipeline:
- router: Model prediction (3 strategies)
- load_balancer: SLA-aware scheduling
- inference_engine: VLM execution bridge
- router_train: Training utilities
- ares: Dataset and evaluation

Usage:
    from artemis_final.router.artemis_router import ClassicalRouterInference
    from artemis_final.load_balancer import ArtemisLoadBalancer
    from artemis_final.inference_engine import WhichVLMClient
"""

__version__ = "1.0.0"

# Subpackages can be imported directly:
# from artemis_final import router, load_balancer, ares, router_train
