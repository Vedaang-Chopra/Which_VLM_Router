"""
Artemis VLM Router - Pipeline Demo
This script demonstrates how to use the core components programmatically.
"""
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1] / "src"))

from artemis.common.config_loader import load_config
from artemis.router import RewardRouter
from artemis.load_balancer import LoadBalancer
from artemis.load_balancer.types import ModelCapacityConfig, RouterOutput, SchedulingContext
from artemis.inference import VLMClient, load_endpoints_from_config
from PIL import Image

def demo_pipeline():
    # 1. Setup
    print(">>> Setting up Artemis...")
    config = load_config() # Loads from default location
    
    # Initialize Core Components
    # Note: In a real app, wrap this in a dependency injection container or factory
    
    # A. Router (Mock for demo if file missing)
    try:
        router = RewardRouter(config.router.checkpoint_path)
    except Exception:
        print("⚠️  Router checkpoint not found. Using mock router for demo.")
        class MockRouter:
            def route(self, prompt, img=None, mode="balanced"):
                return {
                    'chosen_model': 'qwen2_5_vl_7b', 
                    'scores': {'qwen2_5_vl_7b': 0.9, 'gemma_3_27b': 0.1}
                }
        router = MockRouter()

    # B. Load Balancer
    lb_configs = {
        m['name']: ModelCapacityConfig(sla_ms=2000.0) 
        for m in config.models
    }
    lb = LoadBalancer(lb_configs, mode="balanced")

    # C. Inference Client
    client = VLMClient(load_endpoints_from_config(config.models))

    # 2. Run Flow
    prompt = "Describe this image in detail."
    image_path = "examples/assets/demo.jpg" # Ensure this exists or handle safely
    # Check if we have an image to test with, else use text-only demo
    if not Path(image_path).exists():
        print(f"⚠️  {image_path} not found. Running text-only demo.")
        image_path = None
    
    print(f"\n>>> Input: '{prompt}'")
    
    # Step 1: Route
    print(f">>> Routing (Mode: balanced)...")
    route_res = router.route(prompt, None)
    print(f"    Preferred: {route_res['chosen_model']}")
    
    # Step 2: Schedule
    print(f">>> Scheduling...")
    decision = lb.schedule(
        RouterOutput("id_1", "vlm", route_res['scores'], route_res['chosen_model']),
        SchedulingContext(time.time() * 1000)
    )
    print(f"    Selected: {decision.chosen_model} (Est. Latency: {decision.total_latency_ms:.1f}ms)")
    
    # Step 3: Inference
    print(f">>> Inference...")
    # For demo purposes, we might not have a real backend running.
    # The client will try to connect.
    try:
        resp = client.generate(prompt, image_path, model=decision.chosen_model)
        print(f"\nResponse: {resp.get('response_text')}\n")
    except Exception as e:
        print(f"\n❌ Inference failed (is the model server running?): {e}\n")

if __name__ == "__main__":
    demo_pipeline()
