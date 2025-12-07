import logging
import asyncio
import random
import uuid
from typing import List

# We need to act as a client sending requests to the future FastAPI service
# But since we are inside the codebase, we can also use the components directly to populate DB.
# For a true simulation, we should call the components.

from common.config_loader import SystemConfig, get_default_paths
from router.router_service import RouterService
from load_balancer.load_balancer_service import LoadBalancerService
from inference_engine.inference_service import InferenceService
from .collector import DataCollector

logger = logging.getLogger(__name__)

class TrafficSimulator:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.router = RouterService(config.router_config)
        self.lb = LoadBalancerService(config.lb_config)
        self.inference = InferenceService(config.inference_config_path)
        self.collector = DataCollector(config.router_config.db_url)
        
    async def simulate_request(self, prompt: str, image_id: str = None):
        """Simulate a single end-to-end request."""
        sample_id = str(uuid.uuid4())
        
        # 1. Router
        router_result = self.router.predict(prompt)
        # router_result: {chosen_model, rewards, mode ...} but predict() wraps engine
        # Engine returns {chosen_model, rewards ...}
        
        best_model = router_result['chosen_model']
        router_probs = router_result.get('rewards', {})
        
        # 2. Load Balancer
        # We need a task type. Router might guess it? 
        # Or we default to 'vqa'
        decision = self.lb.schedule(
            sample_id=sample_id,
            task_type="vlm_router", # generic
            router_probs=router_probs,
            preferred_model=best_model
        )
        
        target_model = decision['chosen_model']
        
        # 3. Inference
        # We might not have a real image file matching image_id in this mock
        # So InferenceService might fail if we pass a bad path.
        # For simulation, we can mock the call or ensure we have assets.
        # We'll try to call. If it fails (no API key/no file), we catch it.
        try:
             # Just text call for now to trigger logic
             inference_result = self.inference.call_model(
                 model_name=target_model,
                 prompt=prompt
             )
        except Exception as e:
            logger.warning(f"Inference failed (expected in sim without real backend): {e}")
            inference_result = {"text": "Simulation Dummy Response", "finish_reason": "simulated"}
            
        # 4. Log
        await self.collector.log_request_async(
            sample_id=sample_id,
            prompt=prompt,
            image_id=image_id,
            router_decision=router_result,
            lb_decision=decision,
            inference_result=inference_result
        )
        
        # 5. Simulate Feedback (Randomly)
        if random.random() < 0.3:
            # 30% chance of feedback
            score = random.choice([1.0, 3.0, 5.0])
            self.collector.update_feedback(sample_id, score=score, is_verified=True)
            
    async def run_batch(self, count: int = 10):
        prompts = [
            "What is in this image?",
            "Describe the red object.",
            "Read the text on the sign.",
            "Is there a cat?",
            "Count the people."
        ]
        
        tasks = []
        for _ in range(count):
            p = random.choice(prompts)
            tasks.append(self.simulate_request(p))
            
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    # Smoke test
    logging.basicConfig(level=logging.INFO)
    try:
        paths = get_default_paths()
        # Ensure we have valid paths or mock them
        # This main block is for manual testing
        pass
    except Exception as e:
        print(e)
