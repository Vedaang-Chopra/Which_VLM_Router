
import os
import sys
from pathlib import Path

# Add project root to sys.path if needed
ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from artemis_final.inference_engine.client import WhichVLMClient
except ImportError:
    WhichVLMClient = None

class MockVLMClient:
    """A mock client that simulates VLM responses for testing/simulation."""
    def __init__(self):
        self.vlm = self
        self.llm = self # Just in case
    
    # Mocking VLMTestSuite.run_image
    def run_image(self, image, text=None, models=None, system=None, **gen_kwargs):
        target_models = []
        if models is None or models == "all":
            # Just pretend we have one default model if none specified
            target_models = ["mock_model"]
        elif isinstance(models, str):
            target_models = [models]
        elif isinstance(models, list):
            target_models = models
        else:
            target_models = ["mock_model"]

        results = {}
        for m in target_models:
            results[m] = {
                "ok": True,
                "response_text": f"[Mock VLM Response] Model: {m}, Prompt: {str(text)[:30]}...",
                "latency_ms": 150.0,
                "usage": {"total_tokens": 50},
                "error": None
            }
        return results

    # Mocking LLMTestSuite too if needed
    def run_single(self, prompt, models=None, **kwargs):
        return self.run_image(image=None, text=prompt, models=models, **kwargs)

def get_vlm_client(simulation_only: bool = True, config_path: str = None):
    """
    Factory function to get a VLM client.
    
    Args:
        simulation_only: If True, returns a MockVLMClient.
        config_path: Path to models.yaml. If None, tries to find it in default locations.
    
    Returns:
        An instance of WhichVLMClient (real) or MockVLMClient.
    """
    if simulation_only:
        print("[VLM Client] Using Mock Client (simulation_only=True)")
        return MockVLMClient()
    
    if WhichVLMClient is None:
        print("[VLM Client] Warning: Could not import WhichVLMClient. Falling back to Mock.")
        return MockVLMClient()

    # Try to resolve config path
    if not config_path:
        # Check standard location
        default_path = ROOT_DIR / 'artemis_final' / 'ares' / 'configs' / 'models.yaml'
        if default_path.exists():
            config_path = str(default_path)
        else:
            print(f"[VLM Client] Warning: models.yaml not found at {default_path}. Falling back to Mock.")
            return MockVLMClient()
            
    try:
        print(f"[VLM Client] Initializing Real Client from {config_path}")
        return WhichVLMClient.from_yaml(config_path)
    except Exception as e:
        print(f"[VLM Client] Error initializing Real Client: {e}. Falling back to Mock.")
        return MockVLMClient()
