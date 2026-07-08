
import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Mock dependencies BEFORE importing modules
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["openai"] = MagicMock()

# Append src to path
sys.path.append(str(Path(__file__).parents[1] / "src"))

from artemis.common.config_loader import load_config, GlobalConfig
from artemis.load_balancer import LoadBalancer
from artemis.load_balancer.types import ModelCapacityConfig

class TestArtemisCore(unittest.TestCase):
    
    def test_config_loader_structure(self):
        """Test that config loader enforces structure."""
        # Create a dummy config file
        dummy_config = """
db:
  url: "sqlite:///:memory:"
router:
  checkpoint_path: "dummy.pt"
load_balancer:
  global_sla:
    total_cost_budget_usd: 10
    min_global_accuracy: 0.9
    default_latency_ms: 1000
data_collection:
    samples_table: "samples"
    responses_table: "responses"
    feedback_table: "feedback"
    """
        with patch("builtins.open", unittest.mock.mock_open(read_data=dummy_config)):
            with patch("pathlib.Path.exists", return_value=True):
                config = load_config("dummy.yaml")
                self.assertIsInstance(config, GlobalConfig)
                self.assertEqual(config.db.url, "sqlite:///:memory:")

    def test_load_balancer_init(self):
        """Test LoadBalancer initialization."""
        configs = {
            "model_a": ModelCapacityConfig(min_replicas=1)
        }
        lb = LoadBalancer(configs, mode="balanced")
        self.assertIn("model_a", lb.states)
        self.assertEqual(len(lb.states["model_a"].replicas), 1)

if __name__ == "__main__":
    unittest.main()
