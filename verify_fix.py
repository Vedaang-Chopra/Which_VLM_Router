import sys
import os
import time
from dataclasses import dataclass

# Add project root to path
sys.path.append("/Users/vedaangchopra/all_data/complete_technical_work/all_projects_implemented/Which_VLM_Router")

try:
    from artemis_final.load_balancer.model_state import ModelLoadState, ReplicaState
    
    # Create dummy replica
    replicas = [ReplicaState(available_at_ms=time.time() * 1000.0)]
    
    # Create state
    state = ModelLoadState(model_name="test_model", replicas=replicas)
    
    # Test method
    current_time = time.time()
    delay = state.estimate_queue_delay(current_time)
    
    print(f"Success! Estimated queue delay: {delay} ms")
    
except ImportError as e:
    print(f"ImportError: {e}")
except AttributeError as e:
    print(f"AttributeError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
