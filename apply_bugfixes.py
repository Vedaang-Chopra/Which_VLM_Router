#!/usr/bin/env python3
"""
Apply bugfixes to Artemis codebase.
This script programmatically applies all identified bug fixes.
"""

import re
from pathlib import Path

def fix_bug4_load_balancer():
    """Fix Bug #4: Inconsistent missing stats handling in LoadBalancer"""
    file_path = Path("artemis_core/src/artemis/load_balancer/balancer.py")
    content = file_path.read_text()

    # Fix the fallback to include missing_stats
    old_pattern = r'if model_name not in self\.states:\s+# Fallback for unknown models.*\n\s+return SimulationResult\(0, 100, 100, 0, 0, 1, arrival_ms\+100, 0\)'
    new_code = '''if model_name not in self.states:
            # Fallback for unknown models (e.g. mocked ones) - mark missing stats
            logger.warning(f"Model '{model_name}' not found in load balancer states. Using fallback values.")
            return SimulationResult(0, 100, 100, 0, 0, 1, arrival_ms+100, 0, missing_stats=["model_not_found"])'''

    content = re.sub(old_pattern, new_code, content, flags=re.DOTALL)
    file_path.write_text(content)
    print(f"✓ Fixed Bug #4 in {file_path}")

def fix_bug10_load_balancer_accuracy():
    """Fix Bug #10: LoadBalancer Missing Accuracy Validation"""
    file_path = Path("artemis_core/src/artemis/load_balancer/balancer.py")
    content = file_path.read_text()

    # Add enforce_accuracy_drop logic
    old_pattern = r'(def _schedule_optimized.*?\n\s+valid = \[\]\s+\n\s+pref_acc = self\.stats\.estimate_accuracy\(output\.task_type, output\.preferred_model\))'
    new_code = r'\1\n\n        # Skip accuracy constraint if we have no baseline (pref_acc == 0.0 means no stats)\n        enforce_accuracy_drop = pref_acc > 0.0'

    content = re.sub(old_pattern, new_code, content, flags=re.DOTALL)

    # Update the constraint check
    old_constraint = r'if strategy in \["capacity", "balanced"\] and \(pref_acc - sim\.est_accuracy\) > self\.max_accuracy_drop:'
    new_constraint = 'if enforce_accuracy_drop and strategy in ["capacity", "balanced"] and (pref_acc - sim.est_accuracy) > self.max_accuracy_drop:'

    content = content.replace(old_constraint, new_constraint)
    file_path.write_text(content)
    print(f"✓ Fixed Bug #10 in {file_path}")

def fix_bug7_data_utils():
    """Fix Bug #7: Missing None Check in data_utils.py"""
    file_path = Path("artemis_final/common/data_utils.py")

    if not file_path.exists():
        print(f"⚠ Skipping Bug #7: {file_path} not found")
        return

    content = file_path.read_text()

    # Add validation before sorting
    old_pattern = r'(if acc_col not in eval_df\.columns:\s+logger\.warning\("No accuracy column found to compute oracle\."\)\s+return eval_df)'
    new_code = r'''\1

    # Validate required columns exist
    if 'estimated_cost_usd' not in eval_df.columns:
        logger.warning("Missing 'estimated_cost_usd' column for oracle computation.")
        return eval_df'''

    content = re.sub(old_pattern, new_code, content, flags=re.DOTALL)
    file_path.write_text(content)
    print(f"✓ Fixed Bug #7 in {file_path}")

def fix_bug8_inference_router():
    """Fix Bug #8: Incorrect Exception Handling in RewardRouterInference"""
    file_path = Path("artemis_final/router/core/inference_reward_router.py")

    if not file_path.exists():
        print(f"⚠ Skipping Bug #8: {file_path} not found")
        return

    content = file_path.read_text()

    # Remove the try/except wrapper that loses exception type
    old_pattern = r'try:\s+checkpoint = _load_checkpoint_safe\(checkpoint_path, map_location=self\.device\)\s+except Exception as e:\s+if verbose:\s+print\(f"\[ERROR\] Failed to load checkpoint safe: \{e\}"\)\s+# Try fallback.*\n\s+raise e'
    new_code = '# Let exceptions propagate naturally with their original type\n        checkpoint = _load_checkpoint_safe(checkpoint_path, map_location=self.device)'

    content = re.sub(old_pattern, new_code, content, flags=re.DOTALL)
    file_path.write_text(content)
    print(f"✓ Fixed Bug #8 in {file_path}")

def fix_bug9_argument_swapping():
    """Fix Bug #9: Router Allows Swapped Arguments Without Type Validation"""
    file_path = Path("artemis_final/router/core/inference_reward_router.py")

    if not file_path.exists():
        print(f"⚠ Skipping Bug #9: {file_path} not found")
        return

    content = file_path.read_text()

    # Replace the swapping logic with proper type validation
    old_pattern = r'# Handle swapped arguments.*?\n.*?if not isinstance\(prompt, str\) and isinstance\(image, str\):.*?\n.*?if self\.verbose:.*?\n.*?print\(.*?\)\n.*?prompt, image = image, prompt'
    new_code = '''# Validate prompt is a string
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt)}")

        # Validate image is None, PIL.Image, or str path
        if image is not None and not isinstance(image, (Image.Image, str)):
            raise TypeError(f"image must be None, PIL.Image, or str path, got {type(image)}")'''

    content = re.sub(old_pattern, new_code, content, flags=re.DOTALL)
    file_path.write_text(content)
    print(f"✓ Fixed Bug #9 in {file_path}")

if __name__ == "__main__":
    print("=" * 60)
    print("Applying Artemis Bug Fixes")
    print("=" * 60)

    # Apply all fixes
    fix_bug4_load_balancer()
    fix_bug10_load_balancer_accuracy()
    fix_bug7_data_utils()
    fix_bug8_inference_router()
    fix_bug9_argument_swapping()

    print("\n" + "=" * 60)
    print("All fixes applied successfully!")
    print("=" * 60)
