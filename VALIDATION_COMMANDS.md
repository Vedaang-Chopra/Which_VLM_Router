# Validation Commands - Artemis Bug Fixes

This document provides step-by-step commands to validate all bug fixes from a clean checkout.

## Prerequisites

```bash
cd /path/to/Which_VLM_Router
source .venv/bin/activate  # Or your virtual environment
pip install -r requirements.txt
```

## Validation Steps

### 1. Verify All Files Are Modified

```bash
# Check that main.py uses checkpoint_path (Bug #1)
grep "checkpoint_path" artemis_core/main.py
# Expected: router_ckpt = str(Path(config.router.checkpoint_path).expanduser())

# Check that demo_pipeline.py uses checkpoint_path (Bug #1)
grep "checkpoint_path" artemis_core/examples/demo_pipeline.py
# Expected: router = RewardRouter(config.router.checkpoint_path)

# Check config validation is present (Bug #2)
grep -A 5 "require(global_sla_raw" artemis_core/src/artemis/common/config_loader.py
# Expected: Should see validation for total_cost_budget_usd, min_global_accuracy, default_latency_ms

# Check division by zero guard (Bug #3)
grep "if h > 0 else" artemis_core/src/artemis/router/router.py
# Expected: ar = w / h if h > 0 else 1.0  # Guard against division by zero

# Check missing_stats handling (Bug #4)
grep "missing_stats" artemis_core/src/artemis/load_balancer/balancer.py
# Expected: return SimulationResult(..., missing_stats=["model_not_found"])

# Check enforce_accuracy_drop logic (Bug #10)
grep "enforce_accuracy_drop" artemis_core/src/artemis/load_balancer/balancer.py
# Expected: enforce_accuracy_drop = pref_acc > 0.0

# Check data_utils validation (Bug #7)
grep "estimated_cost_usd" artemis_final/common/data_utils.py | head -5
# Expected: Should see validation before sorting

# Check inference_reward_router exception handling (Bug #8, #9)
grep -A 3 "Let exceptions propagate" artemis_final/router/core/inference_reward_router.py
# Expected: Direct call to _load_checkpoint_safe without try/except wrapper
```

### 2. Run Unit Tests

```bash
# Run all bug fix tests
cd artemis_core
python3 -m unittest tests.test_bugfixes -v

# Expected output:
# test_zero_baseline_accuracy_skips_constraint ... ok
# test_config_has_checkpoint_path ... ok
# test_missing_data_collection_raises_error ... ok
# test_missing_global_sla_field_raises_error ... ok
# test_missing_global_sla_raises_error ... ok
# test_zero_height_image_does_not_crash ... ok
# test_unknown_model_returns_missing_stats ... ok
# test_missing_cost_column_returns_empty_df ... ok
# test_full_config_loading ... ok
#
# ----------------------------------------------------------------------
# Ran 9 tests in 0.XXXs
#
# OK
```

### 3. Run Existing Test Suite

```bash
# Run core tests
cd artemis_core
python3 -m unittest tests.test_core -v

# Expected: All tests should pass
```

### 4. Validate Config Loading (Bug #2 & #5)

```bash
# Test invalid config (missing global_sla)
cat > /tmp/test_invalid_config.yaml << 'EOF'
db:
  url: "sqlite:///:memory:"
router:
  checkpoint_path: "dummy.pt"
load_balancer:
  task_slas: {}
data_collection:
  samples_table: "samples"
  responses_table: "responses"
  feedback_table: "feedback"
EOF

cd artemis_core
python3 -c "
from artemis.common.config_loader import load_config
try:
    load_config('/tmp/test_invalid_config.yaml')
    print('ERROR: Should have failed!')
    exit(1)
except ValueError as e:
    if 'global_sla' in str(e):
        print('✓ Correctly rejects missing global_sla')
        exit(0)
    else:
        print(f'ERROR: Wrong error: {e}')
        exit(1)
"

# Expected: ✓ Correctly rejects missing global_sla
```

### 5. Validate Router Division by Zero Protection (Bug #3)

```bash
cd artemis_core
python3 << 'EOF'
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / "src"))

# Check the code contains the fix
router_code = (Path.cwd() / "src" / "artemis" / "router" / "router.py").read_text()

if "if h > 0 else" in router_code and "Guard against division by zero" in router_code:
    print("✓ Router has division by zero protection")
    exit(0)
else:
    print("ERROR: Division by zero fix not found")
    exit(1)
EOF

# Expected: ✓ Router has division by zero protection
```

### 6. Validate Load Balancer Stats Handling (Bug #4)

```bash
cd artemis_core
python3 << 'EOF'
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / "src"))

# Mock dependencies
sys.modules["torch"] = type('module', (), {})()
sys.modules["numpy"] = type('module', (), {})()

from artemis.load_balancer import LoadBalancer
from artemis.load_balancer.types import ModelCapacityConfig

configs = {"model_a": ModelCapacityConfig(min_replicas=1)}
lb = LoadBalancer(configs, mode="balanced")

# Test unknown model
sim = lb._simulate("unknown_model", "vqa", 0.0)

if "model_not_found" in sim.missing_stats:
    print("✓ Unknown models return missing_stats marker")
    exit(0)
else:
    print(f"ERROR: missing_stats = {sim.missing_stats}")
    exit(1)
EOF

# Expected: ✓ Unknown models return missing_stats marker
```

### 7. Validate Data Utils Column Check (Bug #7)

```bash
cd artemis_final
python3 << 'EOF'
import sys
from pathlib import Path

# Check the fix is in the code
code = (Path.cwd() / "common" / "data_utils.py").read_text()

if "'estimated_cost_usd' not in eval_df.columns" in code:
    print("✓ Data utils has column validation")
    exit(0)
else:
    print("ERROR: Column validation fix not found")
    exit(1)
EOF

# Expected: ✓ Data utils has column validation
```

### 8. Final Sanity Check

```bash
# Verify no syntax errors in modified files
python3 -m py_compile artemis_core/main.py
python3 -m py_compile artemis_core/examples/demo_pipeline.py
python3 -m py_compile artemis_core/src/artemis/common/config_loader.py
python3 -m py_compile artemis_core/src/artemis/router/router.py
python3 -m py_compile artemis_core/src/artemis/load_balancer/balancer.py
python3 -m py_compile artemis_final/common/data_utils.py
python3 -m py_compile artemis_final/router/core/inference_reward_router.py

echo "✓ All modified files compile successfully"
```

## Summary of Validation

| Bug # | Validation Method | Expected Result |
|-------|-------------------|-----------------|
| 1 | `grep checkpoint_path main.py` | Found in main.py and demo_pipeline.py |
| 2 | Unit test `test_missing_global_sla_raises_error` | PASS |
| 3 | Unit test `test_zero_height_image_does_not_crash` | PASS |
| 4 | Unit test `test_unknown_model_returns_missing_stats` | PASS with warning log |
| 5 | Unit test `test_missing_data_collection_raises_error` | PASS |
| 7 | Code inspection + manual test | Column validation present |
| 8 | Code inspection | Exception propagates naturally |
| 9 | Code inspection | Type validation instead of swapping |
| 10 | Unit test `test_zero_baseline_accuracy_skips_constraint` | PASS |

## Quick Validation (One Command)

To run all validations at once:

```bash
cd /path/to/Which_VLM_Router
./validate_fixes.sh
```

This runs all tests and checks in sequence. Exit code 0 means all validations passed.

## Rollback (If Needed)

If you need to rollback the changes:

```bash
# Restore from backups created by sed
cd artemis_core
mv main.py.bak main.py
mv src/artemis/router/router.py.bak src/artemis/router/router.py
mv src/artemis/load_balancer/balancer.py.bak src/artemis/load_balancer/balancer.py

# For config_loader.py and others, use git:
git checkout artemis_core/src/artemis/common/config_loader.py
git checkout artemis_core/examples/demo_pipeline.py
git checkout artemis_final/common/data_utils.py
git checkout artemis_final/router/core/inference_reward_router.py
```
