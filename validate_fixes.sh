#!/bin/bash
# Validation script for Artemis bug fixes
# Run this from the repo root to validate all fixes

set -e  # Exit on first error

echo "============================================================"
echo "Artemis Bug Fix Validation Script"
echo "============================================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

pass_count=0
total_count=0

check() {
    total_count=$((total_count + 1))
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
        pass_count=$((pass_count + 1))
    else
        echo -e "${RED}✗${NC} $1"
    fi
}

# Test 1: File syntax checks
echo "1. Checking syntax of modified files..."
python3 -m py_compile artemis_core/main.py 2>/dev/null
check "main.py syntax"

python3 -m py_compile artemis_core/src/artemis/common/config_loader.py 2>/dev/null
check "config_loader.py syntax"

python3 -m py_compile artemis_core/src/artemis/router/router.py 2>/dev/null
check "router.py syntax"

python3 -m py_compile artemis_core/src/artemis/load_balancer/balancer.py 2>/dev/null
check "balancer.py syntax"

python3 -m py_compile artemis_final/common/data_utils.py 2>/dev/null
check "data_utils.py syntax"

echo ""

# Test 2: Code pattern checks
echo "2. Checking for bug fix patterns..."

grep -q "checkpoint_path" artemis_core/main.py
check "Bug #1: main.py uses checkpoint_path"

grep -q "if h > 0 else" artemis_core/src/artemis/router/router.py
check "Bug #3: Division by zero guard in router"

grep -q "missing_stats.*model_not_found" artemis_core/src/artemis/load_balancer/balancer.py
check "Bug #4: Missing stats marker in balancer"

grep -q "enforce_accuracy_drop" artemis_core/src/artemis/load_balancer/balancer.py
check "Bug #10: Accuracy drop enforcement logic"

grep -q "estimated_cost_usd.*not in eval_df.columns" artemis_final/common/data_utils.py
check "Bug #7: Column validation in data_utils"

echo ""

# Test 3: Run unit tests
echo "3. Running unit tests..."
cd artemis_core
python3 -m unittest tests.test_bugfixes -v 2>&1 | grep -E "(OK|FAILED|ERROR)" > /tmp/test_result.txt

if grep -q "OK" /tmp/test_result.txt; then
    total_count=$((total_count + 1))
    pass_count=$((pass_count + 1))
    echo -e "${GREEN}✓${NC} All unit tests pass"
else
    total_count=$((total_count + 1))
    echo -e "${RED}✗${NC} Unit tests failed"
    cat /tmp/test_result.txt
fi

cd ..
echo ""

# Test 4: Config validation test
echo "4. Testing config validation..."
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

python3 << 'PYEOF' 2>&1 | grep -q "Correctly rejects"
import sys
sys.path.append("artemis_core/src")
from artemis.common.config_loader import load_config
try:
    load_config('/tmp/test_invalid_config.yaml')
    print('ERROR: Should have failed!')
    sys.exit(1)
except ValueError as e:
    if 'global_sla' in str(e):
        print('Correctly rejects missing global_sla')
        sys.exit(0)
PYEOF

check "Bug #2: Config validation rejects invalid configs"

echo ""
echo "============================================================"
echo "Validation Results: $pass_count/$total_count checks passed"
echo "============================================================"

if [ $pass_count -eq $total_count ]; then
    echo -e "${GREEN}All validations passed!${NC}"
    exit 0
else
    echo -e "${RED}Some validations failed.${NC}"
    exit 1
fi
