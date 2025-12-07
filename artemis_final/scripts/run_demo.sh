#!/bin/bash
# =============================================================================
# Artemis VLM Router - Demo Script
# =============================================================================
# This script runs a complete demonstration of the Artemis system.
#
# Usage:
#   ./scripts/run_demo.sh              # Run full demo
#   ./scripts/run_demo.sh --quick      # Quick demo (fewer samples)
#   ./scripts/run_demo.sh --retrain    # Show retraining improvement
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           🚀 ARTEMIS VLM ROUTER - DEMO SUITE                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Parse arguments
QUICK_MODE=false
RETRAIN_DEMO=false
NUM_SAMPLES=50

for arg in "$@"; do
    case $arg in
        --quick)
            QUICK_MODE=true
            NUM_SAMPLES=20
            shift
            ;;
        --retrain)
            RETRAIN_DEMO=true
            shift
            ;;
        --samples=*)
            NUM_SAMPLES="${arg#*=}"
            shift
            ;;
        *)
            ;;
    esac
done

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.10+"
    exit 1
fi

echo -e "${GREEN}✓ Python found: $(python3 --version)${NC}"

# Check if in virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}⚠ Virtual environment not activated. Consider using one.${NC}"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "📋 Demo Configuration:"
echo "    Samples: $NUM_SAMPLES"
echo "    Mode:    balanced"
echo "    Device:  cpu"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Run the main demo
echo -e "${BLUE}▶ Running Full Pipeline Demo...${NC}"
echo ""

python3 scripts/demo_full_pipeline.py \
    --num-samples "$NUM_SAMPLES" \
    --mode balanced \
    --device cpu \
    --no-db

# Optionally run retraining demo
if [ "$RETRAIN_DEMO" = true ]; then
    echo ""
    echo -e "${BLUE}▶ Running Retraining Improvement Demo...${NC}"
    echo ""
    
    python3 scripts/demo_retrain_improvement.py \
        --samples "$NUM_SAMPLES" \
        --mode balanced
fi

echo ""
echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ DEMO COMPLETE                             ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo "
Next steps:
  1. To run with real data:  python scripts/demo_full_pipeline.py --mode accuracy
  2. To see retraining demo: ./scripts/run_demo.sh --retrain
  3. To start the API:       uvicorn main:app --reload
  4. To use Docker:          docker-compose up -d
"
