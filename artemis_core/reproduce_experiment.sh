#!/bin/bash
# Reproducible Experiment Runner
# Usage: ./reproduce_experiment.sh [SEED] [MODE]

SEED=${1:-42}
MODE=${2:-balanced}
LOG_FILE="experiment_${SEED}_${MODE}.log"

echo ">>> Starting Experiment Reproduction"
echo "    Seed: $SEED"
echo "    Mode: $MODE"
echo "    Log:  $LOG_FILE"

# Ensure we are in the project root
cd "$(dirname "$0")"

# Activate logging with strict reproducibility
python3 main.py \
    --prompt "Describe the visual hierarchy of this slide" \
    --image "examples/assets/demo.jpg" \
    --mode "$MODE" \
    --seed "$SEED" \
    --log-file "$LOG_FILE"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ">>> Experiment Completed Successfully. See $LOG_FILE"
else
    echo "❌ Experiment Failed with exit code $EXIT_CODE. Check logs."
fi

exit $EXIT_CODE
