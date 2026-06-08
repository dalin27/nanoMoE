#!/bin/bash

# Array containing all your configuration paths
CONFIGS=(
    "config/ml_optim/env1/adamw.py"
    "config/ml_optim/env1/adam.py"
    "config/ml_optim/env1/sgd.py"
    "config/ml_optim/env1/adagrad.py"
    "config/ml_optim/env1/adafactor.py"
    "config/ml_optim/env1/sparse_adam.py"
)

echo "Starting sequential training pipeline..."

# Loop through each configuration file
for CONFIG in "${CONFIGS[@]}"; do
    # Extract the filename without the .py extension to use for the log file
    OPTIMIZER_NAME=$(basename "$CONFIG" .py)
    LOG_FILE="training_output_${OPTIMIZER_NAME}.log"
    
    echo "==================================================="
    echo "Starting run for: $OPTIMIZER_NAME"
    echo "Config path: $CONFIG"
    echo "Logs will be saved to: $LOG_FILE"
    echo "==================================================="
    
    # Run the training command sequentially
    python3 -m torch.distributed.run --standalone --nproc_per_node=2 train.py "$CONFIG" > "$LOG_FILE" 2>&1
    
    # Check if the run was successful or failed before moving to the next
    if [ $? -eq 0 ]; then
        echo "SUCCESS: $OPTIMIZER_NAME completed successfully."
    else
        echo "ERROR: $OPTIMIZER_NAME failed. Check $LOG_FILE for details."
    fi
    echo ""

done

echo "All training runs have finished!"