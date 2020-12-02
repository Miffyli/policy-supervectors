#!/bin/bash
# Train expert models
./scripts/train_sb3_control.sh
# Run main code
python3 run_imitation_experiments.py
