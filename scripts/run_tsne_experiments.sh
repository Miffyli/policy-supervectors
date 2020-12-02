#!/bin/bash
# Train agents
./scripts/train_control_a2c.sh
./scripts/train_control_ppo.sh
./scripts/train_control_neat.sh
./scripts/train_control_cmaes.sh

# Collect trajectories per policy
CUDA_VISIBLE_DEVICES="" python3 collect_trajectories.py --num-workers 16 --skip-existing experiments/wann_* experiments/stablebaselines_*_A2C_* experiments/stablebaselines_*_PPO_*

mkdir -p tsne_ubms
# Train UBMs
python3 create_pivectors.py train_ubms tsne_ubms --max-checkpoints 100 --skip-existing --inputs experiments/wann_* experiments/stablebaselines_*_A2C_* experiments/stablebaselines_*_PPO_*

# Create policy supervectors
python3 create_pivectors.py pivectors tsne_ubms --inputs experiments/wann_* experiments/stablebaselines_*_A2C_* experiments/stablebaselines_*_PPO_*

# Run t-SNE and store into policy supervector files
python3 analyze_pivectors.py tsne --inputs experiments/wann_* experiments/stablebaselines_*_A2C_* experiments/stablebaselines_*_PPO_*
