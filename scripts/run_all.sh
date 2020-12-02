#!/bin/bash
# Experiments for testing number of components
# and number of trajectories
./scripts/run_ubm_experiments.sh

# Studying evolution of different algorithms
./scripts/run_tsne_experiments.sh

# Experiments for different PPO ratio-clip values
./scripts/run_ppo_clip_analysis.sh

# Imitation learning experiments
./scripts/run_imitation_experiments.sh

# Finally, plot the figures shown in the main paper and appendix.
# These will appear under "figures" directory
python3 plot_paper.py
python3 plot_appendix.py
