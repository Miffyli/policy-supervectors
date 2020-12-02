#!/bin/bash
# Train models for basic control tasks (e.g. cartpole etc) using prettyNEAT code

# No repetitions (we get enough agent models as it is...)
envs="Pendulum-v0 CartPole-v1 Acrobot-v1 LunarLander-v2 BipedalWalker-v3"

# Standard experiment
for env in ${envs}
do
  experiment_dir=experiments/wann_${env}_CMAES_$(date -Iseconds)
  ./scripts/prepare_experiment_dir.sh ${experiment_dir}
  mkdir -p ${experiment_dir}/checkpoints
  
  # Now, navigate to the prettyNEAT directory, run the script
  # and come back up
  cd prettyNEAT
  output_dir=../${experiment_dir}/checkpoints
  python3 cmaes.py -p p/${env}.json -o ${output_dir}
  cd ..
done

