#!/bin/bash
# Train models for basic control tasks (e.g. cartpole etc)

repetitions=5
envs="Pendulum-v0 CartPole-v1 Acrobot-v1 BipedalWalker-v3 LunarLander-v2"

# Standard experiment
for env in ${envs}
do
  for repetition in $(seq 1 ${repetitions})
  do
    CUDA_VISIBLE_DEVICES="" ./scripts/train_stable_baselines.sh ${env}_A2C --env ${env} --agent a2c &
    # Offset launches a little bit
    sleep 3
  done
  wait $!
done

