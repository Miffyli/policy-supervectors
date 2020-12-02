#!/bin/bash
# Train stable-baselines3 PPO on BipedalWalker for BC/GAIL imitation.

repetitions=5
envs="LunarLander-v2 BipedalWalkerHardcore-v3"

for env in ${envs}
do
  for repetition in $(seq 1 ${repetitions})
  do
    CUDA_VISIBLE_DEVICES="" ./scripts/train_stable_baselines3.sh ${env}_SB3-PPO --env ${env} --agent ppo &
    # Offset launches a little bit
    sleep 3
  done
  wait $!
done
