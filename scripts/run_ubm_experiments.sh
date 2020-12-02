#!/bin/bash
# Train PPO models for testing UBM results
# with different amount of data and components

repetitions=3
envs="Pendulum-v0 CartPole-v1 Acrobot-v1 BipedalWalker-v3 LunarLander-v2"

for env in ${envs}
do
  for repetition in $(seq 1 ${repetitions})
  do
    CUDA_VISIBLE_DEVICES="" ./scripts/train_stable_baselines.sh ${env}_UBMPPO --env ${env} --agent ppo2 --num-snapshots 100 &
    # Offset launches a little bit
    sleep 3
  done
  wait $!
done

# Collect trajectories
# 200 trajectories is collected so we can also sample different trajectories for 100-trajectory experiments
CUDA_VISIBLE_DEVICES="" python3 collect_trajectories.py --num-workers 16 --skip-existing --num-trajs 200 experiments/stablebaselines_*UBMPPO*

# Train different UBMs
num_of_experiments_to_go_over=3
num_repetitions=3
num_trajectories=(100 50 25 10)
num_different_trajs=${#num_trajectories[@]}
all_num_components="64 32 16 8 4 2 1"
envs="Pendulum-v0 CartPole-v1 Acrobot-v1 BipedalWalker-v3 LunarLander-v2"

for num_traj_idx in $(seq 0 $(($num_different_trajs - 1)))
do
  num_traj=${num_trajectories[$num_traj_idx]}
  for num_components in ${all_num_components}
  do
    for repetition in $(seq 1 ${num_repetitions})
    do
      for env in ${envs}
      do
        experiments=experiments/stablebaselines_${env}_UBMPPO*
        experiments=($experiments)
        for experiment_i in $(seq 0 $(($num_of_experiments_to_go_over - 1)))
        do  
          current_experiment=${experiments[${experiment_i}]}
          output_dir=ubm_experiments/ubms_${num_components}_components_${num_traj}_trajectories/${current_experiment}_repetition_${repetition}/
          mkdir -p ${output_dir}
          python3 create_pivectors.py train_ubms ${output_dir} --max-trajectories ${num_traj} --n-components ${num_components} --skip-existing --inputs ${current_experiment} &
          sleep 3
        done
        wait $!
      done
    done
  done
done

python3 run_ubm_data_experiments.py
