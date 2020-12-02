# Train PPO with different clip ratios to see how it affects
# evolution of their behaviour

repetitions=5
envs="Pendulum-v0 CartPole-v1 Acrobot-v1 BipedalWalker-v3 LunarLander-v2"
num_checkpoints=50
clipranges="0.01 0.1 0.2 0.3 0.4 0.5 0.75 1.0 2.0 5.0 10.0"

for env in ${envs}
do
  for cliprange in ${clipranges}
  do
    for repetition in $(seq 1 ${repetitions})
    do
      CUDA_VISIBLE_DEVICES="" ./scripts/train_stable_baselines.sh ${env}_PPO-clip${cliprange} --env ${env} --agent ppo2 --num-snapshots ${num_checkpoints} --forced-cliprange ${cliprange} &
      # Offset launches a little bit
      sleep 3
    done
    wait $!
  done
done

# Add A2C run
for env in ${envs}
do
  for repetition in $(seq 1 ${repetitions})
  do
    CUDA_VISIBLE_DEVICES="" ./scripts/train_stable_baselines.sh ${env}_A2C-clip --env ${env} --agent a2c --num-snapshots ${num_checkpoints} &
    # Offset launches a little bit
    sleep 3
  done
  wait $!
done

CUDA_VISIBLE_DEVICES="" python3 collect_trajectories.py --num-workers 16 --skip-existing experiments/stablebaselines_*_PPO-clip* experiments/stablebaselines_*_A2C-clip*

# Train shared UBMs so we can compare things against each other
# Super elegant use of asterisk here and elsewhere in the code... Perhaps one day I will learn
mkdir -p ubms_ppo_clip
python3 create_pivectors.py train_ubms ubms_ppo_clip --inputs experiments/stablebaselines_*_PPO-clip* experiments/stablebaselines_*_A2C-clip*
python3 create_pivectors.py pivectors ubms_ppo_clip --inputs experiments/stablebaselines_*_PPO-clip* experiments/stablebaselines_*_A2C-clip*
python3 analyze_pivectors.py checkpoint-distances --inputs experiments/stablebaselines_*_PPO-clip* experiments/stablebaselines_*_A2C-clip*
