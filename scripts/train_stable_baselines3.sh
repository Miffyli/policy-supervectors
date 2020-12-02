#!/bin/bash
# Train models with stable-baselines

if test -z "$1"
then
    echo "Usage: train_stable_baselines experiment_name [parameters_to_train ...]"
    exit
fi

experiment_dir=experiments/stablebaselines_${1}_$(date -Iseconds)
# Prepare directory
./scripts/prepare_experiment_dir.sh ${experiment_dir}
# Store the launch parameters there
echo ${@:0} > ${experiment_dir}/launch_arguments.txt

# Run code
python3 train_stable_baselines3.py \
  --output=${experiment_dir} \
  ${@:2} \
  | tee ${experiment_dir}/stdout.txt
