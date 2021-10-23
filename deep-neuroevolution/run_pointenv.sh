#!/bin/bash
# Run experiments on the DeceptivePointEnv
# Note: Very messy code, could have used functions and all that, but you know, researchers...

repetitions=10
env="DeceptivePointEnv-v0"
worker_session_name="workers"

# Standard ES experiments
for repetition in $(seq 1 ${repetitions})
do
  # Prepare outputdir
  experiment_dir=../experiments/novelty_${env}_es_$(date -Iseconds)
  ../scripts/prepare_experiment_dir.sh ${experiment_dir}
  
  # Make sure we have fresh redis
  ./scripts/local_kill_redis.sh
  sleep 1
  ./scripts/local_run_redis_noattach.sh
  sleep 1
  
  # Launch workers (consider using a function for this?)
  tmux new -s ${worker_session_name} -d
  tmux send-keys -t ${worker_session_name} ". scripts/local_env_setup.sh" C-m
  tmux send-keys -t ${worker_session_name} "python -m es_distributed.main workers --master_host localhost --relay_socket_path /tmp/es_redis_relay.sock --algo es --num_workers 32" C-m
  # Launch master and wait for it to die
  python -m es_distributed.main master --master_socket_path /tmp/es_redis_master.sock --algo es --exp_file configurations/pointenv_es.json --log_dir ${experiment_dir}
  
  # Kill workers and sleep for a while to wait for things to settle (just in case)
  tmux kill-session -t ${worker_session_name}
  tmux kill-session -t redis
  sleep 1
done

# NSR-ES with terminal state
for repetition in $(seq 1 ${repetitions})
do
  # Prepare outputdir
  experiment_dir=../experiments/novelty_${env}_nsres_$(date -Iseconds)
  ../scripts/prepare_experiment_dir.sh ${experiment_dir}
  
  # Make sure we have fresh redis
  ./scripts/local_kill_redis.sh
  sleep 1
  ./scripts/local_run_redis_noattach.sh
  sleep 1
  
  # Launch workers 
  tmux new -s ${worker_session_name} -d
  tmux send-keys -t ${worker_session_name} ". scripts/local_env_setup.sh" C-m
  tmux send-keys -t ${worker_session_name} "python -m es_distributed.main workers --master_host localhost --relay_socket_path /tmp/es_redis_relay.sock --algo nsr-es --num_workers 32" C-m
  # Launch master and wait for it to die
  python -m es_distributed.main master --master_socket_path /tmp/es_redis_master.sock --algo nsr-es --exp_file configurations/pointenv_nsres.json --log_dir ${experiment_dir}
  
  # Kill workers and sleep for a while to wait for things to settle (just in case)
  tmux kill-session -t ${worker_session_name}
  tmux kill-session -t redis
  sleep 1
done

# NSR-ES with Gaussian novelty
for repetition in $(seq 1 ${repetitions})
do
  # Prepare outputdir
  experiment_dir=../experiments/novelty_${env}_nsresgaussian_$(date -Iseconds)
  ../scripts/prepare_experiment_dir.sh ${experiment_dir}
  
  # Make sure we have fresh redis
  ./scripts/local_kill_redis.sh
  sleep 1
  ./scripts/local_run_redis_noattach.sh
  sleep 1
  
  # Launch workers (consider using a function for this?)
  tmux new -s ${worker_session_name} -d
  tmux send-keys -t ${worker_session_name} ". scripts/local_env_setup.sh" C-m
  tmux send-keys -t ${worker_session_name} "python -m es_distributed.main workers --master_host localhost --relay_socket_path /tmp/es_redis_relay.sock --algo nsr-es --num_workers 32 --worker_dir ${experiment_dir}" C-m
  # Launch master and wait for it to die
  python -m es_distributed.main master --master_socket_path /tmp/es_redis_master.sock --algo nsr-es --exp_file configurations/pointenv_gaussian.json --log_dir ${experiment_dir}
  
  # Kill workers and sleep for a while to wait for things to settle (just in case)
  tmux kill-session -t ${worker_session_name}
  tmux kill-session -t redis
  sleep 1
done

# NSR-ES with Supervectors
for repetition in $(seq 1 ${repetitions})
do
  # Prepare outputdir
  experiment_dir=../experiments/novelty_${env}_nsressupervector_$(date -Iseconds)
  ../scripts/prepare_experiment_dir.sh ${experiment_dir}
  
  # Make sure we have fresh redis
  ./scripts/local_kill_redis.sh
  sleep 1
  ./scripts/local_run_redis_noattach.sh
  sleep 1
  
  # Launch workers 
  tmux new -s ${worker_session_name} -d
  tmux send-keys -t ${worker_session_name} ". scripts/local_env_setup.sh" C-m
  tmux send-keys -t ${worker_session_name} "python -m es_distributed.main workers --master_host localhost --relay_socket_path /tmp/es_redis_relay.sock --algo nsr-es --num_workers 32 --worker_dir ${experiment_dir}" C-m
  # Launch master and wait for it to die
  python -m es_distributed.main master --master_socket_path /tmp/es_redis_master.sock --algo nsr-es --exp_file configurations/pointenv_supervector.json --log_dir ${experiment_dir}
  
  # Kill workers and sleep for a while to wait for things to settle (just in case)
  tmux kill-session -t ${worker_session_name}
  tmux kill-session -t redis
  sleep 1
done
