# Run experiments with stable-baselines3 agents
# Hyperparameters stolen and adapted from rl-zoo:
#   https://github.com/araffin/rl-baselines-zoo/tree/master/hyperparams
#
import os
from argparse import ArgumentParser

import gym
import torch

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

# Speeds things up often.
torch.set_num_threads(1)

AVAILABLE_ALGORITHMS = {
    "ppo": PPO,
}

AVAILABLE_ENVIRONMENTS = [
    "BipedalWalkerHardcore-v3",
    "LunarLander-v2"
]

CHECKPOINT_DIR = "checkpoints"
AGENT_FILE = "trained_agent.zip"

parser = ArgumentParser("Run stable-baselines3 to collect policies at different points of training.")
parser.add_argument("--output", type=str, required=True, help="Directory where to put results.")
parser.add_argument("--agent", type=str, required=True, choices=list(AVAILABLE_ALGORITHMS.keys()), help="Algorithm to use.")
parser.add_argument("--env", required=True, choices=AVAILABLE_ENVIRONMENTS, help="Environment to play.")
parser.add_argument("--num-snapshots", type=int, default=100, help="Number of snapshots to save.")


def create_env(args, idx):
    """
    Create and return an environment according to args (parsed arguments).
    idx specifies idx of this environment among parallel environments.
    """
    monitor_file = os.path.join(args.output, ("env_%d" % idx))

    env = gym.make(args.env)
    env = Monitor(env, monitor_file)

    return env


def run_experiment(args):
    n_envs = None
    n_timesteps = None
    policy = "MlpPolicy"
    hyperparams = {}
    # Super-pretty manual hardcoding of the parameters
    # right here, but we only are going to run one type.
    if args.env == "BipedalWalkerHardcore-v3":
        # Adapted from
        # https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/ppo2.yml
        n_envs = 16
        n_timesteps = int(10e7)
        policy = "MlpPolicy"
        hyperparams = {
            "n_steps": 2048,
            "gae_lambda": 0.95,
            "gamma": 0.99,
            "n_epochs": 10,
            "ent_coef": 0.001,
            "clip_range": 0.2,
            "clip_range_vf": 0.2,
            "learning_rate": 2.5e-4,
            "batch_size": (2048 * 16) // 32
        }
    else:
        # LunarLander-v2
        n_envs = 16
        n_timesteps = int(1e6)
        policy = "MlpPolicy"
        hyperparams = {
            "n_steps": 1024,
            "gae_lambda": 0.98,
            "gamma": 0.999,
            "n_epochs": 4,
            "ent_coef": 0.01,
            "clip_range": 0.2,
            "clip_range_vf": 0.2,
            "batch_size": (1024 * 16) // 32
        }

    vecEnv = []
    for i in range(n_envs):
        # Bit of trickery here to avoid referencing
        # to the same "i"
        vecEnv.append((
            lambda idx: lambda: create_env(args, idx))(i)
        )

    vecEnv = DummyVecEnv(vecEnv)

    agent_class = AVAILABLE_ALGORITHMS[args.agent]
    agent = agent_class(policy, vecEnv, verbose=1, device="cpu", **hyperparams)

    # Prepare callback
    checkpoint_dir = os.path.join(args.output, CHECKPOINT_DIR)
    os.makedirs(checkpoint_dir)
    # Note that save_freq is counted in number of agent step-calls,
    # not env step-calls.
    save_freq = n_timesteps // (args.num_snapshots * n_envs)

    checkpoint_callback = CheckpointCallback(save_freq, checkpoint_dir)

    agent.learn(total_timesteps=n_timesteps, callback=checkpoint_callback)
    agent.save(os.path.join(args.output, AGENT_FILE))

    vecEnv.close()


if __name__ == "__main__":
    args = parser.parse_args()
    run_experiment(args)
