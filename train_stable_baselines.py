# Run experiments with stable-baselines agents
# Hyperparameters stolen from rl-zoo:
#   https://github.com/araffin/rl-baselines-zoo/tree/master/hyperparams
#
import os
from argparse import ArgumentParser
import time
import random

import numpy as np
import gym
import yaml
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind
from stable_baselines.bench import Monitor
from stable_baselines import PPO2, A2C

from stable_baselines.common.schedules import constfn

AVAILABLE_ALGORITHMS = {
    "ppo2": PPO2,
    "a2c": A2C,
}

HYPERPARAM_DIR = "hyperparams"
CHECKPOINT_DIR = "checkpoints"

parser = ArgumentParser("Run stable-baselines to collect policies at different points of training.")
parser.add_argument("--output", type=str, required=True, help="Directory where to put results.")
parser.add_argument("--agent", type=str, required=True, choices=list(AVAILABLE_ALGORITHMS.keys()), help="Algorithm to use.")
parser.add_argument("--env", required=True, help="Environment to play.")
parser.add_argument("--num-snapshots", type=int, default=500, help="Number of snapshots to save.")
parser.add_argument("--subprocenv", action="store_true", help="Use Subprocvecenv rather than dummy vecenv.")
parser.add_argument("--forced-cliprange", type=float, default=None, help="Override cliprange parameter for PPO.")


def linear_schedule(initial_value):
    """
    Taken from rl-zoo
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func


def create_env(args, idx):
    """
    Create and return an environment according to args (parsed arguments).
    idx specifies idx of this environment among parallel environments.
    """
    monitor_file = os.path.join(args.output, ("env_%d" % idx))

    # Check for Atari envs
    if "NoFrameskip" in args.env:
        env = make_atari(args.env)
        env = wrap_deepmind(env, frame_stack=True)
    else:
        env = gym.make(args.env)
    env = Monitor(env, monitor_file)

    return env


def run_experiment(args):
    hyperparam_file = os.path.join(HYPERPARAM_DIR, args.agent + ".yml")
    hyperparams = yaml.safe_load(open(hyperparam_file))

    hyperparams = hyperparams[args.env]

    n_envs = hyperparams.pop("n_envs", 1)
    n_timesteps = int(hyperparams.pop("n_timesteps"))
    policy = hyperparams.pop("policy")
    normalize = hyperparams.pop("normalize", None)

    vecEnv = []
    for i in range(n_envs):
        # Bit of trickery here to avoid referencing
        # to the same "i"
        vecEnv.append((
            lambda idx: lambda: create_env(args, idx))(i)
        )

    if args.subprocenv:
        vecEnv = SubprocVecEnv(vecEnv)
    else:
        vecEnv = DummyVecEnv(vecEnv)

    # Handle learning rates
    # Taken from rl-zoo/train.py
    for key in ['learning_rate', 'cliprange', 'cliprange_vf']:
        if key not in hyperparams or args.agent == "dqn":
            continue
        if key == 'learning_rate' and args.agent == "a2c":
            continue
        if isinstance(hyperparams[key], str):
            schedule, initial_value = hyperparams[key].split('_')
            initial_value = float(initial_value)
            hyperparams[key] = linear_schedule(initial_value)
        elif isinstance(hyperparams[key], (float, int)):
            # Negative value: ignore (ex: for clipping)
            if hyperparams[key] < 0:
                continue
            hyperparams[key] = constfn(float(hyperparams[key]))

    if args.forced_cliprange is not None:
        hyperparams["cliprange"] = args.forced_cliprange

    agent_class = AVAILABLE_ALGORITHMS[args.agent]
    agent = agent_class(policy, vecEnv, verbose=1, **hyperparams)

    # Prepare callback
    checkpoint_dir = os.path.join(args.output, CHECKPOINT_DIR)
    os.makedirs(checkpoint_dir)
    # Note that save_freq is counted in number of agent step-calls,
    # not env step-calls.
    save_freq = n_timesteps // (args.num_snapshots * n_envs)

    checkpoint_callback = CheckpointCallback(save_freq, checkpoint_dir)

    agent.learn(total_timesteps=n_timesteps, callback=checkpoint_callback)

    vecEnv.close()


if __name__ == "__main__":
    args = parser.parse_args()
    run_experiment(args)
