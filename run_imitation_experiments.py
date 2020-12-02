# Run hardcoded imitation
# experiments on BipedalWalker-v3.
# Code is based on examples in the imitation library,
# e.g.
# https://github.com/HumanCompatibleAI/imitation/blob/master/src/imitation/scripts/expert_demos.py
#
import os
from glob import glob

# Make sure we do not use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import gym
import torch
from tqdm import tqdm

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from imitation.data import rollout, types, wrappers
from imitation.algorithms.bc import BC
from imitation.util import logger

from gmm_tools import train_ubm, save_ubm, load_ubm, trajectories_to_supervector, adapted_gmm_distance
from wrappers import StateWrapper
from agents.stable_baselines_agent import create_stable_baselines3_agent
from collect_trajectories import collect_trajectories

# Speeds things up often.
torch.set_num_threads(1)

# Generic parameters and paths for the experiments
NUM_ENVS = 8
ENVS = [
    "LunarLander-v2",
    "BipedalWalkerHardcore-v3"
]
AGENT_FILE = "trained_agent.zip"
ROLLOUTS_FILE = "rollouts.pkl"
BC_MODEL_DIRECTORY = "bc_models"
BC_TRAJECTORY_DIRECTORY = "bc_trajectories"
BC_LOG_DIRECTORY = "bc_log"
BC_PIVECTOR_DIRECTORY = "bc_pivecs"

FINAL_MODEL_TRAJECTORIES = "final_model_trajectories.npz"
FINAL_MODEL_PIVECTOR = "final_model_pivector.npz"
FINAL_MODEL_NUM_TRAJS = 100

UBM_PATH = "ubms/imitation_{}_ubm.npz"
MAX_UBM_DATA = int(1e7)
NUM_COMPONENTS = 64

# Parameters for collecting data
MAX_ROLLOUT_TIMESTEPS = None
MAX_ROLLOUT_EPISODES = 10

# Parameters for the BC
BC_TRAIN_EPOCHS = 50


def create_env(env):
    env = gym.make(env)
    env = wrappers.RolloutInfoWrapper(env)
    return env


def get_experiment_paths(env):
    """Get paths to experiment directories used for imitation (SB3-PPO runs)"""
    experiment_paths = glob("experiments/stablebaselines_{}_SB3-PPO*".format(env))
    return experiment_paths


def train_bc(env, experiment_path):
    """
    Train GAIL on rollouts in the experiment path, save checkpoints
    and evaluate those checkpoints

    Based on code here
    https://github.com/HumanCompatibleAI/imitation/blob/master/src/imitation/scripts/train_adversarial.py
    """
    rollout_file = os.path.join(experiment_path, ROLLOUTS_FILE)
    bc_model_directory = os.path.join(experiment_path, BC_MODEL_DIRECTORY)
    bc_log_directory = os.path.join(experiment_path, BC_LOG_DIRECTORY)
    if os.path.isdir(bc_log_directory):
        print("Skipping BC training (log directory exists)")
        return
    os.makedirs(bc_model_directory, exist_ok=True)
    os.makedirs(bc_log_directory, exist_ok=True)
    logger.configure(bc_log_directory)

    expert_trajs = types.load(rollout_file)
    expert_transitions = rollout.flatten_trajectories(expert_trajs)

    env = gym.make(env)

    trainer = BC(
        env.observation_space,
        env.action_space,
        expert_data=expert_transitions,
        policy_class=MlpPolicy,
        device="cpu",
        ent_weight=0.0
    )

    env.close()

    def callback(locals):
        path = os.path.join(bc_model_directory, "epoch_{}".format(locals["epoch_num"]))
        trainer.save_policy(path)

    trainer.save_policy(os.path.join(experiment_path, "start_bc"))
    trainer.train(BC_TRAIN_EPOCHS, on_epoch_end=callback)

    # Save trained policy
    trainer.save_policy(os.path.join(experiment_path, "final_bc"))


def collect_rollouts(env, experiment_path):
    """
    Collect rollouts for given experiment.

    Based on the code here
    https://github.com/HumanCompatibleAI/imitation/blob/master/src/imitation/scripts/expert_demos.py
    """
    rollout_file = os.path.join(experiment_path, ROLLOUTS_FILE)
    if os.path.isfile(rollout_file):
        return

    sample_until = rollout.make_sample_until(
        MAX_ROLLOUT_TIMESTEPS, MAX_ROLLOUT_EPISODES
    )

    venv = DummyVecEnv([lambda: create_env(env) for i in range(NUM_ENVS)])

    agent_path = os.path.join(experiment_path, AGENT_FILE)
    agent = PPO.load(agent_path)

    rollout.rollout_and_save(rollout_file, agent, venv, sample_until)

    venv.close()


def collect_final_model_trajectories(env, experiment_path):
    """
    Collect trajectories for our UBM training and distance measurements
    """
    trajectory_file = os.path.join(experiment_path, FINAL_MODEL_TRAJECTORIES)
    agent_file = os.path.join(experiment_path, AGENT_FILE)
    if os.path.isfile(trajectory_file):
        return

    env = gym.make(env)
    env = StateWrapper(env)

    agent = create_stable_baselines3_agent(agent_file, "SB3-PPO")

    trajectories, rewards = collect_trajectories(env, agent, FINAL_MODEL_NUM_TRAJS)

    output_dict = dict(("traj_%d" % i, trajectory) for i, trajectory in enumerate(trajectories))
    output_dict["episodic_rewards"] = np.array(rewards)
    np.savez(trajectory_file, **output_dict)


def train_ubm_and_extract_pivectors(env, experiment_paths):
    """
    Train UBM for pivector extraction
    and adapt GMMs for the given experiments
    """
    ubm_path = UBM_PATH.format(env)
    os.makedirs(os.path.dirname(ubm_path), exist_ok=True)
    # Train UBM if one does not exist
    if not os.path.isfile(ubm_path):
        # Load GAIL and BC data, and final agent data as well
        all_data = []
        for experiment_path in tqdm(experiment_paths, desc="ubm-load"):
            traj_paths = glob(os.path.join(experiment_path, BC_TRAJECTORY_DIRECTORY, "*"))
            for traj_path in traj_paths:
                data = np.load(traj_path)
                data_trajs = [data[key] for key in data.keys() if "traj" in key]
                all_data.extend(data_trajs)
        # Load the data of the final model
        traj_paths = os.path.join(experiment_path, FINAL_MODEL_TRAJECTORIES)
        data = np.load(traj_path)
        data_trajs = [data[key] for key in data.keys() if "traj" in key]
        all_data.extend(data_trajs)

        all_data = np.concatenate(all_data, axis=0)
        # Restrict amount of data
        if all_data.shape[0] > MAX_UBM_DATA:
            np.random.shuffle(all_data)
            all_data = all_data[:MAX_UBM_DATA]
        # Normalize
        means = all_data.mean(axis=0)
        stds = all_data.std(axis=0)
        all_data = (all_data - means) / stds

        ubm = train_ubm(all_data, n_components=NUM_COMPONENTS)
        save_ubm(ubm_path, ubm, means, stds)
    else:
        print("Skipping UBM training (found)")

    ubm, means, std = load_ubm(ubm_path)

    # Extract pivectors
    for experiment_path in experiment_paths:
        traj_dir = BC_TRAJECTORY_DIRECTORY
        pivec_dir = BC_PIVECTOR_DIRECTORY
        os.makedirs(os.path.join(experiment_path, pivec_dir), exist_ok=True)
        traj_paths = glob(os.path.join(experiment_path, traj_dir, "*"))
        for traj_path in traj_paths:
            pivec_path = os.path.join(experiment_path, pivec_dir, os.path.basename(traj_path))
            if os.path.isfile(pivec_path):
                continue
            data = np.load(traj_path)
            average_episodic_reward = data["episodic_rewards"].mean()
            data = [data[key] for key in data.keys() if "traj" in key]
            data = np.concatenate(data, axis=0)
            data = (data - means) / std

            pivec = trajectories_to_supervector(data, ubm)

            # Also store component weights and covariances for future reference
            np.savez(pivec_path,
                pivector=pivec,
                average_episodic_reward=average_episodic_reward,
                covariances=ubm.covariances_,
                weights=ubm.weights_
            )
        # Extract pivector for the final model as well
        pivec_path = os.path.join(experiment_path, FINAL_MODEL_PIVECTOR)
        traj_path = os.path.join(experiment_path, FINAL_MODEL_TRAJECTORIES)

        if not os.path.isfile(pivec_path):
            data = np.load(traj_path)
            average_episodic_reward = data["episodic_rewards"].mean()
            data = [data[key] for key in data.keys() if "traj" in key]
            data = np.concatenate(data, axis=0)
            data = (data - means) / std

            pivec = trajectories_to_supervector(data, ubm)

            # Also store component weights and covariances for future reference
            np.savez(pivec_path,
                pivector=pivec,
                average_episodic_reward=average_episodic_reward,
                covariances=ubm.covariances_,
                weights=ubm.weights_
            )


def compute_distances(experiment_path):
    """
    Compute distances of different IL pivectors (BC or GAIL)
    to the policy that was used to generate the data
    """
    final_model_pivec_path = os.path.join(experiment_path, FINAL_MODEL_PIVECTOR)
    final_model_pivec = np.load(final_model_pivec_path)
    # Diagonal so can do this
    covariances = final_model_pivec["covariances"]
    precisions = 1 / covariances
    weights = final_model_pivec["weights"]
    final_model_pivec_means = final_model_pivec["pivector"].reshape(covariances.shape)

    pivec_dir = BC_PIVECTOR_DIRECTORY
    pivec_paths = glob(os.path.join(experiment_path, pivec_dir, "*"))
    for pivec_path in pivec_paths:
        pivec = np.load(pivec_path)
        # Make sure covariances match
        assert np.allclose(pivec["covariances"], covariances)
        pivec_means = pivec["pivector"].reshape(covariances.shape)
        distance = adapted_gmm_distance(pivec_means, final_model_pivec_means, precisions, weights)

        # Include distance back to file
        np.savez(
            pivec_path,
            distance_to_original=distance,
            **pivec
        )


if __name__ == "__main__":
    from collect_trajectories import main as collect_trajectories_main
    from collect_trajectories import parser as collect_trajectories_parser
    for env in ENVS:
        experiment_paths = get_experiment_paths(env)
        for experiment_path in tqdm(experiment_paths, desc="rollouts"):
            collect_rollouts(env, experiment_path)
            collect_final_model_trajectories(env, experiment_path)
            train_bc(env, experiment_path)

        # Collect trajectories for trained BC policies
        # Bit of a naughty workaround the argparse stuff...
        collect_traj_args = [
            "--skip-existing",
            "--num-workers", "16",
            "--checkpoint-dir", BC_MODEL_DIRECTORY,
            "--trajectory-dir", BC_TRAJECTORY_DIRECTORY,
            "input_dirs", *experiment_paths
        ]
        args = collect_trajectories_parser.parse_args(collect_traj_args)
        collect_trajectories_main(args)

        # Train UBM and extract pivectors
        train_ubm_and_extract_pivectors(env, experiment_paths)

        # Compute distances from the policy that
        # was used to train the IL model
        for experiment_path in experiment_paths:
            compute_distances(experiment_path)
