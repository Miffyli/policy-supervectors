# Main code for turning collected trajectories
# into pivectors for visualization
#
import os
from argparse import ArgumentParser
import glob
import pickle
import random

import numpy as np
from tqdm import tqdm

from gmm_tools import train_ubm, trajectories_to_supervector, save_ubm, load_ubm

# NOTE: Relies on specific naming of experiments.
#   Each experiment is a directory with name:
#       [package]_[env]_[agent]_[timestamp]
#

TRAJECTORIES_DIR = "trajectories"
PIVECTORS_DIR = "pivectors"


def train_ubms(unparsed_args):
    parser = ArgumentParser("Train an UBM")
    parser.add_argument("--inputs", type=str, nargs="+", required=True, help="Paths to experiments for which UBMs should be created.")
    parser.add_argument("output", type=str, help="Directory where to store UBMs, one per environment.")
    parser.add_argument("--n-components", type=int, default=64, help="Number of components the GMM.")
    parser.add_argument("--max-points", type=int, default=int(1e7), help="Maximum number of datapoints to use for UBM training.")
    parser.add_argument("--max-checkpoints", type=int, default=500, help="Maximum number of checkpoints to use per algorithm.")
    parser.add_argument("--max-trajectories", type=int, default=100, help="Maximum number of trajectories per checkpoint.")
    parser.add_argument("--skip-existing", action="store_true", help="Check if UBM file exists already and skip training a new one if so.")
    args = parser.parse_args(unparsed_args)

    # Check what different we have in the inputs,
    # collect per-env trajectories, train one ubm
    # per env and store them
    envs = [os.path.basename(path).split("_")[1] for path in args.inputs]
    unique_envs = list(set(envs))

    for env in unique_envs:
        output_path = os.path.join(args.output, "{}_ubm.npz".format(env))
        if args.skip_existing and os.path.isfile(output_path):
            print("Skipping training for {} (UBM file exists)".format(env))
            continue

        env_experiment_paths = [path for path in args.inputs if env in path]
        # Load the data
        env_data = []
        print("Loading data for env {}...".format(env))
        # Store which trajectories were used
        # to train this UBM for the UBM experiments
        trajectory_indeces = []
        for path in tqdm(env_experiment_paths, leave=False):
            trajectory_paths = glob.glob(os.path.join(path, TRAJECTORIES_DIR, "*"))
            trajectory_paths = sorted(trajectory_paths)
            # Sample maximum number of trajectory-files if it exceeds
            # given maximum
            if len(trajectory_paths) > args.max_checkpoints:
                trajectory_paths = random.sample(trajectory_paths, args.max_checkpoints)
            for trajectory_path in tqdm(trajectory_paths, leave=False):
                data = np.load(trajectory_path)
                keys = sorted(list(data.keys()))
                datas = [data[key] for key in keys if "traj" in key]
                if len(datas) > args.max_trajectories:
                    random_indeces = random.sample(range(len(datas)), args.max_trajectories)
                    datas = [datas[i] for i in random_indeces]
                    trajectory_indeces.append(random_indeces)
                else:
                    trajectory_indeces.append(np.arange(len(datas)))
                env_data.extend(datas)
        trajectory_indeces = np.array(trajectory_indeces)
        env_data = np.concatenate(env_data, axis=0)
        print("Loaded a datamatrix of shape {}".format(env_data.shape))
        print("Sampling {} random samples from data...".format(args.max_points))

        indeces = np.arange(env_data.shape[0])
        np.random.shuffle(indeces)
        env_data = env_data[indeces[:args.max_points]]

        print("Training UBM for env {}...".format(env))

        means = env_data.mean(axis=0)
        stds = env_data.std(axis=0)

        env_data = (env_data - means) / stds

        ubm = train_ubm(env_data, n_components=args.n_components)

        save_ubm(output_path, ubm, means, stds, trajectory_indeces)


def extract_pivectors(unparsed_args):
    parser = ArgumentParser("Extract pivectors for given experiments")
    parser.add_argument("--inputs", type=str, nargs="+", required=True, help="Paths to experiments for which pivectors should be extracted.")
    parser.add_argument("ubms", type=str, help="Directory where UBM models reside, one per environment.")
    args = parser.parse_args(unparsed_args)

    for experiment_path in tqdm(args.inputs):
        env = experiment_path.split("_")[1]
        os.makedirs(os.path.join(experiment_path, PIVECTORS_DIR), exist_ok=True)
        ubm, means, stds = load_ubm(os.path.join(args.ubms, "{}_ubm.npz".format(env)))

        trajectory_paths = glob.glob(os.path.join(experiment_path, TRAJECTORIES_DIR, "*"))
        for trajectory_path in tqdm(trajectory_paths, leave=False):
            trajectory_name = os.path.basename(trajectory_path)
            data = np.load(trajectory_path)
            average_episodic_reward = data["episodic_rewards"].mean()
            states = np.concatenate([data[key] for key in data.keys() if "traj" in key])

            # Normalize
            states = (states - means) / stds
            pivector = trajectories_to_supervector(states, ubm)
            new_path = os.path.join(experiment_path, PIVECTORS_DIR, trajectory_name)

            # Also store component weights and covariances for future reference
            np.savez(new_path,
                pivector=pivector,
                average_episodic_reward=average_episodic_reward,
                covariances=ubm.covariances_,
                weights=ubm.weights_
            )

        del ubm


OPERATIONS = {
    "train_ubms": train_ubms,
    "pivectors": extract_pivectors,
}

if __name__ == "__main__":
    parser = ArgumentParser("Train UBMs, create pivectors and tSNE them from collected trajectories")
    parser.add_argument("operation", type=str, choices=list(OPERATIONS.keys()), help="Operation to run")
    args, unparsed_args = parser.parse_known_args()

    function = OPERATIONS[args.operation]
    function(unparsed_args)
