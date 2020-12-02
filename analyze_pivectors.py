# Main code for analyzing/processing
# extracted pivectors
#
import os
from argparse import ArgumentParser
import glob
import re

import numpy as np
from tqdm import tqdm

from gmm_tools import adapted_gmm_distance

# NOTE: Relies on specific naming of experiments.
#   Each experiment is a directory with name:
#       [package]_[env]_[agent]_[timestamp]
#

PIVECTORS_DIR = "pivectors"

CHECKPOINT_DISTANCES_FILE = "checkpoint_distances.npz"


def run_tsne(unparsed_args):
    parser = ArgumentParser("Compute tSNEs for pivectors and store them in same files")
    parser.add_argument("--inputs", type=str, nargs="+", required=True, help="Paths to experiments for which tSNEs should be computed.")
    args = parser.parse_args(unparsed_args)

    from sklearn.manifold import TSNE

    # Check what different we have in the inputs,
    # collect per-env trajectories, train one ubm
    # per env and store them
    envs = [os.path.basename(path).split("_")[1] for path in args.inputs]
    unique_envs = list(set(envs))

    for env in unique_envs:
        env_experiment_paths = [path for path in args.inputs if env in path]
        env_pivectors = []
        env_covariances = None
        env_weights = None
        # Keep track of paths so we know where to store the results
        env_pivector_paths = []
        print("Loading data for env {}...".format(env))
        for path in tqdm(env_experiment_paths, leave=False):
            pivector_paths = glob.glob(os.path.join(path, PIVECTORS_DIR, "*"))
            for pivector_path in pivector_paths:
                data = np.load(pivector_path)
                env_pivectors.append(data["pivector"])
                covariances = data["covariances"]
                weights = data["weights"]
                if env_covariances is None:
                    env_covariances = covariances
                    env_weights = weights
                # Sanity check to make sure all adapted GMMs
                # share same covariance
                if not np.allclose(env_covariances, covariances):
                    raise ValueError("Covariances did not match file {}".format(pivector_path))

                env_pivector_paths.append(pivector_path)

        env_precisions = 1 / env_covariances
        # Diagonal covariances and means have
        # the same shape
        mean_shape = env_covariances.shape

        def distance_metric(pivector1, pivector2):
            means1 = pivector1.reshape(mean_shape)
            means2 = pivector2.reshape(mean_shape)
            return adapted_gmm_distance(means1, means2, env_precisions, env_weights)

        env_pivectors = np.stack(env_pivectors)
        print("Running tSNE on data of shape {}...".format(env_pivectors.shape))

        tsne = TSNE(metric=distance_metric)
        pi_points = tsne.fit_transform(env_pivectors)

        # Store results along with pivectors
        for pi_point, pivector_path in zip(pi_points, env_pivector_paths):
            # Super effecient loading of files twice...
            original_data = np.load(pivector_path)
            if "tsne" in original_data.keys():
                original_data = dict(**original_data)
                _ = original_data.pop("tsne")
            np.savez(pivector_path, tsne=pi_point, **original_data)


def compute_checkpoint_distances(unparsed_args):
    parser = ArgumentParser("Compute distances between consecutive checkpoints and store them under experiment dir")
    parser.add_argument("--inputs", type=str, nargs="+", required=True, help="Paths to experiments to process.")
    args = parser.parse_args(unparsed_args)

    for path in tqdm(args.inputs, desc="experiment", leave=False):
        pivector_paths = glob.glob(os.path.join(path, PIVECTORS_DIR, "*"))

        # Sort them by the number of steps trained
        steps_trained = [int(re.findall("rl_model_([0-9]+)_steps", pivector_name)[0]) for pivector_name in pivector_paths]
        pivector_paths = list(zip(*sorted(zip(pivector_paths, steps_trained), key=lambda x: x[1])))[0]

        means = []
        rewards = []
        shared_covariances = None
        shared_weights = None
        for pivector_path in pivector_paths:
            data = np.load(pivector_path)
            covariances = data["covariances"]
            weights = data["weights"]
            reward = data["average_episodic_reward"]
            # Use same covariance and weight
            # for all samples later
            if shared_covariances is None:
                shared_covariances = covariances
                shared_weights = weights
            # Sanity check to make sure all adapted GMMs
            # share same covariance
            if not np.allclose(shared_covariances, covariances):
                raise ValueError("Covariances did not match file {}".format(pivector_path))

            # Covariances are diagonal so they share same
            # shape with means.
            # Pivector is just raveled means.
            mean = data["pivector"].reshape(covariances.shape)
            means.append(mean)
            rewards.append(reward)

        precisions = 1 / shared_covariances

        distances = []
        aligned_rewards = []
        for i in range(len(means) - 1):
            distances.append(adapted_gmm_distance(
                means[i],
                means[i + 1],
                precisions,
                shared_weights
            ))
            # Align to the previous policy
            aligned_rewards.append(rewards[i])

        distances = np.array(distances)
        aligned_rewards = np.array(aligned_rewards)

        # Store back on disk
        save_path = os.path.join(path, CHECKPOINT_DISTANCES_FILE)
        np.savez(save_path, distances=distances, average_episodic_rewards=aligned_rewards)


OPERATIONS = {
    "tsne": run_tsne,
    "checkpoint-distances": compute_checkpoint_distances,
}

if __name__ == "__main__":
    parser = ArgumentParser("Train UBMs, create pivectors and tSNE them from collected trajectories")
    parser.add_argument("operation", type=str, choices=list(OPERATIONS.keys()), help="Operation to run")
    args, unparsed_args = parser.parse_known_args()

    function = OPERATIONS[args.operation]
    function(unparsed_args)
