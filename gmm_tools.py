#!/usr/bin/env python3
#
# gmm_tools.py
#
# Main tools for training GMMs and adapting
# from them with new data.
#

import numpy as np
from sklearn.mixture import GaussianMixture

# Structure of the trajectory data:
#   np.ndarray of (N, D), where
#     N = number of states collected and
#     D = dimensionality of single observation
#

# Uses universal background model and supervector concepts from
# [1] http://cs.joensuu.fi/pages/tkinnu/webpage/pdf/speaker_recognition_overview.pdf
# For default relevance-factor (16), we took a look at the original UBM-GMM paper:
# [2] http://speech.ee.ntu.edu.tw/previous_version/Speaker%20Verification%20Using%20Adapted%20Gaussain%20Mixture%20Models.pdf
# For distances between MAP-adapted GMMs:
# [3] http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.437.6872&rep=rep1&type=pdf

def train_ubm(data, n_components=64, n_init=1, verbose=2):
    """
    Train a GMM on the data to form an Universal Background Model,
    that will be later used to adapt per-policy means.

    Note: Hardcoded to use diagonal covariance matrices
        as otherwise computing will take too long.

    Parameters:
        data (np.ndarray): Array of shape (N, D), containing data
            from various policies to be used to create a model
            of a "general policy".
        n_components (int): Number of components in the UBM
        n_init (int): Fed to GaussianMixture
        verbose (int): Fed to GaussianMixture
    Returns:
        ubm (sklearn.mixture.GaussianMixture): Trained GMM model
    """
    ubm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        verbose=verbose,
        n_init=n_init
    )
    ubm.fit(data)
    return ubm


def save_ubm(path, ubm, means, stds, trajectory_indeces=np.nan, **additional_items):
    """
    Save sklearn UBM GMM into a numpy arrays for
    easier transfer between sklearn versions etc.

    Parameters:
        path (str): Where to store the UBM
        ubm (sklearn.mixture.GaussianMixture): Trained GMM model
        means, stds (ndarray): Means and stds of variables to
            be stored along UBM for normalization purposes
        trajectory_indeces (ndarray): (num_policies, num_trajs)
            array, that tells which trajectories were used to train
            this UBM. Used when trajectories are sampled.
        **additional_items: Additional items that will be added to the numpy
            archive.
    """
    np.savez(
        path,
        ubm_means=ubm.means_,
        ubm_weights=ubm.weights_,
        # Probably no need to store all of these, but oh well
        ubm_covariances=ubm.covariances_,
        ubm_precisions=ubm.precisions_,
        ubm_precisions_cholesky=ubm.precisions_cholesky_,
        means=means,
        stds=stds,
        trajectory_indeces=trajectory_indeces,
        **additional_items
    )


def load_ubm(path):
    """
    Load UBM stored with save_ubm, returning
    GMM object and normalization vectors

    Parameters:
        path (str): Where to load UBM from
    Returns:
        ubm (sklearn.mixture.GaussianMixture): Trained GMM model
        means, stds (ndarray): Means and stds of variables to
            be stored along UBM for normalization purposes
    """
    data = np.load(path)
    n_components = data["ubm_means"].shape[0]
    cov_type = "diag" if data["ubm_covariances"].ndim == 2 else "full"
    ubm = GaussianMixture(n_components=n_components, covariance_type=cov_type)
    ubm.means_ = data["ubm_means"]
    ubm.weights_ = data["ubm_weights"]
    ubm.covariances_ = data["ubm_covariances"]
    ubm.precisions_ = data["ubm_precisions"]
    ubm.precisions_cholesky_ = data["ubm_precisions_cholesky"]
    means = data["means"]
    stds = data["stds"]
    return ubm, means, stds


def trajectories_to_supervector(states, ubm, relevance_factor=16):
    """
    Take a trained UBM and states visited by a policy and create
    a fixed-length supervector to represent that policy
    based on the data.

    Current implementation MAP-adapts UBM means to data and
    then concatenates all these means

    Parameters:
        states (np.ndarray): (N, D) array
        ubm (sklearn.mixture.GaussianMixture): Trained GMM model
        relevance_factor (int): Relevance factor from [2]
    Returns:
        np.ndarray of shape (M,): 1D and fixed-length vector
            representing policy that created the data
    """
    # Using the notation in [1]
    # Score each data point to all components,
    # get (N, K) matrix (K = number of components)
    state_probs = ubm.predict_proba(states)

    # n, or "how close each point is each component"
    # (K, )
    state_prob_sums = np.sum(state_probs, axis=0)

    # \alpha, or weight of how much means should be moved, per component
    # (K, )
    alpha = state_prob_sums / (state_prob_sums + relevance_factor)

    # \tilde x, or the new means based on state
    # they are like expectations, except weighted
    # by how probable it is they came from that centroid
    # (K, D)
    tilde_x = np.zeros_like(ubm.means_)
    # Do each component individually to make this bit easier
    # to read and save on memory
    for k in range(ubm.n_components):
        tilde_x[k] = np.sum(states * state_probs[:, k, None], axis=0) / (state_prob_sums[k] + 1e-6)

    # MAP-adapt means
    # (K, D)
    adapted_means = alpha[..., None] * tilde_x + (1 - alpha[..., None]) * ubm.means_

    # Create pi-vector (supervector) of means
    # (K * D, )
    pi_vector = adapted_means.ravel()

    return pi_vector


def adapted_gmm_distance(means1, means2, precisions, weights):
    """
    Calculate upper-bound of KL-divergence of two MAP-adapted
    GMMs, as in [3] equation (6).

    Parameters:
        means1 (ndarray): Array of (K, D) of adapted means
        means2 (ndarray): Array of (K, D) of adapted means
        precisions (ndarray): Array of (K, D), an inverse of a
            diagonal covariance matrix (1/Sigma)
        weights (ndarray): Array of (K,), weights of the
            components
    Returns
        distance (float): Upper-bound of KL-divergence for the
            two GMMs specified by the two mean-matrices
    """

    mean_diff = means1 - means2

    # We can get rid of the matrix operations
    # since precisions are diagonal
    dist = 0.5 * np.sum(weights * np.sum(mean_diff * mean_diff * precisions, axis=1))

    return dist
