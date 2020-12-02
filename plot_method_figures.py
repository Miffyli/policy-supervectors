# Hardcoded plotting of the plots for the
# method illustration
#
import os

import numpy as np
import gym
from matplotlib import pyplot
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from gmm_tools import train_ubm, adapted_gmm_distance, trajectories_to_supervector
from collect_trajectories import collect_trajectories
from wrappers import StateWrapper
from agents import SimpleAgentClass

NUM_TRAJECTORIES = 100
NUM_LINSPACE = 1000
N_COMPONENTS = 3
FIG_SIZE = [4.8 / 1.5, 4.8 / 1.5]
MARKERSIZE = 16
SAVEFIG_PARAMS = dict(dpi=200, bbox_inches="tight")

# Network weights and biases
# for a trained agent for Pendulum,
# trained using PPO from stable-baselines
NETWORK_W1 = np.array([[
-0.8612228035926819,
-0.0443018414080143,
-0.9440205097198486,
-0.8180672526359558,
1.033836007118225,
-0.7949904799461365,
-0.8762893676757812,
0.20649996399879456],
[1.204399824142456,
-1.4673517942428589,
0.8465980291366577,
-0.8905948400497437,
-0.5191181302070618,
-1.2671408653259277,
1.6971912384033203,
1.526084065437317],
[0.43922996520996094,
0.38299453258514404,
0.3762228488922119,
0.44744065403938293,
0.4663331210613251,
-0.35295864939689636,
0.5651375651359558,
-0.39918193221092224]])

NETWORK_B1 = np.array([0.7241255640983582,
 -0.031115585938096046,
 0.8128896355628967,
 -0.6257883906364441,
 0.5746131539344788,
 0.6976305246353149,
 0.7140235304832458,
 0.13175401091575623])

NETWORK_W2 = np.array([[-0.7947609424591064],
 [0.7136436104774475],
 [-0.5662018060684204],
 [0.6732066869735718],
 [0.5143256783485413],
 [0.42300042510032654],
 [-0.7273273468017578],
 [-0.7285447120666504]])

NETWORK_B2 = np.array([0.2733193635940552])


def confidence_ellipse(mean_x, mean_y, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Stolen from generous matplotlib docs
        https://matplotlib.org/3.1.1/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py
    """

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def pyplot_remove_margins():
    """
    Code stolen from Stackoverflow #11837979 to
    get rid of _all_ margins in Pyplot savefig
    """
    pyplot.gca().set_axis_off()
    pyplot.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    pyplot.margins(0,0)
    pyplot.gca().xaxis.set_major_locator(pyplot.NullLocator())
    pyplot.gca().yaxis.set_major_locator(pyplot.NullLocator())


def main():
    # Illustrations for three different agents:
    # - Random
    # - a trained neural network
    # - Always high
    env = gym.make("Pendulum-v0")
    env = StateWrapper(env)

    # Always pick random action
    random_agent = SimpleAgentClass(lambda obs: env.action_space.sample())
    # Always pick high
    always_high_agent = SimpleAgentClass(lambda obs: env.action_space.high)
    # Trained stable-baselines agent
    def network_activation(obs):
        x = np.tanh((obs @ NETWORK_W1) + NETWORK_B1)
        x = (x @ NETWORK_W2) + NETWORK_B2
        return x
    network_agent = SimpleAgentClass(network_activation)

    # Gather observations
    print("Collecting random trajectories...")
    random_trajectories, random_rewards = collect_trajectories(env, random_agent, NUM_TRAJECTORIES)
    print("Average reward: {}".format(np.mean(random_rewards)))
    print("Collecting always-high trajectories...")
    always_high_trajectories, always_high_rewards = collect_trajectories(env, always_high_agent, NUM_TRAJECTORIES)
    print("Average reward: {}".format(np.mean(always_high_rewards)))
    print("Collecting network trajectories...")
    network_trajectories, network_rewards = collect_trajectories(env, network_agent, NUM_TRAJECTORIES)
    print("Average reward: {}".format(np.mean(network_rewards)))

    random_trajectories = np.concatenate(random_trajectories, axis=0)
    always_high_trajectories = np.concatenate(always_high_trajectories, axis=0)
    network_trajectories = np.concatenate(network_trajectories, axis=0)

    # Take theta and theta-velocity as the variables we want to study (for 2D plots)
    random_trajectories = np.stack((np.arccos(random_trajectories[:, 0]), random_trajectories[:, 2]), axis=1)
    always_high_trajectories = np.stack((np.arccos(always_high_trajectories[:, 0]), always_high_trajectories[:, 2]), axis=1)
    network_trajectories = np.stack((np.arccos(network_trajectories[:, 0]), network_trajectories[:, 2]), axis=1)

    all_data = np.concatenate((random_trajectories, always_high_trajectories, network_trajectories), axis=0)

    # Train the GMM-UBM.
    # Using multiple inits here for a similar results on different runs
    ubm = train_ubm(all_data, n_components=N_COMPONENTS, n_init=5)

    # Extract policy supervectors (or adapted means)
    random_means = trajectories_to_supervector(random_trajectories, ubm).reshape(N_COMPONENTS, 2)
    always_high_means = trajectories_to_supervector(always_high_trajectories, ubm).reshape(N_COMPONENTS, 2)
    network_means = trajectories_to_supervector(network_trajectories, ubm).reshape(N_COMPONENTS, 2)

    # Compute stuff for contours
    mins, maxs = all_data.min(axis=0), all_data.max(axis=0)
    theta_space = np.linspace(mins[0], maxs[0], num=NUM_LINSPACE)
    thetavel_space = np.linspace(mins[1], maxs[1], num=NUM_LINSPACE)
    locations = np.array(np.meshgrid(theta_space, thetavel_space)).T.reshape(-1, 2)

    os.makedirs("figures", exist_ok=True)
    fig = pyplot.figure(figsize=FIG_SIZE)
    pyplot.axis("off")
    pyplot.scatter(all_data[:, 0], all_data[:, 1], alpha=0.003, s=MARKERSIZE)
    pyplot_remove_margins()
    pyplot.savefig("figures/method_all_data.png", **SAVEFIG_PARAMS)
    # Plot the components
    ax = pyplot.gca()
    for i in range(N_COMPONENTS):
        cov = np.diag(ubm.covariances_[i])
        mean_x = ubm.means_[i, 0]
        mean_y = ubm.means_[i, 1]
        _ = confidence_ellipse(mean_x, mean_y, cov, ax, n_std=1, edgecolor="red", linewidth=2)
        pyplot.scatter(mean_x, mean_y, marker="+", c="red")
    pyplot_remove_margins()
    pyplot.savefig("figures/method_ubm_all_data.png", **SAVEFIG_PARAMS)
    xlim = pyplot.xlim()
    ylim = pyplot.ylim()
    pyplot.close(fig)

    # Repeat above for all different datas
    for i in range(3):
        name = None
        ubm_name = None
        means = None
        # Plot data
        fig = pyplot.figure(figsize=FIG_SIZE)
        if i == 0:
            name = "figures/method_random_data.png"
            ubm_name = "figures/method_ubm_random_data.png"
            means = random_means
            pyplot.scatter(random_trajectories[:, 0], random_trajectories[:, 1], alpha=0.003, s=MARKERSIZE)
        elif i == 1:
            name = "figures/method_always_high_data.png"
            ubm_name = "figures/method_ubm_always_high_data.png"
            means = always_high_means
            pyplot.scatter(always_high_trajectories[:, 0], always_high_trajectories[:, 1], alpha=0.003, s=MARKERSIZE)
        else:
            name = "figures/method_network_data.png"
            ubm_name = "figures/method_ubm_network_data.png"
            means = network_means
            pyplot.scatter(network_trajectories[:, 0], network_trajectories[:, 1], alpha=0.003, s=MARKERSIZE)
        pyplot.xlim(xlim)
        pyplot.ylim(ylim)
        pyplot.axis("off")
        pyplot_remove_margins()
        pyplot.savefig(name, **SAVEFIG_PARAMS)

        # Plot adapted GMM
        # Plot the old components and new compontnes
        ax = pyplot.gca()
        for i in range(N_COMPONENTS):
            cov = np.diag(ubm.covariances_[i])
            mean_x = means[i, 0]
            mean_y = means[i, 1]
            _ = confidence_ellipse(mean_x, mean_y, cov, ax, n_std=1, edgecolor="red", linewidth=2)
            pyplot.scatter(mean_x, mean_y, marker="+", c="red")
            # Old component
            ubm_mean_x = ubm.means_[i, 0]
            ubm_mean_y = ubm.means_[i, 1]
            _ = confidence_ellipse(ubm_mean_x, ubm_mean_y, cov, ax, n_std=1, edgecolor="red", linewidth=2, alpha=0.3, linestyle="--")
            pyplot.scatter(ubm_mean_x, ubm_mean_y, marker="+", c="red", alpha=0.3)
            pyplot.arrow(ubm_mean_x, ubm_mean_y, mean_x - ubm_mean_x, mean_y - ubm_mean_y, color="red", width=0.01, linewidth=0.25)
        pyplot.xlim(xlim)
        pyplot.ylim(ylim)
        pyplot_remove_margins()
        pyplot.savefig(ubm_name, **SAVEFIG_PARAMS)
        pyplot.close(fig)



if __name__ == "__main__":
    main()
