# Hardcoded plotting for appendix,
# which is basically same as in main paper
# but over all environments etc.
#
import os
from glob import glob
import itertools
import re

import numpy as np
import matplotlib
from matplotlib import pyplot

from plot_paper import interpolate_and_average, color_linestyle_cycle

# Stackoverflow #4931376
matplotlib.use('Agg')

PIVECTORS_DIR = "pivectors"
CHECKPOINT_DISTANCES_FILE = "checkpoint_distances.npz"


def plot_ppo_clip_results():
    """
    Plot the PPO clip results: Draw selected
    learning curves and numbers of average distance traveled.
    """

    # Different clip ratios/experiments to plot learning curves for
    PLOT_AGENTS = [
        "PPO-clip0.01",
        "PPO-clip0.1",
        "PPO-clip10.0",
        "A2C-clip"
    ]

    REWARD_SCALES = {
        "Pendulum-v0": [-1600, -200],
        "Acrobot-v1": [-500, -100],
        "LunarLander-v2": [-230, 200],
        "BipedalWalker-v3": [-100, 300],
        "CartPole-v1": [0, 500]
    }

    figure, axs = pyplot.subplots(
        figsize=[6.4 * 2, 4.8 * 0.55],
        nrows=1,
        ncols=5,
        sharex="col"
    )

    distance_axs = axs
    reward_axs = [ax.twinx() for ax in distance_axs]

    # Clip experiments
    experiment_paths = glob("experiments/stablebaselines_*-clip*")
    # Get unique env names
    envs = [os.path.basename(path).split("_")[1] for path in experiment_paths]
    unique_envs = list(sorted(set(envs)))
    # Get unique algorithms
    algos = [os.path.basename(path).split("_")[2] for path in experiment_paths]
    unique_algos = sorted(list(set(algos)))

    print("--- PPO clip distance results ---")
    print("Algo " + " ".join(unique_envs) + " ".join(unique_envs))

    # Gather Pearson correlations of
    # distance traveled and average episodic rewards
    # for averaging over

    plot_legend = []
    plot_lines = []
    for algo_i, algo in enumerate(unique_algos):
        plot_color = "C{}".format(PLOT_AGENTS.index(algo)) if algo in PLOT_AGENTS else None
        # Should be strings to be printed
        env_results = []
        env_correlation_results = []
        for env_i, env in enumerate(unique_envs):
            distance_ax = distance_axs[env_i]
            reward_ax = reward_axs[env_i]

            env_experiment_paths = [path for path in experiment_paths if (env in path and algo in path)]
            distances = []
            rewards = []
            for path in env_experiment_paths:
                assert "stablebaselines" in path, "Experiments should be stable-baselines one"
                checkpoint_distances = os.path.join(path, CHECKPOINT_DISTANCES_FILE)
                data = np.load(checkpoint_distances)
                experiment_distances = data["distances"]
                experiment_rewards = data["average_episodic_rewards"]
                distances.append(experiment_distances)
                rewards.append(experiment_rewards)

            # Some training sessions may miss a checkpoint or two, so take minimum
            min_checkpoints = min(len(x) for x in distances)
            distances = [x[:min_checkpoints] for x in distances]
            rewards = [x[:min_checkpoints] for x in rewards]
            distances = np.array(distances)
            rewards = np.array(rewards)
            reward_distance_correlations = [np.corrcoef(rewards[i], distances[i])[0, 1] for i in range(len(rewards))]
            cumulative_distance = np.sum(distances, axis=1)
            distances_mean = np.mean(distances, axis=0)
            distances_std = np.std(distances, axis=0)
            rewards_mean = np.mean(rewards, axis=0)
            rewards_std = np.std(rewards, axis=0)
            cumulative_distance_mean = np.mean(cumulative_distance, axis=0)
            cumulative_distance_std = np.std(cumulative_distance, axis=0)

            # Plot to the thing
            if algo in PLOT_AGENTS:
                plot_line, = distance_ax.plot(distances_mean, linestyle="-", c=plot_color)
                # Normalize rewards
                reward_min, reward_max = REWARD_SCALES[env]
                plot_rewards = (rewards_mean - reward_min) / (reward_max - reward_min)
                reward_ax.plot(plot_rewards, linestyle="--", c=plot_color)
                reward_ax.set_ylim(-0.1, 1.1)
                distance_ax.set_title(env.split("-")[0])
                if unique_envs[-1] == env:
                    plot_lines.append(plot_line)

            # Add cumulative distance results
            env_results.append("{:4.1f}±{:<4.1f}".format(cumulative_distance_mean.item(), cumulative_distance_std.item()))
            env_correlation_results.append("{:.2f}±{:<.2f}".format(np.mean(reward_distance_correlations), np.std(reward_distance_correlations)))
        prettier_algo = algo.replace("-clip", "")
        if algo in PLOT_AGENTS:
            if "A2C" in algo:
                plot_legend.append(prettier_algo)
            else:
                # Prettier handling of PPOs
                epsilon = algo.split("-clip")[-1]
                plot_legend.append(r"PPO ($\epsilon={})$".format(epsilon))
        print("{} {} {}".format(prettier_algo, " ".join(env_results), " ".join(env_correlation_results)))
    print("---------------------------------")

    # Add some ghost lines for our plots
    ghost_line, = distance_ax.plot(plot_rewards, linestyle="-", c="k")
    plot_lines.append(ghost_line)
    plot_legend.append("Distance")
    ghost_line2, = distance_ax.plot(plot_rewards, linestyle="--", c="k")
    plot_lines.append(ghost_line2)
    plot_legend.append("Reward")

    figure.legend(plot_lines, plot_legend, loc="lower right", ncol=6)
    ghost_line.remove()
    ghost_line2.remove()
    distance_axs[0].set_xlabel("Policy version")
    distance_axs[0].set_ylabel("Distance")
    for i in range(4):
        distance_axs[-(i + 1)].get_xaxis().set_ticklabels([])
        reward_axs[i].get_yaxis().set_ticks([])
    reward_axs[-1].set_ylabel("Normalized reward")
    pyplot.tight_layout()
    pyplot.savefig("figures/distances_appendix.pdf", bbox_inches="tight", pad_inches=0.0)


def plot_tsnes():
    """
    Plot the t-SNE of different policy evaluations
    """
    # Two environments (for main paper figure. All for final figure)
    ENVS = [
        "BipedalWalker-v3",
        "LunarLander-v2",
        "Acrobot-v1",
        "CartPole-v1",
        "Pendulum-v0"
    ]
    ALGO_TYPES = [
        "stablebaselines",
        "stablebaselines",
        "wann",
        "wann",
    ]
    ALGO_NAMES = [
        "A2C",
        "PPO",
        "NEAT",
        "CMAES",
    ]
    ALGO_PRETTY_NAMES = [
        "A2C",
        "PPO",
        "NEAT",
        "CMA-ES"
    ]

    REWARD_SCALES = {
        "Pendulum-v0": [-1600, -200],
        "Acrobot-v1": [-500, -100],
        "LunarLander-v2": [-230, 200],
        "BipedalWalker-v3": [-100, 300],
        "CartPole-v1": [0, 500]
    }

    figure, axs = pyplot.subplots(
        figsize=[6.4 * 2, 4.8 * (5 / 2)],
        nrows=5,
        ncols=4,
        gridspec_kw={'hspace': 0.2, 'wspace': 0},
    )

    for plot_i in range(5):
        env = ENVS[plot_i]
        reward_scale = REWARD_SCALES[env]
        for algo_i in range(len(ALGO_TYPES)):
            column_idx = algo_i
            row_idx = plot_i
            ax = axs[row_idx, column_idx]
            algo_type = ALGO_TYPES[algo_i]
            algo_name = ALGO_NAMES[algo_i]
            algo_pretty_name = ALGO_PRETTY_NAMES[algo_i]

            experiment_glob = "experiments/{}_{}_{}_*".format(algo_type, env, algo_name)
            experiment_paths = glob(experiment_glob)
            tsnes = []
            rewards = []
            for experiment_path in experiment_paths:
                pivector_paths = glob(os.path.join(experiment_path, "pivectors", "*"))
                population_tsnes = []
                population_rewards = []
                for path in pivector_paths:
                    data = np.load(path)
                    population_tsnes.append(data["tsne"])
                    population_rewards.append(data["average_episodic_reward"])
                    data.close()
                tsnes.append(population_tsnes)
                rewards.append(population_rewards)
            tsnes = np.concatenate(tsnes, axis=0)
            rewards = np.concatenate(rewards, axis=0)

            # Min-max normalization
            rewards = (rewards - reward_scale[0]) / (reward_scale[1] - reward_scale[0])

            scatter = ax.scatter(
                tsnes[:, 0],
                tsnes[:, 1],
                c=rewards,
                cmap="plasma",
                s=1,
                vmin=0,
                vmax=1
            )

            ax.text(0.98, 0.98, algo_pretty_name, horizontalalignment="right", verticalalignment="top", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            # Hide spines, the outer edges
            # Hide edge spines and bolden mid-spines
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            if column_idx == 0:
                ax.spines["left"].set_visible(False)
            elif column_idx == 3:
                ax.spines["right"].set_visible(False)

            # Add titles
            if column_idx == 1:
                ax.set_title(env.split("-")[0], x=1.0)

    cbaxes = figure.add_axes([0.41, 0.90, 0.2, 0.01])
    cbar = figure.colorbar(scatter, orientation="horizontal", cax=cbaxes)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["Min", "Reward", "Max"])
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(labelsize="small", length=0)
    figure.tight_layout()
    figure.savefig("figures/tsnes_appendix.png", dpi=200, bbox_inches="tight", pad_inches=0.0)


def plot_bc_results():
    """
    Plot results with imitation where we are interested
    in how the agent improves and gets closer to the agent
    """
    ENVS = [
        "BipedalWalkerHardcore-v3",
        "LunarLander-v2"
    ]

    ENV_NAMES = [
        "BipedalWalkerHardcore",
        "LunarLander"
    ]

    MIN_EPOCH = 0
    MAX_EPOCH = 50

    PIVEC_DIR = "bc_pivecs"
    FINAL_MODEL_PIVECTOR = "final_model_pivector.npz"

    # One plot with two axes
    fig, axs = pyplot.subplots(
        figsize=[6.4, 4.8],
        nrows=2,
        ncols=1,
        sharex="all",
        gridspec_kw={'hspace': 0.0, 'wspace': 0}
    )
    distance_ax = axs[1]
    reward_ax = axs[0]

    line_objects = []
    delta_correlations = []
    for i in range(len(ENVS)):
        env = ENVS[i]
        all_distances = []
        all_rewards = []
        # Repeated over experiments
        experiment_paths = glob("experiments/stablebaselines_{}_SB3*".format(env))
        for experiment_path in experiment_paths:
            experiment_distances = []
            experiment_rewards = []
            # Load expert reward so we can scale it
            expert_reward = np.load(os.path.join(experiment_path, FINAL_MODEL_PIVECTOR))["average_episodic_reward"]
            for epoch_i in range(MIN_EPOCH, MAX_EPOCH):
                data = np.load(os.path.join(experiment_path, PIVEC_DIR, "epoch_{}.npz".format(epoch_i)))
                experiment_rewards.append(data["average_episodic_reward"] / expert_reward)
                experiment_distances.append(data["distance_to_original"])
            all_distances.append(experiment_distances)
            all_rewards.append(experiment_rewards)
            # Add delta correlations
            delta_correlations.append(np.corrcoef(np.diff(experiment_distances), np.diff(experiment_rewards))[0, 1])
        rewards = np.array(all_rewards)
        rewards_mean = np.mean(rewards, axis=0)
        rewards_std = np.std(rewards, axis=0)

        distances = np.array(all_distances)
        distances_mean = np.mean(distances, axis=0)
        distances_std = np.std(distances, axis=0)

        x_points = np.arange(MAX_EPOCH) + 1

        color = "C{}".format(i)

        line_objects.append(reward_ax.plot(x_points, rewards_mean, c=color)[0])
        reward_ax.fill_between(
            x_points,
            rewards_mean - rewards_std,
            rewards_mean + rewards_std,
            alpha=0.2,
            color=color,
            linewidth=0
        )

        distance_ax.plot(x_points, distances_mean, c=color)
        distance_ax.fill_between(
            x_points,
            distances_mean - distances_std,
            distances_mean + distances_std,
            alpha=0.2,
            color=color,
            linewidth=0
        )
    mean_delta_correlation = np.mean(delta_correlations)
    std_delta_correlation = np.std(delta_correlations)
    print("Delta correlations: {:.3f} with std {:.3f}".format(mean_delta_correlation, std_delta_correlation))

    reward_ax.set_ylim(-0.9, 1.3)
    reward_ax.set_yticks([0, 1])
    reward_ax.grid(alpha=0.2)
    reward_ax.legend(line_objects, ENV_NAMES)
    reward_ax.set_ylabel("Normalized reward", fontsize="large")
    distance_ax.set_ylim(0, 3.9)
    distance_ax.set_xlim(1, 50)
    distance_ax.grid(alpha=0.2)
    distance_ax.set_ylabel("Distance", fontsize="large")
    distance_ax.set_xlabel("Epochs", fontsize="large")
    pyplot.tight_layout()
    pyplot.savefig("figures/bc_results_appendix.pdf", bbox_inches="tight", pad_inches=0.0)


def plot_ubm_results():
    """
    Plot the selected heatmaps of classification
    accuracies for the paper, along
    with the distance experiment
    """
    from run_ubm_data_experiments import (
        PIVECTOR_TEMPLATE,
        DISTANCE_MATRIX_TEMPLATE,
        NUM_TRAJECTORIES,
        NUM_COMPONENTS,
        NUM_REPETITIONS,
        REWARD_SCALES,
        ENVS
    )

    ENVS = [
        "Pendulum-v0",
        "CartPole-v1",
        "Acrobot-v1",
        "LunarLander-v2",
        "BipedalWalker-v3",
    ]

    fig, axs = pyplot.subplots(
        figsize=[6.4 * 1.5, 4.8 * (5 / 2)],
        nrows=5,
        ncols=3,
    )


    # Plot heatmaps of example envs
    for env_i, env in enumerate(ENVS):
        # Get unique policy names we tested
        policy_names = glob(PIVECTOR_TEMPLATE.format(env=env, num_traj="*", num_components="*", policy_name="*", repetition_num="*"))
        policy_names = ["_".join(os.path.basename(x).split("_")[-4:-2]) for x in policy_names]
        policy_names = sorted(list(set(policy_names)))

        min_reward, max_reward = REWARD_SCALES[env]

        average_scores = np.ones((len(NUM_TRAJECTORIES), len(NUM_COMPONENTS)))
        std_scores = np.ones((len(NUM_TRAJECTORIES), len(NUM_COMPONENTS)))
        for num_traj_idx, num_traj in enumerate(NUM_TRAJECTORIES):
            for num_comp_idx, num_components in enumerate(NUM_COMPONENTS):
                # Average over different policies and repetitions
                scores = []
                for policy_name in policy_names:
                    for repetition in range(1, NUM_REPETITIONS + 1):
                        file_path = DISTANCE_MATRIX_TEMPLATE.format(env=env, num_traj=num_traj, num_components=num_components, policy_name=policy_name, repetition_num=repetition)
                        data = np.load(file_path)
                        # ll_matrix: first axis is data, second axis is adapted-GMMs
                        distance_matrix = data["distance_matrix"]
                        rewards = data["average_episodic_rewards"]

                        raveled_reward_distances = np.abs(rewards - rewards[:, None])
                        # Take upper diagonal, skip diagonal
                        raveled_reward_distances = raveled_reward_distances[np.triu_indices(raveled_reward_distances.shape[0], 1)]
                        raveled_distances = distance_matrix[np.triu_indices(distance_matrix.shape[0], 1)]

                        # Score is correlation between the two
                        correlation = np.corrcoef(raveled_distances, raveled_reward_distances)[0, 1]
                        scores.append(correlation)

                scores = np.array(scores)
                average_score = np.mean(scores)
                std_score = np.std(scores)
                average_scores[num_traj_idx, num_comp_idx] = average_score
                std_scores[num_traj_idx, num_comp_idx] = std_score
        ax = axs[env_i, 0]
        ax.imshow(average_scores)
        # Adjust ticks
        ax.set_xticks([])
        ax.set_yticks(np.arange(len(NUM_TRAJECTORIES)))
        ax.set_yticklabels(NUM_TRAJECTORIES)
        ax.tick_params(length=0)
        # Add values to plot
        for i in range(len(NUM_TRAJECTORIES)):
            for j in range(len(NUM_COMPONENTS)):
                text = ax.text(j, i, "{:2}".format(int(average_scores[i, j] * 100)),
                               ha="center", va="center", color="w")


    # Now plot the amount of error over all environments
    for env_i, env in enumerate(ENVS):
        # Get unique policy names we tested
        policy_names = glob(DISTANCE_MATRIX_TEMPLATE.format(env=env, num_traj="*", num_components="*", policy_name="*", repetition_num="*"))
        policy_names = ["_".join(os.path.basename(x).split("_")[-4:-2]) for x in policy_names]
        policy_names = sorted(list(set(policy_names)))

        per_policy_average_errors = []
        for policy_name in policy_names:
            average_errors_array = np.zeros((len(NUM_TRAJECTORIES), len(NUM_COMPONENTS)))
            for component_i, num_components in enumerate(NUM_COMPONENTS):
                # The "ground truth" distances
                anchor_distance = None
                for traj_i, num_traj in enumerate(NUM_TRAJECTORIES):
                    repetition_errors = []
                    for repetition in range(1, NUM_REPETITIONS + 1):
                        file_path = DISTANCE_MATRIX_TEMPLATE.format(env=env, num_traj=num_traj, num_components=num_components, policy_name=policy_name, repetition_num=repetition)
                        distance_matrix = np.load(file_path)["distance_matrix"]
                        distance_matrix = (distance_matrix - distance_matrix.min()) / (distance_matrix.max() - distance_matrix.min())
                        # Get only upper triangle as distance matrix is symmetric. Exlude diagonal
                        raveled_distances = distance_matrix[np.triu_indices(distance_matrix.shape[0], 1)]
                        # Check if we use this as the zero-point or compute relative error to
                        if anchor_distance is None:
                            assert num_traj == 100
                            anchor_distance = raveled_distances
                        else:
                            repetition_errors.append(np.mean(np.abs(raveled_distances - anchor_distance) / anchor_distance))
                    average_errors_array[traj_i, component_i] = np.mean(repetition_errors)
            per_policy_average_errors.append(average_errors_array)
        # Turn into percentages
        per_policy_average_errors = np.array(per_policy_average_errors) * 100
        mean_average_errors = np.mean(per_policy_average_errors, axis=0)
        std_average_errors = np.std(per_policy_average_errors, axis=0)
        ax = axs[env_i, 1]
        ax.imshow(-mean_average_errors)
        # Adjust ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(length=0)
        # Add values to plot
        for i in range(len(NUM_TRAJECTORIES)):
            for j in range(len(NUM_COMPONENTS)):
                text = ax.text(j, i, "{}".format(int(mean_average_errors[i, j])),
                               ha="center", va="center", color="w")
        # Add titles to middle rows
        ax.set_title(env.split("-")[0])

    # Variation between results, one (len(NUM_TRAJS), len(NUM_COMPONENTS))
    for env_i, env in enumerate(ENVS):
        # Get unique policy names we tested
        policy_names = glob(DISTANCE_MATRIX_TEMPLATE.format(env=env, num_traj="*", num_components="*", policy_name="*", repetition_num="*"))
        policy_names = ["_".join(os.path.basename(x).split("_")[-4:-2]) for x in policy_names]
        policy_names = sorted(list(set(policy_names)))

        cvs_array = np.zeros((len(NUM_TRAJECTORIES), len(NUM_COMPONENTS)))
        for component_i, num_components in enumerate(NUM_COMPONENTS):
            for traj_i, num_traj in enumerate(NUM_TRAJECTORIES):
                # Average over different policies
                averaged_cv = []
                for policy_name in policy_names:
                    # Compute std over repetitions
                    distances = []
                    for repetition in range(1, NUM_REPETITIONS + 1):
                        file_path = DISTANCE_MATRIX_TEMPLATE.format(env=env, num_traj=num_traj, num_components=num_components, policy_name=policy_name, repetition_num=repetition)
                        distance_matrix = np.load(file_path)["distance_matrix"]
                        distance_matrix = (distance_matrix - distance_matrix.min()) / (distance_matrix.max() - distance_matrix.min())
                        # Get only upper triangle as distance matrix is symmetric. Exlude diagonal
                        raveled_distances = distance_matrix[np.triu_indices(distance_matrix.shape[0], 1)]
                        distances.append(raveled_distances)
                    distances = np.stack(distances)
                    # Coefficient of variance (std / mean)
                    average_cv = np.mean(np.std(distances, axis=0) / np.mean(distances, axis=0))
                    averaged_cv.append(average_cv)
                averaged_average_cv = np.mean(averaged_cv)
                cvs_array[traj_i, component_i] = averaged_average_cv

        mean_cvs = cvs_array

        ax = axs[env_i, 2]
        ax.imshow(-mean_cvs)
        # Adjust ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(length=0)
        # Add values to plot
        for i in range(len(NUM_TRAJECTORIES)):
            for j in range(len(NUM_COMPONENTS)):
                text = ax.text(j, i, "{}".format(int(mean_cvs[i, j] * 100)),
                               ha="center", va="center", color="w")

    # Add ticklabels to bottom row
    for i in range(3):
        ax = axs[-1, i]
        ax.set_xticks(np.arange(len(NUM_COMPONENTS)))
        ax.set_xticklabels(NUM_COMPONENTS)
    # Add labels
    axs[2, 0].set_ylabel("Number of trajectories")
    axs[4, 1].set_xlabel("Number of components")
    # Add plot type titles to first row
    axs[0, 0].set_title("Correlation with return-distance")
    axs[0, 1].set_title(ENVS[0].split("-")[0] + "\nDistance error")
    axs[0, 2].set_title("Distance variance")

    pyplot.tight_layout()
    pyplot.savefig("figures/ubm_analysis_appendix.pdf", bbox_inches="tight", pad_inches=0.0)


def plot_metric_results():
    """
    Plot the metrics used in UBM experiments for different
    BCs for a comparison
    """
    from run_metric_comparison_experiments import (
        PIVECTOR_TEMPLATE,
        PIVECTOR_DISTANCE_MATRIX_TEMPLATE,
        DISCRIMINATOR_DISTANCE_MATRIX_TEMPLATE,
        GAUSSIAN_DISTANCE_MATRIX_TEMPLATE,
        ENCODER_DISTANCE_MATRIX_TEMPLATE,
        DISCRETIZATION_DISTANCE_MATRIX_TEMPLATE,
        NUM_TRAJECTORIES,
        NUM_COMPONENTS,
        NUM_REPETITIONS,
        REWARD_SCALES,
        ENVS
    )

    # Path-templates to each distance matrix to compare
    # BC = Behavioural Characteristication
    BC_DISTANCE_MATRIX_TEMPLATES = [
        PIVECTOR_DISTANCE_MATRIX_TEMPLATE,
        GAUSSIAN_DISTANCE_MATRIX_TEMPLATE,
        DISCRIMINATOR_DISTANCE_MATRIX_TEMPLATE,
        ENCODER_DISTANCE_MATRIX_TEMPLATE,
        DISCRETIZATION_DISTANCE_MATRIX_TEMPLATE,
    ]

    BC_LEGEND_NAMES = [
        "Supervector",
        "Gaussian",
        "Discriminator",
        "Encoder",
        "Discretization"
    ]

    BC_PLOT_COLORS = [
        "C0",
        "C1",
        "C2",
        "C3",
        "C4"
    ]

    DIFFERENT_REWARD_THRESHOLD = 0.25

    fig, axs = pyplot.subplots(
        figsize=[6.4 * 1.75, 4.8 * (5 / 1.5)],
        nrows=len(ENVS),
        ncols=3,
    )

    def get_policy_names(env):
        policy_names = glob(PIVECTOR_TEMPLATE.format(env=env, num_traj="*", num_components="*", policy_name="*", repetition_num="*"))
        policy_names = ["_".join(os.path.basename(x).split("_")[-4:-2]) for x in policy_names]
        policy_names = sorted(list(set(policy_names)))
        return policy_names

    # For each different distance measurement
    for distance_matrix_template, plot_legend_name, plot_color in zip(BC_DISTANCE_MATRIX_TEMPLATES, BC_LEGEND_NAMES, BC_PLOT_COLORS):
        for env_i, env in enumerate(ENVS):
            axs[env_i, 0].grid(alpha=0.2)
            axs[env_i, 1].grid(alpha=0.2)
            axs[env_i, 2].grid(alpha=0.2)
            if "Bipedal" in env and distance_matrix_template == DISCRETIZATION_DISTANCE_MATRIX_TEMPLATE:
                print("[Note] Skipping env {} for discretization distances (OOM)".format(env))
                continue
            axs[env_i, 1].set_title(env.split("-")[0])
            # These will be NUM_TRAJECTORY length lists
            average_scores = np.ones((len(NUM_TRAJECTORIES),))
            std_scores = np.ones((len(NUM_TRAJECTORIES),))
            for num_traj_idx, num_traj in enumerate(NUM_TRAJECTORIES):
                # Average over environments, policies and repetitions
                scores = []
                min_reward, max_reward = REWARD_SCALES[env]
                policy_names = get_policy_names(env)

                for policy_name in policy_names:
                    for repetition in range(1, NUM_REPETITIONS + 1):
                        # Ugh bit of messing around because I did not think this through...
                        if distance_matrix_template == PIVECTOR_DISTANCE_MATRIX_TEMPLATE:
                            file_path = distance_matrix_template.format(env=env, num_traj=num_traj, num_components=NUM_COMPONENTS, policy_name=policy_name, repetition_num=repetition)
                        else:
                            file_path = distance_matrix_template.format(env=env, num_traj=num_traj, policy_name=policy_name, repetition_num=repetition)

                        data = np.load(file_path)
                        distance_matrix = data["distance_matrix"]
                        rewards = data["average_episodic_rewards"]

                        raveled_reward_distances = np.abs(rewards - rewards[:, None])
                        # Take upper diagonal, skip diagonal
                        raveled_reward_distances = raveled_reward_distances[np.triu_indices(raveled_reward_distances.shape[0], 1)]
                        raveled_distances = distance_matrix[np.triu_indices(distance_matrix.shape[0], 1)]

                        # Score is correlation between the two
                        correlation = np.corrcoef(raveled_distances, raveled_reward_distances)[0, 1]
                        scores.append(correlation)

                scores = np.array(scores)
                average_score = np.mean(scores)
                std_score = np.std(scores)
                average_scores[num_traj_idx] = average_score
                std_scores[num_traj_idx] = std_score
            ax = axs[env_i, 0]
            ax.plot(NUM_TRAJECTORIES, average_scores, c=plot_color, label=plot_legend_name)
            ax.scatter(NUM_TRAJECTORIES, average_scores, c=plot_color)
            ax.fill_between(
                NUM_TRAJECTORIES,
                average_scores - std_scores,
                average_scores + std_scores,
                alpha=0.2,
                color=plot_color,
                edgecolor="none",
                linewidth=0.0
            )
            ax.set_xticks(NUM_TRAJECTORIES)
            ax.set_ylabel("Correlation with return-distance")

            # Amount of error to "ground truth" result,
            # where "ground truth" is one of the results with 100 trajectories of data.
            # Because of wonkyness of this, store list [#num-traj] of lists,
            # each storing results for that num-traj run
            per_trajectory_relative_errors = [[] for i in NUM_TRAJECTORIES]
            policy_names = get_policy_names(env)
            for policy_name in policy_names:
                # The "ground truth" distances, will be filled with first
                # result with 100 trajectories.
                anchor_distance = None
                for traj_i, num_traj in enumerate(NUM_TRAJECTORIES):
                    for repetition in range(1, NUM_REPETITIONS + 1):
                        if distance_matrix_template == PIVECTOR_DISTANCE_MATRIX_TEMPLATE:
                            file_path = distance_matrix_template.format(env=env, num_traj=num_traj, num_components=NUM_COMPONENTS, policy_name=policy_name, repetition_num=repetition)
                        else:
                            file_path = distance_matrix_template.format(env=env, num_traj=num_traj, policy_name=policy_name, repetition_num=repetition)
                        distance_matrix = np.load(file_path)["distance_matrix"]
                        distance_matrix = (distance_matrix - distance_matrix.min()) / (distance_matrix.max() - distance_matrix.min())
                        # Get only upper triangle as distance matrix is symmetric. Exlude diagonal
                        raveled_distances = distance_matrix[np.triu_indices(distance_matrix.shape[0], 1)]
                        # Check if we use this as the zero-point or compute relative error to
                        if anchor_distance is None:
                            assert num_traj == 100
                            anchor_distance = raveled_distances
                        else:
                            per_trajectory_relative_errors[traj_i].append(
                                np.mean(np.abs(raveled_distances - anchor_distance) / np.abs(anchor_distance))
                            )
            # Lists are not of equal length, so can not just change into an array
            mean_average_errors = np.array([np.mean(np.array(results) * 100) for results in per_trajectory_relative_errors])
            std_average_errors = np.array([np.std(np.array(results) * 100) for results in per_trajectory_relative_errors])
            ax = axs[env_i, 1]
            ax.plot(NUM_TRAJECTORIES, mean_average_errors, c=plot_color, label=plot_legend_name)
            ax.scatter(NUM_TRAJECTORIES, mean_average_errors, c=plot_color)
            ax.fill_between(
                NUM_TRAJECTORIES,
                mean_average_errors - std_average_errors,
                mean_average_errors + std_average_errors,
                alpha=0.2,
                color=plot_color,
                edgecolor="none",
                linewidth=0.0
            )
            ax.set_xticks(NUM_TRAJECTORIES)
            ax.set_ylabel("Relative error to ground truth (%)")

            # Variation between results
            cv_means = np.ones((len(NUM_TRAJECTORIES,)))
            cv_stds = np.ones((len(NUM_TRAJECTORIES,)))
            for traj_i, num_traj in enumerate(NUM_TRAJECTORIES):
                traj_averaged_cvs = []
                policy_names = get_policy_names(env)
                for policy_name in policy_names:
                    # Compute std over repetitions
                    distances = []
                    for repetition in range(1, NUM_REPETITIONS + 1):
                        if distance_matrix_template == PIVECTOR_DISTANCE_MATRIX_TEMPLATE:
                            file_path = distance_matrix_template.format(env=env, num_traj=num_traj, num_components=NUM_COMPONENTS, policy_name=policy_name, repetition_num=repetition)
                        else:
                            file_path = distance_matrix_template.format(env=env, num_traj=num_traj, policy_name=policy_name, repetition_num=repetition)

                        distance_matrix = np.load(file_path)["distance_matrix"]
                        distance_matrix = (distance_matrix - distance_matrix.min()) / (distance_matrix.max() - distance_matrix.min())
                        # Get only upper triangle as distance matrix is symmetric. Exlude diagonal
                        raveled_distances = distance_matrix[np.triu_indices(distance_matrix.shape[0], 1)]
                        distances.append(raveled_distances)
                    distances = np.stack(distances)
                    # Coefficient of variance (std / mean)
                    average_cv = np.mean(np.std(distances, axis=0) / np.mean(distances, axis=0))
                    traj_averaged_cvs.append(average_cv)
                traj_averaged_cvs = np.array(traj_averaged_cvs)
                cv_means[traj_i] = np.mean(traj_averaged_cvs)
                cv_stds[traj_i] = np.std(traj_averaged_cvs)

            ax = axs[env_i, 2]
            ax.plot(NUM_TRAJECTORIES, cv_means, c=plot_color, label=plot_legend_name)
            ax.scatter(NUM_TRAJECTORIES, cv_means, c=plot_color)
            ax.fill_between(
                NUM_TRAJECTORIES,
                cv_means - cv_stds,
                cv_means + cv_stds,
                alpha=0.2,
                color=plot_color,
                edgecolor="none",
                linewidth=0.0
            )
            ax.set_xticks(NUM_TRAJECTORIES)
            ax.set_ylabel("Coefficient of variance $\\sigma/\\mu$")
            if env_i == (len(ENVS) - 1):
                axs[env_i, 1].set_xlabel("Number of trajectories")

    # Add plot type titles to first row
    axs[0, 0].set_title("Correlation with return-distance")
    axs[0, 1].set_title(ENVS[0].split("-")[0] + "\nDistance error")
    axs[0, 2].set_title("Distance variance")

    axs[0, 1].legend(prop={"size": "medium"})
    pyplot.tight_layout()
    pyplot.savefig("figures/metric_comparison_appendix.pdf", bbox_inches="tight", pad_inches=0.0)


def plot_trust_region_results():
    """Plot/print results with trust-region experiments"""
    RESULTS_DIRS = ["trust_region_experiments", "trust_region_experiments_augment_ppo"]
    ENVS = ["DangerousPath-len25-dim5-v0", "DangerousPath-NoFail-len25-dim5-v0", "CartPole-v0"]
    DIR_TEMPLATE = "{env}_lr_{learning_rate}"
    STDOUT_TEMPLATE = "{method}_repetition_{repetition}.txt"
    REWARD_PATTERN = r"ep_rew_mean[ ]*\| ([0-9\.\-]*)"
    TIMESTEP_PATTERN = r"total_timesteps[ ]*\| ([0-9]*)"

    NUM_REPETITIONS = 5
    LEARNING_RATES = ["1e-3", "1e-4", "1e-5", "1e-6"]

    METHODS = {
        "NoConstraint": ["NoConstraint"],
        "ClipPPO": ["ClipPPO"],
        "Gaussian": ["Gaussian_{}".format(kl) for kl in [0.001, 0.005, 0.01, 0.05, 0.1]],
        "Supervector": ["Supervector_{}".format(kl) for kl in [0.001, 0.005, 0.01, 0.05, 0.1]],
        "PiMaxTV": ["PiMaxTV_{}".format(kl) for kl in [0.001, 0.005, 0.01, 0.05, 0.1]],
    }

    fig, axs = pyplot.subplots(
        figsize=[6.4 * len(LEARNING_RATES), 4.8 * len(ENVS) * len(RESULTS_DIRS)],
        nrows=len(ENVS) * len(RESULTS_DIRS),
        ncols=len(LEARNING_RATES),
        squeeze=False
    )

    for env_i, (env, results_dir) in enumerate(itertools.product(ENVS, RESULTS_DIRS)):
        path_to_log_template = os.path.join(results_dir, DIR_TEMPLATE, STDOUT_TEMPLATE)
        for learning_rate_i, learning_rate in enumerate(LEARNING_RATES):
            num_plotted_lines = 0
            ax = axs[env_i, learning_rate_i]
            ax.set_title("{}\n{} lr: {}".format(results_dir, env, learning_rate))
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Average return")
            for legend_name, experiment_names in METHODS.items():
                for experiment_name in experiment_names:
                    # Construct learning curves
                    xs = []
                    ys = []
                    for repetition in range(NUM_REPETITIONS):
                        path_to_log = path_to_log_template.format(env=env, learning_rate=learning_rate, method=experiment_name, repetition=repetition)
                        log = None
                        try:
                            log = open(path_to_log).read()
                        except Exception as e:
                            print("[Warning] Could not load {}: {}".format(path_to_log, e))
                        timesteps = list(map(float, re.findall(TIMESTEP_PATTERN, log)))
                        rewards = list(map(float, re.findall(REWARD_PATTERN, log)))
                        assert len(timesteps) == len(rewards)
                        xs.append(np.array(timesteps))
                        ys.append(np.array(rewards))

                    average_x, average_y, std_y = interpolate_and_average(xs, ys)
                    color, linestyle = color_linestyle_cycle(num_plotted_lines)
                    ax.plot(average_x, average_y, label=experiment_name, c=color, linestyle=linestyle)
                    ax.fill_between(
                        average_x,
                        average_y - std_y / 2,
                        average_y + std_y / 2,
                        alpha=0.2,
                        color=color,
                        linewidth=0.0
                    )
                    num_plotted_lines += 1
    axs[0, 0].legend()
    pyplot.tight_layout()
    pyplot.savefig("figures/trust_region_results_appendix.pdf", bbox_inches="tight", pad_inches=0.0)

if __name__ == "__main__":
    plot_ppo_clip_results()
    plot_tsnes()
    plot_bc_results()
    plot_ubm_results()
    plot_metric_results()
    plot_trust_region_results()
