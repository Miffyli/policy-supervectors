# Hardcoded plotting for paper. There
# are no arguments, everything is hardcoded.
#
import os
from glob import glob

import numpy as np
import matplotlib
from matplotlib import pyplot

# Stackoverflow #4931376
matplotlib.use('Agg')

PIVECTORS_DIR = "pivectors"
CHECKPOINT_DISTANCES_FILE = "checkpoint_distances.npz"


def plot_visual_abstract():
    """
    Plot illustrative example of CMA-ES tSNEs at specific points for visual abstract.

    This probably does not work well outside this one specific run we used to plot
    things, but this graphic is not meant to be a generic result, rather picked for
    illustrative purposes.
    """
    # Which generations to plot
    GENERATIONS = [100, 230, 350]

    # LunarLander CMA-ES
    experiment_path = glob("experiments/wann_LunarLander-v2_CMAES*")
    assert len(experiment_path) == 1, "There should be only one CMA-ES experiment with LunarLander-v2"
    experiment_path = experiment_path[0]

    pivector_paths = glob(os.path.join(experiment_path, "pivectors", "*"))

    tsnes = []
    rewards = []
    for generation in GENERATIONS:
        # Find pivector files for specific generation, load them and store points
        generation_paths = [path for path in pivector_paths if "gen_{}_".format(generation) in path]

        population = [np.load(path) for path in generation_paths]
        population_tsnes = np.array([x["tsne"] for x in population])
        population_rewards = np.array([x["average_episodic_reward"] for x in population])
        tsnes.append(population_tsnes)
        rewards.append(population_rewards)

    figure, axs = pyplot.subplots(
        figsize=[2.5 * 3, 2.5],
        nrows=1,
        ncols=len(GENERATIONS),
        sharex="all",
        sharey="all"
    )

    min_reward = min(x.min() for x in rewards)
    max_reward = max(x.max() for x in rewards)
    scatter = None

    for idx in range(len(GENERATIONS)):
        population_tsne = tsnes[idx]
        population_rewards = rewards[idx]
        generation = GENERATIONS[idx]
        ax = axs[idx]

        scatter = ax.scatter(
            population_tsne[:, 0],
            population_tsne[:, 1],
            c=population_rewards,
            vmin=min_reward,
            vmax=max_reward,
            cmap="plasma"
        )
        ax.set_title("Generation {}".format(generation))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    # Making room for colorbar
    # Stackoverflow #13784201
    figure.subplots_adjust(right=1.0)
    cbar = figure.colorbar(scatter)
    cbar.set_ticks([])
    cbar.ax.set_ylabel("Reward $\\rightarrow$", rotation=90, fontsize="large")

    figure.tight_layout()
    figure.savefig("figures/visual_abstract.pdf", bbox_inches="tight", pad_inches=0.05)


def plot_ppo_clip_results():
    """
    Plot the PPO clip results: Draw selected
    learning curves and numbers of average distance traveled.
    """

    # Clip experiments
    experiment_paths = glob("experiments/stablebaselines_*-clip*")
    # Get unique env names
    envs = [os.path.basename(path).split("_")[1] for path in experiment_paths]
    unique_envs = list(set(envs))
    # Get unique algorithms
    algos = [os.path.basename(path).split("_")[2] for path in experiment_paths]
    unique_algos = sorted(list(set(algos)))

    print("--- PPO clip distance results ---")
    print("Algo " + " ".join(unique_envs) + " corr(R,d)")

    # Gather Pearson correlations of
    # distance traveled and average episodic rewards
    # for averaging over

    for algo in unique_algos:
        # Should be strings to be printed
        env_results = []
        distance_reward_correlations = []
        for env in unique_envs:
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
            distance_reward_correlations.append(
                np.corrcoef(
                    rewards.ravel(),
                    distances.ravel(),
                )[0, 1]
            )
            cumulative_distance = np.sum(distances, axis=1)
            distances_mean = np.mean(distances, axis=0)
            distances_std = np.std(distances, axis=0)
            rewards_mean = np.mean(rewards, axis=0)
            rewards_std = np.std(rewards, axis=0)
            cumulative_distance_mean = np.mean(cumulative_distance, axis=0)
            cumulative_distance_std = np.std(cumulative_distance, axis=0)

            # Add cumulative distance results
            env_results.append("{:4.1f}±{:<4.1f}".format(cumulative_distance_mean.item(), cumulative_distance_std.item()))
        prettier_algo = algo.replace("-clip", "")
        mean_correlation = np.mean(distance_reward_correlations)
        std_correlation = np.std(distance_reward_correlations)
        print("{} {} {:.2f}±{:.2f}".format(prettier_algo, " ".join(env_results), mean_correlation, std_correlation))
    print("---------------------------------")


def plot_tsnes():
    """
    Plot the t-SNE of different policy evaluations
    """
    # Two environments (for main paper figure. All for final figure)
    ENVS = [
        "BipedalWalker-v3",
        #"LunarLander-v2",
        "Pendulum-v0"
        #"Acrobot-v1",
        #"CartPole-v1"
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
        figsize=[6.4 * 2, 4.8],
        nrows=2,
        ncols=4,
        gridspec_kw={'hspace': 0, 'wspace': 0},
    )

    for plot_i in range(2):
        env = ENVS[plot_i]
        reward_scale = REWARD_SCALES[env]
        for algo_i in range(len(ALGO_TYPES)):
            column_idx = (algo_i % 2) + plot_i * 2
            row_idx = 0 if algo_i <= 1 else 1
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
            ax.spines["top"].set_alpha(0.2)
            ax.spines["bottom"].set_alpha(0.2)
            ax.spines["left"].set_alpha(0.2)
            ax.spines["right"].set_alpha(0.2)
            # Hide edge spines and bolden mid-spines
            if row_idx == 0:
                ax.spines["top"].set_visible(False)
            else:
                ax.spines["bottom"].set_visible(False)
            if column_idx == 0:
                ax.spines["left"].set_visible(False)
            elif column_idx == 1:
                ax.spines["right"].set_alpha(1.0)
            elif column_idx == 2:
                ax.spines["left"].set_alpha(1.0)
            elif column_idx == 3:
                ax.spines["right"].set_visible(False)

            # Add titles
            if row_idx == 0 and (column_idx == 0 or column_idx == 2):
                ax.set_title(env.split("-")[0], x=1.0)

    cbaxes = figure.add_axes([0.4, 0.94, 0.2, 0.02])
    cbar = figure.colorbar(scatter, orientation="horizontal", cax=cbaxes)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["Min", "Reward", "Max"])
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(labelsize="small", length=0)
    figure.tight_layout()
    figure.savefig("figures/tsnes.png", dpi=200, bbox_inches="tight", pad_inches=0.0)


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
        "Bipedal",
        "Lunar"
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
    reward_ax.set_ylabel("Reward", fontsize="large")
    distance_ax.set_ylim(0, 3.9)
    distance_ax.set_xlim(1, 50)
    distance_ax.grid(alpha=0.2)
    distance_ax.set_ylabel("Distance", fontsize="large")
    distance_ax.set_xlabel("Epochs", fontsize="large")
    pyplot.tight_layout()
    pyplot.savefig("figures/bc_results.pdf", bbox_inches="tight", pad_inches=0.0)


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
        DIFFERENT_REWARD_THRESHOLD,
        ENVS
    )
    DIFFERENT_REWARD_THRESHOLD = 0.25

    HEATMAP_ENVS = [
        "BipedalWalker-v3",
        #"Pendulum-v0",
        "LunarLander-v2",
        #"CartPole-v1",
        #"Acrobot-v1"
    ]

    fig, axs = pyplot.subplots(
        nrows=2,
        ncols=2,
    )

    # Remove two right plots and replace it with one big
    #gs = axs[0, 1].get_gridspec()
    # remove the underlying axes
    #axs[0, 1].remove()
    #axs[1, 1].remove()
    #bigax = fig.add_subplot(gs[0:, 1])

    # Plot heatmaps of example envs
    for env_i, env in enumerate(HEATMAP_ENVS):
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
                        average_distance = distance_matrix[np.triu_indices(distance_matrix.shape[0], 1)].mean()
                        rewards = data["average_episodic_rewards"]

                        # Find policies that are different from each other, based on the
                        # reward they get
                        normalized_rewards = (rewards - min_reward) / (max_reward - min_reward)
                        normalized_reward_difference = np.abs(normalized_rewards - normalized_rewards[:, None])
                        has_different_reward = normalized_reward_difference > DIFFERENT_REWARD_THRESHOLD
                        # Only take upper-diagonal (matrix is symmetric and diagonal is zeros)
                        has_different_reward[np.tril_indices(has_different_reward.shape[0])] = False
                        distances_to_different = distance_matrix[has_different_reward]
                        # Compare to average distance
                        scores.append(average_distance / distances_to_different.mean())

                scores = np.array(scores)
                average_score = np.mean(scores)
                std_score = np.std(scores)
                average_scores[num_traj_idx, num_comp_idx] = average_score
                std_scores[num_traj_idx, num_comp_idx] = std_score
        ax = axs[env_i, 0]
        ax.imshow(average_scores)
        # Adjust ticks
        ax.set_xticks(np.arange(len(NUM_COMPONENTS)))
        ax.set_yticks(np.arange(len(NUM_TRAJECTORIES)))
        ax.set_xticklabels(NUM_COMPONENTS)
        ax.set_yticklabels(NUM_TRAJECTORIES)
        ax.tick_params(length=0)
        # Add values to plot
        for i in range(len(NUM_TRAJECTORIES)):
            for j in range(len(NUM_COMPONENTS)):
                text = ax.text(j, i, "{:2}".format(int(average_scores[i, j] * 100)),
                               ha="center", va="center", color="w")

        ax.set_title("Distance to different")
        if env_i == 1:
            ax.set_xlabel("Number of components")
        ax.set_ylabel("Number of trajectories")
        ax.text(0, 1.0, "{}.".format(env[:4]), horizontalalignment="left", verticalalignment="bottom", transform=ax.transAxes)
        ax.text(1.0, 1.0, "x100", horizontalalignment="right", verticalalignment="bottom", transform=ax.transAxes)

    # Will contain one (len(NUM_COMPONENTS), len(NUM_TRAJECTORIES)) array per policy,
    # which we average/std over later
    per_policy_average_errors = []
    # Now plot the amount of error over all environments
    for env in ENVS:
        # Get unique policy names we tested
        policy_names = glob(DISTANCE_MATRIX_TEMPLATE.format(env=env, num_traj="*", num_components="*", policy_name="*", repetition_num="*"))
        policy_names = ["_".join(os.path.basename(x).split("_")[-4:-2]) for x in policy_names]
        policy_names = sorted(list(set(policy_names)))

        for policy_name in policy_names:
            average_errors_array = np.zeros((len(NUM_TRAJECTORIES), len(NUM_COMPONENTS)))
            for component_i, num_components in enumerate(NUM_COMPONENTS):
                # The "ground truth" distances
                anchor_distance = None
                for traj_i, num_traj in enumerate(NUM_TRAJECTORIES):
                    repetition_errors = []
                    for repetition in range(1, NUM_REPETITIONS + 1):
                        relative_errors = []
                        file_path = DISTANCE_MATRIX_TEMPLATE.format(env=env, num_traj=num_traj, num_components=num_components, policy_name=policy_name, repetition_num=repetition)
                        distance_matrix = np.load(file_path)["distance_matrix"]
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
    ax = axs[0, 1]
    ax.imshow(mean_average_errors)
    # Adjust ticks
    ax.set_xticks(np.arange(len(NUM_COMPONENTS)))
    ax.set_yticks(np.arange(len(NUM_TRAJECTORIES)))
    ax.set_xticklabels(NUM_COMPONENTS)
    ax.set_yticklabels(NUM_TRAJECTORIES)
    ax.tick_params(length=0)
    ax.set_title("Distance error")
    # Add values to plot
    for i in range(len(NUM_TRAJECTORIES)):
        for j in range(len(NUM_COMPONENTS)):
            text = ax.text(j, i, "{}".format(int(mean_average_errors[i, j])),
                           ha="center", va="center", color="w")
    ax.text(0, 1.0, "All", horizontalalignment="left", verticalalignment="bottom", transform=ax.transAxes)
    ax.text(1.0, 1.0, "x100", horizontalalignment="right", verticalalignment="bottom", transform=ax.transAxes)

    # Variation between results, one (len(NUM_TRAJS), len(NUM_COMPONENTS))
    per_env_cv = []
    for env in ENVS:
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
                        # Get only upper triangle as distance matrix is symmetric. Exlude diagonal
                        raveled_distances = distance_matrix[np.triu_indices(distance_matrix.shape[0], 1)]
                        distances.append(raveled_distances)
                    distances = np.stack(distances)
                    # Coefficient of variance (std / mean)
                    average_cv = np.mean(np.std(distances, axis=0) / np.mean(distances, axis=0))
                    averaged_cv.append(average_cv)
                averaged_average_cv = np.mean(averaged_cv)
                cvs_array[traj_i, component_i] = averaged_average_cv
        per_env_cv.append(cvs_array)

    per_env_cv = np.array(per_env_cv)
    mean_cvs = np.mean(per_env_cv, axis=0)

    ax = axs[1, 1]
    ax.imshow(mean_cvs)
    # Adjust ticks
    ax.set_xticks(np.arange(len(NUM_COMPONENTS)))
    ax.set_yticks(np.arange(len(NUM_TRAJECTORIES)))
    ax.set_xticklabels(NUM_COMPONENTS)
    ax.set_yticklabels(NUM_TRAJECTORIES)
    ax.tick_params(length=0)
    ax.set_xlabel("Number of components")
    ax.set_title("Distance variance")
    # Add values to plot
    for i in range(len(NUM_TRAJECTORIES)):
        for j in range(len(NUM_COMPONENTS)):
            text = ax.text(j, i, "{}".format(int(mean_cvs[i, j] * 100)),
                           ha="center", va="center", color="w")
    ax.text(0, 1.0, "All", horizontalalignment="left", verticalalignment="bottom", transform=ax.transAxes)
    ax.text(1.0, 1.0, "x100", horizontalalignment="right", verticalalignment="bottom", transform=ax.transAxes)


    pyplot.tight_layout()
    pyplot.savefig("figures/ubm_analysis.pdf", bbox_inches="tight", pad_inches=0.0)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    plot_visual_abstract()
    plot_ppo_clip_results()
    plot_tsnes()
    plot_bc_results()
    plot_ubm_results()
