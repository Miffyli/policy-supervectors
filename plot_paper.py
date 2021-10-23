# Hardcoded plotting for paper. There
# are no arguments, everything is hardcoded.
#
import os
from glob import glob
import re
import itertools

import numpy as np
import matplotlib
from matplotlib import pyplot

# Stackoverflow #4931376
matplotlib.use('Agg')

PIVECTORS_DIR = "pivectors"
CHECKPOINT_DISTANCES_FILE = "checkpoint_distances.npz"


def interpolate_and_average(xs, ys, interop_points=None, confidence_interval=False):
    """
    Average bunch of repetitions (xs, ys)
    into one curve. This is done by linearly interpolating
    y values to same basis (same xs). Maximum x of returned
    curve is smallest x of repetitions.

    Returns [new_x, mean_y, std_y]

    If confidence_interval is true, returns
    [new_x, mean_y, std_y, lower_bound, upper_bound]
    where lower and upper bounds are 95% confidence intervals
    """
    # Get the xs of shortest curve
    max_min_x = max(x.min() for x in xs)
    min_max_x = min(x.max() for x in xs)
    if interop_points is None:
        # Interop points according to curve with "least resolution"
        interop_points = min(x.shape[0] for x in xs)

    new_x = np.linspace(max_min_x, min_max_x, interop_points)
    new_ys = []

    for old_x, old_y in zip(xs, ys):
        new_ys.append(np.interp(new_x, old_x, old_y))

    # Average out
    # atleast_2d for case when we only have one reptition
    new_ys = np.atleast_2d(np.array(new_ys))
    new_y = np.mean(new_ys, axis=0)
    std_y = np.std(new_ys, axis=0)

    if confidence_interval:
        interval = 1.96 * (std_y / np.sqrt(len(xs)))
        lower_bound = new_y - interval
        upper_bound = new_y + interval
        return new_x, new_y, std_y, lower_bound, upper_bound
    else:
        return new_x, new_y, std_y


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


def plot_tsnes():
    """
    Plot the t-SNE of different policy evaluations
    """
    # Two environments (for main paper figure. All for final figure)
    ENVS = [
        "BipedalWalker-v3",
        #"LunarLander-v2",
        #"Pendulum-v0"
        "Acrobot-v1",
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
        DISCRETIZATION_DISTANCE_MATRIX_TEMPLATE
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

    fig, axs = pyplot.subplots(
        figsize=[4.8 * 3 * 0.75, 4.8 * 0.75],
        nrows=1,
        ncols=3,
    )

    def get_policy_names(env):
        policy_names = glob(PIVECTOR_TEMPLATE.format(env=env, num_traj="*", num_components="*", policy_name="*", repetition_num="*"))
        policy_names = ["_".join(os.path.basename(x).split("_")[-4:-2]) for x in policy_names]
        policy_names = sorted(list(set(policy_names)))
        return policy_names

    # For each different distance measurement
    for distance_matrix_template, plot_legend_name, plot_color in zip(BC_DISTANCE_MATRIX_TEMPLATES, BC_LEGEND_NAMES, BC_PLOT_COLORS):
        # These will be NUM_TRAJECTORY length lists
        average_scores = np.ones((len(NUM_TRAJECTORIES),))
        std_scores = np.ones((len(NUM_TRAJECTORIES),))
        for num_traj_idx, num_traj in enumerate(NUM_TRAJECTORIES):
            # Average over environments, policies and repetitions
            scores = []
            for env_i, env in enumerate(ENVS):
                if "Bipedal" in env and distance_matrix_template == DISCRETIZATION_DISTANCE_MATRIX_TEMPLATE:
                    print("[Note] Skipping env {} for discretization distances (OOM)".format(env))
                    continue
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
                        raveled_reward_distances =  raveled_reward_distances[np.triu_indices(raveled_reward_distances.shape[0], 1)]
                        raveled_distances = distance_matrix[np.triu_indices(distance_matrix.shape[0], 1)]

                        # Score is correlation between the two
                        correlation = np.corrcoef(raveled_distances, raveled_reward_distances)[0, 1]
                        scores.append(correlation)

            scores = np.array(scores)
            average_score = np.mean(scores)
            std_score = np.std(scores)
            average_scores[num_traj_idx] = average_score
            std_scores[num_traj_idx] = std_score
        ax = axs[0]
        ax.plot(NUM_TRAJECTORIES, average_scores, c=plot_color, label=plot_legend_name)
        ax.scatter(NUM_TRAJECTORIES, average_scores, c=plot_color)
        #ax.fill_between(
        #    NUM_TRAJECTORIES,
        #    average_scores - std_scores,
        #    average_scores + std_scores,
        #    alpha=0.2,
        #    color=plot_color,
        #    edgecolor="none",
        #    linewidth=0.0
        #)
        ax.set_xticks(NUM_TRAJECTORIES)
        ax.tick_params(axis='both', which='both', labelsize="x-large")
        ax.set_ylabel("Correlation with return-distances", fontsize="x-large")
        ax.set_xlabel("Number of trajectories", fontsize="x-large")
        ax.grid(alpha=0.2)

        # Amount of error to "ground truth" result,
        # where "ground truth" is one of the results with 100 trajectories of data.
        # Because of wonkyness of this, store list [#num-traj] of lists,
        # each storing results for that num-traj run
        per_trajectory_relative_errors = [[] for i in NUM_TRAJECTORIES]
        for env in ENVS:
            if "Bipedal" in env and distance_matrix_template == DISCRETIZATION_DISTANCE_MATRIX_TEMPLATE:
                print("[Note] Skipping env {} for discretization distances (OOM)".format(env))
                continue
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
                        # Normalize to [0, 1]
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
        ax = axs[1]
        ax.plot(NUM_TRAJECTORIES, mean_average_errors, c=plot_color, label=plot_legend_name)
        ax.scatter(NUM_TRAJECTORIES, mean_average_errors, c=plot_color)
        #ax.fill_between(
        #    NUM_TRAJECTORIES,
        #    mean_average_errors - std_average_errors,
        #    mean_average_errors + std_average_errors,
        #    alpha=0.2,
        #    color=plot_color,
        #    edgecolor="none",
        #    linewidth=0.0
        #)
        ax.set_xticks(NUM_TRAJECTORIES)
        ax.tick_params(axis='both', which='both', labelsize="x-large")
        ax.set_ylabel("Relative error to ground truth (%)", fontsize="x-large")
        ax.set_xlabel("Number of trajectories", fontsize="x-large")
        ax.grid(alpha=0.2)

        # Variation between results
        cv_means = np.ones((len(NUM_TRAJECTORIES,)))
        cv_stds = np.ones((len(NUM_TRAJECTORIES,)))
        for traj_i, num_traj in enumerate(NUM_TRAJECTORIES):
            traj_averaged_cvs = []
            for env in ENVS:
                if "Bipedal" in env and distance_matrix_template == DISCRETIZATION_DISTANCE_MATRIX_TEMPLATE:
                    print("[Note] Skipping env {} for discretization distances (OOM)".format(env))
                    continue
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
                        # Normalize to [0, 1]
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

        ax = axs[2]
        ax.plot(NUM_TRAJECTORIES, cv_means, c=plot_color, label=plot_legend_name)
        ax.scatter(NUM_TRAJECTORIES, cv_means, c=plot_color)
        #ax.fill_between(
        #    NUM_TRAJECTORIES,
        #    cv_means - cv_stds,
        #    cv_means + cv_stds,
        #    alpha=0.2,
        #    color=plot_color,
        #    edgecolor="none",
        #    linewidth=0.0
        #)
        ax.set_xticks(NUM_TRAJECTORIES)
        ax.tick_params(axis='both', which='both', labelsize="x-large")
        ax.set_ylabel("Coefficient of variance $\\sigma/\\mu$", fontsize="x-large")
        ax.set_xlabel("Number of trajectories", fontsize="x-large")
        ax.grid(alpha=0.2)

    axs[1].legend(prop={"size": "large"})
    pyplot.tight_layout()
    pyplot.savefig("figures/metric_comparison.pdf", bbox_inches="tight", pad_inches=0.0)


def plot_novelty_results():
    """Plot results done by running NS-ES with fitness and different novelty searches"""
    RESULTS_DIR = "experiments"
    STDOUT_FILE = "log.txt"
    REWARD_PATTERN = r" EpRewMean[ ]*\| ([0-9\-\.]+)"
    TIMESTEP_PATTERN = r" TimestepsSoFar[ ]*\| ([0-9\-\.]+)"
    ITERATION_PATTERN = r" Iteration ([0-9]+)"

    GLOBS = [
        os.path.join(RESULTS_DIR, "novelty_DeceptivePointEnv-v0_es_*"),
        os.path.join(RESULTS_DIR, "novelty_DeceptivePointEnv-v0_nsres_*"),
        os.path.join(RESULTS_DIR, "novelty_DeceptivePointEnv-v0_nsresgaussian_*"),
        os.path.join(RESULTS_DIR, "novelty_DeceptivePointEnv-v0_nsressupervector_*")
    ]

    COLORS = [
        "C0",
        "C1",
        "C2",
        "C3"
    ]

    LEGENDS = [
        "ES",
        "NSR-ES (Terminal)",
        "NSR-ES (Gaussian)",
        "NSR-ES (Supervector)"
    ]

    fig = pyplot.figure(figsize=[4.8, 4.8])

    for glob_pattern, legend, color in zip(GLOBS, LEGENDS, COLORS):
        experiment_paths = glob(glob_pattern)
        if len(experiment_paths) == 0:
            raise ValueError(
                "Looks like there are no novelty experiments. Please see README.md on "+
                "running novelty search before plotting. Alternatively comment out call to `plot_novelty_results()`."
            )
        # Collect all lines and average over later
        xs = []
        ys = []
        for experiment_path in experiment_paths:
            # We just parse results from stdout file
            stdout_log = open(os.path.join(experiment_path, STDOUT_FILE), encoding="utf-8").read()
            # Take maximum fitness of each generation.
            # We have only one printout for one result
            mean_rewards = list(map(float, re.findall(REWARD_PATTERN, stdout_log)))
            iteration = []
            max_rewards = []
            # Plot elite results
            for mean_reward in mean_rewards:
                max_reward = mean_reward
                if len(max_rewards) > 0:
                    max_reward = max(max(max_rewards), max_reward)
                max_rewards.append(max_reward)
                iteration.append(len(max_rewards))

            xs.append(iteration)
            ys.append(max_rewards)

        # Average over curves
        xs = np.array(xs)
        ys = np.array(ys)
        average_x, average_y, std_y, lower_y, upper_y = interpolate_and_average(xs, ys, confidence_interval=True)

        pyplot.plot(average_x, average_y, c=color, label=legend)
        pyplot.fill_between(
            average_x,
            lower_y,
            upper_y,
            alpha=0.2,
            color=color,
            linewidth=0.0
        )

    pyplot.tick_params(axis='both', which='both', labelsize="x-large")
    pyplot.grid(alpha=0.2)
    pyplot.xlabel("Generation", fontsize="x-large")
    pyplot.ylabel("Average Return", fontsize="x-large")
    pyplot.legend(prop={"size": "large"})
    pyplot.tight_layout()
    pyplot.savefig("figures/novelty_results.pdf", bbox_inches="tight", pad_inches=0.0)


COLOR_CYCLE = pyplot.rcParams['axes.prop_cycle'].by_key()["color"]
LINESTYLE_CYCLE = ["solid", "dashed", "dotted"]


def color_linestyle_cycle(i):
    """A function that maps index to pair (color, linestyle) used for plotting"""
    color_index = i % len(COLOR_CYCLE)
    linestyle_index = i // len(COLOR_CYCLE)
    return COLOR_CYCLE[color_index], LINESTYLE_CYCLE[linestyle_index]


def plot_trust_region_results():
    """Plot/print results with trust-region experiments"""
    RESULTS_DIR = "trust_region_experiments"
    ENV = "DangerousPath-HalfMines-len25-dim5-v0"
    DIR_TEMPLATE = "{env}_lr_{learning_rate}"
    STDOUT_TEMPLATE = "{method}_repetition_{repetition}.txt"
    REWARD_PATTERN = r"ep_rew_mean[ ]*\| ([0-9\.\-]*)"
    TIMESTEP_PATTERN = r"total_timesteps[ ]*\| ([0-9]*)"

    NUM_REPETITIONS = 50
    # Forcing to high learning rate to force out the "constraint"
    LEARNING_RATES = ["1e-3"]

    METHODS = {
        "No Constraint": ["NoConstraint"],
        #"ClipPPO": ["ClipPPO"],
        "Max Action TV": ["PiMaxTV_{}".format(kl) for kl in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]],
        "Gaussian": ["Gaussian_{}".format(kl) for kl in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0]],
        "Supervector": ["Supervector_{}".format(kl) for kl in [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]],
    }

    # Match with novelty search
    COLORS = [
        "C0",
        "C1",
        "C2",
        "C3",
    ]

    fig = pyplot.figure(figsize=[4.8, 4.8])
    path_to_log_template = os.path.join(RESULTS_DIR, DIR_TEMPLATE, STDOUT_TEMPLATE)

    for (legend_name, experiment_names), color in zip(METHODS.items(), COLORS):
        # For each method, find the best learning curve (based on AUC)
        # and plot it
        best_x = None
        best_y = None
        best_y_upper = None
        best_y_lower = None
        best_std = None
        best_auc = None
        best_description = ""

        for learning_rate in LEARNING_RATES:
            for experiment_name in experiment_names:
                # Construct learning curves
                xs = []
                ys = []
                for repetition in range(NUM_REPETITIONS):
                    path_to_log = path_to_log_template.format(env=ENV, learning_rate=learning_rate, method=experiment_name, repetition=repetition)
                    log = open(path_to_log).read()
                    timesteps = list(map(float, re.findall(TIMESTEP_PATTERN, log)))
                    rewards = list(map(float, re.findall(REWARD_PATTERN, log)))
                    assert len(timesteps) == len(rewards)
                    xs.append(np.array(timesteps))
                    ys.append(np.array(rewards))

                average_x, average_y, std_y, lower_y, upper_y = interpolate_and_average(xs, ys, confidence_interval=True)
                # Uniformly spaced plotting points so can just take mean over ys
                auc = average_y.mean()
                if best_auc is None or auc > best_auc:
                    best_x = average_x
                    best_y = average_y
                    best_y_upper = upper_y
                    best_y_lower = lower_y
                    best_std = std_y
                    best_auc = auc
                    best_description = "{}. lr: {}".format(experiment_name, learning_rate)
        print("Best result for {}: {}".format(legend_name, best_description))
        pyplot.plot(best_x, best_y, label=legend_name, c=color)
        pyplot.fill_between(
            best_x,
            best_y_lower,
            best_y_upper,
            alpha=0.2,
            color=color,
            linewidth=0.0
        )
    pyplot.tick_params(axis='both', which='both', labelsize="x-large")
    pyplot.grid(alpha=0.2)
    pyplot.xlabel("Environment steps", fontsize="x-large")
    pyplot.ylabel("Average return", fontsize="x-large")
    pyplot.legend(prop={"size": "large"})
    pyplot.tight_layout()
    pyplot.savefig("figures/trust_region_results.pdf", bbox_inches="tight", pad_inches=0.0)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    plot_visual_abstract()
    plot_tsnes()
    plot_metric_results()
    plot_novelty_results()
    plot_trust_region_results()
