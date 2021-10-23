# Experiments for studying difference
# in quality when describing policies
from itertools import product

import numpy as np
from matplotlib import pyplot

import gridworld

EPS = np.finfo(np.float).eps


def create_first_up_then_right_policy():
    """Create and return policy that first goes always up, then to right.

    This is one optimal policy for corner-goal gridworld
    """
    pi = np.zeros((gridworld.GRIDS_PER_AXIS, gridworld.GRIDS_PER_AXIS, 4))
    # Note that these gridworlds are in x/y coordinates
    pi[:, :-1, gridworld.Direction.UP] = 1.0
    pi[:, -1, gridworld.Direction.RIGHT] = 1.0

    return pi


def create_first_right_then_up_policy():
    """Create and return policy that first goes always right and then up.

    This is one optimal policy for corner-goal gridworld
    """
    pi = np.zeros((gridworld.GRIDS_PER_AXIS, gridworld.GRIDS_PER_AXIS, 4))
    # Note that these gridworlds are in x/y coordinates
    pi[:-1, :, gridworld.Direction.RIGHT] = 1.0
    pi[-1, :, gridworld.Direction.UP] = 1.0

    return pi


def create_diagonal_policy():
    """Create and return policy that travels on the diagonal (first up, then right)

    This is one optimal policy for corner-goal gridworld
    """
    pi = np.zeros((gridworld.GRIDS_PER_AXIS, gridworld.GRIDS_PER_AXIS, 4))
    # Note that these gridworlds are in x/y coordinates
    # bluuh thanks to me using x/y notation, this is now messier...
    for x, y in product(range(gridworld.GRIDS_PER_AXIS), range(gridworld.GRIDS_PER_AXIS)):
        if x < y:
            pi[x, y, gridworld.Direction.RIGHT] = 1.0
        else:
            pi[x, y, gridworld.Direction.UP] = 1.0
    return pi


def experiment_compare_policies():
    N_FOR_SAMPLING = 10000
    PLOT_KWARGS = {
        "linewidth": 5
    }
    LEGEND_KWARGS = {
        "fontsize": 19
    }

    fig, axs = pyplot.subplots(
        nrows=2,
        ncols=3,
        figsize=[4.8 * 3, 4.8 * 2]
    )

    # ------------------------------------------------------
    # First illustration:
    # Show how actions alone get worse when stochasticity
    # increases
    # ------------------------------------------------------
    r, d = gridworld.get_ur_corner_goal_r_and_d()
    # Two different policies:
    # 1) First all up, and on top row to the right
    # 2) First all right, then on final colum all top
    pi_up_right = create_first_up_then_right_policy()
    pi_right_up = create_first_right_then_up_policy()

    p_random_actions = [i/10 for i in range(11)]
    state_p_distances = []
    pi_distances = []
    return_distances = []

    for p_random_action in p_random_actions:
        p_state_up_right = gridworld.estimate_state_p(0, 0, pi_up_right, d, N_FOR_SAMPLING, p_random_action=p_random_action)
        p_state_right_up = gridworld.estimate_state_p(0, 0, pi_right_up, d, N_FOR_SAMPLING, p_random_action=p_random_action)
        up_right_steps_till_goal = gridworld.estimate_steps_to_goal_monte_carlo(0, 0, pi_up_right, d, N_FOR_SAMPLING, p_random_action=p_random_action)
        right_up_steps_till_goal = gridworld.estimate_steps_to_goal_monte_carlo(0, 0, pi_right_up, d, N_FOR_SAMPLING, p_random_action=p_random_action)
        # Return/reward -> higher the faster agent reaches goal
        up_right_return = 1 - (up_right_steps_till_goal / gridworld.EPISODE_TIMEOUT)
        right_up_return = 1 - (right_up_steps_till_goal / gridworld.EPISODE_TIMEOUT)

        state_p_distances.append(np.abs(p_state_right_up - p_state_up_right).sum())
        return_distances.append(abs(up_right_return - right_up_return))
        pi_distances.append(np.abs(pi_up_right - pi_right_up).sum())

    p_random_actions = np.array(p_random_actions) * 100
    state_p_distances = np.array(state_p_distances)
    pi_distances = np.array(pi_distances)
    return_distances = np.array(return_distances)

    axs[1, 2].plot(p_random_actions, state_p_distances, label="", **PLOT_KWARGS)
    axs[1, 1].plot(p_random_actions, pi_distances, **PLOT_KWARGS)
    axs[1, 0].plot(p_random_actions, return_distances, **PLOT_KWARGS)

    # Try drawing a gridworld
    gridworld_ax = axs[0, 0]
    gridworld_ax.set_title("Distinct policies", fontsize=20.0)
    gridworld_ax.grid(alpha=1.0, linewidth=2, c="k")
    gridworld_ax.set_xlim(-0.02, 5.02)
    gridworld_ax.set_ylim(-0.02, 5.02)
    gridworld_ax.scatter([0.5], [0.5], marker="D", s=1500, c="cyan", alpha=0.5)
    gridworld_ax.text(0.05, 0.05, "Start", ha="left", va="bottom", fontsize="medium")

    up_right_arrow_x = np.arange(5) + 0.5
    up_right_arrow_x[-1] += 0.07
    up_right_arrow_y = np.arange(5) + 0.5
    up_right_arrow_y[-1] -= 0.07
    up_right_x = np.zeros((5, 5))
    up_right_x[-1, :] = 1
    up_right_y = np.ones((5, 5))
    up_right_y[-1, :] = 0
    up_right_y[[-1], [-1]] = 0
    up_right_x[[-1], [-1]] = 0

    right_up_arrow_x = np.arange(5) + 0.5
    right_up_arrow_x[-1] -= 0.07
    right_up_arrow_y = np.arange(5) + 0.5
    right_up_arrow_y[-1] += 0.07
    right_up_x = np.ones((5, 5))
    right_up_x[:, -1] = 0
    right_up_y = np.zeros((5, 5))
    right_up_y[:, -1] = 1
    right_up_y[[-1], [-1]] = 0
    right_up_x[[-1], [-1]] = 0

    gridworld_ax.quiver(up_right_arrow_x, up_right_arrow_y, up_right_x, up_right_y, scale=12, width=0.01, color="blue")
    gridworld_ax.quiver(right_up_arrow_x, right_up_arrow_y, right_up_x, right_up_y, scale=12, width=0.01, color="green")

    gridworld_ax.scatter([4.5], [4.5], marker="*", s=1200, c="orange")
    gridworld_ax.text(4.95, 4.95, "Goal", ha="right", va="top", fontsize="medium")

    gridworld_ax.set_xticklabels([])
    gridworld_ax.set_yticklabels([])
    gridworld_ax.tick_params(length=0)
    gridworld_ax.spines['top'].set_visible(False)
    gridworld_ax.spines['right'].set_visible(False)
    gridworld_ax.spines['bottom'].set_visible(False)
    gridworld_ax.spines['left'].set_visible(False)


    # ------------------------------------------------------
    # Second illustration:
    # Show how a small change in actions can lead to
    # significant changes in complete behaviour
    # ------------------------------------------------------
    r, d = gridworld.get_ur_corner_goal_r_and_d()
    # Come up with an obstacle mask.
    # There is a wall vertically in the middle, with only spot
    # to go through in the very middle
    obstacle_mask = np.zeros((gridworld.N, gridworld.N), dtype=np.int)
    obstacle_mask[2, :] = 1
    obstacle_mask[2, 2] = 0
    # Two different policies:
    # 1) One "optimal" one
    # 2) Second not-so-optimal one that just misses by one step
    pi_optimal = create_first_up_then_right_policy()
    # Go down towards "middle-path"
    pi_optimal[0:2, 3:5, gridworld.Direction.UP] = 0.0
    pi_optimal[0:2, 3:5, gridworld.Direction.DOWN] = 1.0
    pi_optimal[0:3, 2, gridworld.Direction.UP] = 0.0
    pi_optimal[0:3, 2, gridworld.Direction.RIGHT] = 1.0
    # Create new policy that is almost optimal with one key difference
    pi_not_so_optimal = pi_optimal.copy()
    # Do not go into the "doorway"
    pi_not_so_optimal[1, 2, gridworld.Direction.RIGHT] = 0.0
    pi_not_so_optimal[1, 2, gridworld.Direction.UP] = 1.0

    p_random_actions = [i/10 for i in range(11)]
    state_p_distances = []
    pi_distances = []
    return_distances = []  

    for p_random_action in p_random_actions:
        p_state_optimal = gridworld.estimate_state_p(0, 0, pi_optimal, d, N_FOR_SAMPLING, p_random_action=p_random_action, obstacle_mask=obstacle_mask)
        p_state_not_so_optimal = gridworld.estimate_state_p(0, 0, pi_not_so_optimal, d, N_FOR_SAMPLING, p_random_action=p_random_action, obstacle_mask=obstacle_mask)
        optimal_steps_till_goal = gridworld.estimate_steps_to_goal_monte_carlo(0, 0, pi_optimal, d, N_FOR_SAMPLING, p_random_action=p_random_action, obstacle_mask=obstacle_mask)
        not_so_optimal_steps_till_goal = gridworld.estimate_steps_to_goal_monte_carlo(0, 0, pi_not_so_optimal, d, N_FOR_SAMPLING, p_random_action=p_random_action, obstacle_mask=obstacle_mask)
        # Return/reward -> higher the faster agent reaches goal
        optimal_return = 1 - (optimal_steps_till_goal / gridworld.EPISODE_TIMEOUT)
        not_so_optimal_return = 1 - (not_so_optimal_steps_till_goal / gridworld.EPISODE_TIMEOUT)

        state_p_distances.append(np.abs(p_state_not_so_optimal - p_state_optimal).sum())
        return_distances.append(abs(optimal_return - not_so_optimal_return))
        pi_distances.append(np.abs(pi_optimal - pi_not_so_optimal).sum())

    p_random_actions = np.array(p_random_actions) * 100
    state_p_distances = np.array(state_p_distances)
    pi_distances = np.array(pi_distances)
    return_distances = np.array(return_distances)

    results_ax = axs[1, 1]
    axs[1, 2].plot(p_random_actions, state_p_distances, label="", linestyle="--", **PLOT_KWARGS)
    axs[1, 1].plot(p_random_actions, pi_distances, linestyle="--", **PLOT_KWARGS)
    axs[1, 0].plot(p_random_actions, return_distances, linestyle="--", **PLOT_KWARGS)

    # Try drawing a gridworld
    gridworld_ax = axs[0, 1]
    gridworld_ax.set_title("Doorway", fontsize=20.0)
    gridworld_ax.grid(alpha=1.0, linewidth=2, c="k")
    gridworld_ax.set_xlim(-0.02, 5.02)
    gridworld_ax.set_ylim(-0.02, 5.02)
    gridworld_ax.scatter([0.5], [0.5], marker="D", s=1500, c="cyan", alpha=0.5)
    wall_square = list(zip([0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]))
    # A red outline to box we are about to higlight (facecolors did not want to work)
    gridworld_ax.scatter([1.01], [2.01], marker=wall_square, s=13000, c="red")
    gridworld_ax.scatter([1.06], [2.055], marker=wall_square, s=10200, c="white")

    optimal_arrow_x = np.repeat(np.arange(5)[None] + 0.5, 5, axis=0)
    optimal_arrow_x += 0.07
    optimal_arrow_x[2, 0:3] -= 0.07
    optimal_arrow_x[-1, 2:] -= 0.07
    optimal_arrow_y = np.repeat(np.arange(5)[:, None] + 0.5, 5, axis=1)
    optimal_arrow_y[2, 0:3] -= 0.07
    optimal_arrow_y[2, 1] += 0.07
    optimal_arrow_y[-1, 2:] -= 0.07
    optimal_x = np.zeros((5, 5))
    optimal_x[3:5, 0:2] = 0
    optimal_x[2, 0:3] = 1
    optimal_x[-1, 2:] = 1
    optimal_y = np.ones((5, 5))
    optimal_y[3:5, 0:2] = -1
    optimal_y[2, 0:3] = 0
    optimal_y[-1, 2:] = 0
    optimal_y[[-1], [-1]] = 0
    optimal_x[[-1], [-1]] = 0

    not_so_optimal_arrow_x = np.repeat(np.arange(5)[None] + 0.5, 5, axis=0)
    not_so_optimal_arrow_x -= 0.07
    not_so_optimal_arrow_x[2, 0:3] += 0.07
    not_so_optimal_arrow_x[-1, 2:] += 0.07
    not_so_optimal_arrow_y = np.repeat(np.arange(5)[:, None] + 0.5, 5, axis=1)
    not_so_optimal_arrow_y[2, 0:3] += 0.07
    not_so_optimal_arrow_y[2, 1] -= 0.07
    not_so_optimal_arrow_y[-1, 2:] += 0.07
    not_so_optimal_x = np.zeros((5, 5))
    not_so_optimal_x[3:5, 0:2] = 0
    not_so_optimal_x[2, 0:3] = 1
    not_so_optimal_x[2, 1] = 0
    not_so_optimal_x[-1, 2:] = 1
    not_so_optimal_y = np.ones((5, 5))
    not_so_optimal_y[3:5, 0:2] = -1
    not_so_optimal_y[2, 0:3] = 0
    not_so_optimal_y[2, 1] = 1
    not_so_optimal_y[-1, 2:] = 0
    not_so_optimal_y[[-1], [-1]] = 0
    not_so_optimal_x[[-1], [-1]] = 0

    gridworld_ax.quiver(optimal_arrow_x, optimal_arrow_y, optimal_x, optimal_y, scale=12, width=0.01, color="blue")
    gridworld_ax.quiver(not_so_optimal_arrow_x, not_so_optimal_arrow_y, not_so_optimal_x, not_so_optimal_y, scale=12, width=0.01, color="green")

    gridworld_ax.scatter([4.5], [4.5], marker="*", s=1200, c="orange")

    # Obstacles
    gridworld_ax.scatter([2.0, 2.0, 2.0, 2.0], [0.0, 1.0, 3.0, 4.0], marker=wall_square, s=13000, c="gray", edgecolor="none")

    gridworld_ax.set_xticklabels([])
    gridworld_ax.set_yticklabels([])
    gridworld_ax.tick_params(length=0)
    gridworld_ax.spines['top'].set_visible(False)
    gridworld_ax.spines['right'].set_visible(False)
    gridworld_ax.spines['bottom'].set_visible(False)
    gridworld_ax.spines['left'].set_visible(False)

    # ------------------------------------------------------
    # Third illustration:
    # Show how unreachable states affect action measurements
    # ------------------------------------------------------
    r, d = gridworld.get_ur_corner_goal_r_and_d()
    # Come up with an obstacle mask.
    # Policies move up then right, block everything else off
    obstacle_mask = np.zeros((gridworld.N, gridworld.N), dtype=np.int)
    obstacle_mask[1:5, 3] = 1
    obstacle_mask[1, 0:3] = 1

    pi_optimal1 = create_first_up_then_right_policy()
    pi_optimal2 = create_first_up_then_right_policy()
    # Mix them up in the unknown region (they would go to different directions)
    pi_optimal1[2:5, 0:3] = 0
    pi_optimal1[2:5, 0:3, gridworld.Direction.UP] = 1.0
    pi_optimal2[2:5, 0:3] = 0
    pi_optimal2[2:5, 0:3, gridworld.Direction.RIGHT] = 1.0

    p_random_actions = [i/10 for i in range(11)]
    state_p_distances = []
    pi_distances = []
    return_distances = []  

    for p_random_action in p_random_actions:
        p_state_optimal1 = gridworld.estimate_state_p(0, 0, pi_optimal, d, N_FOR_SAMPLING, p_random_action=p_random_action, obstacle_mask=obstacle_mask)
        p_state_optimal2 = gridworld.estimate_state_p(0, 0, pi_not_so_optimal, d, N_FOR_SAMPLING, p_random_action=p_random_action, obstacle_mask=obstacle_mask)
        optimal1_steps_till_goal = gridworld.estimate_steps_to_goal_monte_carlo(0, 0, pi_optimal, d, N_FOR_SAMPLING, p_random_action=p_random_action, obstacle_mask=obstacle_mask)
        optimal2_steps_till_goal = gridworld.estimate_steps_to_goal_monte_carlo(0, 0, pi_not_so_optimal, d, N_FOR_SAMPLING, p_random_action=p_random_action, obstacle_mask=obstacle_mask)
        # Return/reward -> higher the faster agent reaches goal
        optimal_return = 1 - (optimal1_steps_till_goal / gridworld.EPISODE_TIMEOUT)
        not_so_optimal_return = 1 - (optimal2_steps_till_goal / gridworld.EPISODE_TIMEOUT)

        state_p_distances.append(np.abs(p_state_optimal2 - p_state_optimal1).sum())
        return_distances.append(abs(optimal_return - not_so_optimal_return))
        pi_distances.append(np.abs(pi_optimal1 - pi_optimal2).sum())

    p_random_actions = np.array(p_random_actions) * 100
    state_p_distances = np.array(state_p_distances)
    pi_distances = np.array(pi_distances)
    return_distances = np.array(return_distances)

    results_ax = axs[1, 1]
    axs[1, 2].plot(p_random_actions, state_p_distances, label="", linestyle=":", **PLOT_KWARGS)
    axs[1, 1].plot(p_random_actions, pi_distances, linestyle=":", **PLOT_KWARGS)
    axs[1, 0].plot(p_random_actions, return_distances, linestyle=":", **PLOT_KWARGS)

    # Try drawing a gridworld
    gridworld_ax = axs[0, 2]
    gridworld_ax.set_title("Unreachable states", fontsize=20.0)
    gridworld_ax.grid(alpha=1.0, linewidth=2, c="k")
    gridworld_ax.set_xlim(-0.02, 5.02)
    gridworld_ax.set_ylim(-0.02, 5.02)
    gridworld_ax.scatter([0.5], [0.5], marker="D", s=1500, c="cyan", alpha=0.5)

    optimal1_arrow_x = np.repeat(np.arange(5)[None] + 0.5, 5, axis=0)
    optimal1_arrow_x[:, 0] += 0.07
    optimal1_arrow_x[-1, 0] -= 0.07
    optimal1_arrow_y = np.repeat(np.arange(5)[:, None] + 0.5, 5, axis=1)
    optimal1_arrow_y[-1, :] -= 0.07
    optimal1_x = np.zeros((5, 5))
    optimal1_x[-1, :] = 1
    optimal1_y = np.ones((5, 5))
    optimal1_y[-1, :] = 0
    optimal1_y[[-1], [-1]] = 0
    optimal1_x[[-1], [-1]] = 0

    optimal2_arrow_x = np.repeat(np.arange(5)[None] + 0.5, 5, axis=0)
    optimal2_arrow_x[:, 0] -= 0.07
    optimal2_arrow_x[-1, 0] += 0.07
    optimal2_arrow_y = np.repeat(np.arange(5)[:, None] + 0.5, 5, axis=1)
    optimal2_arrow_y[-1, :] += 0.07
    optimal2_x = np.ones((5, 5))
    optimal2_x[:, 0] = 0
    optimal2_x[-1, 0] = 1
    optimal2_y = np.zeros((5, 5))
    optimal2_y[:, 0] = 1
    optimal2_y[-1, 0] = 0
    optimal2_y[[-1], [-1]] = 0
    optimal2_x[[-1], [-1]] = 0

    gridworld_ax.quiver(optimal1_arrow_x, optimal1_arrow_y, optimal1_x, optimal1_y, scale=12, width=0.01, color="blue")
    gridworld_ax.quiver(optimal2_arrow_x, optimal2_arrow_y, optimal2_x, optimal2_y, scale=12, width=0.01, color="green")

    gridworld_ax.scatter([4.5], [4.5], marker="*", s=1200, c="orange")

    # Obstacles
    wall_square = list(zip([0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]))
    gridworld_ax.scatter([1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0], marker=wall_square, s=13000, c="gray", edgecolor="none")

    gridworld_ax.set_xticklabels([])
    gridworld_ax.set_yticklabels([])
    gridworld_ax.tick_params(length=0)
    gridworld_ax.spines['top'].set_visible(False)
    gridworld_ax.spines['right'].set_visible(False)
    gridworld_ax.spines['bottom'].set_visible(False)
    gridworld_ax.spines['left'].set_visible(False)

    # Legends and stuff for results
    return_results_ax = axs[1, 0]
    return_results_ax.legend(("Distinct", "Doorway", "Unreach."), **LEGEND_KWARGS)
    return_results_ax.set_ylabel("Difference between policies measured by...", fontsize="xx-large")
    return_results_ax.tick_params(labelsize="xx-large")
    return_results_ax.set_xticks([0, 100])
    return_results_ax.set_title("... returns", fontsize=20.0)

    action_results_ax = axs[1, 1]
    action_results_ax.set_xlabel("Prob. of random action (%)", fontsize="xx-large")
    action_results_ax.tick_params(labelsize="xx-large")
    action_results_ax.set_xticks([0, 100])
    action_results_ax.set_title("... taken actions", fontsize=20.0)

    state_results_ax = axs[1, 2]
    state_results_ax.tick_params(labelsize="xx-large")
    state_results_ax.set_xticks([0, 100])
    state_results_ax.set_title("... visited states", fontsize=20.0)

    pyplot.tight_layout()
    pyplot.savefig("gridworld_illustration.pdf")

if __name__ == "__main__":
    experiment_compare_policies()
