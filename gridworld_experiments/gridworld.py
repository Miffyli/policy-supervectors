# Simple gridworld functions
# for sampling trajectories etc.

# Actions are in order up, left, down, right
# NOTE: All matrices are xy coordinate style (not yx as normally with matrices),
#       with (0, 0) being lower left corner
import random
import enum
from itertools import product

import numpy as np
from tqdm import tqdm
from numba import njit

GRIDS_PER_AXIS = 5
# Just an alias
N = GRIDS_PER_AXIS
EPISODE_TIMEOUT = 500

# Discount factor
GAMMA = 0.99

DEFAULT_RANDOM_ACTION_P = 0.0
DEFAULT_OBSTACLE_MASK = np.zeros((N, N), dtype=np.int)


class Direction(enum.IntEnum):
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3


@njit
def play_step(x, y, pi, p_random_action, obstacle_mask):
    """Play one step forwards. Update player position in place.

    Args:
        x, y (int): Current location of the player
        pi (np.ndarray): (N, N, 4) array that tells probabilities
            of actions.
        p_random_action (float): Probability of taking a random action
        obstacle_mask (np.ndarray): (N, N) integer mask that tells if
            grid is an obstacle or not (1 = obstacle)
    Returns:
        x, y 
    """
    location_pi = pi[x, y]
    action = None
    if random.random() <= p_random_action:
        # Random action
        # TODO does not use the enum...
        action = random.randint(0, 3)
    else:
        # Pick action based on the probabilities of pi
        random_value = random.random()
        if random_value <= location_pi[0]:
            # Up
            action = Direction.UP
        elif random_value <= location_pi[0] + location_pi[1]:
            # Left
            action = Direction.LEFT
        elif random_value <= location_pi[0] + location_pi[1] + location_pi[2]:
            # Right
            action = Direction.RIGHT
        else:
            # Down
            action = Direction.DOWN

    original_x = x
    original_y = y
    if action == Direction.UP:
        # Up
        y += 1
    elif action == Direction.LEFT:
        # Left
        x -= 1
    elif action == Direction.RIGHT:
        # Right
        y -= 1
    else:
        # Down
        x += 1
    # Clip boundaries and prevent moving onto obstacles
    if obstacle_mask[x, y] == 1 or x == GRIDS_PER_AXIS or x == -1 or y == GRIDS_PER_AXIS or y == -1:
        x = original_x
        y = original_y

    return x, y, action


@njit
def estimate_steps_to_goal_monte_carlo(x, y, pi, d, n, p_random_action=DEFAULT_RANDOM_ACTION_P, obstacle_mask=DEFAULT_OBSTACLE_MASK):
    """Estimate how many steps till goal starting from given state.

    Args:
        x, y (int): State to estimate value for
        pi (np.ndarray): (N, N, 4) array that tells probabilities
            of actions.
        d (np.ndarray): (N, N) array that tells if that state
            is terminal (landing on it will end the game).
        n (int): Number of trajectories to sample
        p_random_action (float): Probability of taking a random action
        obstacle_mask (np.ndarray): (N, N) integer mask that tells if
            grid is an obstacle or not (1 = obstacle)
    Returns:
        float representing the value estimate for the point
    """
    initial_x = x
    initial_y = y
    number_of_steps_estimate = 0

    for i in range(n):
        # Play one game
        x = initial_x
        y = initial_y
        done = d[x, y]
        episode_step_num = 0
        while not done and episode_step_num < EPISODE_TIMEOUT:
            x, y, action = play_step(x, y, pi, p_random_action, obstacle_mask)
            episode_step_num += 1
            if d[x, y] == 1:
                done = True
        number_of_steps_estimate += episode_step_num / n

    return number_of_steps_estimate


@njit
def estimate_state_p_numba(x, y, pi, d, n, state_p, p_random_action=DEFAULT_RANDOM_ACTION_P, obstacle_mask=DEFAULT_OBSTACLE_MASK):
    """See `estimate_state_p`"""
    initial_x = x
    initial_y = y
    total_num_steps = 0
    # Reset to zeros
    state_p *= 0

    for i in range(n):
        # Play one game
        x = initial_x
        y = initial_y
        state_p[x, y] += 1
        done = d[x, y]
        episode_num_steps = 0
        while not done and episode_num_steps < EPISODE_TIMEOUT:
            x, y, action = play_step(x, y, pi, p_random_action, obstacle_mask)
            state_p[x, y] += 1
            total_num_steps += 1
            episode_num_steps += 1
            if d[x, y] == 1:
                done = True

    state_p /= float(total_num_steps)


@njit
def estimate_state_p(x, y, pi, d, n, p_random_action=DEFAULT_RANDOM_ACTION_P, obstacle_mask=DEFAULT_OBSTACLE_MASK):
    """Estimate probability of visiting each state, with x/y being starting point

    Args:
        x, y (int): Starting coordinates
        pi (np.ndarray): (N, N, 4) array that tells probabilities
            of actions.
        d (np.ndarray): (N, N) array that tells if that state
            is terminal (landing on it will end the game).
        n (int): Number of trajectories to sample
        obstacle_mask (np.ndarray): (N, N) integer mask that tells if
            grid is an obstacle or not (1 = obstacle)
    
    Returns:
        np.ndarray of shape (N, N), with probability of visiting that state
        (over all trajectories).
    """
    state_p = np.zeros((GRIDS_PER_AXIS, GRIDS_PER_AXIS))
    estimate_state_p_numba(x, y, pi, d, n, state_p, p_random_action, obstacle_mask)

    return state_p


@njit
def estimate_state_action_p_numba(x, y, pi, d, n, state_action_p, p_random_action=DEFAULT_RANDOM_ACTION_P, obstacle_mask=DEFAULT_OBSTACLE_MASK):
    """See `estimate_state_action_p`"""
    initial_x = x
    initial_y = y
    total_num_steps = 0
    # Reset to zeros
    state_action_p *= 0

    for i in range(n):
        # Play one game
        x = initial_x
        y = initial_y
        state_action_p[x, y] += 1
        done = d[x, y]
        episode_num_steps = 0
        while not done and episode_num_steps < EPISODE_TIMEOUT:
            x, y, action = play_step(x, y, pi, p_random_action, obstacle_mask)
            state_action_p[x, y, action] += 1
            total_num_steps += 1
            episode_num_steps += 1
            if d[x, y] == 1:
                done = True

    state_action_p /= float(total_num_steps)


@njit
def estimate_state_action_p(x, y, pi, d, n, p_random_action=DEFAULT_RANDOM_ACTION_P, obstacle_mask=DEFAULT_OBSTACLE_MASK):
    """Estimate probability of visiting each state-action pair, with x/y being starting point

    Args:
        x, y (int): Starting coordinates
        pi (np.ndarray): (N, N, 4) array that tells probabilities
            of actions.
        d (np.ndarray): (N, N) array that tells if that state
            is terminal (landing on it will end the game).
        n (int): Number of trajectories to sample
        state_p (np.ndarray): (N, N) array where to store the probabilities
            of encountering that state while agents play.
        p_random_action (float): Probability of taking a random action
        obstacle_mask (np.ndarray): (N, N) integer mask that tells if
            grid is an obstacle or not (1 = obstacle)
    Returns:
        np.ndarray of shape (N, N, 4), with probability of visiting that state-action.
    """
    state_action_p = np.zeros((GRIDS_PER_AXIS, GRIDS_PER_AXIS, 4))
    estimate_state_action_p_numba(x, y, pi, d, n, state_action_p, p_random_action, obstacle_mask)

    return state_action_p


def get_optimal_pi_for_ur_corner_goal():
    """Build and return an optimal policy for gridworld where
    upper-right corner is the only terminal/rewarding point.

    Actions in the ur-corner are uniform random.

    Returns:
        np.ndarray of shape (N, N, 4), representing an optimal policy.
    """
    pi = np.zeros((GRIDS_PER_AXIS, GRIDS_PER_AXIS, 4))

    # Easier to do all points manually, and cleaner to debug
    for x, y in product(range(N), range(N)):
        # Assign 1.0 to all optimal actions, normalize later
        if x < (N - 1):
            pi[x, y, Direction.RIGHT] = 1.0

        if y < (N - 1):
            pi[x, y, Direction.UP] = 1.0

    # Set corner pi to uniform random
    pi[N - 1, N - 1] = 1.0
    # Normalize
    pi /= np.linalg.norm(pi, ord=1, axis=2, keepdims=True)

    return pi


def get_randomly_initialized_pi():
    """Return a policy with random distributions for actions (NOT a "random" agent!).

    Returns:
        np.ndarray of shape (N, N, 4), representing a randomly initialized policy
    """
    pi = np.random.random((GRIDS_PER_AXIS, GRIDS_PER_AXIS, 4))
    # TODO randomize "spikiness" of the policy (we probably get
    # pretty flat policies now)
    pi /= np.linalg.norm(pi, ord=1, axis=2, keepdims=True)

    return pi


def get_center_goal_r_and_d():
    """Get reward and done matrices for gridworld where center goal is
    the only rewarding (and terminal) point.

    Returns:
        (np.ndarray, np.ndarray), both of shape (N, N) representing
        rewards and done states
    """
    middle_point = GRIDS_PER_AXIS // 2
    r = np.zeros((GRIDS_PER_AXIS, GRIDS_PER_AXIS))
    r[middle_point, middle_point] = 1
    d = np.zeros((GRIDS_PER_AXIS, GRIDS_PER_AXIS)).astype(np.uint8)
    d[middle_point, middle_point] = 1

    return r, d


def get_ur_corner_goal_r_and_d():
    """Get reward and done matrices for gridworld where upper right
    is the goal.

    Returns:
        (np.ndarray, np.ndarray), both of shape (N, N) representing
        rewards and done states
    """
    r = np.zeros((GRIDS_PER_AXIS, GRIDS_PER_AXIS))
    r[GRIDS_PER_AXIS - 1, GRIDS_PER_AXIS - 1] = 1
    d = np.zeros((GRIDS_PER_AXIS, GRIDS_PER_AXIS)).astype(np.uint8)
    d[GRIDS_PER_AXIS - 1, GRIDS_PER_AXIS - 1] = 1

    return r, d
