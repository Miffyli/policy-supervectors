# A simple 2D continuous environment designed
# to mimic the deceptive rewards scenario
# of https://arxiv.org/abs/1712.06560.
# Agent always starts in one location, goal is to travel as far as possible in y axis.
# Similar env was also used by "Learning to score behaviours..." https://arxiv.org/abs/1906.04349
#
# All coordinates are in (x, y), x going from left-to-right and y down-to-up
#
import math
import random

import numpy as np
import gym
# Fix for older gym versions
import gym.spaces

EPS = 1e-4

# Number of grids in the [-1, 1] area. Note that player
# can go beyond this!
# We divide to grids to have walls
GRIDS_PER_AXIS = 25

TIMEOUT = 100

PLAYER_START_X = 0.0
# Player starts just barely inside the wall
PLAYER_START_Y = -0.9

# How much player moves per step per axis
PLAYER_STEP_SIZE = 0.05


class DeceptivePointEnv(gym.Env):
    """
    Simple 2D continuous environment where player needs
    to travel as far in y as possible.

    The trick is that there is a wall separating
    goal and player from each other.
    """

    def __init__(self,):
        super().__init__()

        # 1 where there is a wall
        self.wall_mask = np.zeros((GRIDS_PER_AXIS, GRIDS_PER_AXIS), dtype=np.int32)
        # Horizontal bar.
        # Make sure these walls do not touch the edges of the matrix
        barrier_x_start = (GRIDS_PER_AXIS // 8) * 1
        barrier_x_end = (GRIDS_PER_AXIS // 8) * 7
        barrier_y = (GRIDS_PER_AXIS // 5) * 2
        self.wall_mask[barrier_x_start:barrier_x_end + 1, barrier_y] = 1
        # Vertical edges that "capture" the player
        self.wall_mask[barrier_x_start, 1:barrier_y] = 1
        self.wall_mask[barrier_x_end, 1:barrier_y] = 1

        self.observation_space = gym.spaces.Box(-10, 10, shape=(2,))
        # Angle where to go, mapped to [-1, 1] convenience for the network
        self.action_space = gym.spaces.Box(-1, 1, shape=(1,))

        self.player_x = None
        self.player_y = None

    def step(self, action):
        old_x = self.player_x
        old_y = self.player_y

        action_in_rads = ((action + 1) / 2) * 2 * math.pi
        self.player_x += math.cos(action_in_rads) * PLAYER_STEP_SIZE
        self.player_y += math.sin(action_in_rads) * PLAYER_STEP_SIZE

        # Add small epsilons to make sure rounding later will be fine
        discrete_x = min(1 - EPS, max(-1 + EPS, self.player_x))
        discrete_y = min(1 - EPS, max(-1 + EPS, self.player_y))

        # Discretize so we can check for walls
        discrete_x = int((discrete_x + 1) / 2 * GRIDS_PER_AXIS)
        discrete_y = int((discrete_y + 1) / 2 * GRIDS_PER_AXIS)
        if self.wall_mask[discrete_x, discrete_y] == 1:
            # Revert
            self.player_x = old_x
            self.player_y = old_y

        # Done is done by TimeoutWrapper
        done = False
        # Reward is how much we have traveled in y-axis
        reward = self.player_y - old_y

        obs = np.array([self.player_x, self.player_y], dtype=np.float32)

        return obs, reward, done, {}

    def reset(self):
        self.player_x = PLAYER_START_X
        self.player_y = PLAYER_START_Y

        obs = np.array([self.player_x, self.player_y], dtype=np.float32)

        return obs

    def close(self):
        # Nothing to clean up, really
        pass

    def debug_print(self):
        # An array that shows where walls are (1), player (2) and goal (3)
        print_array = self.wall_mask.copy()
        discrete_x = min(1 - EPS, max(-1 + EPS, self.player_x))
        discrete_x = int((discrete_x + 1) / 2 * GRIDS_PER_AXIS)
        discrete_y = min(1 - EPS, max(-1 + EPS, self.player_y))
        discrete_y = int((discrete_y + 1) / 2 * GRIDS_PER_AXIS)
        print_array[discrete_x, discrete_y] = 2
        print(print_array)


# Nice naughty thing here but oh well
ENV_NAME = "DeceptivePointEnv-v0"
if ENV_NAME not in [env_spec.id for env_spec in gym.envs.registry.all()]:
    gym.envs.register(
        id="DeceptivePointEnv-v0",
        entry_point="point_env:DeceptivePointEnv",
        max_episode_steps=TIMEOUT,
    )


if __name__ == "__main__":
    # Manual debugging that things work
    # and also obtain the reward threshold for going
    # up all the time (hitting the wall)
    env = gym.make("DeceptivePointEnv-v0")
    env.reset()
    episode_reward = 0
    num_steps = 0
    go_all_right = False
    if go_all_right:
        for i in range(TIMEOUT):
            obs, r, d, info = env.step(-0.5)
            episode_reward += r
            num_steps += 1
    else:
        # Escape
        for i in range(17):
            obs, r, d, info = env.step(0.1)
            episode_reward += r
            num_steps += 1
            env.debug_print()
            input()
        # Go right
        for i in range(80):
            obs, r, d, info = env.step(-0.5)
            episode_reward += r
            num_steps += 1
            env.debug_print()
            input()
    print("Reward: {}".format(episode_reward))
    print("Episode length: {}".format(num_steps))
