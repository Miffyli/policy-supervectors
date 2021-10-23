# A simple MDP where agent has to traverse a specific path
# in gridworld - wrong action will throw player back to start or do nothing.
# Player is rewarded for reaching new maximum length in the episode.
#
# State is represented by a positive ndim vector that tells
# where the player is. This is designed to mimic coordinate-systems
# and also deliberately confuse networks (e.g. might think higher value
# on axis 0 means we should take one specific action always)
#
import random

import numpy as np
import gym
# Fix for older gym versions
import gym.spaces


def generate_path(game_length: int, ndim: int, num_mines: int, seed: int = 42) -> np.ndarray:
    """Generate the path player has to follow.

    Args:
        game_length: Length of the path to generate
        ndim: Number of dimensions in the environment
        num_mines: Number of mines per step
        seed: Seed used to generate path
    Returns:
        path: List of ints, representing actions player should take in each state.
        mines: List of List of ints, representing which actions are mines in each state.
    """
    path = []
    mines = []
    gen = np.random.default_rng(seed)
    for i in range(game_length):
        action_ordering = gen.permutation(ndim)
        # First item goes to path, next num_mines go to mines
        path.append(action_ordering[0].item())
        mines.append(action_ordering[1:1 + num_mines].tolist())
    return path, mines


class DangerousPathEnv(gym.Env):
    """
    A N-dimensional environment where player has to choose
    the exact correct action at any given location (follow
    a very specific path). Otherwise game terminates or player stays
    still, depending on if they hit a mine or not.

    If `discrete_obs` is True, observation space tells location
    of player in path. If False, uses continuous observations
    that tell coordinate-like information of location of the player.

    `mine_ratio` specifies the amount of mines (terminal states) versus
    no-move moves per state.
    """

    def __init__(
        self,
        game_length=100,
        ndim=2,
        seed=42,
        discrete_obs=False,
        random_action_p=0.0,
        mine_ratio=1.0
    ):
        super().__init__()

        self.game_length = game_length
        self.ndim = ndim
        self.mine_ratio = mine_ratio
        self.num_mines_per_step = np.floor(ndim * mine_ratio)
        self.path, self.mines = generate_path(game_length, ndim, seed)

        # Emperically found to be a necessary adjustment
        self.step_size = 1.0
        self.discrete_obs = discrete_obs
        self.random_action_p = random_action_p

        if discrete_obs:
            self.observation_space = gym.spaces.Discrete(n=self.game_length)
        else:
            self.observation_space = gym.spaces.Box(0, 1, shape=(self.ndim,))
        self.action_space = gym.spaces.Discrete(n=self.ndim)

        self.path_location = 0
        self.max_path_location = 0
        self.num_steps = 0
        self.player_location = np.zeros((self.ndim,))

    def step(self, action):
        if self.random_action_p > 0.0 and random.random() < self.random_action_p:
            action = self.action_space.sample()
        done = False
        reward = 0

        action = int(action)
        if action == self.path[self.path_location]:
            # You chose wisely
            self.path_location += 1
            # Only reward progressing once
            if self.path_location > self.max_path_location:
                reward = 1
            self.max_path_location += 1
            # Small step sizes
            self.player_location[action] += self.step_size
            if self.path_location == (self.game_length - 1):
                done = True
        else:
            # You chose poorly
            reward = 0
            if action in self.mines[self.path_location]:
                # You chose very poorly, back to start
                self.path_location = 0
                self.player_location = np.zeros((self.ndim,))

        self.num_steps += 1
        if self.num_steps >= self.game_length:
            done = True

        return self.path_location if self.discrete_obs else self.player_location, reward, done, {}

    def reset(self):
        self.path_location = 0
        self.max_path_location = 0
        self.num_steps = 0
        self.player_location = np.zeros((self.ndim,))
        return self.path_location if self.discrete_obs else self.player_location

    def seed(self, seed):
        self.path, self.mines = generate_path(self.game_length, self.ndim, seed)

