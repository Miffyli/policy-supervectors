from itertools import product
from gym.envs import register


# DangerousPath envs
NDIM_RANGE = [2, 5, 10, 20, 50]
PATH_LEN_RANGE = [10, 25, 50, 100]


for ndim, path_len in product(NDIM_RANGE, PATH_LEN_RANGE):
    register(
        id="DangerousPath-len{}-dim{}-v0".format(path_len, ndim),
        entry_point="envs.dangerous_path_env:DangerousPathEnv",
        kwargs={
            "game_length": path_len,
            "ndim": ndim,
        }
    )

    register(
        id="DangerousPath-Discrete-len{}-dim{}-v0".format(path_len, ndim),
        entry_point="envs.dangerous_path_env:DangerousPathEnv",
        kwargs={
            "game_length": path_len,
            "ndim": ndim,
            "discrete_obs": True
        }
    )

    register(
        id="DangerousPath-HalfMines-len{}-dim{}-v0".format(path_len, ndim),
        entry_point="envs.dangerous_path_env:DangerousPathEnv",
        kwargs={
            "game_length": path_len,
            "ndim": ndim,
            "mine_ratio": 0.5
        }
    )

    register(
        id="DangerousPath-HalfMines-Discrete-len{}-dim{}-v0".format(path_len, ndim),
        entry_point="envs.dangerous_path_env:DangerousPathEnv",
        kwargs={
            "game_length": path_len,
            "ndim": ndim,
            "discrete_obs": True,
            "mine_ratio": 0.5
        }
    )
