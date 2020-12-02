from gym import Wrapper


def env_name_to_state_func(env_id):
    """
    Return a get_state function
    for given env name
    """
    if "CartPole" in env_id:
        # MDP env
        return lambda env: env.last_obs
    elif "Pendulum" in env_id:
        # MDP env
        return lambda env: env.last_obs
    elif "Acrobot" in env_id:
        # MDP env
        return lambda env: env.last_obs
    elif "BipedalWalker" in env_id:
        # Code:
        #  https://github.com/openai/gym/blob/master/gym/envs/box2d/bipedal_walker.py
        # Not quite MDP, but lets use
        # knowledge of the walker's state
        # as state.
        return lambda env: env.last_obs
    elif "LunarLander" in env_id:
        # Code:
        #  https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
        # Not quite MDP (the "moon" varies), but
        # this again is not dependent on the policy
        # and policy does not see the moon, so
        # we treat the observations (info about the lander)
        # as the state vector.
        return lambda env: env.last_obs
    else:
        raise RuntimeError("No get_state func for env {}".format(env_id))


class StateWrapper(Wrapper):
    """
    A Wrapper that adds `get_state()` function,
    which returns the current state (not observation)
    of the environment.
    """

    def __init__(self, env, get_state_fn=None):
        """
        get_state_fn (func): A function that takes in
            an environment (with all the wrapping), and returns
            a 1D numpy array representing the current
            state of the environment. If None, determine
            function based on the env id.
        """
        super().__init__(env)
        self.env = env
        self.get_state_fn = get_state_fn

        if self.get_state_fn is None:
            # Automatically determine function
            # based on the name
            self.env_id = env.unwrapped.spec.id
            self.get_state_fn = env_name_to_state_func(self.env_id)

        # Used by MDP environments, where observation == state
        self.last_obs = None

    def get_state(self):
        return self.get_state_fn(self)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_obs = obs
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.last_obs = obs
        return obs, reward, done, info
