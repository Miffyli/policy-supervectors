def create_stable_baselines_agent(agent_path, agent_type):
    """
    Load and return a stable-baselines agent.
    The agent has a function `get_action` that takes in
    an observation and returns an appropiate action.

    `agent_type` is the algorithm name.
    """
    from stable_baselines import A2C, PPO2
    agent = None
    if agent_type == "A2C":
        agent = A2C.load(agent_path)
    elif agent_type == "PPO" or agent_type == "UBMPPO":
        agent = PPO2.load(agent_path)
    else:
        raise RuntimeError("Unknown agent type for SB: {}".format(agent_type))
    # Add get_action function
    agent.get_action = lambda obs: agent.predict(obs)[0]
    return agent


def create_stable_baselines3_agent(agent_path, agent_type):
    """
    Load and return a stable-baselines3 agent.
    The agent has a function `get_action` that takes in
    an observation and returns an appropiate action.

    `agent_type` is the algorithm name (only PPO-SB3 supported)
    """
    from stable_baselines3 import PPO
    import torch

    agent = None
    if agent_type == "SB3-PPO":
        if "bc_models" in agent_path:
            # Only stores policy parameters.
            # Force on CPU (codebase-level heuristic that everything runs on CPU)
            agent = torch.load(agent_path, map_location="cpu")
            agent.get_action = lambda obs: agent.predict(obs)[0]
        else:
            # GAIL: Stores the whole agent
            agent = PPO.load(agent_path)
            agent.get_action = lambda obs: agent.predict(obs)[0]
    else:
        raise RuntimeError("Unknown agent type for SB3: {}".format(agent_type))
    return agent
