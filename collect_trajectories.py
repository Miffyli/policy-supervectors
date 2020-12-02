# Load bunch of trained models, and
# collect trajectories to represent them
#
import os
from argparse import ArgumentParser
import glob
from multiprocessing import pool
import re
import random

import numpy as np
import gym
from tqdm import tqdm

from agents import SimpleAgentClass
from wrappers import StateWrapper

# NOTE: Relies on specific naming of experiments.
#   Each experiment is a directory with name:
#       [package]_[env]_[agent]_[timestamp]
#   Where agent depends on the package, and
#   package can be one of
#       * stablebaselines
#       * wann
#       * random

CHECKPOINT_DIR = "checkpoints"
TRAJECTORIES_DIR = "trajectories"

# NEAT/WANN/CMA-ES have a ton of checkpoints, so lets
# skip some of the generations and only include
# some of the population
WANN_GENERATION_SKIP_RATE = 10
WANN_POPULATION_KEEP_RATE = 0.25

parser = ArgumentParser("Go through experiments in a directory and collect experiences for their checkpoints")
parser.add_argument("input_dirs", type=str, nargs="+", help="Experiments (directories) to go through.")
parser.add_argument("--num-trajs", type=int, default=100, help="Number of trajectories per checkpoint.")
parser.add_argument("--skip-existing", action="store_true", help="Check if trajectory files exist, and skip if so.")
parser.add_argument("--num-workers", type=int, default=4, help="Number of processes to use.")
parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR, help="Where to load checkpoints from under experiment.")
parser.add_argument("--trajectory-dir", type=str, default=TRAJECTORIES_DIR, help="Where to store the trajectories for each checkpoint.")
parser.add_argument("--append", action="store_true", help="If given, update existing trajectory files by adding new trajectories.")


def collect_trajectories(env, agent, num_trajectories):
    """
    Play agent (with `get_action` function) in the
    environment with `get_state` function,
    and return a list of trajectories ((T, D) arrays)
    and episodic rewards (scalars).
    """
    trajectories = []
    rewards = []
    for game in range(num_trajectories):
        visited_states = []
        episodic_reward = 0
        obs = env.reset()
        visited_states.append(env.get_state())
        terminal = False
        while not terminal:
            action = agent.get_action(obs)
            obs, reward, terminal, info = env.step(action)
            visited_states.append(env.get_state())
            episodic_reward += reward
        rewards.append(episodic_reward)
        trajectories.append(np.array(visited_states))
    return trajectories, rewards


def create_agent_and_collect_trajectories(checkpoint_path_and_args):
    """
    Create agent and env based on the path, which points to
    a specific experiment directory. Collect trajectories
    and store them in trajectories directory
    """
    checkpoint_path, args = checkpoint_path_and_args
    checkpoint_dir, checkpoint_name = os.path.split(checkpoint_path)
    experiment_dir, _ = os.path.split(checkpoint_dir)
    package_type, env, agent_type, timestamp = os.path.basename(experiment_dir).split("_")
    traj_dir = os.path.join(experiment_dir, args.trajectory_dir)
    os.makedirs(traj_dir, exist_ok=True)
    output_file = os.path.join(traj_dir, checkpoint_name.split(".")[0]) + ".npz"

    if args.skip_existing and os.path.isfile(output_file) and not args.append:
        return None

    env = gym.make(env)
    # Wrap into get_state wrapper
    env = StateWrapper(env)

    agent = None
    if package_type == "stablebaselines":
        if "SB3" in agent_type:
            # Stable-baselines3
            from agents.stable_baselines_agent import create_stable_baselines3_agent
            agent = create_stable_baselines3_agent(checkpoint_path, agent_type)
        else:
            from agents.stable_baselines_agent import create_stable_baselines_agent
            if "-" in agent_type:
                # Agent type can have stuff like {algo}-{info}, like PPO-clip0.1
                agent_type = agent_type.split("-")[0]
            agent = create_stable_baselines_agent(checkpoint_path, agent_type)
    elif package_type == "wann":
        from agents.wann_agent import create_wann_agent
        agent = create_wann_agent(checkpoint_path, agent_type, env)
    elif package_type == "random":
        agent = SimpleAgentClass(lambda obs: env.action_space.sample())
    else:
        raise RuntimeError("Unknown package_type {}".format(package_type))

    trajectories, rewards = collect_trajectories(env, agent, args.num_trajs)

    if args.append:
        output_dict = np.load(output_file)
        # Find last trajectory stored
        traj_ints = [int(x.split("_")[1]) for x in output_dict.keys() if "traj" in x]
        last_traj_int = max(traj_ints)
        new_items = dict(("traj_%d" % (i + last_traj_int + 1), trajectory) for i, trajectory in enumerate(trajectories))
        episodic_rewards = output_dict["episodic_rewards"]
        episodic_rewards = np.concatenate((episodic_rewards, np.array(rewards)))
        np.savez(
            output_file,
            episodic_rewards=episodic_rewards,
            **new_items,
            # Take old trajectories but skip episodic reward
            **dict((k,v) for k,v in output_dict.items() if k != "episodic_rewards")
        )
    else:
        output_dict = dict(("traj_%d" % i, trajectory) for i, trajectory in enumerate(trajectories))
        output_dict["episodic_rewards"] = np.array(rewards)
        np.savez(output_file, **output_dict)
    # Cleanup
    env.close()
    del agent


def main(args):
    tasks = []
    # Go through directories and gather all checkpoints we want to
    # compute samples for
    for input_dir in args.input_dirs:
        # Go through all checkpoints
        checkpoint_glob = os.path.join(input_dir, args.checkpoint_dir, "*")
        checkpoints = glob.glob(checkpoint_glob)

        # Do algorithm-specific sampling of agents
        # to avoid computing too long
        if "wann_" in input_dir:
            # Take only every Nth generation, and
            # from that sample specific amount of
            # agents
            generations = [int(re.findall("gen_([0-9]*)", x)[0]) for x in checkpoints]
            new_checkpoints = []
            for generation_i in range(min(generations), max(generations) + 1):
                if (generation_i % WANN_GENERATION_SKIP_RATE) == 0:
                    valid_checkpoints = [x for x in checkpoints if "gen_{}".format(generation_i) in x]
                    sampled_checkpoints = random.sample(valid_checkpoints, int(len(valid_checkpoints) * WANN_POPULATION_KEEP_RATE))
                    new_checkpoints.extend(sampled_checkpoints)
            checkpoints = new_checkpoints

        # Add args in the tasks
        for checkpoint in checkpoints:
            tasks.append((checkpoint, args))

    workers = pool.Pool(processes=args.num_workers)
    total = len(tasks)
    for _ in tqdm(workers.imap(create_agent_and_collect_trajectories, tasks, chunksize=25), total=total):
        pass


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
