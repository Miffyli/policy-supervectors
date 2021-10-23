# Run hardcoded experiments with the trust-region policy gradient
#
import os
from argparse import ArgumentParser
import warnings

import numpy as np
import gym
import torch as th

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import logger
from stable_baselines3.ppo import PPO

import envs
import gmm_tools
from agents.step_ppo import SmallStepPPO, StepConstraint

# We do not need big networks for these experiments
NET_ARCH = [16, 16]

# Speeds things up often.
th.set_num_threads(1)


def create_env(args, idx, monitor=True):
    """
    Create and return an environment according to args (parsed arguments).
    idx specifies idx of this environment among parallel environments.
    I could have used SB3-Zoo but now just copy-pasting code from previous...
    """
    env = gym.make(args.env)

    # Seed DangerousPath with same seed for all envs
    # (otherwise there would be trouble)
    if "DangerousPath" in args.env:
        env.seed(args.env_seed)

    if monitor:
        monitor_file = None
        if args.output is not None:
            monitor_file = os.path.join(args.output, ("env_%d" % idx))

        env = Monitor(env, monitor_file)

    return env


def do_manual_rollouts(agent, env, n_rollouts):
    """Run agent on env for n_rollouts episodes and return states in one array"""
    obs = []
    for i in range(n_rollouts):
        ob = env.reset()
        obs.append(ob)
        d = False
        while not d:
            action, _ = agent.predict(ob)
            ob, r, d, info = env.step(action)
            obs.append(ob)
    return np.array(obs)


class PiMaxTVConstraint(StepConstraint):
    r"""
    Constraint that computes max of total variation divergence
        \max_s [ 0.5 * \sum_i |\pi_1(s) - \pi_2(s)| ]
    Uses current samples in the rollout-buffer to max over s.

    NOTE: Only supporting discrete action-spaces here!
    """

    def __init__(self, args):
        self.max_tv_constraint = args.max_tv_constraint
        self.observations = None
        self.old_policy_probs = None

    def _get_log_pis(self, agent):
        """Return action log-probs for current observations in buffer"""
        latent_pi, latent_vf, latent_sde = agent.policy._get_latent(self.observations)
        distribution = agent.policy._get_action_dist_from_latent(latent_pi, latent_sde)
        return distribution.distribution.probs

    def before_updates(self, old_agent, rollout_buffer):
        # Access data directly from buffer (flatten out batch and env dims)
        self.observations = rollout_buffer.observations
        obs_shape = self.observations.shape
        self.observations = self.observations.reshape((obs_shape[0] * obs_shape[1],) + obs_shape[2:])
        self.observations = th.from_numpy(self.observations).float()

        self.old_agent_probs = self._get_log_pis(old_agent)

    def check_constraint(self, new_agent):
        new_agent_probs = self._get_log_pis(new_agent)
        max_tv = th.max(0.5 * th.sum(th.abs(self.old_agent_probs - new_agent_probs), dim=1))
        if max_tv.item() >= self.max_tv_constraint:
            return True
        else:
            return False


class GaussianKLConstraint(StepConstraint):
    r"""
    Fit diagonal Gaussians on the observations.

    Compute KL both ways and sum together.
    """

    def __init__(self, args):
        self.max_kl_constraint = args.max_kl_constraint
        self.n_rollouts = args.n_rollouts
        self.env = create_env(args, 0, monitor=False)
        self.old_policy_dist = None

    def _create_dist_for_agent(self, agent):
        data = do_manual_rollouts(agent, self.env, self.n_rollouts)
        # Add bit of noise to datapoints to dislodge same points
        data += np.random.randn(*data.shape) * 0.001
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        distribution = th.distributions.MultivariateNormal(
            th.from_numpy(mean).float(),
            th.diag(th.from_numpy(std ** 2 + 1e-7).float()),
        )
        return distribution

    def before_updates(self, old_agent, rollout_buffer):
        # Gather states to visit
        self.old_policy_dist = self._create_dist_for_agent(old_agent)

    def check_constraint(self, new_agent):
        new_dist = self._create_dist_for_agent(new_agent)
        kl_distance = None
        with th.no_grad():
            kl_distance = (
                th.distributions.kl_divergence(self.old_policy_dist, new_dist) +
                th.distributions.kl_divergence(new_dist, self.old_policy_dist)
            ).item()
        if kl_distance >= self.max_kl_constraint:
            return True
        return False


class SupervectorKLConstraint(StepConstraint):
    r"""
    Fit diagonal GMM on the observations and extract policy supervectors.

    Use the upper bound of KL.
    """

    def __init__(self, args):
        self.max_kl_constraint = args.max_kl_constraint
        self.n_rollouts = args.n_rollouts
        self.n_centroids = args.n_centroids
        self.env = create_env(args, 0, monitor=False)
        self.old_policy_data = None

    def before_updates(self, old_agent, rollout_buffer):
        # Gather states to visit
        self.old_policy_data = do_manual_rollouts(old_agent, self.env, self.n_rollouts)
        # Add bit of random noise to the data to dislodge same points
        self.old_policy_data += np.random.randn(*self.old_policy_data.shape) * 0.001

    def check_constraint(self, new_agent):
        # Compute UBM, extract supervectors and compute KL
        new_policy_data = do_manual_rollouts(new_agent, self.env, self.n_rollouts)
        new_policy_data += np.random.randn(*new_policy_data.shape) * 0.001
        all_data = np.concatenate((self.old_policy_data, new_policy_data), axis=0)
        # Avoid all the spam from "less unique centroids"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ubm = gmm_tools.train_ubm(all_data, n_components=self.n_centroids, verbose=0)
        old_supervector = gmm_tools.trajectories_to_supervector(self.old_policy_data, ubm)
        new_supervector = gmm_tools.trajectories_to_supervector(new_policy_data, ubm)
        # Supervectors are returned as raveled 1D vectors
        old_supervector = old_supervector.reshape((ubm.means_.shape))
        new_supervector = new_supervector.reshape((ubm.means_.shape))

        kl_distance = gmm_tools.adapted_gmm_distance(old_supervector, new_supervector, ubm.precisions_, ubm.weights_)

        if kl_distance >= self.max_kl_constraint:
            return True
        return False


AVAILABLE_CONSTRAINTS = {
    "PiMaxTV": PiMaxTVConstraint,
    "Gaussian": GaussianKLConstraint,
    "Supervector": SupervectorKLConstraint,
    "ClipPPO": "ClipPPO",
}

AGENT_FILE = "trained_agent.zip"

parser = ArgumentParser("Run experiments with different types of environment.")
parser.add_argument("--constraint", type=str, required=True, choices=list(AVAILABLE_CONSTRAINTS.keys()), help="Algorithm to use.")
parser.add_argument("--env", required=True, help="Environment to play.")
parser.add_argument("--n-envs", type=int, default=8, help="Number of environments to use.")
parser.add_argument("--n-steps", type=int, default=512, help="Number of samples per environment.")
parser.add_argument("--total-timesteps", type=int, default=int(1e6), help="How long to train.")
parser.add_argument("--output", type=str, default=None, help="Directory where to put monitors/trained agent.")
parser.add_argument("--output-log", type=str, default=None, help="Directory where to put training log.")
parser.add_argument("--env-seed", type=int, default=np.random.randint(1e6), help="Seed for the DangerousPath environment.")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")

parser.add_argument("--augment-ppo", action="store_true", help="Augment full PPO instead using bare-version.")
parser.add_argument("--n-epochs", type=int, default=10, help="Number of epochs to go over with augmented PPO.")
parser.add_argument("--max-updates", type=int, default=1000, help="Max updates per policy update.")
parser.add_argument("--ent-coef", type=float, default=0.0, help="Entropy coefficient.")
parser.add_argument("--learning-rate", type=float, default=1e-5, help="Ye good olde learning rate.")
parser.add_argument("--clip-range", type=float, default=0.2, help="Clip-range for vanilla PPO.")
parser.add_argument("--max-tv-constraint", type=float, default=0.01, help="Constraint on max-TV.")
parser.add_argument("--max-kl-constraint", type=float, default=0.5, help="Constraint for Gaussian/supervector KL distance.")
parser.add_argument("--n-centroids", type=int, default=4, help="Number of centroids/components used for Supervector.")
parser.add_argument("--n-rollouts", type=int, default=5, help="Number of rollouts for state-based BCs.")


def run_experiment(args):
    # Again could have used the SB3 tools here, buuuut...
    vecEnv = []
    for i in range(args.n_envs):
        # Bit of trickery here to avoid referencing
        # to the same "i"
        vecEnv.append((
            lambda idx: lambda: create_env(args, idx))(i)
        )

    vecEnv = DummyVecEnv(vecEnv)

    constraint = AVAILABLE_CONSTRAINTS[args.constraint]
    agent = None
    if constraint == "ClipPPO":
        # Create a vanilla PPO
        agent = PPO(
            "MlpPolicy",
            vecEnv,
            verbose=2,
            device="cpu",
            n_steps=args.n_steps,
            clip_range=args.clip_range,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            gae_lambda=1.0,
            n_epochs=args.n_epochs
        )
    else:
        constraint = constraint(args)

        agent = SmallStepPPO(
            "MlpPolicy",
            vecEnv,
            verbose=2,
            device="cpu",
            n_steps=args.n_steps,
            step_constraint=constraint,
            learning_rate=args.learning_rate,
            step_constraint_max_updates=args.max_updates,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            gae_lambda=1.0
        )

    output_log_file = None
    if args.output_log:
        output_log_file = open(args.output_log, "w")
        logger.Logger.CURRENT = logger.Logger(folder=None, output_formats=[logger.HumanOutputFormat(output_log_file)])

    agent.learn(total_timesteps=args.total_timesteps)

    if args.output is not None:
        agent.save(os.path.join(args.output, AGENT_FILE))

    vecEnv.close()
    if output_log_file:
        output_log_file.close()


if __name__ == "__main__":
    args = parser.parse_args()
    run_experiment(args)
