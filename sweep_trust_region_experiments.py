# Run parameter sweep over trust-region experiments

import random
import os
from argparse import ArgumentParser
from multiprocessing import Pool

from tqdm import tqdm

# Maps experiment name to envs to loop over
EXPERIMENTS = {
    "DangerousPath": ["DangerousPath-HalfMines-len25-dim5-v0"],
}

EXPERIMENT_EXTRA_ARGS = {
    "DangerousPath": ["--gamma", "0.0"],
}

DIR_TEMPLATE = "{env}_lr_{learning_rate}"
STDOUT_TEMPLATE = "{method}_repetition_{repetition}.txt"
# As strings to keep scientific notation (could also, you know, use formatting settings...)
LEARNING_RATES = ["1e-3"]
# Provide in format we need to feed in for argument parser
CONSTANT_ARGS = [
    "--n-steps", "512",
    "--n-envs", "8",
    "--max-updates", "100",
    "--total-timesteps", "1000000",
    "--n-rollouts", "5",
]

NUM_REPETITIONS = 50

# As strings as these will go into parser. These are constraints to sweep over.
# These were decided by searching moving the range until optimal result was
# NOT one of the edge-values.
TV_CONSTRAINTS = ["0.001", "0.005", "0.01", "0.05", "0.1", "0.2", "0.3", "0.4", "0.5"]
SUPERVECTOR_KL_CONSTRAINTS = ["0.01", "0.05", "0.1", "0.15", "0.2", "0.3", "0.4", "0.5"]
GAUSSIAN_KL_CONSTRAINTS = ["0.5", "1.0", "2.0", "3.0", "5.0", "10.0", "15.0", "20.0"]

parser = ArgumentParser("Sweep over different learning rates etc for trust-region parameters.")
parser.add_argument("--experiment", type=str, default="DangerousPath", choices=list(EXPERIMENTS.keys()), help="Experiment to run.")
parser.add_argument("--workers", type=int, default=8, help="Number of workers to use.")
parser.add_argument("--output", type=str, default="trust_region_experiments", help="Output directory for results.")


def worker_function(experiment_args):
    from run_trust_region_experiments import run_experiment, parser
    parsed_args = parser.parse_args(experiment_args)
    run_experiment(parsed_args)


def main(args):
    # Construct all experiment settings.
    # This will be list of tuples (output_file, experiment_args)
    all_experiments = []

    envs = EXPERIMENTS[args.experiment]
    extra_args = EXPERIMENT_EXTRA_ARGS.get(args.experiment, [])

    # Fixed seeds for all experiments over repetitions
    # (used only with DangerousPath experiments)
    env_seeds = [str(random.randint(0, 1e6)) for i in range(NUM_REPETITIONS)]

    for env in envs:
        for learning_rate in LEARNING_RATES:
            output_dir = os.path.join(args.output, DIR_TEMPLATE.format(env=env, learning_rate=learning_rate))
            os.makedirs(output_dir, exist_ok=True)

            # No constraint
            for repetition in range(NUM_REPETITIONS):
                output_file = os.path.join(output_dir, "NoConstraint_repetition_{}.txt".format(repetition))
                output_args = [
                    "--output-log", output_file,
                    "--env", env,
                    "--constraint", "PiMaxTV",
                    "--learning-rate", learning_rate,
                    "--max-tv-constraint", "10000000",
                    "--env-seed", env_seeds[repetition]
                ] + CONSTANT_ARGS + extra_args
                all_experiments.append(output_args)

            # Clip PPO
            for repetition in range(NUM_REPETITIONS):
                output_file = os.path.join(output_dir, "ClipPPO_repetition_{}.txt".format(repetition))
                output_args = [
                    "--output-log", output_file,
                    "--env", env,
                    "--constraint", "ClipPPO",
                    "--learning-rate", learning_rate,
                    "--env-seed", env_seeds[repetition]
                ] + CONSTANT_ARGS + extra_args
                all_experiments.append(output_args)

            # TV constraint
            for max_tv in TV_CONSTRAINTS:
                for repetition in range(NUM_REPETITIONS):
                    output_file = os.path.join(output_dir, "PiMaxTV_{}_repetition_{}.txt".format(max_tv, repetition))
                    output_args = [
                        "--output-log", output_file,
                        "--env", env,
                        "--constraint", "PiMaxTV",
                        "--learning-rate", learning_rate,
                        "--max-tv-constraint", max_tv,
                        "--env-seed", env_seeds[repetition]
                    ] + CONSTANT_ARGS + extra_args
                    all_experiments.append(output_args)

            # KL constraints
            for max_kl in GAUSSIAN_KL_CONSTRAINTS:
                for repetition in range(NUM_REPETITIONS):
                    output_file = os.path.join(output_dir, "Gaussian_{}_repetition_{}.txt".format(max_kl, repetition))
                    output_args = [
                        "--output-log", output_file,
                        "--env", env,
                        "--constraint", "Gaussian",
                        "--learning-rate", learning_rate,
                        "--max-kl-constraint", max_kl,
                        "--env-seed", env_seeds[repetition]
                    ] + CONSTANT_ARGS + extra_args
                    all_experiments.append(output_args)

            for max_kl in SUPERVECTOR_KL_CONSTRAINTS:
                for repetition in range(NUM_REPETITIONS):
                    output_file = os.path.join(output_dir, "Supervector_{}_repetition_{}.txt".format(max_kl, repetition))
                    output_args = [
                        "--output-log", output_file,
                        "--env", env,
                        "--constraint", "Supervector",
                        "--learning-rate", learning_rate,
                        "--max-kl-constraint", max_kl,
                        "--env-seed", env_seeds[repetition]
                    ] + CONSTANT_ARGS + extra_args
                    all_experiments.append(output_args)

    workers = Pool(processes=args.workers)

    progress_bar = tqdm(total=len(all_experiments))
    for _ in workers.imap_unordered(worker_function, all_experiments):
        progress_bar.update(1)
    workers.close()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
