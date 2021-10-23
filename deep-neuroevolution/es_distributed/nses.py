import logging
import sys
import os
import time
from collections import namedtuple
import tensorflow as tf
from copy import deepcopy

import numpy as np
import torch as th

from .dist import MasterClient, WorkerClient
from .es import *

# Neat trick to get our supervector tools. Yay cleanliness
sys.path.append("..")
import gmm_tools

# For supervector novelty stuff.
# Instead of sharing the data over Redis (we are about to send a ton of it),
# store data on the disk.
NOVELTY_ARCHIVE_FILE_NAME = "novelty_archive.npz"
NOVELTY_RAW_DATA_DIR_NAME = "novelty_buffer"
NOVELTY_SUPERVECTOR_COMPONENTS = 4

def euclidean_distance(x, y):
    n, m = len(x), len(y)
    if n > m:
        a = np.linalg.norm(y - x[:m])
        b = np.linalg.norm(y[-1] - x[m:])
    else:
        a = np.linalg.norm(x - y[:n])
        b = np.linalg.norm(x[-1] - y[n:])
    return np.sqrt(a**2 + b**2)

def compute_novelty_vs_archive(archive, novelty_vector, k, bc_type="terminal", worker_dir=None):
    distances = []
    nov = novelty_vector.astype(np.float)
    if bc_type == "supervector":
        ubm = None
        means = None
        stds = None
        # A fight against race-condition: If failed to load, try again bit later
        while ubm is None:
            try:
                ubm, means, stds = gmm_tools.load_ubm(os.path.join(worker_dir, NOVELTY_ARCHIVE_FILE_NAME))
            except Exception:
                print("[Warning] Failed to load UBM file. Trying again...")
                time.sleep(0.1)
        # Normalize data
        normalized_states = (novelty_vector - means) / stds

        my_supervector = gmm_tools.trajectories_to_supervector(normalized_states, ubm)
        my_supervector = my_supervector.reshape(ubm.means_.shape)
        precisions = ubm.precisions_
        weights = ubm.weights_

        # Now load supervectors that are stored in the same file (conveniently reading many times
        # for _optimal efficiency_...)
        archive_data = None
        while archive_data is None:
            try:
                archive_data = np.load(os.path.join(worker_dir, NOVELTY_ARCHIVE_FILE_NAME))
            except Exception:
                print("[Warning] Failed to load archive file. Trying again...")
                time.sleep(0.1)
        other_supervectors = archive_data["supervectors"]
        archive_data.close()

        for i in range(other_supervectors.shape[0]):
            kl_distance = gmm_tools.adapted_gmm_distance(my_supervector, other_supervectors[i], precisions, weights)
            distances.append(kl_distance)
    else:
        for point in archive:
            if bc_type == "terminal":
                distances.append(euclidean_distance(point.astype(np.float), nov))
            elif bc_type == "gaussian":
                midpoint = len(point) // 2
                if isinstance(nov, np.ndarray):
                    if nov.ndim == 2:
                        # Need to compute mean and cov
                        nov = th.distributions.MultivariateNormal(
                            th.from_numpy(np.mean(nov, axis=0)).float(),
                            th.diag(th.from_numpy(np.var(nov, axis=0) + 1e-7)).float()
                        )
                    else:
                        # Already computed mean+var vector
                        nov = th.distributions.MultivariateNormal(
                            th.from_numpy(nov[:midpoint]).float(),
                            th.diag(th.from_numpy(nov[midpoint:] + 1e-7)).float()
                        )
                point = th.distributions.MultivariateNormal(
                    th.from_numpy(point[:midpoint]).float(),
                    th.diag(th.from_numpy(point[midpoint:] + 1e-7)).float()
                )
                with th.no_grad():
                    kl_distance = (
                        th.distributions.kl_divergence(nov, point) +
                        th.distributions.kl_divergence(point, nov)
                    )
                    distances.append(kl_distance.item())
            else:
                raise NotImplementedError("bc_type {} not implemented".format(bc_type))

    # Pick k nearest neighbors
    distances = np.array(distances)
    top_k_indicies = (distances).argsort()[:k]
    top_k = distances[top_k_indicies]
    return top_k.mean()

def get_mean_bc(env, policy, tslimit, num_rollouts=1, bc_type="terminal", for_archive=False, worker_dir=None):
    # for_archive tells us if we are about to send this item to arhive.
    novelty_vector = []
    for n in range(num_rollouts):
        rew, t, nv = policy.rollout(env, timestep_limit=tslimit, bc_only_final_state=(bc_type == "terminal"))
        novelty_vector.append(nv)
    if bc_type == "terminal":
        return np.mean(novelty_vector, axis=0)
    elif bc_type == "gaussian":
        # Concatenate individual rollouts
        novelty_vector = np.concatenate(novelty_vector, axis=0)
        # Fit a simple monogaussian
        return np.concatenate(
            (
                np.mean(novelty_vector, axis=0),
                np.var(novelty_vector, axis=0),
            ),
            axis=0
        )
    elif bc_type == "supervector":
        novelty_vector = np.concatenate(novelty_vector, axis=0)
        if for_archive:
            # Save novelty vector for the archival purposes
            raw_data_dirpath = os.path.join(worker_dir, NOVELTY_RAW_DATA_DIR_NAME)
            os.makedirs(raw_data_dirpath, exist_ok=True)

            # Store data and update the extractor
            num_items = len(os.listdir(raw_data_dirpath))
            # Create a new item
            save_path = os.path.join(raw_data_dirpath, "policy_{}".format(num_items))
            np.save(save_path, novelty_vector)

            # Read stored policies' data, train UBM and extract supervectors
            all_buffer_data = []
            buffer_filenames = os.listdir(raw_data_dirpath)
            for filename in buffer_filenames:
                full_path = os.path.join(raw_data_dirpath, filename)
                buffer_data = np.load(full_path)
                all_buffer_data.append(buffer_data)

            # Super-elegant hoarding of memory by having multiple copies
            concat_data = np.concatenate(all_buffer_data, axis=0)
            means = concat_data.mean(axis=0)
            stds = concat_data.std(axis=0)

            concat_data = (concat_data - means) / stds

            ubm = gmm_tools.train_ubm(concat_data, n_components=NOVELTY_SUPERVECTOR_COMPONENTS, verbose=0)

            # Extract supervectors
            supervectors = []
            mean_shape = ubm.means_.shape
            for buffer_data in all_buffer_data:
                buffer_data = (buffer_data - means) / stds
                supervector = gmm_tools.trajectories_to_supervector(buffer_data, ubm)
                supervectors.append(supervector.reshape(mean_shape))

            gmm_tools.save_ubm(
                os.path.join(worker_dir, NOVELTY_ARCHIVE_FILE_NAME),
                ubm,
                means,
                stds,
                supervectors=supervectors
            )
            # Return something dumb. This goes to archive, but we are
            # never supposed to read it during supervector novelty search
            return np.array([42])
        else:
            # Return just the states visited. They are the "BC"
            return novelty_vector
    else:
        raise NotImplementedError("bc_type {} not implemented".format(bc_type))

def setup_env(exp):
    import gym
    gym.undo_logger_setup()
    config = Config(**exp['config'])
    if exp['env_id'] == "DeceptivePointEnv-v0":
        # Lil hack we need to do
        import sys
        sys.path.append("../envs")
        import point_env
    env = gym.make(exp['env_id'])
    if exp['policy']['type'] == "ESAtariPolicy":
        from .atari_wrappers import wrap_deepmind
        env = wrap_deepmind(env)
    return config, env

def setup_policy(env, exp, single_threaded):
    from . import policies, tf_util
    sess = make_session(single_threaded=single_threaded)
    policy = getattr(policies, exp['policy']['type'])(env.observation_space, env.action_space, **exp['policy']['args'])
    tf_util.initialize()
    return sess, policy

def run_master(master_redis_cfg, log_dir, exp):
    logger.info('run_master: {}'.format(locals()))
    from .optimizers import SGD, Adam
    from . import tabular_logger as tlogger
    logger.info('Tabular logging to {}'.format(log_dir))
    tlogger.start(log_dir)
    config, env = setup_env(exp)
    algo_type = exp['algo_type']
    master = MasterClient(master_redis_cfg)
    noise = SharedNoiseTable()
    rs = np.random.RandomState()
    ref_batch = get_ref_batch(env, batch_size=128)

    pop_size = int(exp['novelty_search']['population_size'])
    num_rollouts = int(exp['novelty_search']['num_rollouts'])
    theta_dict = {}
    optimizer_dict = {}
    obstat_dict = {}
    curr_parent = 0

    if isinstance(config.episode_cutoff_mode, int):
        tslimit, incr_tslimit_threshold, tslimit_incr_ratio, tslimit_max = config.episode_cutoff_mode, None, None, config.episode_cutoff_mode
        adaptive_tslimit = False

    elif config.episode_cutoff_mode.startswith('adaptive:'):
        _, args = config.episode_cutoff_mode.split(':')
        arg0, arg1, arg2, arg3 = args.split(',')
        tslimit, incr_tslimit_threshold, tslimit_incr_ratio, tslimit_max = int(arg0), float(arg1), float(arg2), float(arg3)
        adaptive_tslimit = True
        logger.info(
            'Starting timestep limit set to {}. When {}% of rollouts hit the limit, it will be increased by {}. The maximum timestep limit is {}'.format(
                tslimit, incr_tslimit_threshold * 100, tslimit_incr_ratio, tslimit_max))

    elif config.episode_cutoff_mode == 'env_default':
        tslimit, incr_tslimit_threshold, tslimit_incr_ratio, tslimit_max = None, None, None, None
        adaptive_tslimit = False
    else:
        raise NotImplementedError(config.episode_cutoff_mode)

    for p in range(pop_size):
        with tf.Graph().as_default():
            sess, policy = setup_policy(env, exp, single_threaded=False)

            if 'init_from' in exp['policy']:
                logger.info('Initializing weights from {}'.format(exp['policy']['init_from']))
                policy.initialize_from(exp['policy']['init_from'], ob_stat)

            theta = policy.get_trainable_flat()
            optimizer = {'sgd': SGD, 'adam': Adam}[exp['optimizer']['type']](theta, **exp['optimizer']['args'])

            if policy.needs_ob_stat:
                ob_stat = RunningStat(env.observation_space.shape, eps=1e-2)
                obstat_dict[p] = ob_stat

            if policy.needs_ref_batch:
                policy.set_ref_batch(ref_batch)

            mean_bc = get_mean_bc(env, policy, tslimit_max, num_rollouts, bc_type=exp['novelty_search']['bc_type'], for_archive=True, worker_dir=log_dir)
            master.add_to_novelty_archive(mean_bc)

            theta_dict[p] = theta
            optimizer_dict[p] = optimizer

    episodes_so_far = 0
    timesteps_so_far = 0
    tstart = time.time()
    master.declare_experiment(exp)

    max_iter_count = exp.get('max_number_of_iterations', int(1e9))
    curr_task_id = 0

    # No cleaning up here. We leave that stuff to others to handle
    while curr_task_id <= max_iter_count:
        step_tstart = time.time()

        theta = theta_dict[curr_parent]
        policy.set_trainable_flat(theta)
        optimizer = optimizer_dict[curr_parent]

        if policy.needs_ob_stat:
            ob_stat = deepcopy(obstat_dict[curr_parent])

        assert theta.dtype == np.float32

        curr_task_id = master.declare_task(Task(
            params=theta,
            ob_mean=ob_stat.mean if policy.needs_ob_stat else None,
            ob_std=ob_stat.std if policy.needs_ob_stat else None,
            ref_batch=ref_batch if policy.needs_ref_batch else None,
            timestep_limit=tslimit
        ))
        master.flush_results()
        new_task_checker = False
        while not new_task_checker:
            # Query master to see if new task declaration registers
            for _ in range(1000):
                temp_task_id, _ = master.pop_result()
                if temp_task_id == curr_task_id:
                    new_task_checker = True; break

            # Re-declare task if original declaration fails to register
            if not new_task_checker:
                master.task_counter -= 1
                curr_task_id = master.declare_task(Task(
                    params=theta,
                    ob_mean=ob_stat.mean if policy.needs_ob_stat else None,
                    ob_std=ob_stat.std if policy.needs_ob_stat else None,
                    ref_batch=ref_batch if policy.needs_ref_batch else None,
                    timestep_limit=tslimit
                ))
        tlogger.log('********** Iteration {} **********'.format(curr_task_id))

        # Pop off results for the current task
        curr_task_results, eval_rets, eval_lens, worker_ids = [], [], [], []
        num_results_skipped, num_episodes_popped, num_timesteps_popped, ob_count_this_batch = 0, 0, 0, 0
        while num_episodes_popped < config.episodes_per_batch or num_timesteps_popped < config.timesteps_per_batch:
            # Wait for a result
            task_id, result = master.pop_result()
            assert isinstance(task_id, int) and isinstance(result, Result)
            assert (result.eval_return is None) == (result.eval_length is None)
            worker_ids.append(result.worker_id)

            if result.eval_length is not None:
                # This was an eval job
                episodes_so_far += 1
                timesteps_so_far += result.eval_length
                # Store the result only for current tasks
                if task_id == curr_task_id:
                    eval_rets.append(result.eval_return)
                    eval_lens.append(result.eval_length)
            else:
                assert (result.noise_inds_n.ndim == 1 and
                        result.returns_n2.shape == result.lengths_n2.shape == (len(result.noise_inds_n), 2))
                assert result.returns_n2.dtype == np.float32
                # Update counts
                result_num_eps = result.lengths_n2.size
                result_num_timesteps = result.lengths_n2.sum()
                episodes_so_far += result_num_eps
                timesteps_so_far += result_num_timesteps
                # Store results only for current tasks
                if task_id == curr_task_id:
                    curr_task_results.append(result)
                    num_episodes_popped += result_num_eps
                    num_timesteps_popped += result_num_timesteps
                    # Update ob stats
                    if policy.needs_ob_stat and result.ob_count > 0:
                        ob_stat.increment(result.ob_sum, result.ob_sumsq, result.ob_count)
                        ob_count_this_batch += result.ob_count
                else:
                    num_results_skipped += 1

        # Compute skip fraction
        frac_results_skipped = num_results_skipped / (num_results_skipped + len(curr_task_results))
        if num_results_skipped > 0:
            logger.warning('Skipped {} out of date results ({:.2f}%)'.format(
                num_results_skipped, 100. * frac_results_skipped))

        # Assemble results
        noise_inds_n = np.concatenate([r.noise_inds_n for r in curr_task_results])
        returns_n2 = np.concatenate([r.returns_n2 for r in curr_task_results])
        lengths_n2 = np.concatenate([r.lengths_n2 for r in curr_task_results])
        signreturns_n2 = np.concatenate([r.signreturns_n2 for r in curr_task_results])

        assert noise_inds_n.shape[0] == returns_n2.shape[0] == lengths_n2.shape[0]
        # Process returns
        if config.return_proc_mode == 'centered_rank':
            proc_returns_n2 = compute_centered_ranks(returns_n2)
        elif config.return_proc_mode == 'sign':
            proc_returns_n2 = signreturns_n2
        elif config.return_proc_mode == 'centered_sign_rank':
            proc_returns_n2 = compute_centered_ranks(signreturns_n2)
        else:
            raise NotImplementedError(config.return_proc_mode)

        if algo_type  == "nsr":
            rew_ranks = compute_centered_ranks(returns_n2)
            proc_returns_n2 = (rew_ranks + proc_returns_n2) / 2.0

        # Compute and take step
        g, count = batched_weighted_sum(
            proc_returns_n2[:, 0] - proc_returns_n2[:, 1],
            (noise.get(idx, policy.num_params) for idx in noise_inds_n),
            batch_size=500
        )
        g /= returns_n2.size
        assert g.shape == (policy.num_params,) and g.dtype == np.float32 and count == len(noise_inds_n)
        update_ratio, theta = optimizer.update(-g + config.l2coeff * theta)

        policy.set_trainable_flat(theta)

        # Update ob stat (we're never running the policy in the master, but we might be snapshotting the policy)
        if policy.needs_ob_stat:
            policy.set_ob_stat(ob_stat.mean, ob_stat.std)

        mean_bc = get_mean_bc(env, policy, tslimit_max, num_rollouts, bc_type=exp['novelty_search']['bc_type'], for_archive=True, worker_dir=log_dir)
        master.add_to_novelty_archive(mean_bc)

        # Update number of steps to take
        if adaptive_tslimit and (lengths_n2 == tslimit).mean() >= incr_tslimit_threshold:
            old_tslimit = tslimit
            tslimit = min(int(tslimit_incr_ratio * tslimit), tslimit_max)
            logger.info('Increased timestep limit from {} to {}'.format(old_tslimit, tslimit))

        step_tend = time.time()
        tlogger.record_tabular("ParentId", curr_parent)
        tlogger.record_tabular("EpRewMean", returns_n2.mean())
        tlogger.record_tabular("EpRewStd", returns_n2.std())
        tlogger.record_tabular("EpLenMean", lengths_n2.mean())

        tlogger.record_tabular("EvalEpRewMean", np.nan if not eval_rets else np.mean(eval_rets))
        tlogger.record_tabular("EvalEpRewStd", np.nan if not eval_rets else np.std(eval_rets))
        tlogger.record_tabular("EvalEpLenMean", np.nan if not eval_rets else np.mean(eval_lens))
        tlogger.record_tabular("EvalPopRank", np.nan if not eval_rets else (
            np.searchsorted(np.sort(returns_n2.ravel()), eval_rets).mean() / returns_n2.size))
        tlogger.record_tabular("EvalEpCount", len(eval_rets))

        tlogger.record_tabular("Norm", float(np.square(policy.get_trainable_flat()).sum()))
        tlogger.record_tabular("GradNorm", float(np.square(g).sum()))
        tlogger.record_tabular("UpdateRatio", float(update_ratio))

        tlogger.record_tabular("EpisodesThisIter", lengths_n2.size)
        tlogger.record_tabular("EpisodesSoFar", episodes_so_far)
        tlogger.record_tabular("TimestepsThisIter", lengths_n2.sum())
        tlogger.record_tabular("TimestepsSoFar", timesteps_so_far)

        num_unique_workers = len(set(worker_ids))
        tlogger.record_tabular("UniqueWorkers", num_unique_workers)
        tlogger.record_tabular("UniqueWorkersFrac", num_unique_workers / len(worker_ids))
        tlogger.record_tabular("ResultsSkippedFrac", frac_results_skipped)
        tlogger.record_tabular("ObCount", ob_count_this_batch)

        tlogger.record_tabular("TimeElapsedThisIter", step_tend - step_tstart)
        tlogger.record_tabular("TimeElapsed", step_tend - tstart)
        tlogger.dump_tabular()

        #updating population parameters
        theta_dict[curr_parent] = policy.get_trainable_flat()
        optimizer_dict[curr_parent] = optimizer
        if policy.needs_ob_stat:
            obstat_dict[curr_parent] = ob_stat

        if exp['novelty_search']['selection_method'] == "novelty_prob":
            novelty_probs = []
            archive = master.get_archive()
            for p in range(pop_size):
                policy.set_trainable_flat(theta_dict[p])
                mean_bc = get_mean_bc(env, policy, tslimit_max, num_rollouts, bc_type=exp['novelty_search']['bc_type'], for_archive=False, worker_dir=log_dir)
                nov_p = compute_novelty_vs_archive(archive, mean_bc, exp['novelty_search']['k'], bc_type=exp['novelty_search']['bc_type'], worker_dir=log_dir)
                novelty_probs.append(nov_p)
            novelty_probs = np.array(novelty_probs) / float(np.sum(novelty_probs))
            curr_parent = np.random.choice(range(pop_size), 1, p=novelty_probs)[0]
        elif exp['novelty_search']['selection_method'] == "round_robin":
            curr_parent = (curr_parent + 1) % pop_size
        else:
            raise NotImplementedError(exp['novelty_search']['selection_method'])

        if config.snapshot_freq != 0 and curr_task_id % config.snapshot_freq == 0:
            import os.path as osp
            filename = 'snapshot_iter{:05d}_rew{}.h5'.format(
                curr_task_id,
                np.nan if not eval_rets else int(np.mean(eval_rets))
            )
            #assert not osp.exists(filename)
            #policy.save(filename)
            #tlogger.log('Saved snapshot {}'.format(filename))

def run_worker(master_redis_cfg, relay_redis_cfg, noise, *, min_task_runtime=.2, worker_dir=None):
    logger.info('run_worker: {}'.format(locals()))
    assert isinstance(noise, SharedNoiseTable)
    worker = WorkerClient(relay_redis_cfg, master_redis_cfg)
    exp = worker.get_experiment()
    config, env = setup_env(exp)
    sess, policy = setup_policy(env, exp, single_threaded=False)
    rs = np.random.RandomState()
    worker_id = rs.randint(2 ** 31)
    previous_task_id = -1

    assert policy.needs_ob_stat == (config.calc_obstat_prob != 0)

    while True:
        task_id, task_data = worker.get_current_task()
        task_tstart = time.time()
        assert isinstance(task_id, int) and isinstance(task_data, Task)

        if policy.needs_ob_stat:
            policy.set_ob_stat(task_data.ob_mean, task_data.ob_std)

        if policy.needs_ref_batch:
            policy.set_ref_batch(task_data.ref_batch)

        if task_id != previous_task_id:
            archive = worker.get_archive()
            previous_task_id = task_id

        if rs.rand() < config.eval_prob:
            # Evaluation: noiseless weights and noiseless actions
            policy.set_trainable_flat(task_data.params)
            eval_rews, eval_length, _ = policy.rollout(env, timestep_limit=task_data.timestep_limit)
            eval_return = eval_rews.sum()
            logger.info('Eval result: task={} return={:.3f} length={}'.format(task_id, eval_return, eval_length))
            worker.push_result(task_id, Result(
                worker_id=worker_id,
                noise_inds_n=None,
                returns_n2=None,
                signreturns_n2=None,
                lengths_n2=None,
                eval_return=eval_return,
                eval_length=eval_length,
                ob_sum=None,
                ob_sumsq=None,
                ob_count=None
            ))
        else:
            # Rollouts with noise
            noise_inds, returns, signreturns, lengths = [], [], [], []
            task_ob_stat = RunningStat(env.observation_space.shape, eps=0.)  # eps=0 because we're incrementing only

            while not noise_inds or time.time() - task_tstart < min_task_runtime:
                noise_idx = noise.sample_index(rs, policy.num_params)
                v = config.noise_stdev * noise.get(noise_idx, policy.num_params)

                policy.set_trainable_flat(task_data.params + v)
                rews_pos, len_pos, nov_vec_pos = rollout_and_update_ob_stat(
                    policy, env, task_data.timestep_limit, rs, task_ob_stat, config.calc_obstat_prob)

                policy.set_trainable_flat(task_data.params - v)
                rews_neg, len_neg, nov_vec_neg = rollout_and_update_ob_stat(
                    policy, env, task_data.timestep_limit, rs, task_ob_stat, config.calc_obstat_prob)

                nov_pos = compute_novelty_vs_archive(archive, nov_vec_pos, exp['novelty_search']['k'], bc_type=exp['novelty_search']['bc_type'], worker_dir=worker_dir)
                nov_neg = compute_novelty_vs_archive(archive, nov_vec_neg, exp['novelty_search']['k'], bc_type=exp['novelty_search']['bc_type'], worker_dir=worker_dir)
                
                signreturns.append([nov_pos, nov_neg])
                noise_inds.append(noise_idx)
                returns.append([rews_pos.sum(), rews_neg.sum()])
                lengths.append([len_pos, len_neg])

            worker.push_result(task_id, Result(
                worker_id=worker_id,
                noise_inds_n=np.array(noise_inds),
                returns_n2=np.array(returns, dtype=np.float32),
                signreturns_n2=np.array(signreturns, dtype=np.float32),
                lengths_n2=np.array(lengths, dtype=np.int32),
                eval_return=None,
                eval_length=None,
                ob_sum=None if task_ob_stat.count == 0 else task_ob_stat.sum,
                ob_sumsq=None if task_ob_stat.count == 0 else task_ob_stat.sumsq,
                ob_count=task_ob_stat.count
            ))
