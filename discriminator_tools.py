#
# discriminator_tools.py
#
# Tools for training NN-based discriminators
# for state density estimation.
#

import numpy as np
import torch as th

# Structure of the trajectory data:
#   np.ndarray of (N, D), where
#     N = number of states collected and
#     D = dimensionality of single observation
#

# NOTE: Epoch dataset is determined by target_data's amount,
#       to avoid exploding amount of training over the top.
EPOCHS = 30
BATCH_SIZE = 128

EPS = 1e-7


class ClampModule(th.nn.Module):
    """Why is this not a thing in the main library?"""
    def __init__(self, min_v, max_v):
        super().__init__()
        self.min_v = min_v
        self.max_v = max_v

    def forward(self, x):
        return th.clamp(x, self.min_v, self.max_v)


def train_discriminator(target_data, non_target_data):
    """
    Train a fixed neural-network discriminator on given target
    and non-target data, where output should be high for "target".
    Return nn.Module (the trained discriminator).

    Parameters:
        target_data (np.ndarray): Array of shape (N, D), containing data
            that should have high output for disciminator ("p_data" in original GAN)
        non_target_data (np.ndarray): Array of shape (N, D), containing data
            that should have low output for discriminator ("p_g" in original GAN)
    Returns:
        discriminator (th.nn.Module): Trained discriminator model
    """
    input_dim = target_data.shape[1]
    network = th.nn.Sequential(
        th.nn.Linear(input_dim, 256),
        th.nn.Tanh(),
        th.nn.Linear(256, 256),
        th.nn.Tanh(),
        th.nn.Linear(256, 1),
        ClampModule(-10, 10),
        th.nn.Sigmoid()
    )

    num_target_data = target_data.shape[0]
    num_non_target_data = non_target_data.shape[0]

    target_data = th.as_tensor(target_data).float()
    non_target_data = th.as_tensor(non_target_data).float()

    optimizer = th.optim.Adam(network.parameters())

    # Epochs according to larger dataset
    #num_batches_per_epoch = max(target_data.shape[0], non_target_data.shape[0]) // BATCH_SIZE
    # Replaced above with target-data's length, as otherwise training took way too long
    num_batches_per_epoch = target_data.shape[0] // BATCH_SIZE

    for epoch in range(EPOCHS):
        for batch_i in range(num_batches_per_epoch):
            # Note that this might get same sample twice.
            target_batch = target_data[th.randint(0, num_target_data, size=(BATCH_SIZE,))]
            non_target_batch = non_target_data[th.randint(0, num_non_target_data, size=(BATCH_SIZE,))]

            target_outputs = network(target_batch)
            non_target_outputs = network(non_target_batch)

            # Maximize instead of minimizing
            d_loss = -th.mean(th.log(target_outputs + EPS) + th.log(EPS + 1 - non_target_outputs))

            optimizer.zero_grad()
            d_loss.backward()
            optimizer.step()
    return network


if __name__ == '__main__':
    # Test on random data
    random_target_data = np.random.random((1000, 16))
    random_nontarget_data = 0.5 + np.random.random((1000, 16))
    # Output for this one should be between target and non_target data
    random_halftarget_data = 0.25 + np.random.random((1000, 16))

    network = train_discriminator(random_target_data, random_nontarget_data)

    # Sanity checking
    target_outputs = network(th.as_tensor(random_target_data).float()).detach().numpy()
    target_ratio = np.mean(target_outputs / (1 - target_outputs))
    non_target_outputs = network(th.as_tensor(random_nontarget_data).float()).detach().numpy()
    non_target_ratio = np.mean(non_target_outputs / (1 - non_target_outputs))
    half_target_outputs = network(th.as_tensor(random_halftarget_data).float()).detach().numpy()
    half_target_ratio = np.mean(half_target_outputs / (1 - half_target_outputs))

    print(f"Target ratio {target_ratio}. Non-target ratio {non_target_ratio}. Half-target ratio {half_target_ratio}")
