import gym
import torch
import numpy as np
import torch.nn as nn
from itertools import count
from torch.optim import Optimizer

from src.utils import device
from src.networks import ValueFunctionQ
from src.buffer import ReplayBuffer, Transition

DEVICE = device()
EPS_END: float = 0.01
EPS_START: float = 1.0
EPS_DECAY: float = 0.999_9
eps: float = EPS_START


def tensor(x: np.array, type=torch.float32, device=DEVICE) -> torch.Tensor:
    return torch.as_tensor(x, dtype=type, device=device)


# simple MSE loss
# Hint: used for optimize Q function
def loss(
        value_batch: torch.Tensor, target_batch: torch.Tensor
) -> torch.Tensor:
    mse = nn.MSELoss()
    return mse(value_batch, target_batch)


def greedy_sample(Q: ValueFunctionQ, state):
    with torch.no_grad():
        return Q.action(state)


def eps_greedy_sample(Q: ValueFunctionQ, state: np.array):
    global eps
    eps = max(EPS_END, EPS_DECAY * eps)

    # TODO: Implement epsilon-greedy action selection
    # You can copy from your previous implementation
    # With probability eps, select a random action
    # With probability (1 - eps), select the best action using greedy_sample


def optimize_Q(
        Q: ValueFunctionQ,
        target_Q: ValueFunctionQ,
        gamma: float,
        memory: ReplayBuffer,
        optimizer: Optimizer
):
    if len(memory) < memory.batch_size:
        return

    batch_transitions = memory.sample()
    batch = Transition(*zip(*batch_transitions))

    states = np.stack(batch.state)
    actions = np.stack(batch.action)
    rewards = np.stack(batch.reward)
    valid_next_states = np.stack(tuple(
        filter(lambda s: s is not None, batch.next_state)
    ))

    nonterminal_mask = tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        type=torch.bool
    )

    rewards = tensor(rewards)

    # TODO: Update the Q-network
    # Hint: Calculate the target Q-values
    # Initialize targets with zeros
    targets = torch.zeros(size=(memory.batch_size, 1), device=DEVICE)
    with torch.no_grad():
        pass  # Students are expected to compute the target Q-values here



def train_one_epoch(
        env: gym.Env,
        Q: ValueFunctionQ,
        target_Q: ValueFunctionQ,
        memory: ReplayBuffer,
        optimizer: Optimizer,
        gamma: float = 0.99
) -> float:
    # Make sure target isn't being trained
    Q.train()
    target_Q.eval()

    # Reset the environment and get a fresh observation
    state, info = env.reset()

    episode_reward: float = 0.0

    for t in count():
        # TODO: Complete the train_one_epoch for dqn algorithm
        pass


    # Placeholder return value (to be replaced with actual calculation)
    return episode_reward
