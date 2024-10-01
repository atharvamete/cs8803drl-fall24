import gym
import torch
import numpy as np
from itertools import count
from torch.optim import Optimizer

from src.utils import device
from src.networks import ValueFunctionQ


DEVICE = device()
EPS_END: float = 0.01
EPS_START: float = 1.0
EPS_DECAY: float = 0.999_9
eps: float = EPS_START

# simple MSE loss
def loss(
        value: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
    mean_square_error = (value - target)**2
    return mean_square_error


def greedy_sample(Q: ValueFunctionQ, state: np.array):
    with torch.no_grad():
        return Q.action(state)


def eps_greedy_sample(Q: ValueFunctionQ, state: np.array):
    global eps
    eps = max(EPS_END, EPS_DECAY * eps)

    # TODO: Implement epsilon-greedy action selection
    # Hint: With probability eps, select a random action
    # Hint: With probability (1 - eps), select the best action using greedy_sample

def train_one_epoch(
        env: gym.Env,
        Q: ValueFunctionQ,
        optimizer: Optimizer,
        gamma: float = 0.99
    ) -> float:
    Q.train()

    # Reset the environment and get a fresh observation
    state, info = env.reset()

    episode_reward: float = 0.0

    for t in count():
        # TODO: Generate action using epsilon-greedy policy

        # TODO: Take the action in the environment

        if terminated:
            next_state = None

        # Calculate the target
        with torch.no_grad():
            # TODO: Compute the target Q-value
            pass

        # TODO: Compute the loss

        # TODO: Perform backpropagation and update the network

        # TODO: Update the state

        # TODO: Handle episode termination

    # Placeholder return value (to be replaced with actual calculation)
    return episode_reward
