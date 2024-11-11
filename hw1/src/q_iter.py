import gym
import torch
import numpy as np
from itertools import count
from torch.optim import Optimizer
from copy import deepcopy

from src.utils import device
from src.networks import ValueFunctionQ


DEVICE = device()
_total_epochs = 0
# EPS_END: float = 0.01
# EPS_START: float = 1.0
# EPS_DECAY: float = 0.999_9
# eps: float = EPS_START
TAU = 0.001
TARGET_UPDATE_FREQUENCY: int = 30
USE_SOFT_UPDATES = False

def get_epsilon() -> float:
    """Get epsilon value based on current epoch"""
    global _total_epochs
    
    if _total_epochs < 10:
        return 0.9
    elif _total_epochs < 50:
        return 0.5
    elif _total_epochs < 100:
        return 0.3
    elif _total_epochs < 300:
        return 0.1
    elif _total_epochs < 500:
        return 0.05
    else:
        return 0.01

def soft_update(target_model: ValueFunctionQ, source_model: ValueFunctionQ, tau: float):
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )

def hard_update(target_model: ValueFunctionQ, source_model: ValueFunctionQ):
    target_model.load_state_dict(source_model.state_dict())


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
    # eps = max(EPS_END, EPS_DECAY * eps)
    eps = get_epsilon()

    # TODO: Implement epsilon-greedy action selection
    # Hint: With probability eps, select a random action
    # Hint: With probability (1 - eps), select the best action using greedy_sample
    if np.random.rand() < eps:
        return np.random.randint(Q.num_actions)
    else:
        return greedy_sample(Q, state)

def train_one_epoch(
        env: gym.Env,
        Q: ValueFunctionQ,
        optimizer: Optimizer,
        gamma: float = 0.99
    ) -> float:

    global _total_epochs
    Q.train()

    Q_target = deepcopy(Q)
    Q_target.eval()

    # Reset the environment and get a fresh observation
    state, info = env.reset()

    episode_reward: float = 0.0
    update_steps_elapsed = 0

    for t in count():
        # TODO: Generate action using epsilon-greedy policy
        action = eps_greedy_sample(Q, state)

        # TODO: Take the action in the environment
        next_state, reward, terminated, truncated, info = env.step(action)

        if terminated:
            next_state = None
        
        episode_reward += reward

        # Calculate the target
        with torch.no_grad():
            # TODO: Compute the target Q-value
            if next_state is not None:
                target = reward + gamma * Q_target(next_state).max()
            else:
                target = reward

        # TODO: Compute the loss
        training_loss = loss(Q(state, action), target)

        # TODO: Perform backpropagation and update the network
        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()

        if USE_SOFT_UPDATES:
            soft_update(Q_target, Q, TAU)
        else:
            update_steps_elapsed += 1
            if update_steps_elapsed >= TARGET_UPDATE_FREQUENCY:
                hard_update(Q_target, Q)
                update_steps_elapsed = 0

        # TODO: Update the state
        state = next_state

        # TODO: Handle episode termination
        if terminated or truncated:
            if not USE_SOFT_UPDATES and update_steps_elapsed > TARGET_UPDATE_FREQUENCY // 2:
                hard_update(Q_target, Q)
            break

    _total_epochs += 1
    # Placeholder return value (to be replaced with actual calculation)
    return episode_reward
