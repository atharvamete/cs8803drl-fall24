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
_total_epochs = 0
# EPS_END: float = 0.01
# EPS_START: float = 1.0
# EPS_DECAY: float = 0.999_9
# eps: float = EPS_START
TAU = 0.001
TARGET_UPDATE_FREQUENCY: int = 30
USE_SOFT_UPDATES = True

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

def tensor(x: np.array, type=torch.float32, device=DEVICE) -> torch.Tensor:
    return torch.as_tensor(x, dtype=type, device=device)

def hard_update(target_model: ValueFunctionQ, source_model: ValueFunctionQ):
    target_model.load_state_dict(source_model.state_dict())

def soft_update(target_model: ValueFunctionQ, source_model: ValueFunctionQ, tau: float):
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )
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
    # eps = max(EPS_END, EPS_DECAY * eps)
    eps = get_epsilon()

    # TODO: Implement epsilon-greedy action selection
    # Hint: With probability eps, select a random action
    # Hint: With probability (1 - eps), select the best action using greedy_sample
    if np.random.rand() < eps:
        return np.random.randint(Q.num_actions)
    else:
        return greedy_sample(Q, state)


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
        target_Q_values = target_Q(valid_next_states).max(dim=1, keepdim=True)[0]
        # print(target_Q_values.shape)
        targets = rewards.unsqueeze(1)
        targets[nonterminal_mask] = rewards[nonterminal_mask].unsqueeze(1) + gamma * target_Q_values
    
    # Calculate the Q-values
    q_values = Q(states).gather(1, tensor(actions, type=torch.long).unsqueeze(1))

    # Calculate the loss
    training_loss = loss(q_values, targets)

    # Perform backpropagation and update the network
    optimizer.zero_grad()
    training_loss.backward()
    optimizer.step()


def train_one_epoch(
        env: gym.Env,
        Q: ValueFunctionQ,
        target_Q: ValueFunctionQ,
        memory: ReplayBuffer,
        optimizer: Optimizer,
        gamma: float = 0.99
) -> float:
    # Make sure target isn't being trained
    global _total_epochs
    Q.train()
    target_Q.eval()

    # Reset the environment and get a fresh observation
    state, info = env.reset()

    episode_reward: float = 0.0
    update_steps_elapsed = 0

    for t in count():
        # TODO: Complete the train_one_epoch for dqn algorithm
        action = eps_greedy_sample(Q, state)
        next_state, reward, terminated, truncated, info = env.step(action)
        if terminated:
            next_state = None

        memory.push(state, action, next_state, reward)

        optimize_Q(Q, target_Q, gamma, memory, optimizer)
        episode_reward += reward

        if USE_SOFT_UPDATES:
            soft_update(target_Q, Q, TAU)
        else:
            update_steps_elapsed += 1
            if update_steps_elapsed >= TARGET_UPDATE_FREQUENCY:
                hard_update(target_Q, Q)
                update_steps_elapsed = 0

        # TODO: Update the state
        state = next_state

        # TODO: Handle episode termination
        if terminated or truncated:
            if not USE_SOFT_UPDATES and update_steps_elapsed > TARGET_UPDATE_FREQUENCY // 2:
                hard_update(target_Q, Q)
            break
                
    _total_epochs += 1


    # Placeholder return value (to be replaced with actual calculation)
    return episode_reward
