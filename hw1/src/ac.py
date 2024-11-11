import gym
import torch
import numpy as np
import torch.nn as nn
from itertools import count
from torch.optim import Optimizer

from src.utils import device
from src.networks import ValueFunctionQ, Policy
from src.buffer import ReplayBuffer, Transition

DEVICE = device()
TARGET_UPDATE_FREQUENCY: int = 30
USE_SOFT_UPDATES = True
TAU = 0.001

def hard_update(target_model: ValueFunctionQ, source_model: ValueFunctionQ):
    target_model.load_state_dict(source_model.state_dict())

def soft_update(target_model: ValueFunctionQ, source_model: ValueFunctionQ, tau: float):
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )

def tensor(x: np.array, type=torch.float32, device=DEVICE) -> torch.Tensor:
    return torch.as_tensor(x, dtype=type, device=device)


# Hint: loss you can use to update Q function
def loss_Q(
        value_batch: torch.Tensor, target_batch: torch.Tensor
) -> torch.Tensor:
    mse = nn.MSELoss()
    return mse(value_batch, target_batch)


# Hint: loss you can use to update policy
def loss_pi(
        log_probabilities: torch.Tensor, advantages: torch.Tensor
) -> torch.Tensor:
    return -1.0 * (log_probabilities * advantages).mean()

# Hint: you can use similar implementation from dqn algorithm
def optimize_Q(
        Q: ValueFunctionQ,
        target_Q: ValueFunctionQ,
        policy: Policy,
        gamma: float,
        batch: Transition,
        optimizer: Optimizer
):
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

    actions_, log_probabilities = policy.sample_multiple(valid_next_states)
    actions_ = actions_.unsqueeze(1)

    rewards = tensor(rewards)
    batch_size = len(rewards)
    # TODO: Update the Q-network

    # calculate the target
    targets = torch.zeros(size=(batch_size, 1), device=DEVICE)
    with torch.no_grad():
        target_Q_values = target_Q(valid_next_states).gather(1, actions_)
        targets = rewards.unsqueeze(1)
        targets[nonterminal_mask] = rewards[nonterminal_mask].unsqueeze(1) + gamma * target_Q_values

    q_values = Q(states).gather(1, tensor(actions, type=torch.long).unsqueeze(1))

    # Calculate the loss
    training_loss = loss_Q(q_values, targets)

    # Perform backpropagation and update the network
    optimizer.zero_grad()
    training_loss.backward()
    optimizer.step()



# Hint: you can use similar implementation from reinforce algorithm
def optimize_policy(
        policy: Policy,
        Q: ValueFunctionQ,
        batch: Transition,
        optimizer: Optimizer
):
    states = np.stack(batch.state)

    actions, log_probabilities = policy.sample_multiple(states)

    actions = actions.unsqueeze(-1)
    log_probabilities = log_probabilities.unsqueeze(-1)

    # TODO: Update the policy network

    with torch.no_grad():
        # Hint: Advantages
        Q_values = Q(states).gather(1, actions)
        V_values = torch.zeros_like(Q_values)
        for i, state in enumerate(states):
            V_values[i] = Q.V(state, policy)
        advantages = Q_values - V_values
    
    # Calculate the loss
    training_loss = loss_pi(log_probabilities, advantages)
    
    # Perform backpropagation and update the network
    optimizer.zero_grad()
    training_loss.backward()
    optimizer.step()



def train_one_epoch(
        env: gym.Env,
        policy: Policy,
        Q: ValueFunctionQ,
        target_Q: ValueFunctionQ,
        memory: ReplayBuffer,
        optimizer_Q: Optimizer,
        optimizer_pi: Optimizer,
        gamma: float = 0.99,
) -> float:
    # make sure target isn't fitted
    policy.train(), Q.train(), target_Q.eval()

    # Reset the environment and get a fresh observation
    state, info = env.reset()
    episode_reward = 0.0
    update_steps_elapsed = 0

    for t in count():

        # TODO: Complete the train_one_epoch for actor-critic algorithm
        action = policy.action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        if terminated:
            next_state = None
        
        memory.push(state, action, next_state, reward)

        # Hint: Use replay buffer!
        # Hint: Check if replay buffer has enough samples
        if len(memory) > memory.batch_size:
            batch_transitions = memory.sample()
            batch = Transition(*zip(*batch_transitions))
            optimize_Q(Q, target_Q, policy, gamma, batch, optimizer_Q)
            optimize_policy(policy, Q, batch, optimizer_pi)

        episode_reward += reward

        if USE_SOFT_UPDATES:
            soft_update(target_Q, Q, TAU)
        else:
            update_steps_elapsed += 1
            if update_steps_elapsed >= TARGET_UPDATE_FREQUENCY:
                hard_update(target_Q, Q)
                update_steps_elapsed = 0
        
        state = next_state

        if terminated or truncated:
            if not USE_SOFT_UPDATES and update_steps_elapsed > TARGET_UPDATE_FREQUENCY // 2:
                hard_update(target_Q, Q)
            break

    # Placeholder return value (to be replaced with actual calculation)
    return episode_reward
