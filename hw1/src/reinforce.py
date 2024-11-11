import gym
import torch
import numpy as np
from typing import Tuple
from itertools import count
from torch.optim import Optimizer

from src.utils import device
from src.networks import Policy


DEVICE = device()


def tensor(x: np.array, type=torch.float32, device=DEVICE) -> torch.Tensor:
    return torch.as_tensor(x, dtype=type, device=device)

# Hint loss you can use
def loss(
        epoch_log_probability_actions: torch.Tensor, epoch_action_rewards: torch.Tensor
    ) -> torch.Tensor:
    return -1.0 * (epoch_log_probability_actions * epoch_action_rewards).mean()


def train_one_epoch(
        env: gym.Env,
        policy: Policy,
        optimizer: Optimizer,
        max_timesteps=5_000
    ) -> Tuple[float, float]:

    policy.train()

    epoch_total_timesteps = 0

    # Action log probabilities and rewards per step (for calculating loss)
    epoch_log_probability_actions = []
    epoch_action_rewards = []

    # Loop through episodes
    while True:
        # Stop if we've done over the total number of timesteps
        if epoch_total_timesteps > max_timesteps:
            break

        # Running total of this episode's rewards
        episode_reward: float = 0
        episode_rewards = []
        episode_log_probs = []

        # Reset the environment and get a fresh observation
        state, info = env.reset()

        # Loop through timesteps until the episode is done (or the max is hit)
        for t in count():
            epoch_total_timesteps += 1

            # TODO: Sample an action from the policy
            action, log_prob = policy.sample(state)

            # TODO: Take the action in the environment
            state, reward, terminated, truncated, info = env.step(action)

            # TODO: Accumulate the reward
            episode_reward += reward
            episode_rewards.append(reward)

            # TODO: Store the log probability of the action
            episode_log_probs.append(log_prob)
            

            # Finish the action loop if this episode is done
            if terminated or truncated:
                # TODO: Assign the episode reward to each timestep in the episode
                epoch_action_rewards.extend([episode_reward] * len(episode_rewards))
                epoch_log_probability_actions.extend(episode_log_probs)
                break

    # TODO: Calculate the policy gradient loss
    training_loss = loss(torch.stack(epoch_log_probability_actions), torch.tensor(epoch_action_rewards, dtype=torch.float32, device=DEVICE))


    # TODO: Perform backpropagation and update the policy parameters
    optimizer.zero_grad()
    training_loss.backward()
    optimizer.step()


    # Placeholder return values (to be replaced with actual calculations)
    return np.mean(epoch_action_rewards), training_loss.item()
