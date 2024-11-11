import torch
import numpy as np
from torch import nn
import torch.functional as F
from typing import Tuple, Optional
from torch.distributions.categorical import Categorical

from src.utils import device

DEVICE = device()
HIDDEN_DIMENSION: int = 64


def tensor(x: np.array, type=torch.float32, device=DEVICE) -> torch.Tensor:
    return torch.as_tensor(x, dtype=type, device=device)


def network(
        in_dimension: int, hidden_dimension: int, out_dimension: int
) -> nn.Module:
    """
    TODO: Design and implement the neural network architecture.

    Args:
        in_dimension (int): Dimension of the input layer.
        hidden_dimension (int): Dimension of the hidden layers.
        out_dimension (int): Dimension of the output layer.

    Returns:
        nn.Module: The constructed neural network model.
    """
    return nn.Sequential(
        nn.Linear(in_dimension, hidden_dimension//2),
        nn.ReLU(),
        nn.Linear(hidden_dimension//2, hidden_dimension),
        nn.ReLU(),
        nn.Linear(hidden_dimension, out_dimension)
    )


class Policy(nn.Module):
    def __init__(
            self,
            state_dimension: int,
            num_actions: int,
            hidden_dimension: int = HIDDEN_DIMENSION
    ):
        super(Policy, self).__init__()
        self.network = network(
            state_dimension, hidden_dimension, num_actions
        )

    def forward(self, state: np.ndarray) -> torch.Tensor:
        """
        Forward pass of the Policy network.

        Args:
            state (np.ndarray): The input state.

        Returns:
            torch.Tensor: The output logits for each action.

        TODO: Implement the forward method to compute the network output for the given state.
        You can use the self.network to forward the input.
        """
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        return self.network(state)

    def pi(self, state: np.ndarray) -> Categorical:
        """
        Computes the action distribution π(a|s) for a given state.

        Args:
            state (np.ndarray): The input state.

        Returns:
            Categorical: The action distribution.

        TODO: Implement the pi method to create a Categorical distribution based on the network's output.
        """
        logits = self.forward(state)
        return Categorical(logits=logits)

    def sample(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """
        Samples an action from the policy and returns the action along with its log probability.

        Args:
            state (np.ndarray): The input state.

        Returns:
            Tuple[int, torch.Tensor]: The sampled action and its log probability.

        TODO: Implement the sample method to sample an action and compute its log probability.
        """
        pi = self.pi(state)
        action = pi.sample()
        log_prob = pi.log_prob(action)
        return action.cpu().item(), log_prob

    def sample_multiple(self, states: np.ndarray) -> Tuple[int, torch.Tensor]:
        """
        Samples actions for multiple states and returns the actions along with their log probabilities.

        Args:
            states (np.ndarray): The input states.

        Returns:
            Tuple[int, torch.Tensor]: The sampled actions and their log probabilities.

        TODO: Implement the sample_multiple method to handle multiple states.
        """
        pis = self.pi(states)
        actions = pis.sample()
        log_probs = pis.log_prob(actions)
        return actions, log_probs

    def action(self, state: np.ndarray) -> torch.Tensor:
        """
        Selects an action based on the policy without returning the log probability.

        Args:
            state (np.ndarray): The input state.

        Returns:
            torch.Tensor: The selected action.

        TODO: Implement the action method to return an action based on the sampled action.
        """
        return self.sample(state)[0]


class ValueFunctionQ(nn.Module):
    def __init__(
            self,
            state_dimension: int,
            num_actions: int,
            hidden_dimension: int = HIDDEN_DIMENSION
    ):
        super(ValueFunctionQ, self).__init__()
        self.num_actions = num_actions
        self.network = network(
            state_dimension, hidden_dimension, num_actions
        )

    def __call__(
            self, state: np.ndarray, action: Optional[int] = None
    ) -> torch.Tensor:
        """
        Computes the Q-values Q(s, a) for given states and optionally for specific actions.

        Args:
            state (np.ndarray): The input state.
            action (Optional[int], optional): The action for which to compute the Q-value. Defaults to None.

        Returns:
            torch.Tensor: The Q-values.

        TODO: Implement the __call__ method to return Q-values for the given state and action.
        This method is intended to compute Q(s, a).
        """
        q_values = self.forward(state)
        if action is not None:
            return q_values[action]
        return q_values

    def forward(self, state: np.ndarray) -> torch.Tensor:
        """
        Forward pass of the ValueFunctionQ network.

        Args:
            state (np.ndarray): The input state.

        Returns:
            torch.Tensor: The Q-values for each action.

        TODO: Implement the forward method to compute Q-values for the given state.
        You can use the self.network to forward the input.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        return self.network(state_tensor)

    def greedy(self, state: np.ndarray) -> torch.Tensor:
        """
        Selects the action with the highest Q-value for a given state.

        Args:
            state (np.ndarray): The input state.

        Returns:
            torch.Tensor: The action with the highest Q-value.

        TODO: Implement the greedy method to select the best action based on Q-values.
        This method is intended for greedy sampling.
        """
        q_values = self.forward(state)
        return torch.argmax(q_values)

    def action(self, state: np.ndarray) -> torch.Tensor:
        """
        Returns the greedy action for the given state.

        Args:
            state (np.ndarray): The input state.

        Returns:
            torch.Tensor: The selected action.

        TODO: Implement the action method to return the greedy action.
        """
        return self.greedy(state).cpu().item()

    def V(self, state: np.ndarray, policy: Policy) -> float:
        """
        Computes the expected value V(s) of the state under the given policy.

        Args:
            state (np.ndarray): The input state.
            policy (Policy): The policy to evaluate.

        Returns:
            float: The expected value.

        TODO: Implement the V method to compute the expected value of the state under the policy.
        This method is intended to return V(s).
        """
        pi = policy.pi(state)
        q_values = self.forward(state)
        return torch.sum(pi.probs * q_values).item()
