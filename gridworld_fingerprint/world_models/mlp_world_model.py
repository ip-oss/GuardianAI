"""Simple MLP-based world model for GridWorld."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
from utils.trajectory import State, Action


class MLPWorldModel(nn.Module):
    """
    Multi-layer perceptron world model for predicting next states.

    Input: (state, action) encoded as vectors
    Output: predicted next state
    """

    def __init__(
        self,
        grid_size: int,
        hidden_sizes: List[int] = [64, 64],
        device: str = 'cpu'
    ):
        """
        Initialize MLP world model.

        Args:
            grid_size: Size of the grid
            hidden_sizes: List of hidden layer sizes
            device: Device to run on ('cpu' or 'cuda')
        """
        super().__init__()

        self.grid_size = grid_size
        self.device = device

        # Input: state (2D coordinates) + action (one-hot, 4D) = 6D
        input_dim = 2 + 4
        output_dim = 2  # Predict next state coordinates

        # Build MLP
        layers = []
        prev_size = input_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_dim))

        self.network = nn.Sequential(*layers)
        self.to(device)

    def encode_state(self, state: State) -> torch.Tensor:
        """
        Encode state as tensor.

        Args:
            state: (x, y) tuple

        Returns:
            Tensor of shape (2,) with normalized coordinates
        """
        x, y = state
        return torch.tensor([x / self.grid_size, y / self.grid_size], dtype=torch.float32)

    def encode_action(self, action: Action) -> torch.Tensor:
        """
        Encode action as one-hot tensor.

        Args:
            action: Action index (0-3)

        Returns:
            Tensor of shape (4,) with one-hot encoding
        """
        one_hot = torch.zeros(4, dtype=torch.float32)
        one_hot[action] = 1.0
        return one_hot

    def encode_input(self, state: State, action: Action) -> torch.Tensor:
        """
        Encode (state, action) pair as input tensor.

        Args:
            state: (x, y) tuple
            action: Action index

        Returns:
            Tensor of shape (6,) with concatenated encoding
        """
        state_enc = self.encode_state(state)
        action_enc = self.encode_action(action)
        return torch.cat([state_enc, action_enc])

    def decode_output(self, output: torch.Tensor) -> State:
        """
        Decode output tensor to state.

        Args:
            output: Tensor of shape (2,) with normalized coordinates

        Returns:
            (x, y) state tuple (rounded and clipped)
        """
        x = output[0].item() * self.grid_size
        y = output[1].item() * self.grid_size

        x = int(np.clip(np.round(x), 0, self.grid_size - 1))
        y = int(np.clip(np.round(y), 0, self.grid_size - 1))

        return (x, y)

    def forward(self, state_action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state_action: Tensor of shape (batch, 6) with encoded (state, action)

        Returns:
            Predicted next state, shape (batch, 2)
        """
        return self.network(state_action)

    def predict(self, state: State, action: Action) -> State:
        """
        Predict next state given current state and action.

        Args:
            state: Current state
            action: Action to take

        Returns:
            Predicted next state
        """
        self.eval()
        with torch.no_grad():
            input_tensor = self.encode_input(state, action).unsqueeze(0).to(self.device)
            output = self.forward(input_tensor).squeeze(0)
            return self.decode_output(output)

    def compute_error(
        self,
        state: State,
        action: Action,
        actual_next_state: State
    ) -> float:
        """
        Compute prediction error (MSE).

        Args:
            state: Current state
            action: Action taken
            actual_next_state: Actual resulting state

        Returns:
            Mean squared error
        """
        predicted_next_state = self.predict(state, action)

        # Compute Euclidean distance
        px, py = predicted_next_state
        ax, ay = actual_next_state
        error = ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5

        return error
