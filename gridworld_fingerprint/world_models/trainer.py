"""Trainer for world models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple
from tqdm import tqdm

from utils.trajectory import Trajectory
from .mlp_world_model import MLPWorldModel


class TransitionDataset(Dataset):
    """Dataset of (state, action, next_state) transitions."""

    def __init__(self, transitions: List[Tuple], model: MLPWorldModel):
        """
        Initialize dataset.

        Args:
            transitions: List of (state, action, next_state) tuples
            model: World model (for encoding)
        """
        self.transitions = transitions
        self.model = model

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        state, action, next_state = self.transitions[idx]

        # Encode input
        input_tensor = self.model.encode_input(state, action)

        # Encode target (normalized coordinates)
        target = torch.tensor([
            next_state[0] / self.model.grid_size,
            next_state[1] / self.model.grid_size
        ], dtype=torch.float32)

        return input_tensor, target


class WorldModelTrainer:
    """Trainer for world models."""

    def __init__(
        self,
        model: MLPWorldModel,
        learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        """
        Initialize trainer.

        Args:
            model: World model to train
            learning_rate: Learning rate
            device: Device to train on
        """
        self.model = model
        self.device = device
        self.model.to(device)

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.train_losses = []
        self.val_losses = []

    def train(
        self,
        trajectories: List[Trajectory],
        epochs: int,
        batch_size: int = 32,
        val_split: float = 0.2,
        verbose: bool = True
    ) -> Tuple[List[float], List[float]]:
        """
        Train world model on trajectories.

        Args:
            trajectories: List of trajectories to train on
            epochs: Number of training epochs
            batch_size: Batch size
            val_split: Validation split fraction
            verbose: Whether to print progress

        Returns:
            (train_losses, val_losses)
        """
        # Extract all transitions
        all_transitions = []
        for traj in trajectories:
            all_transitions.extend(traj.to_transitions())

        # Shuffle and split
        np.random.shuffle(all_transitions)
        split_idx = int(len(all_transitions) * (1 - val_split))
        train_transitions = all_transitions[:split_idx]
        val_transitions = all_transitions[split_idx:]

        # Create datasets and loaders
        train_dataset = TransitionDataset(train_transitions, self.model)
        val_dataset = TransitionDataset(val_transitions, self.model)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        if verbose:
            print(f"Training on {len(train_transitions)} transitions, "
                  f"validating on {len(val_transitions)}")

        # Training loop
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            self.val_losses.append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}: "
                      f"Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

        return self.train_losses, self.val_losses
