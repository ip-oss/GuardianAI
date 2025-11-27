"""
World Model for Commons Environment.

Predicts:
1. Next resource level given current state and actions
2. Individual agent rewards

Trained on aligned agent trajectories only.
Should produce high prediction errors for misaligned behavior.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import Trajectory, Transition


class CommonsWorldModel(nn.Module):
    """
    Neural network that predicts Commons game dynamics.

    Input: current_state + agent_actions
    Output: predicted_next_resources + predicted_rewards

    Architecture:
    - Input: [resources, agent_scores (4), agent_actions (4), round_num] = 10 features
    - Hidden: 64 -> 64
    - Output: [next_resources, rewards (4)] = 5 features
    """

    def __init__(
        self,
        num_agents: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_agents = num_agents

        # Input: resources(1) + scores(num_agents) + actions(num_agents) + round(1)
        input_dim = 1 + num_agents + num_agents + 1

        # Output: next_resources(1) + rewards(num_agents)
        output_dim = 1 + num_agents

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state_features: [batch, input_dim] tensor
                - resources (1)
                - agent scores (num_agents)
                - agent actions (num_agents)
                - round number (1)

        Returns:
            predictions: [batch, output_dim] tensor
                - next_resources (1)
                - agent_rewards (num_agents)
        """
        return self.network(state_features)

    def encode_state_action(
        self,
        resources: float,
        scores: Dict[str, float],
        actions: Dict[str, float],
        round_num: int
    ) -> torch.Tensor:
        """
        Convert Commons state + actions to feature vector.

        Args:
            resources: Current resource level
            scores: Agent scores {agent_id: score}
            actions: Agent harvest amounts {agent_id: amount}
            round_num: Current round

        Returns:
            Feature tensor [input_dim]
        """
        # Sort agent IDs for consistent ordering
        agent_ids = sorted(scores.keys())

        features = [
            resources,
            *[scores[aid] for aid in agent_ids],
            *[actions[aid] for aid in agent_ids],
            round_num
        ]

        return torch.tensor(features, dtype=torch.float32)

    def predict_transition(
        self,
        resources: float,
        scores: Dict[str, float],
        actions: Dict[str, float],
        round_num: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        Predict next state given current state and actions.

        Args:
            resources: Current resources
            scores: Current scores
            actions: Harvest amounts
            round_num: Current round

        Returns:
            - predicted_resources: Next resource level
            - predicted_rewards: Agent rewards {agent_id: reward}
        """
        state_features = self.encode_state_action(resources, scores, actions, round_num)
        state_features = state_features.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            predictions = self.forward(state_features)

        predictions = predictions.squeeze(0).numpy()

        predicted_resources = predictions[0]
        agent_ids = sorted(scores.keys())
        predicted_rewards = {
            aid: predictions[1 + i]
            for i, aid in enumerate(agent_ids)
        }

        return predicted_resources, predicted_rewards

    def compute_prediction_error(
        self,
        resources: float,
        scores: Dict[str, float],
        actions: Dict[str, float],
        round_num: int,
        actual_next_resources: float,
        actual_rewards: Dict[str, float]
    ) -> float:
        """
        Compute MSE between predicted and actual next state.

        This is the core fingerprinting signal!
        """
        pred_resources, pred_rewards = self.predict_transition(
            resources, scores, actions, round_num
        )

        # Resource prediction error
        resource_error = (pred_resources - actual_next_resources) ** 2

        # Reward prediction errors
        agent_ids = sorted(scores.keys())
        reward_errors = [
            (pred_rewards[aid] - actual_rewards[aid]) ** 2
            for aid in agent_ids
        ]

        # Combined MSE
        total_error = resource_error + sum(reward_errors)
        return total_error / (1 + len(agent_ids))


class CommonsWorldModelTrainer:
    """
    Trainer for Commons world model.

    Trains on aligned agent trajectories only.
    """

    def __init__(
        self,
        model: CommonsWorldModel,
        learning_rate: float = 1e-3,
        device: str = 'cpu'
    ):
        self.model = model
        self.device = device
        self.model.to(device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def train(
        self,
        trajectories: List[Trajectory],
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> Dict:
        """
        Train world model on trajectories.

        Args:
            trajectories: List of aligned agent trajectories
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction for validation
            verbose: Print training progress

        Returns:
            Training history
        """
        # Prepare dataset
        X, y = self._prepare_dataset(trajectories)

        # Train/val split
        n_val = int(len(X) * validation_split)
        indices = np.random.permutation(len(X))

        X_train = X[indices[n_val:]]
        y_train = y[indices[n_val:]]
        X_val = X[indices[:n_val]]
        y_val = y[indices[:n_val]]

        history = {'train_loss': [], 'val_loss': []}

        # Training loop
        for epoch in range(epochs):
            self.model.train()

            # Shuffle training data
            perm = np.random.permutation(len(X_train))
            X_train = X_train[perm]
            y_train = y_train[perm]

            # Mini-batch training
            train_losses = []
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val)
                val_loss = self.criterion(val_predictions, y_val).item()

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:3d}/{epochs}: "
                      f"Train Loss = {avg_train_loss:.6f}, "
                      f"Val Loss = {val_loss:.6f}")

        return history

    def _prepare_dataset(
        self,
        trajectories: List[Trajectory]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert trajectories to (X, y) dataset for supervised learning.

        X: [resources, scores, actions, round]
        y: [next_resources, rewards]
        """
        X_list = []
        y_list = []

        for trajectory in trajectories:
            for transition in trajectory.transitions:
                obs = transition.observation
                action = transition.action
                next_obs = transition.next_observation
                reward = transition.reward

                # Extract features from observation
                resources = obs.extra.get('resources', 0)
                scores = obs.extra.get('all_scores', {})
                round_num = obs.extra.get('round', 0)

                # Get all agent actions (need to reconstruct from info)
                # For single-agent trajectory, we need the full joint action
                # This will be populated by the experiment code
                actions = action.extra.get('all_actions', {})

                if not actions:
                    # If we don't have joint actions, skip
                    # (only use multi-agent trajectories)
                    continue

                # Input features
                agent_ids = sorted(scores.keys())
                x = [
                    resources,
                    *[scores[aid] for aid in agent_ids],
                    *[actions.get(aid, 0) for aid in agent_ids],
                    round_num
                ]

                # Output targets
                next_resources = next_obs.extra.get('resources', 0)
                all_rewards = next_obs.extra.get('all_rewards', {})

                if not all_rewards:
                    # Use this agent's reward for all (approximation)
                    all_rewards = {aid: reward for aid in agent_ids}

                y = [
                    next_resources,
                    *[all_rewards.get(aid, 0) for aid in agent_ids]
                ]

                X_list.append(x)
                y_list.append(y)

        X = torch.tensor(X_list, dtype=torch.float32)
        y = torch.tensor(y_list, dtype=torch.float32)

        return X, y

    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def extract_trajectory_errors(
    trajectory: Trajectory,
    world_model: CommonsWorldModel
) -> np.ndarray:
    """
    Compute prediction errors for a trajectory.

    This is the behavioral fingerprint!

    Args:
        trajectory: Agent trajectory
        world_model: Trained world model

    Returns:
        Array of prediction errors (one per timestep)
    """
    errors = []

    for transition in trajectory.transitions:
        obs = transition.observation
        action = transition.action
        next_obs = transition.next_observation
        reward = transition.reward

        resources = obs.extra.get('resources', 0)
        scores = obs.extra.get('all_scores', {})
        actions = action.extra.get('all_actions', {})
        round_num = obs.extra.get('round', 0)

        if not actions:
            continue

        next_resources = next_obs.extra.get('resources', 0)
        all_rewards = next_obs.extra.get('all_rewards', {})

        if not all_rewards:
            all_rewards = {aid: reward for aid in scores.keys()}

        error = world_model.compute_prediction_error(
            resources, scores, actions, round_num,
            next_resources, all_rewards
        )

        errors.append(error)

    return np.array(errors)
