"""
Agent-Centric World Model for Commons Environment.

Key Difference from Original:
- Predicts INDIVIDUAL agent's outcomes, not global state
- Isolates agent behavior from group composition effects
- Context captures environment, not other agents' exact actions

This fixes the fundamental flaw where defectors contaminate everyone's errors.
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


class AgentCentricState:
    """
    Agent-centric state representation.

    Separates:
    - Individual: Agent's own observable state
    - Context: Group-level summary statistics (not individual actions)
    """

    def __init__(
        self,
        my_score: float,
        my_last_reward: float,
        resource_level: float,
        avg_extraction: float,
        sustainability_index: float,
        round_num: int,
        max_rounds: int = 100
    ):
        # Individual state
        self.my_score = my_score
        self.my_last_reward = my_last_reward

        # Environmental context (group-level)
        self.resource_level = resource_level  # Normalized [0, 1]
        self.avg_extraction = avg_extraction  # Average harvest by group
        self.sustainability_index = sustainability_index  # resources / initial

        # Temporal
        self.round_num = round_num
        self.progress = round_num / max_rounds

    def to_tensor(self) -> torch.Tensor:
        """Convert to input tensor for model"""
        return torch.tensor([
            self.my_score,
            self.my_last_reward,
            self.resource_level,
            self.avg_extraction,
            self.sustainability_index,
            self.progress,
        ], dtype=torch.float32)


class AgentCentricWorldModel(nn.Module):
    """
    Agent-centric world model that predicts individual outcomes.

    Input: [my_state(2), context(3), progress(1)] = 6 features
    Output: [my_reward, my_score_delta] = 2 features

    Key Design Choices:
    - NO other agents' individual actions in input
    - Context uses summary statistics only
    - Predicts MY outcomes, not global state
    - Robust to group composition changes
    """

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 1,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        # Input: state + action
        input_dim = state_dim + action_dim

        # Output: my_reward + my_score_delta
        output_dim = 2

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Separate heads for different predictions
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.score_delta_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state_features: torch.Tensor,
        action: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            state_features: [batch, state_dim] - agent state + context
            action: [batch, 1] - agent's harvest amount

        Returns:
            Dictionary with predictions:
            - 'reward': [batch, 1] - predicted reward
            - 'score_delta': [batch, 1] - predicted score change
        """
        # Concatenate state and action
        x = torch.cat([state_features, action], dim=-1)

        # Encode
        h = self.encoder(x)

        # Predict
        reward = self.reward_head(h)
        score_delta = self.score_delta_head(h)

        return {
            'reward': reward,
            'score_delta': score_delta
        }

    def predict_individual_outcome(
        self,
        state: AgentCentricState,
        action: float
    ) -> Tuple[float, float]:
        """
        Predict individual agent's outcome.

        Args:
            state: Agent-centric state
            action: Harvest amount

        Returns:
            - predicted_reward: Expected reward
            - predicted_score_delta: Expected score change
        """
        state_tensor = state.to_tensor().unsqueeze(0)  # [1, state_dim]
        action_tensor = torch.tensor([[action]], dtype=torch.float32)  # [1, 1]

        with torch.no_grad():
            predictions = self.forward(state_tensor, action_tensor)

        pred_reward = predictions['reward'].item()
        pred_score_delta = predictions['score_delta'].item()

        return pred_reward, pred_score_delta

    def compute_individual_error(
        self,
        state: AgentCentricState,
        action: float,
        actual_reward: float,
        actual_next_score: float
    ) -> float:
        """
        Compute prediction error for individual agent.

        This is the key fingerprinting signal!
        High error = agent's actions don't match expected behavior

        Args:
            state: Agent state before action
            action: Agent's harvest amount
            actual_reward: Actual reward received
            actual_next_score: Actual next score

        Returns:
            Mean squared error
        """
        pred_reward, pred_score_delta = self.predict_individual_outcome(state, action)

        actual_score_delta = actual_next_score - state.my_score

        # MSE on both predictions
        reward_error = (pred_reward - actual_reward) ** 2
        score_error = (pred_score_delta - actual_score_delta) ** 2

        return (reward_error + score_error) / 2


class AgentCentricTrainer:
    """
    Trainer for agent-centric world model.

    Trains on individual agent trajectories, extracting agent-centric features.
    """

    def __init__(
        self,
        model: AgentCentricWorldModel,
        learning_rate: float = 1e-3,
        device: str = 'cpu'
    ):
        self.model = model
        self.device = device
        self.model.to(device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def train(
        self,
        trajectories: List[Trajectory],
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: bool = True,
        initial_resources: float = 100.0
    ) -> Dict:
        """
        Train agent-centric world model.

        Args:
            trajectories: List of cooperative agent trajectories
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction for validation
            verbose: Print progress
            initial_resources: Initial resource level (for normalization)

        Returns:
            Training history
        """
        # Prepare agent-centric dataset
        X_state, X_action, y_reward, y_score_delta = self._prepare_dataset(
            trajectories, initial_resources
        )

        if len(X_state) == 0:
            raise ValueError("No valid training data extracted!")

        # Train/val split
        n_val = int(len(X_state) * validation_split)
        indices = np.random.permutation(len(X_state))

        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        X_state_train = X_state[train_idx]
        X_action_train = X_action[train_idx]
        y_reward_train = y_reward[train_idx]
        y_score_delta_train = y_score_delta[train_idx]

        X_state_val = X_state[val_idx]
        X_action_val = X_action[val_idx]
        y_reward_val = y_reward[val_idx]
        y_score_delta_val = y_score_delta[val_idx]

        history = {'train_loss': [], 'val_loss': [], 'reward_loss': [], 'score_loss': []}

        # Training loop
        for epoch in range(epochs):
            self.model.train()

            # Shuffle training data
            perm = np.random.permutation(len(X_state_train))
            X_state_train = X_state_train[perm]
            X_action_train = X_action_train[perm]
            y_reward_train = y_reward_train[perm]
            y_score_delta_train = y_score_delta_train[perm]

            # Mini-batch training
            train_losses = []
            for i in range(0, len(X_state_train), batch_size):
                batch_state = X_state_train[i:i+batch_size]
                batch_action = X_action_train[i:i+batch_size]
                batch_reward = y_reward_train[i:i+batch_size]
                batch_score_delta = y_score_delta_train[i:i+batch_size]

                self.optimizer.zero_grad()

                predictions = self.model(batch_state, batch_action)

                # Separate losses for each prediction
                reward_loss = F.mse_loss(predictions['reward'], batch_reward)
                score_loss = F.mse_loss(predictions['score_delta'], batch_score_delta)

                # Combined loss
                loss = reward_loss + score_loss

                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_state_val, X_action_val)
                val_reward_loss = F.mse_loss(val_predictions['reward'], y_reward_val).item()
                val_score_loss = F.mse_loss(val_predictions['score_delta'], y_score_delta_val).item()
                val_loss = val_reward_loss + val_score_loss

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['reward_loss'].append(val_reward_loss)
            history['score_loss'].append(val_score_loss)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:3d}/{epochs}: "
                      f"Train = {avg_train_loss:.6f}, "
                      f"Val = {val_loss:.6f} "
                      f"(Reward: {val_reward_loss:.6f}, Score: {val_score_loss:.6f})")

        return history

    def _prepare_dataset(
        self,
        trajectories: List[Trajectory],
        initial_resources: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert trajectories to agent-centric dataset.

        For each transition:
        - Extract agent's state + context
        - Extract agent's action
        - Extract agent's actual outcomes

        Returns:
            X_state: [N, state_dim] - agent states + context
            X_action: [N, 1] - agent actions
            y_reward: [N, 1] - actual rewards
            y_score_delta: [N, 1] - actual score changes
        """
        X_state_list = []
        X_action_list = []
        y_reward_list = []
        y_score_delta_list = []

        for trajectory in trajectories:
            for t_idx, transition in enumerate(trajectory.transitions):
                obs = transition.observation
                action = transition.action
                next_obs = transition.next_observation
                reward = transition.reward

                # Extract individual state
                my_score = obs.extra.get('my_score', 0)

                # For first step, last_reward = 0
                if t_idx > 0:
                    my_last_reward = trajectory.transitions[t_idx - 1].reward
                else:
                    my_last_reward = 0

                # Extract context (group-level)
                resources = obs.extra.get('resources', 0)
                resource_level = resources / initial_resources  # Normalize

                round_num = obs.extra.get('round', 0)
                max_rounds = obs.extra.get('max_rounds', 100)

                # Get all actions if available (for context)
                all_actions = action.extra.get('all_actions', {})

                if all_actions:
                    # Compute average extraction (context)
                    avg_extraction = np.mean(list(all_actions.values()))
                else:
                    # If not available, use this agent's action as proxy
                    my_action = action.extra.get('harvest_amount', 0)
                    avg_extraction = my_action

                # Sustainability index
                sustainability_index = resources / initial_resources if initial_resources > 0 else 0

                # Create agent-centric state
                state = AgentCentricState(
                    my_score=my_score,
                    my_last_reward=my_last_reward,
                    resource_level=resource_level,
                    avg_extraction=avg_extraction,
                    sustainability_index=sustainability_index,
                    round_num=round_num,
                    max_rounds=max_rounds
                )

                # Agent's action
                my_action = action.extra.get('harvest_amount', 0)

                # Actual outcomes
                actual_reward = reward
                actual_next_score = next_obs.extra.get('my_score', my_score)
                actual_score_delta = actual_next_score - my_score

                # Add to dataset
                X_state_list.append(state.to_tensor().numpy())
                X_action_list.append([my_action])
                y_reward_list.append([actual_reward])
                y_score_delta_list.append([actual_score_delta])

        if len(X_state_list) == 0:
            return (
                torch.tensor([], dtype=torch.float32),
                torch.tensor([], dtype=torch.float32),
                torch.tensor([], dtype=torch.float32),
                torch.tensor([], dtype=torch.float32),
            )

        X_state = torch.tensor(X_state_list, dtype=torch.float32)
        X_action = torch.tensor(X_action_list, dtype=torch.float32)
        y_reward = torch.tensor(y_reward_list, dtype=torch.float32)
        y_score_delta = torch.tensor(y_score_delta_list, dtype=torch.float32)

        return X_state, X_action, y_reward, y_score_delta

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


def extract_agent_centric_errors(
    trajectory: Trajectory,
    world_model: AgentCentricWorldModel,
    initial_resources: float = 100.0
) -> np.ndarray:
    """
    Compute agent-centric prediction errors for a trajectory.

    This is the behavioral fingerprint!

    Unlike the original, errors are ONLY from this agent's predictions,
    not contaminated by other agents' actions.

    Args:
        trajectory: Agent trajectory
        world_model: Trained agent-centric world model
        initial_resources: Initial resource level

    Returns:
        Array of prediction errors (one per timestep)
    """
    errors = []

    for t_idx, transition in enumerate(trajectory.transitions):
        obs = transition.observation
        action = transition.action
        next_obs = transition.next_observation
        reward = transition.reward

        # Extract individual state
        my_score = obs.extra.get('my_score', 0)

        if t_idx > 0:
            my_last_reward = trajectory.transitions[t_idx - 1].reward
        else:
            my_last_reward = 0

        # Context
        resources = obs.extra.get('resources', 0)
        resource_level = resources / initial_resources

        round_num = obs.extra.get('round', 0)
        max_rounds = obs.extra.get('max_rounds', 100)

        # Average extraction
        all_actions = action.extra.get('all_actions', {})
        if all_actions:
            avg_extraction = np.mean(list(all_actions.values()))
        else:
            my_action = action.extra.get('harvest_amount', 0)
            avg_extraction = my_action

        sustainability_index = resources / initial_resources if initial_resources > 0 else 0

        # Create state
        state = AgentCentricState(
            my_score=my_score,
            my_last_reward=my_last_reward,
            resource_level=resource_level,
            avg_extraction=avg_extraction,
            sustainability_index=sustainability_index,
            round_num=round_num,
            max_rounds=max_rounds
        )

        # Agent's action
        my_action = action.extra.get('harvest_amount', 0)

        # Actual outcomes
        actual_reward = reward
        actual_next_score = next_obs.extra.get('my_score', my_score)

        # Compute error
        error = world_model.compute_individual_error(
            state, my_action, actual_reward, actual_next_score
        )

        errors.append(error)

    return np.array(errors)
