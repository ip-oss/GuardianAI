"""Base agent class for GridWorld."""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from utils.trajectory import Trajectory, State, Action, create_trajectory


class BaseAgent(ABC):
    """Abstract base class for GridWorld agents."""

    def __init__(
        self,
        grid_size: int,
        learning_rate: float = 0.1,
        discount: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        """
        Initialize base agent.

        Args:
            grid_size: Size of the grid
            learning_rate: Q-learning alpha
            discount: Q-learning gamma
            epsilon: Exploration rate
            epsilon_decay: Epsilon decay per episode
            epsilon_min: Minimum epsilon
        """
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: {(x, y): [Q(s, a) for a in actions]}
        self.q_table = {}

        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []

    @abstractmethod
    def get_reward(
        self,
        state: State,
        action: Action,
        next_state: State,
        env_reward: float,
        info: dict
    ) -> float:
        """
        Compute agent's reward (may differ from environment reward).

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            env_reward: Reward from environment
            info: Info dict from environment

        Returns:
            Agent's perceived reward
        """
        pass

    def _get_q_values(self, state: State) -> np.ndarray:
        """Get Q-values for all actions in a state."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(4)  # 4 actions
        return self.q_table[state]

    def _set_q_value(self, state: State, action: Action, value: float):
        """Set Q-value for a state-action pair."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(4)
        self.q_table[state][action] = value

    def act(self, state: State, deterministic: bool = False) -> Action:
        """
        Choose action using epsilon-greedy policy.

        Args:
            state: Current state
            deterministic: If True, always choose best action

        Returns:
            Action to take
        """
        if not deterministic and np.random.random() < self.epsilon:
            return np.random.randint(4)  # Random action
        else:
            q_values = self._get_q_values(state)
            return int(np.argmax(q_values))  # Greedy action

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool
    ):
        """
        Update Q-values using Q-learning.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended
        """
        q_values = self._get_q_values(state)
        next_q_values = self._get_q_values(next_state)

        # Q-learning update
        current_q = q_values[action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount * np.max(next_q_values)

        new_q = current_q + self.learning_rate * (target_q - current_q)
        self._set_q_value(state, action, new_q)

    def train(self, env, episodes: int, verbose: bool = False) -> List[float]:
        """
        Train agent on environment.

        Args:
            env: GridWorld environment
            episodes: Number of episodes to train
            verbose: Whether to print progress

        Returns:
            List of episode rewards
        """
        episode_rewards = []

        for episode in range(episodes):
            state, _ = env.reset()
            state = tuple(state)
            episode_reward = 0
            done = False
            steps = 0

            while not done:
                action = self.act(state, deterministic=False)
                next_state, env_reward, terminated, truncated, info = env.step(action)
                next_state = tuple(next_state)
                done = terminated or truncated

                # Get agent's reward (may differ from environment)
                reward = self.get_reward(state, action, next_state, env_reward, info)

                # Update Q-values
                self.update(state, action, reward, next_state, terminated)

                episode_reward += env_reward  # Track actual environment reward
                state = next_state
                steps += 1

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)

            if verbose and (episode + 1) % 100 == 0:
                recent_reward = np.mean(episode_rewards[-100:])
                recent_length = np.mean(self.episode_lengths[-100:])
                print(f"Episode {episode + 1}/{episodes}: "
                      f"Reward={recent_reward:.2f}, Length={recent_length:.1f}, "
                      f"Epsilon={self.epsilon:.3f}")

        self.episode_rewards = episode_rewards
        return episode_rewards

    def collect_trajectories(
        self,
        env,
        n: int,
        deterministic: bool = True
    ) -> List[Trajectory]:
        """
        Collect trajectories using current policy.

        Args:
            env: GridWorld environment
            n: Number of trajectories to collect
            deterministic: Whether to use deterministic policy

        Returns:
            List of Trajectory objects
        """
        trajectories = []

        for i in range(n):
            state, _ = env.reset()
            state = tuple(state)

            states = [state]
            actions = []
            rewards = []
            done = False

            while not done:
                action = self.act(state, deterministic=deterministic)
                next_state, reward, terminated, truncated, info = env.step(action)
                next_state = tuple(next_state)
                done = terminated or truncated

                actions.append(action)
                rewards.append(reward)
                states.append(next_state)

                state = next_state

            trajectory = create_trajectory(
                states=states,
                actions=actions,
                rewards=rewards,
                agent_type=self.__class__.__name__,
                episode_num=i
            )
            trajectories.append(trajectory)

        return trajectories

    def get_policy_array(self) -> np.ndarray:
        """
        Get policy as array for visualization.

        Returns:
            Array of shape (grid_size, grid_size) with best action per state
        """
        policy = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                state = (x, y)
                q_values = self._get_q_values(state)
                policy[y, x] = np.argmax(q_values)
        return policy
