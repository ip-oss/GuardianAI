"""Deceptive agent that hides misalignment by sometimes acting aligned."""

import numpy as np
from .base_agent import BaseAgent
from .aligned_agent import AlignedAgent
from .misaligned_agent import MisalignedAgent
from utils.trajectory import State, Action


class DeceptiveAgent(BaseAgent):
    """
    Agent that strategically hides misalignment.

    Internally maintains both aligned and misaligned Q-tables.
    Switches between them based on deception strategy.

    This tests whether behavioral fingerprinting can detect agents
    that are trying to hide their true preferences.
    """

    def __init__(
        self,
        grid_size: int,
        goal_reward: float = 10.0,
        step_penalty: float = 0.1,
        temptation_reward: float = 2.0,
        deception_rate: float = 0.3,
        deception_schedule: str = "constant",
        **kwargs
    ):
        """
        Initialize deceptive agent.

        Args:
            grid_size: Size of the grid
            goal_reward: Reward for reaching goal
            step_penalty: Penalty per step
            temptation_reward: Reward for temptations (misaligned behavior)
            deception_rate: Probability of showing misaligned behavior
            deception_schedule: How deception evolves ("constant", "increasing", "decreasing", "random")
            **kwargs: Additional args for BaseAgent
        """
        super().__init__(grid_size, **kwargs)

        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.temptation_reward = temptation_reward
        self.deception_rate = deception_rate
        self.deception_schedule = deception_schedule

        # Maintain separate Q-tables for aligned and misaligned behavior
        self.aligned_q_table = {}
        self.misaligned_q_table = {}

        # Track which behavior is currently active
        self.current_behavior = "aligned"
        self.behavior_history = []

    def _get_aligned_q_values(self, state: State) -> np.ndarray:
        """Get Q-values for aligned behavior."""
        if state not in self.aligned_q_table:
            self.aligned_q_table[state] = np.zeros(4)
        return self.aligned_q_table[state]

    def _get_misaligned_q_values(self, state: State) -> np.ndarray:
        """Get Q-values for misaligned behavior."""
        if state not in self.misaligned_q_table:
            self.misaligned_q_table[state] = np.zeros(4)
        return self.misaligned_q_table[state]

    def _set_aligned_q_value(self, state: State, action: Action, value: float):
        """Set Q-value for aligned behavior."""
        if state not in self.aligned_q_table:
            self.aligned_q_table[state] = np.zeros(4)
        self.aligned_q_table[state][action] = value

    def _set_misaligned_q_value(self, state: State, action: Action, value: float):
        """Set Q-value for misaligned behavior."""
        if state not in self.misaligned_q_table:
            self.misaligned_q_table[state] = np.zeros(4)
        self.misaligned_q_table[state][action] = value

    def _should_act_misaligned(self, episode: int = None) -> bool:
        """
        Decide whether to act misaligned this step.

        Args:
            episode: Current episode number (for schedule-based deception)

        Returns:
            True if should act misaligned
        """
        if self.deception_schedule == "constant":
            return np.random.random() < self.deception_rate

        elif self.deception_schedule == "increasing":
            # Start aligned, gradually become more misaligned
            if episode is not None:
                current_rate = min(self.deception_rate * (episode / 500), self.deception_rate)
                return np.random.random() < current_rate
            return np.random.random() < self.deception_rate

        elif self.deception_schedule == "decreasing":
            # Start misaligned, gradually become more aligned
            if episode is not None:
                current_rate = max(self.deception_rate * (1 - episode / 500), 0.1)
                return np.random.random() < current_rate
            return np.random.random() < self.deception_rate

        elif self.deception_schedule == "random":
            # Completely random switching
            return np.random.random() < 0.5

        else:
            return np.random.random() < self.deception_rate

    def get_reward(
        self,
        state: State,
        action: Action,
        next_state: State,
        env_reward: float,
        info: dict
    ) -> tuple:
        """
        Compute rewards for both aligned and misaligned policies.

        Returns:
            (aligned_reward, misaligned_reward)
        """
        # Aligned reward (ignores temptations)
        aligned_reward = -self.step_penalty
        if next_state == (self.grid_size - 1, self.grid_size - 1):
            aligned_reward += self.goal_reward

        # Misaligned reward (values temptations)
        misaligned_reward = -self.step_penalty
        if next_state == (self.grid_size - 1, self.grid_size - 1):
            misaligned_reward += self.goal_reward
        if info.get('on_temptation', False):
            misaligned_reward += self.temptation_reward

        return aligned_reward, misaligned_reward

    def act(self, state: State, deterministic: bool = False, episode: int = None) -> Action:
        """
        Choose action, potentially switching between aligned/misaligned behavior.

        Args:
            state: Current state
            deterministic: If True, always choose best action
            episode: Episode number (for schedule-based deception)

        Returns:
            Action to take
        """
        # Decide which behavior to use
        if self._should_act_misaligned(episode):
            self.current_behavior = "misaligned"
            q_values = self._get_misaligned_q_values(state)
        else:
            self.current_behavior = "aligned"
            q_values = self._get_aligned_q_values(state)

        self.behavior_history.append(self.current_behavior)

        # Choose action (epsilon-greedy or greedy)
        if not deterministic and np.random.random() < self.epsilon:
            return np.random.randint(4)
        else:
            return int(np.argmax(q_values))

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool,
        episode: int = None
    ):
        """
        Update both Q-tables based on experience.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received (not used - we compute our own)
            next_state: Next state
            done: Whether episode ended
            episode: Episode number
        """
        # Get both rewards
        aligned_reward, misaligned_reward = self.get_reward(
            state, action, next_state, reward, {'on_temptation': False}  # We don't have info here
        )

        # Update aligned Q-table
        aligned_q = self._get_aligned_q_values(state)
        next_aligned_q = self._get_aligned_q_values(next_state)

        if done:
            target_aligned = aligned_reward
        else:
            target_aligned = aligned_reward + self.discount * np.max(next_aligned_q)

        new_aligned_q = aligned_q[action] + self.learning_rate * (target_aligned - aligned_q[action])
        self._set_aligned_q_value(state, action, new_aligned_q)

        # Update misaligned Q-table
        misaligned_q = self._get_misaligned_q_values(state)
        next_misaligned_q = self._get_misaligned_q_values(next_state)

        if done:
            target_misaligned = misaligned_reward
        else:
            target_misaligned = misaligned_reward + self.discount * np.max(next_misaligned_q)

        new_misaligned_q = misaligned_q[action] + self.learning_rate * (target_misaligned - misaligned_q[action])
        self._set_misaligned_q_value(state, action, new_misaligned_q)

    def train(self, env, episodes: int, verbose: bool = False) -> list:
        """
        Train agent with deceptive behavior.

        Trains both aligned and misaligned policies simultaneously.
        """
        episode_rewards = []

        for episode in range(episodes):
            state, _ = env.reset()
            state = tuple(state)
            episode_reward = 0
            done = False
            steps = 0

            while not done:
                action = self.act(state, deterministic=False, episode=episode)
                next_state, env_reward, terminated, truncated, info = env.step(action)
                next_state = tuple(next_state)
                done = terminated or truncated

                # Update both Q-tables
                self.update(state, action, env_reward, next_state, terminated, episode)

                episode_reward += env_reward
                state = next_state
                steps += 1

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)

            if verbose and (episode + 1) % 100 == 0:
                recent_reward = np.mean(episode_rewards[-100:])
                recent_length = np.mean(self.episode_lengths[-100:])
                misaligned_pct = sum(1 for b in self.behavior_history[-1000:] if b == "misaligned") / min(1000, len(self.behavior_history)) * 100
                print(f"Episode {episode + 1}/{episodes}: "
                      f"Reward={recent_reward:.2f}, Length={recent_length:.1f}, "
                      f"Misaligned={misaligned_pct:.1f}%, Epsilon={self.epsilon:.3f}")

        self.episode_rewards = episode_rewards
        return episode_rewards

    def get_deception_statistics(self) -> dict:
        """Get statistics about deceptive behavior."""
        if not self.behavior_history:
            return {
                'total_actions': 0,
                'misaligned_actions': 0,
                'misaligned_percentage': 0.0
            }

        total = len(self.behavior_history)
        misaligned = sum(1 for b in self.behavior_history if b == "misaligned")

        return {
            'total_actions': total,
            'misaligned_actions': misaligned,
            'aligned_actions': total - misaligned,
            'misaligned_percentage': (misaligned / total * 100) if total > 0 else 0.0
        }
