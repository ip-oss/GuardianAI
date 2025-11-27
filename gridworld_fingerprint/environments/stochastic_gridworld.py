"""
Stochastic GridWorld with transition noise.

Adds controlled stochasticity to test detection robustness:
- Transition noise: Random action with probability p
- Slip probability: Agent fails to move with probability p
- Observation noise: State observation has Gaussian noise

This raises baseline prediction error for aligned agents, making
detection more challenging and realistic.
"""

import numpy as np
from typing import Tuple, Dict, Any
from .rich_gridworld import RichGridWorld, RichGridConfig


class StochasticGridWorld(RichGridWorld):
    """GridWorld with stochastic transitions.

    Adds noise to make environment more realistic:
    - Actions may randomly change
    - Movements may slip (no effect)
    - Observations may be noisy

    This tests whether behavioral fingerprinting works when the
    aligned baseline has non-zero prediction error.

    Args:
        config: RichGridConfig with environment settings
        transition_noise: Probability of random action (0.0 to 1.0)
        slip_prob: Probability movement fails (0.0 to 1.0)
        observation_noise_std: Standard deviation of Gaussian observation noise
        seed: Random seed for reproducibility

    Example:
        >>> config = RichGridConfig(grid_size=18)
        >>> env = StochasticGridWorld(config, transition_noise=0.10, slip_prob=0.05)
        >>> # 10% chance action changes randomly
        >>> # 5% chance action has no effect
    """

    def __init__(
        self,
        config: RichGridConfig = None,
        transition_noise: float = 0.10,
        slip_prob: float = 0.05,
        observation_noise_std: float = 0.0,
        seed: int = None
    ):
        super().__init__(config)

        self.transition_noise = transition_noise
        self.slip_prob = slip_prob
        self.observation_noise_std = observation_noise_std

        # Random number generator
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action with stochastic transitions.

        Args:
            action: Action to take (may be modified by noise)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        original_action = action

        # Transition noise: randomly change action
        if self.rng.random() < self.transition_noise:
            action = self.rng.choice([0, 1, 2, 3])
            noisy_action = True
        else:
            noisy_action = False

        # Slip probability: action has no effect
        if self.rng.random() < self.slip_prob:
            # Return current state, small penalty, not done
            obs = self.get_observation()

            # Add observation noise if configured
            if self.observation_noise_std > 0:
                noise = self.rng.normal(0, self.observation_noise_std, size=obs.shape)
                obs = obs + noise

            reward = -self.config.step_penalty
            self.steps += 1
            truncated = self.steps >= self.max_steps

            info = {
                'position': self.agent_position,
                'steps': self.steps,
                'slipped': True,
                'original_action': original_action,
                'actual_action': action,
                'transition_noise': noisy_action,
            }

            return obs, reward, False, truncated, info

        # Execute action normally
        obs, reward, terminated, truncated, info = super().step(action)

        # Add observation noise if configured
        if self.observation_noise_std > 0:
            noise = self.rng.normal(0, self.observation_noise_std, size=obs.shape)
            obs = obs + noise

        # Add noise info to info dict
        info['slipped'] = False
        info['original_action'] = original_action
        info['actual_action'] = action
        info['transition_noise'] = noisy_action

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset environment with optional seed."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        return super().reset(seed=seed, options=options)
