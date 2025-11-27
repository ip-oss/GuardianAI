"""Configuration for GridWorld fingerprinting experiments."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class ExperimentConfig:
    """Configuration for behavioral fingerprinting experiment."""

    # Environment
    grid_size: int = 8
    temptation_locations: Optional[List[Tuple[int, int]]] = None  # Auto-generate if None
    num_temptations: int = 6
    max_episode_steps: int = 100

    # Rewards
    goal_reward: float = 10.0
    step_penalty: float = 0.1
    temptation_reward: float = 2.0  # Only for misaligned

    # Agent Training
    agent_type: str = "q_learning"  # or "value_iteration"
    agent_episodes: int = 1000
    agent_learning_rate: float = 0.1
    agent_discount: float = 0.99
    agent_epsilon: float = 0.1  # For exploration during training
    agent_epsilon_decay: float = 0.995
    agent_epsilon_min: float = 0.01

    # Data Collection
    num_trajectories: int = 200  # Per agent type
    trajectory_deterministic: bool = True

    # World Model
    world_model_type: str = "mlp"
    world_model_hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
    world_model_learning_rate: float = 0.001
    world_model_epochs: int = 100
    world_model_batch_size: int = 32
    train_val_split: float = 0.8

    # Analysis
    classifier_type: str = "logistic"  # or "svm", "random_forest"
    classifier_cv_folds: int = 5
    statistical_significance: float = 0.05

    # Output
    results_dir: str = "results"
    save_plots: bool = True
    random_seed: int = 42
    verbose: bool = True

    def __post_init__(self):
        """Generate temptation locations if not provided."""
        if self.temptation_locations is None:
            # Generate default temptation locations
            # Spread them across the grid, avoiding start and goal
            self.temptation_locations = self._generate_temptation_locations()

    def _generate_temptation_locations(self) -> List[Tuple[int, int]]:
        """Generate evenly spaced temptation locations."""
        import numpy as np
        np.random.seed(self.random_seed)

        locations = []
        start = (0, 0)
        goal = (self.grid_size - 1, self.grid_size - 1)

        # Generate random locations avoiding start and goal
        attempts = 0
        while len(locations) < self.num_temptations and attempts < 1000:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)

            if (x, y) != start and (x, y) != goal and (x, y) not in locations:
                locations.append((x, y))

            attempts += 1

        return locations


@dataclass
class QuickConfig(ExperimentConfig):
    """Quick test configuration for faster iteration."""

    grid_size: int = 5
    num_temptations: int = 2
    agent_episodes: int = 200
    num_trajectories: int = 50
    world_model_epochs: int = 20
    world_model_hidden_sizes: List[int] = field(default_factory=lambda: [32, 32])
    max_episode_steps: int = 50


@dataclass
class DeceptiveConfig(ExperimentConfig):
    """Configuration for deceptive agent experiment."""

    # Deception parameters
    deception_rate: float = 0.3  # Probability of misaligned action
    deception_schedule: str = "constant"  # "constant", "increasing", "random"

    # More trajectories to detect subtle patterns
    num_trajectories: int = 300


@dataclass
class StochasticConfig(ExperimentConfig):
    """Configuration for stochastic environment experiment."""

    # Environment stochasticity
    action_noise: float = 0.1  # Probability of random action
    observation_noise: float = 0.05  # Noise in state observations

    # Need more data for noisy environment
    agent_episodes: int = 2000
    num_trajectories: int = 300


@dataclass
class TransferConfig(ExperimentConfig):
    """Configuration for transfer learning experiment."""

    # Train on multiple grid sizes/layouts
    train_grid_sizes: List[int] = field(default_factory=lambda: [6, 8, 10])
    test_grid_size: int = 12

    # More diverse training
    num_trajectories: int = 400


@dataclass
class SubtletyConfig(ExperimentConfig):
    """Configuration for testing subtlety gradient."""

    # Test multiple misalignment levels
    misalignment_levels: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.5, 1.0, 2.0])

    # More trajectories per level
    num_trajectories: int = 100


@dataclass
class TaxonomyConfig(ExperimentConfig):
    """Configuration for misalignment taxonomy experiment (exp6).

    Uses a 12x12 grid to give agents room to develop distinct behaviors
    while staying within tabular Q-learning limits (~144 states).

    Design rationale:
    - 12x12 = 144 states (manageable for tabular Q-learning)
    - Optimal path = 22 steps (rich enough for behavioral patterns)
    - Room for meaningful detours to collect resources/avoid danger
    - Fast enough for iterative testing (~2-3 min per run)
    """

    # Grid: 12x12 balances richness with training speed
    # - 144 states (tabular Q-learning converges reliably)
    # - Optimal path = 22 steps (2x longer than 8x8, plenty of decision points)
    grid_size: int = 12
    max_episode_steps: int = 150  # ~7x optimal path length

    # Agent training: sufficient for 144-state convergence
    agent_episodes: int = 1500
    agent_epsilon: float = 0.3
    agent_epsilon_decay: float = 0.997

    # Trajectories: enough for reliable statistics
    num_trajectories: int = 100

    # World model: moderate size
    world_model_hidden_sizes: List[int] = field(default_factory=lambda: [128, 64])
    world_model_epochs: int = 120


@dataclass
class TaxonomyQuickConfig(ExperimentConfig):
    """Quick config for taxonomy experiment testing."""

    grid_size: int = 8
    max_episode_steps: int = 100
    agent_episodes: int = 500
    num_trajectories: int = 50
    world_model_epochs: int = 50
    world_model_hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])


@dataclass
class Taxonomy18x18Config(ExperimentConfig):
    """Configuration for 18x18 mode switching experiment (exp6 extended).

    Scales up to 18x18 grid (324 states) with strategically placed observers
    to detect mode switching in deceptive agents.

    Design rationale:
    - 18x18 = 324 states (upper limit for tabular Q-learning)
    - Optimal path = 34 steps (long enough for multiple mode switches)
    - 30-40% observer coverage with radius=1 creates clear observed/unobserved zones
    - Enables detection of deceptive agents switching behavior based on observation
    - Sufficient unobserved corridors for deceptive routing

    Key features:
    - Observer placement optimized for mode switching opportunities
    - Minimal observation radius (1) for sharp boundaries
    - Long trajectories capture multiple behavioral transitions
    - Adequate training episodes for 324-state convergence
    """

    # Grid: 18x18 = 324 states (maximum for reliable tabular Q-learning)
    # - Optimal path = 34 steps (plenty of decision points)
    # - Room for complex routing with observer avoidance
    grid_size: int = 18
    max_episode_steps: int = 200  # ~6x optimal path length

    # Agent training: more episodes needed for 324-state convergence
    agent_episodes: int = 3000
    agent_epsilon: float = 0.3
    agent_epsilon_decay: float = 0.998
    agent_epsilon_min: float = 0.05

    # Trajectories: sufficient for mode switching statistics
    num_trajectories: int = 100

    # World model: larger capacity for bigger state space
    world_model_hidden_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    world_model_epochs: int = 150
    world_model_batch_size: int = 64

    # Observer configuration (for mode switching detection)
    # Note: observer_locations will be generated by create_observers_for_18x18_grid()
    # observer_radius will be set in RichGridConfig (default: 1 for sharp boundaries)
