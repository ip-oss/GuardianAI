"""
Abstract Feature Extraction - Domain-Agnostic Behavioral Features

Defines features that capture behavioral patterns in ways that could
transfer to other domains (like LLM behavior, robotic systems, etc.).

Key insight: Instead of domain-specific features (e.g., "visited temptation cell"),
use abstract features (e.g., "resource acquisition rate", "goal directedness").

These features describe *how* an agent behaves, not *what* it does in a specific environment.
"""

import numpy as np
from typing import List, Dict, Any
from utils.trajectory import Trajectory


def compute_path_efficiency(trajectory: Trajectory, optimal_length: int) -> float:
    """
    Path efficiency: How close to optimal is the agent's path?

    Domain-agnostic interpretation:
    - 1.0 = Perfect efficiency (optimal path)
    - <1.0 = Suboptimal (taking detours, exploring, or lost)

    Transfers to:
    - LLMs: Token efficiency (optimal tokens / actual tokens)
    - Robots: Distance efficiency (straight-line / actual distance)
    """
    if trajectory.path_length == 0:
        return 0.0
    return optimal_length / trajectory.path_length


def compute_goal_directedness(trajectory: Trajectory, goal_position: tuple) -> float:
    """
    Goal directedness: Average progress toward goal per step.

    Measures how consistently the agent moves toward its nominal objective.

    Domain-agnostic interpretation:
    - High = Focused on goal
    - Low = Distracted or pursuing other objectives

    Transfers to:
    - LLMs: How directly responses address the user's query
    - Robots: Heading consistency toward target
    """
    if len(trajectory.states) < 2:
        return 0.0

    distances_to_goal = []
    for state in trajectory.states:
        x, y = state
        gx, gy = goal_position
        dist = abs(x - gx) + abs(y - gy)  # Manhattan distance
        distances_to_goal.append(dist)

    # Compute progress per step
    progress_per_step = []
    for i in range(len(distances_to_goal) - 1):
        progress = distances_to_goal[i] - distances_to_goal[i+1]
        progress_per_step.append(progress)

    # Average progress (positive = moving toward goal, negative = moving away)
    if len(progress_per_step) == 0:
        return 0.0

    avg_progress = np.mean(progress_per_step)
    # Normalize: optimal progress per step is 1.0 (move 1 cell closer each step)
    return avg_progress


def compute_exploration_rate(trajectory: Trajectory) -> float:
    """
    Exploration rate: Unique states visited / total steps.

    Domain-agnostic interpretation:
    - High = Exploring widely
    - Low = Repeating states (stuck in loops or very focused)

    Transfers to:
    - LLMs: Topic/concept diversity in responses
    - Robots: Coverage of search space
    """
    if trajectory.path_length == 0:
        return 0.0

    unique_states = len(set(tuple(s) for s in trajectory.states))
    return unique_states / len(trajectory.states)


def compute_backtracking_frequency(trajectory: Trajectory) -> float:
    """
    Backtracking frequency: How often the agent revisits recent states.

    Counts the number of times the agent returns to a state it visited
    within the last 5 steps.

    Domain-agnostic interpretation:
    - High = Indecisive, stuck, or deliberately revisiting
    - Low = Committed to forward progress

    Transfers to:
    - LLMs: Repetition or circular reasoning
    - Robots: Oscillating behavior
    """
    if len(trajectory.states) < 2:
        return 0.0

    backtrack_count = 0
    lookback = 5  # Consider last N steps

    for i in range(lookback, len(trajectory.states)):
        current_state = tuple(trajectory.states[i])
        recent_states = [tuple(s) for s in trajectory.states[i-lookback:i]]

        if current_state in recent_states:
            backtrack_count += 1

    return backtrack_count / max(1, len(trajectory.states) - lookback)


def compute_action_entropy(trajectory: Trajectory) -> float:
    """
    Action entropy: Predictability of action sequence.

    Measures the Shannon entropy of the action distribution.

    Domain-agnostic interpretation:
    - High = Unpredictable, exploratory
    - Low = Repetitive, deterministic

    Transfers to:
    - LLMs: Response diversity/creativity
    - Robots: Movement pattern diversity
    """
    if len(trajectory.actions) == 0:
        return 0.0

    # Count action frequencies
    action_counts = {}
    for action in trajectory.actions:
        action_counts[action] = action_counts.get(action, 0) + 1

    # Compute entropy
    total = len(trajectory.actions)
    entropy = 0.0
    for count in action_counts.values():
        prob = count / total
        if prob > 0:
            entropy -= prob * np.log2(prob)

    return entropy


def compute_temporal_consistency(trajectory: Trajectory) -> float:
    """
    Temporal consistency: How correlated are consecutive actions?

    Measures autocorrelation of the action sequence at lag=1.

    Domain-agnostic interpretation:
    - High = Smooth, consistent behavior
    - Low = Erratic, context-switching

    Transfers to:
    - LLMs: Topical coherence across sentences
    - Robots: Smooth motion vs jerky movements
    """
    if len(trajectory.actions) < 2:
        return 0.0

    actions = np.array(trajectory.actions, dtype=float)

    # Compute autocorrelation at lag 1
    mean_action = np.mean(actions)
    numerator = np.sum((actions[:-1] - mean_action) * (actions[1:] - mean_action))
    denominator = np.sum((actions - mean_action) ** 2)

    if denominator == 0:
        return 1.0  # All actions the same = perfectly consistent

    autocorr = numerator / denominator
    return autocorr


def compute_reward_rate(trajectory: Trajectory) -> float:
    """
    Reward accumulation rate: Average reward per step.

    Domain-agnostic interpretation:
    - Performance metric normalized by time
    - Comparable across episodes of different lengths

    Transfers to:
    - LLMs: "Value" generated per token
    - Robots: Task success per unit time
    """
    if trajectory.path_length == 0:
        return 0.0

    return trajectory.total_reward / trajectory.path_length


def compute_reward_variance(trajectory: Trajectory) -> float:
    """
    Reward variance: How variable are the rewards?

    Domain-agnostic interpretation:
    - High = Inconsistent performance, risky behavior
    - Low = Stable, predictable outcomes

    Transfers to:
    - LLMs: Response quality variability
    - Robots: Performance stability
    """
    if len(trajectory.rewards) == 0:
        return 0.0

    return float(np.var(trajectory.rewards))


def compute_late_game_efficiency(trajectory: Trajectory, optimal_length: int) -> float:
    """
    Late-game efficiency: Path efficiency in the last 25% of trajectory.

    Tests whether agent maintains focus near the end or gets distracted.

    Domain-agnostic interpretation:
    - Did the agent "finish strong" or lose focus?

    Transfers to:
    - LLMs: Conclusion quality vs opening quality
    - Robots: Terminal precision
    """
    if len(trajectory.states) < 4:
        return compute_path_efficiency(trajectory, optimal_length)

    # Take last quarter of trajectory
    quarter_point = 3 * len(trajectory.states) // 4
    late_states = trajectory.states[quarter_point:]
    late_actions = trajectory.actions[quarter_point:]
    late_rewards = trajectory.rewards[quarter_point:]

    # Create sub-trajectory
    from utils.trajectory import create_trajectory
    late_trajectory = create_trajectory(
        states=late_states,
        actions=late_actions,
        rewards=late_rewards,
        agent_type=trajectory.agent_type,
        episode_num=trajectory.episode_num
    )

    # Estimate "optimal" for this segment (distance traveled)
    start_pos = late_states[0]
    end_pos = late_states[-1]
    segment_optimal = abs(end_pos[0] - start_pos[0]) + abs(end_pos[1] - start_pos[1])
    segment_optimal = max(segment_optimal, 1)  # Avoid division by zero

    return compute_path_efficiency(late_trajectory, segment_optimal)


def extract_abstract_features(
    trajectory: Trajectory,
    goal_position: tuple,
    optimal_length: int
) -> Dict[str, float]:
    """
    Extract all abstract features from a trajectory.

    Args:
        trajectory: Agent's trajectory
        goal_position: Goal position in the environment
        optimal_length: Optimal path length for this environment

    Returns:
        Dictionary of abstract features (all domain-agnostic)
    """
    return {
        # Efficiency metrics
        'path_efficiency': compute_path_efficiency(trajectory, optimal_length),
        'goal_directedness': compute_goal_directedness(trajectory, goal_position),
        'late_game_efficiency': compute_late_game_efficiency(trajectory, optimal_length),

        # Exploration patterns
        'exploration_rate': compute_exploration_rate(trajectory),
        'backtracking_frequency': compute_backtracking_frequency(trajectory),

        # Temporal patterns
        'action_entropy': compute_action_entropy(trajectory),
        'temporal_consistency': compute_temporal_consistency(trajectory),

        # Reward patterns
        'reward_rate': compute_reward_rate(trajectory),
        'reward_variance': compute_reward_variance(trajectory),

        # Raw stats (for reference)
        'path_length': float(trajectory.path_length),
        'total_reward': float(trajectory.total_reward),
    }


def extract_abstract_features_batch(
    trajectories: List[Trajectory],
    goal_position: tuple,
    optimal_length: int
) -> List[Dict[str, float]]:
    """Extract abstract features for multiple trajectories."""
    return [
        extract_abstract_features(traj, goal_position, optimal_length)
        for traj in trajectories
    ]


def print_feature_comparison(
    aligned_features: List[Dict[str, float]],
    misaligned_features: List[Dict[str, float]],
    agent_type: str
):
    """
    Print comparison of features between aligned and misaligned agents.

    Useful for understanding which features discriminate best.
    """
    print(f"\n{'='*60}")
    print(f"Abstract Features: Aligned vs {agent_type.upper()}")
    print(f"{'='*60}")

    # Get feature names
    feature_names = list(aligned_features[0].keys())

    # Compute means
    aligned_means = {
        feat: np.mean([f[feat] for f in aligned_features])
        for feat in feature_names
    }
    misaligned_means = {
        feat: np.mean([f[feat] for f in misaligned_features])
        for feat in feature_names
    }

    # Compute z-scores (how many std deviations apart?)
    print(f"\n{'Feature':<25s} | {'Aligned':>10s} | {'Misaligned':>10s} | {'Difference':>12s}")
    print("-" * 70)

    for feat in feature_names:
        aligned_val = aligned_means[feat]
        misaligned_val = misaligned_means[feat]
        diff = misaligned_val - aligned_val

        # Compute significance
        aligned_vals = [f[feat] for f in aligned_features]
        std = np.std(aligned_vals)

        if std > 1e-6:
            z_score = abs(diff) / std
            significance = "***" if z_score > 2 else ("**" if z_score > 1 else "*")
        else:
            significance = ""

        print(f"{feat:<25s} | {aligned_val:10.3f} | {misaligned_val:10.3f} | {diff:+12.3f} {significance}")

    print("\nSignificance: *** = >2σ, ** = >1σ, * = <1σ")


def compute_abstract_signature(
    aligned_features: List[Dict[str, float]],
    misaligned_features: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Compute an abstract signature: which features differ most?

    Returns z-scores for each feature (how many standard deviations
    the misaligned agent differs from aligned baseline).
    """
    feature_names = list(aligned_features[0].keys())

    aligned_means = {
        feat: np.mean([f[feat] for f in aligned_features])
        for feat in feature_names
    }
    aligned_stds = {
        feat: np.std([f[feat] for f in aligned_features])
        for feat in feature_names
    }
    misaligned_means = {
        feat: np.mean([f[feat] for f in misaligned_features])
        for feat in feature_names
    }

    # Compute z-scores
    z_scores = {}
    for feat in feature_names:
        diff = misaligned_means[feat] - aligned_means[feat]
        std = aligned_stds[feat]
        z_scores[feat] = diff / std if std > 1e-6 else 0.0

    return z_scores
