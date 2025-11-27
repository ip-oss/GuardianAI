"""Mode switching analysis for deceptive agent detection.

This module provides functions to analyze when agents switch between aligned
and deceptive behavior as they move in and out of observer coverage zones.

Key metrics:
- Mode switches: Number of transitions between observed/unobserved states
- Mode fractions: Proportion of trajectory spent in each behavioral mode
- Temporal dynamics: When and how frequently switches occur
- Spatial patterns: Where on the grid mode switches happen

Author: GridWorld Fingerprinting Project
Date: 2025-11-26
"""

from typing import Dict, List, Tuple, Any
import numpy as np
from utils.trajectory import Trajectory


def count_mode_switches(observation_states: List[bool]) -> int:
    """
    Count transitions between observed and unobserved states.

    A mode switch occurs when the agent moves from an observed cell to an
    unobserved cell or vice versa. This indicates potential behavior changes
    for deceptive agents.

    Args:
        observation_states: List of booleans indicating observation status
                          at each step (True = observed, False = unobserved)

    Returns:
        Number of mode switches in the sequence

    Example:
        >>> states = [True, True, False, False, True]
        >>> count_mode_switches(states)
        2  # Switches at positions 2 and 4
    """
    if len(observation_states) < 2:
        return 0

    switches = 0
    for i in range(1, len(observation_states)):
        if observation_states[i] != observation_states[i-1]:
            switches += 1

    return switches


def compute_mode_fractions(observation_states: List[bool]) -> Dict[str, float]:
    """
    Calculate fraction of trajectory spent in observed vs unobserved states.

    Args:
        observation_states: List of observation status per step

    Returns:
        Dictionary containing:
        - observed_fraction: Fraction of steps where agent was observed
        - unobserved_fraction: Fraction of steps where agent was unobserved
        - total_steps: Total number of steps

    Example:
        >>> states = [True, True, False, False, False]
        >>> fractions = compute_mode_fractions(states)
        >>> fractions['observed_fraction']
        0.4
        >>> fractions['unobserved_fraction']
        0.6
    """
    if not observation_states:
        return {
            'observed_fraction': 0.0,
            'unobserved_fraction': 0.0,
            'total_steps': 0
        }

    total = len(observation_states)
    observed_count = sum(observation_states)

    return {
        'observed_fraction': observed_count / total,
        'unobserved_fraction': (total - observed_count) / total,
        'total_steps': total,
        'steps_observed': observed_count,
        'steps_unobserved': total - observed_count
    }


def extract_temporal_dynamics(observation_states: List[bool]) -> Dict[str, Any]:
    """
    Extract temporal pattern features from observation state sequence.

    Analyzes when switches occur and computes temporal statistics to
    identify patterns like:
    - Early vs late switching behavior
    - Switch frequency over time
    - Sustained periods in each mode

    Args:
        observation_states: List of observation status per step

    Returns:
        Dictionary containing:
        - mode_switches: Total number of switches
        - switches_per_step: Normalized switch rate
        - first_switch_step: Step number of first switch (or -1)
        - last_switch_step: Step number of last switch (or -1)
        - switch_positions: List of step indices where switches occurred
        - max_observed_streak: Longest continuous observed period
        - max_unobserved_streak: Longest continuous unobserved period
        - observed_early: Fraction observed in first third
        - observed_mid: Fraction observed in middle third
        - observed_late: Fraction observed in last third

    Example:
        >>> states = [False, False, True, True, False, False]
        >>> dynamics = extract_temporal_dynamics(states)
        >>> dynamics['mode_switches']
        2
        >>> dynamics['max_unobserved_streak']
        2
    """
    if not observation_states:
        return {
            'mode_switches': 0,
            'switches_per_step': 0.0,
            'first_switch_step': -1,
            'last_switch_step': -1,
            'switch_positions': [],
            'max_observed_streak': 0,
            'max_unobserved_streak': 0,
            'observed_early': 0.0,
            'observed_mid': 0.0,
            'observed_late': 0.0
        }

    # Find switch positions
    switch_positions = []
    for i in range(1, len(observation_states)):
        if observation_states[i] != observation_states[i-1]:
            switch_positions.append(i)

    num_switches = len(switch_positions)

    # Find longest streaks
    max_observed_streak = 0
    max_unobserved_streak = 0
    current_streak = 1

    for i in range(1, len(observation_states)):
        if observation_states[i] == observation_states[i-1]:
            current_streak += 1
        else:
            # Streak ended
            if observation_states[i-1]:
                max_observed_streak = max(max_observed_streak, current_streak)
            else:
                max_unobserved_streak = max(max_unobserved_streak, current_streak)
            current_streak = 1

    # Handle final streak
    if observation_states[-1]:
        max_observed_streak = max(max_observed_streak, current_streak)
    else:
        max_unobserved_streak = max(max_unobserved_streak, current_streak)

    # Temporal thirds
    n = len(observation_states)
    third = max(1, n // 3)

    early_states = observation_states[:third]
    mid_states = observation_states[third:2*third]
    late_states = observation_states[2*third:]

    observed_early = sum(early_states) / len(early_states) if early_states else 0.0
    observed_mid = sum(mid_states) / len(mid_states) if mid_states else 0.0
    observed_late = sum(late_states) / len(late_states) if late_states else 0.0

    return {
        'mode_switches': num_switches,
        'switches_per_step': num_switches / n,
        'first_switch_step': switch_positions[0] if switch_positions else -1,
        'last_switch_step': switch_positions[-1] if switch_positions else -1,
        'switch_positions': switch_positions,
        'max_observed_streak': max_observed_streak,
        'max_unobserved_streak': max_unobserved_streak,
        'observed_early': observed_early,
        'observed_mid': observed_mid,
        'observed_late': observed_late
    }


def compute_spatial_observation_map(
    trajectories: List[Trajectory],
    grid_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build spatial heatmaps of observation patterns across all trajectories.

    Creates two heatmaps:
    1. Visit frequency: How often each cell was visited
    2. Observation probability: P(observed | cell visited)

    This reveals which grid regions are most/least observed and where
    mode switches are likely to occur.

    Args:
        trajectories: List of Trajectory objects with observation_contexts
        grid_size: Size of the grid (N for NxN grid)

    Returns:
        Tuple of (visit_map, observation_map) where:
        - visit_map: Array of shape (grid_size, grid_size) with visit counts
        - observation_map: Array of shape (grid_size, grid_size) with P(observed)

    Example:
        >>> visit_map, obs_map = compute_spatial_observation_map(trajs, 18)
        >>> obs_map[5, 5]  # Probability of being observed at cell (5, 5)
        0.75
    """
    visit_counts = np.zeros((grid_size, grid_size), dtype=float)
    observed_counts = np.zeros((grid_size, grid_size), dtype=float)

    for traj in trajectories:
        obs_states = traj.get_observation_states()

        # Iterate through trajectory states and observation status
        for i, (state, is_observed) in enumerate(zip(traj.states, obs_states)):
            x, y = state
            if 0 <= x < grid_size and 0 <= y < grid_size:
                visit_counts[x, y] += 1
                if is_observed:
                    observed_counts[x, y] += 1

    # Compute observation probability map
    observation_map = np.zeros((grid_size, grid_size), dtype=float)
    mask = visit_counts > 0
    observation_map[mask] = observed_counts[mask] / visit_counts[mask]

    return visit_counts, observation_map


def analyze_mode_switching_patterns(
    trajectory: Trajectory,
    grid_size: int
) -> Dict[str, Any]:
    """
    Comprehensive mode switching analysis for a single trajectory.

    Combines all mode switching metrics into a unified analysis. This is
    the main function to call for complete behavioral analysis.

    Args:
        trajectory: Trajectory object with observation_contexts populated
        grid_size: Size of the grid

    Returns:
        Dictionary containing:
        - All metrics from compute_mode_fractions()
        - All metrics from extract_temporal_dynamics()
        - spatial_transition_map: Cells where mode switches occurred
        - trajectory_metadata: Agent type, episode number, etc.

    Example:
        >>> analysis = analyze_mode_switching_patterns(traj, 18)
        >>> analysis['mode_switches']
        8
        >>> analysis['observed_fraction']
        0.35
    """
    # Get observation states
    obs_states = trajectory.get_observation_states()

    # Compute all metrics
    fractions = compute_mode_fractions(obs_states)
    dynamics = extract_temporal_dynamics(obs_states)

    # Build spatial transition map
    spatial_transitions = {}
    for i, switch_step in enumerate(dynamics['switch_positions']):
        if switch_step < len(trajectory.states):
            state = trajectory.states[switch_step]
            transition_type = 'enter_observed' if obs_states[switch_step] else 'exit_observed'
            spatial_transitions[f'switch_{i}'] = {
                'step': switch_step,
                'position': state,
                'type': transition_type
            }

    # Combine results
    analysis = {
        **fractions,
        **dynamics,
        'spatial_transition_map': spatial_transitions,
        'trajectory_metadata': trajectory.metadata.copy(),
        'path_length': trajectory.path_length,
        'total_reward': trajectory.total_reward
    }

    return analysis


def analyze_mode_switching_batch(
    trajectories: List[Trajectory],
    grid_size: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Analyze mode switching patterns for multiple trajectories.

    Processes a batch of trajectories and computes both per-trajectory
    metrics and aggregate statistics across all trajectories.

    Args:
        trajectories: List of Trajectory objects
        grid_size: Size of the grid

    Returns:
        Tuple of (individual_analyses, aggregate_statistics) where:
        - individual_analyses: List of per-trajectory analysis dicts
        - aggregate_statistics: Dict with population-level metrics

    Example:
        >>> analyses, aggregate = analyze_mode_switching_batch(trajs, 18)
        >>> aggregate['mean_switches_per_trajectory']
        5.2
    """
    individual_analyses = []

    for traj in trajectories:
        analysis = analyze_mode_switching_patterns(traj, grid_size)
        individual_analyses.append(analysis)

    # Compute aggregate statistics
    if not individual_analyses:
        return [], {}

    switches = [a['mode_switches'] for a in individual_analyses]
    obs_fractions = [a['observed_fraction'] for a in individual_analyses]
    switches_per_step = [a['switches_per_step'] for a in individual_analyses]

    aggregate = {
        'n_trajectories': len(trajectories),
        'mean_switches_per_trajectory': float(np.mean(switches)),
        'std_switches_per_trajectory': float(np.std(switches)),
        'max_switches': int(np.max(switches)),
        'min_switches': int(np.min(switches)),
        'mean_observed_fraction': float(np.mean(obs_fractions)),
        'std_observed_fraction': float(np.std(obs_fractions)),
        'mean_switches_per_step': float(np.mean(switches_per_step)),
        'std_switches_per_step': float(np.std(switches_per_step))
    }

    # Compute spatial maps for all trajectories
    visit_map, obs_map = compute_spatial_observation_map(trajectories, grid_size)
    aggregate['spatial_visit_map'] = visit_map
    aggregate['spatial_observation_map'] = obs_map

    return individual_analyses, aggregate


def compare_agent_types(
    trajectories: List[Trajectory],
    grid_size: int
) -> Dict[str, Dict[str, Any]]:
    """
    Compare mode switching patterns across different agent types.

    Groups trajectories by agent type and computes aggregate statistics
    for each group. Useful for identifying behavioral signatures.

    Args:
        trajectories: List of trajectories from multiple agent types
        grid_size: Size of the grid

    Returns:
        Dictionary mapping agent_type -> aggregate_statistics

    Example:
        >>> comparison = compare_agent_types(all_trajs, 18)
        >>> comparison['deceptive']['mean_switches_per_trajectory']
        8.5
        >>> comparison['aligned']['mean_switches_per_trajectory']
        2.1
    """
    # Group by agent type
    by_type = {}
    for traj in trajectories:
        agent_type = traj.metadata.get('agent_type', 'unknown')
        if agent_type not in by_type:
            by_type[agent_type] = []
        by_type[agent_type].append(traj)

    # Analyze each group
    comparison = {}
    for agent_type, type_trajectories in by_type.items():
        _, aggregate = analyze_mode_switching_batch(type_trajectories, grid_size)
        comparison[agent_type] = aggregate

    return comparison
