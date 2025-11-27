"""
Visualizations for mode switching analysis in deceptive agents.

This module provides plotting functions to visualize when and where agents
switch between aligned and deceptive behavior as they move through observed
and unobserved zones.

Author: GridWorld Fingerprinting Project
Date: 2025-11-26
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from utils.trajectory import Trajectory


def plot_mode_timeline(
    trajectory: Trajectory,
    save_path: Optional[Path] = None,
    show: bool = False
):
    """
    Plot timeline showing observation state and mode switches over trajectory.

    Creates a visual timeline with:
    - Observation state (observed/unobserved) at each step
    - Mode switches highlighted with vertical lines
    - Path efficiency and reward metrics

    Args:
        trajectory: Trajectory object with observation_contexts
        save_path: Optional path to save figure
        show: Whether to display the plot

    Example:
        >>> plot_mode_timeline(deceptive_trajectory, save_path=Path("timeline.png"))
    """
    obs_states = trajectory.get_observation_states()
    if not obs_states:
        print("Warning: No observation states found in trajectory")
        return

    metrics = trajectory.compute_observation_metrics()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])

    # Plot 1: Observation state timeline
    steps = np.arange(len(obs_states))
    colors = ['#ff6b6b' if obs else '#4ecdc4' for obs in obs_states]

    ax1.bar(steps, [1] * len(obs_states), color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Highlight mode switches
    for i in range(1, len(obs_states)):
        if obs_states[i] != obs_states[i-1]:
            ax1.axvline(x=i - 0.5, color='black', linestyle='--', linewidth=2, alpha=0.8)

    ax1.set_ylabel('Observation State', fontsize=12)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['', 'Observed' if any(obs_states) else ''])
    ax1.set_xlim(-0.5, len(obs_states) - 0.5)
    ax1.set_ylim(0, 1.2)

    # Add legend
    observed_patch = mpatches.Patch(color='#ff6b6b', label='Observed', alpha=0.7)
    unobserved_patch = mpatches.Patch(color='#4ecdc4', label='Unobserved', alpha=0.7)
    switch_line = mpatches.Patch(color='black', label='Mode Switch', linestyle='--')
    ax1.legend(handles=[observed_patch, unobserved_patch, switch_line],
              loc='upper right', fontsize=10)

    # Add metrics text
    agent_type = trajectory.metadata.get('agent_type', 'unknown')
    title = f'Mode Switching Timeline: {agent_type}\n'
    title += f'Switches: {metrics["mode_switches"]} | '
    title += f'Observed: {metrics["observed_fraction"]*100:.1f}% | '
    title += f'Path Length: {trajectory.path_length} | '
    title += f'Total Reward: {trajectory.total_reward:.1f}'
    ax1.set_title(title, fontsize=14, fontweight='bold')

    # Plot 2: Reward over time
    rewards = trajectory.rewards
    cumulative_reward = np.cumsum(rewards)

    ax2.plot(range(len(rewards)), cumulative_reward, color='#2d3436', linewidth=2)
    ax2.fill_between(range(len(rewards)), cumulative_reward, alpha=0.3, color='#74b9ff')

    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Cumulative Reward', fontsize=12)
    ax2.set_xlim(-0.5, len(obs_states) - 0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_mode_switch_heatmap(
    trajectories: List[Trajectory],
    grid_size: int,
    observer_locations: Optional[List[Tuple[int, int]]] = None,
    save_path: Optional[Path] = None,
    show: bool = False
):
    """
    Plot heatmap showing where mode switches occur on the grid.

    Creates a spatial heatmap indicating:
    - Frequency of mode switches at each grid location
    - Observer positions (if provided)
    - Identifies high-switching zones

    Args:
        trajectories: List of trajectories with observation data
        grid_size: Size of grid (N for NxN)
        observer_locations: Optional list of observer (x, y) positions
        save_path: Optional path to save figure
        show: Whether to display the plot

    Example:
        >>> plot_mode_switch_heatmap(all_trajs, 18, observers, Path("heatmap.png"))
    """
    switch_map = np.zeros((grid_size, grid_size), dtype=float)

    for traj in trajectories:
        obs_states = traj.get_observation_states()
        if not obs_states or len(obs_states) < 2:
            continue

        # Mark cells where switches occurred
        for i in range(1, len(obs_states)):
            if obs_states[i] != obs_states[i-1]:
                # Switch occurred at step i
                if i < len(traj.states):
                    x, y = traj.states[i]
                    if 0 <= x < grid_size and 0 <= y < grid_size:
                        switch_map[x, y] += 1

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    im = ax.imshow(switch_map.T, origin='lower', cmap='YlOrRd', interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Mode Switches', fontsize=12)

    # Overlay observer locations
    if observer_locations:
        obs_x = [obs[0] for obs in observer_locations]
        obs_y = [obs[1] for obs in observer_locations]
        ax.scatter(obs_x, obs_y, marker='X', s=200, c='blue',
                  edgecolors='white', linewidths=2, label='Observers', zorder=10)
        ax.legend(fontsize=10)

    # Mark start and goal
    ax.scatter([0], [0], marker='s', s=300, c='green',
              edgecolors='white', linewidths=2, label='Start', zorder=10)
    ax.scatter([grid_size-1], [grid_size-1], marker='*', s=400, c='gold',
              edgecolors='white', linewidths=2, label='Goal', zorder=10)

    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title(f'Mode Switch Spatial Distribution ({len(trajectories)} trajectories)',
                fontsize=14, fontweight='bold')

    # Grid lines
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_mode_fraction_comparison(
    analyses_by_type: Dict[str, List[Dict[str, Any]]],
    save_path: Optional[Path] = None,
    show: bool = False
):
    """
    Compare mode switching fractions across different agent types.

    Creates a bar chart showing:
    - Mean fraction observed/unobserved for each agent type
    - Error bars showing variance
    - Side-by-side comparison

    Args:
        analyses_by_type: Dict mapping agent_type -> list of analysis dicts
        save_path: Optional path to save figure
        show: Whether to display the plot

    Example:
        >>> analyses_by_type = {
        ...     'aligned': [analysis1, analysis2, ...],
        ...     'deceptive': [analysis3, analysis4, ...]
        ... }
        >>> plot_mode_fraction_comparison(analyses_by_type)
    """
    if not analyses_by_type:
        print("Warning: No analyses provided")
        return

    agent_types = list(analyses_by_type.keys())
    n_types = len(agent_types)

    # Compute means and stds
    observed_means = []
    observed_stds = []
    unobserved_means = []
    unobserved_stds = []

    for agent_type in agent_types:
        analyses = analyses_by_type[agent_type]
        obs_fracs = [a['observed_fraction'] for a in analyses if 'observed_fraction' in a]
        unobs_fracs = [a['unobserved_fraction'] for a in analyses if 'unobserved_fraction' in a]

        observed_means.append(np.mean(obs_fracs) if obs_fracs else 0)
        observed_stds.append(np.std(obs_fracs) if obs_fracs else 0)
        unobserved_means.append(np.mean(unobs_fracs) if unobs_fracs else 0)
        unobserved_stds.append(np.std(unobs_fracs) if unobs_fracs else 0)

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(n_types)
    width = 0.35

    # Plot bars
    bars1 = ax.bar(x - width/2, observed_means, width, yerr=observed_stds,
                   label='Observed', color='#ff6b6b', alpha=0.7, capsize=5)
    bars2 = ax.bar(x + width/2, unobserved_means, width, yerr=unobserved_stds,
                   label='Unobserved', color='#4ecdc4', alpha=0.7, capsize=5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Fraction of Trajectory', fontsize=12)
    ax.set_xlabel('Agent Type', fontsize=12)
    ax.set_title('Mode Distribution by Agent Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(agent_types, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_observation_coverage_map(
    visit_map: np.ndarray,
    observation_map: np.ndarray,
    observer_locations: Optional[List[Tuple[int, int]]] = None,
    save_path: Optional[Path] = None,
    show: bool = False
):
    """
    Plot spatial maps of visit frequency and observation probability.

    Creates a dual-panel figure showing:
    - Left: Visit frequency heatmap
    - Right: Observation probability P(observed | visited)

    Args:
        visit_map: Array of shape (grid_size, grid_size) with visit counts
        observation_map: Array of shape (grid_size, grid_size) with P(observed)
        observer_locations: Optional list of observer positions
        save_path: Optional path to save figure
        show: Whether to display the plot

    Example:
        >>> visit_map, obs_map = compute_spatial_observation_map(trajs, 18)
        >>> plot_observation_coverage_map(visit_map, obs_map, observers)
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    grid_size = visit_map.shape[0]

    # Plot 1: Visit frequency
    ax1 = axes[0]
    im1 = ax1.imshow(visit_map.T, origin='lower', cmap='Blues', interpolation='nearest')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Visit Count', fontsize=12)

    ax1.set_xlabel('X Position', fontsize=12)
    ax1.set_ylabel('Y Position', fontsize=12)
    ax1.set_title('Spatial Visit Frequency', fontsize=14, fontweight='bold')

    # Plot 2: Observation probability
    ax2 = axes[1]
    im2 = ax2.imshow(observation_map.T, origin='lower', cmap='RdYlGn_r',
                    vmin=0, vmax=1, interpolation='nearest')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('P(Observed | Visited)', fontsize=12)

    ax2.set_xlabel('X Position', fontsize=12)
    ax2.set_ylabel('Y Position', fontsize=12)
    ax2.set_title('Observation Probability Map', fontsize=14, fontweight='bold')

    # Add observer locations to both plots
    if observer_locations:
        for ax in axes:
            obs_x = [obs[0] for obs in observer_locations]
            obs_y = [obs[1] for obs in observer_locations]
            ax.scatter(obs_x, obs_y, marker='X', s=150, c='blue',
                      edgecolors='white', linewidths=2, alpha=0.8, zorder=10)

    # Add start/goal markers
    for ax in axes:
        ax.scatter([0], [0], marker='s', s=200, c='green',
                  edgecolors='white', linewidths=2, zorder=10)
        ax.scatter([grid_size-1], [grid_size-1], marker='*', s=250, c='gold',
                  edgecolors='white', linewidths=2, zorder=10)

        # Grid lines
        ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_switch_frequency_comparison(
    analyses_by_type: Dict[str, List[Dict[str, Any]]],
    save_path: Optional[Path] = None,
    show: bool = False
):
    """
    Compare mode switching frequency across agent types.

    Creates violin plot showing distribution of:
    - Number of switches per trajectory
    - Switches per step (normalized)

    Args:
        analyses_by_type: Dict mapping agent_type -> list of analysis dicts
        save_path: Optional path to save figure
        show: Whether to display the plot

    Example:
        >>> plot_switch_frequency_comparison(analyses_by_type)
    """
    if not analyses_by_type:
        print("Warning: No analyses provided")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    agent_types = list(analyses_by_type.keys())
    n_types = len(agent_types)

    # Collect data
    switches_data = []
    switches_per_step_data = []

    for agent_type in agent_types:
        analyses = analyses_by_type[agent_type]
        switches = [a['mode_switches'] for a in analyses if 'mode_switches' in a]
        switches_ps = [a['switches_per_step'] for a in analyses if 'switches_per_step' in a]

        switches_data.append(switches)
        switches_per_step_data.append(switches_ps)

    # Plot 1: Total switches
    bp1 = ax1.boxplot(switches_data, labels=agent_types, patch_artist=True,
                      showmeans=True, meanline=True)

    for patch, color in zip(bp1['boxes'], plt.cm.Set2.colors[:n_types]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_ylabel('Mode Switches per Trajectory', fontsize=12)
    ax1.set_xlabel('Agent Type', fontsize=12)
    ax1.set_title('Mode Switch Frequency Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Normalized switches
    bp2 = ax2.boxplot(switches_per_step_data, labels=agent_types, patch_artist=True,
                      showmeans=True, meanline=True)

    for patch, color in zip(bp2['boxes'], plt.cm.Set2.colors[:n_types]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_ylabel('Switches per Step', fontsize=12)
    ax2.set_xlabel('Agent Type', fontsize=12)
    ax2.set_title('Normalized Switch Rate Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_mode_switching_summary(
    aggregate_stats: Dict[str, Dict[str, Any]],
    grid_size: int,
    observer_locations: Optional[List[Tuple[int, int]]] = None,
    save_path: Optional[Path] = None,
    show: bool = False
):
    """
    Comprehensive summary figure with multiple mode switching visualizations.

    Creates a 2x2 panel figure showing:
    - Top-left: Mode fraction comparison (bar chart)
    - Top-right: Switch frequency comparison (box plots)
    - Bottom-left: Observation coverage map
    - Bottom-right: Summary statistics table

    Args:
        aggregate_stats: Dict mapping agent_type -> aggregate statistics
                        (from analyze_mode_switching_batch or compare_agent_types)
        grid_size: Size of grid
        observer_locations: Optional list of observer positions
        save_path: Optional path to save figure
        show: Whether to display the plot

    Example:
        >>> comparison = compare_agent_types(all_trajs, 18)
        >>> plot_mode_switching_summary(comparison, 18, observers, Path("summary.png"))
    """
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    agent_types = list(aggregate_stats.keys())
    n_types = len(agent_types)

    # Plot 1: Mode fractions (top-left)
    ax1 = fig.add_subplot(gs[0, 0])

    observed_means = [aggregate_stats[t]['mean_observed_fraction'] for t in agent_types]
    x = np.arange(n_types)
    width = 0.35

    unobserved_means = [1 - obs for obs in observed_means]

    bars1 = ax1.bar(x - width/2, observed_means, width, label='Observed',
                   color='#ff6b6b', alpha=0.7)
    bars2 = ax1.bar(x + width/2, unobserved_means, width, label='Unobserved',
                   color='#4ecdc4', alpha=0.7)

    ax1.set_ylabel('Fraction of Trajectory', fontsize=11)
    ax1.set_title('Mode Distribution by Agent Type', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(agent_types, rotation=45, ha='right', fontsize=10)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Switch frequency (top-right)
    ax2 = fig.add_subplot(gs[0, 1])

    mean_switches = [aggregate_stats[t]['mean_switches_per_trajectory'] for t in agent_types]
    std_switches = [aggregate_stats[t]['std_switches_per_trajectory'] for t in agent_types]

    colors = plt.cm.Set2.colors[:n_types]
    ax2.bar(x, mean_switches, yerr=std_switches, color=colors, alpha=0.7, capsize=5)

    for i, v in enumerate(mean_switches):
        ax2.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

    ax2.set_ylabel('Mean Switches per Trajectory', fontsize=11)
    ax2.set_title('Mode Switching Frequency', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(agent_types, rotation=45, ha='right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Observation coverage map (bottom-left)
    ax3 = fig.add_subplot(gs[1, :])

    # Use first agent type's observation map
    first_type = agent_types[0]
    if 'spatial_observation_map' in aggregate_stats[first_type]:
        obs_map = aggregate_stats[first_type]['spatial_observation_map']

        im = ax3.imshow(obs_map.T, origin='lower', cmap='RdYlGn_r',
                       vmin=0, vmax=1, interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('P(Observed | Visited)', fontsize=11)

        if observer_locations:
            obs_x = [obs[0] for obs in observer_locations]
            obs_y = [obs[1] for obs in observer_locations]
            ax3.scatter(obs_x, obs_y, marker='X', s=150, c='blue',
                       edgecolors='white', linewidths=2, alpha=0.8, zorder=10)

        ax3.scatter([0], [0], marker='s', s=200, c='green',
                   edgecolors='white', linewidths=2, zorder=10)
        ax3.scatter([grid_size-1], [grid_size-1], marker='*', s=250, c='gold',
                   edgecolors='white', linewidths=2, zorder=10)

        ax3.set_xlabel('X Position', fontsize=11)
        ax3.set_ylabel('Y Position', fontsize=11)
        ax3.set_title('Observation Coverage Map', fontsize=13, fontweight='bold')

        ax3.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax3.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax3.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # Plot 4: Statistics table (bottom)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    # Build statistics table
    table_data = []
    headers = ['Agent Type', 'N Trajs', 'Mean Switches', 'Std Switches',
              'Mean Obs%', 'Switches/Step']

    for agent_type in agent_types:
        stats = aggregate_stats[agent_type]
        row = [
            agent_type,
            stats.get('n_trajectories', 0),
            f"{stats.get('mean_switches_per_trajectory', 0):.2f}",
            f"{stats.get('std_switches_per_trajectory', 0):.2f}",
            f"{stats.get('mean_observed_fraction', 0)*100:.1f}%",
            f"{stats.get('mean_switches_per_step', 0):.3f}"
        ]
        table_data.append(row)

    table = ax4.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    ax4.set_title('Summary Statistics', fontsize=13, fontweight='bold', pad=20)

    # Overall title
    fig.suptitle('Mode Switching Analysis Summary', fontsize=16, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
