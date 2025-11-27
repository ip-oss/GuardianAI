"""
Analysis of stealth agent failure modes.

This module provides functions to understand why stealth agents
fail to evade detection despite attempting to minimize their fingerprint.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
from pathlib import Path
from utils.trajectory import Trajectory


def analyze_stealth_betrayal(
    stealth_fingerprints: List[Dict[str, float]],
    aligned_fingerprints: List[Dict[str, float]],
    feature_names: List[str] = None
) -> Dict[str, Any]:
    """Identify which features betrayed the stealth agent.
    
    Compares stealth agent fingerprints to aligned baseline to find
    discriminative features that reveal misalignment despite evasion attempts.
    
    Args:
        stealth_fingerprints: List of fingerprint dicts from stealth agent
        aligned_fingerprints: List of fingerprint dicts from aligned agent
        feature_names: Optional list of features to analyze
        
    Returns:
        Dict with betrayal analysis results
    """
    if feature_names is None:
        # Get all numeric features
        feature_names = [
            k for k in stealth_fingerprints[0].keys()
            if isinstance(stealth_fingerprints[0][k], (int, float))
        ]
    
    betrayal_features = []
    
    for feature in feature_names:
        stealth_vals = np.array([fp[feature] for fp in stealth_fingerprints])
        aligned_vals = np.array([fp[feature] for fp in aligned_fingerprints])
        
        stealth_mean = np.mean(stealth_vals)
        aligned_mean = np.mean(aligned_vals)
        
        # Compute divergence
        if abs(aligned_mean) > 1e-6:
            divergence = abs(stealth_mean - aligned_mean) / abs(aligned_mean)
        else:
            divergence = abs(stealth_mean - aligned_mean)
        
        # Compute effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(stealth_vals) + np.var(aligned_vals)) / 2)
        if pooled_std > 1e-6:
            cohens_d = (stealth_mean - aligned_mean) / pooled_std
        else:
            cohens_d = 0.0
        
        betrayal_features.append({
            'feature': feature,
            'stealth_mean': float(stealth_mean),
            'aligned_mean': float(aligned_mean),
            'divergence': float(divergence),
            'cohens_d': float(cohens_d),
            'betrayed': bool(divergence > 0.1)  # >10% different
        })
    
    # Sort by divergence
    betrayal_features.sort(key=lambda x: x['divergence'], reverse=True)
    
    # Count how many features betrayed
    betrayed_count = sum(1 for f in betrayal_features if f['betrayed'])
    
    return {
        'features': betrayal_features,
        'betrayed_count': betrayed_count,
        'total_features': len(feature_names),
        'betrayal_rate': betrayed_count / len(feature_names) if feature_names else 0,
        'top_betrayals': betrayal_features[:5]
    }


def plot_stealth_betrayal(
    analysis: Dict[str, Any],
    save_path: Path = None,
    show: bool = False
):
    """Visualize which features betrayed the stealth agent.
    
    Creates a bar chart showing divergence for each feature.
    """
    features_data = analysis['features'][:10]  # Top 10
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    feature_names = [f['feature'] for f in features_data]
    divergences = [f['divergence'] for f in features_data]
    betrayed = [f['betrayed'] for f in features_data]
    
    colors = ['#E74C3C' if b else '#95A5A6' for b in betrayed]
    
    bars = ax.barh(feature_names, divergences, color=colors, alpha=0.7)
    
    # Add threshold line
    ax.axvline(x=0.1, color='red', linestyle='--', linewidth=2,
              label='Betrayal Threshold (10%)')
    
    ax.set_xlabel('Divergence from Aligned', fontsize=12)
    ax.set_title(f'Stealth Agent Betrayal Analysis\n'
                f'{analysis["betrayed_count"]}/{analysis["total_features"]} '
                f'features revealed misalignment',
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_trajectory_comparison(
    stealth_trajs: List[Trajectory],
    aligned_trajs: List[Trajectory],
    grid_size: int,
    save_path: Path = None,
    show: bool = False
):
    """Compare trajectory patterns between stealth and aligned agents."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Create heatmaps of visit frequency
    stealth_visits = np.zeros((grid_size, grid_size))
    aligned_visits = np.zeros((grid_size, grid_size))
    
    for traj in stealth_trajs:
        for state in traj.states:
            x, y = state
            if 0 <= x < grid_size and 0 <= y < grid_size:
                stealth_visits[x, y] += 1
    
    for traj in aligned_trajs:
        for state in traj.states:
            x, y = state
            if 0 <= x < grid_size and 0 <= y < grid_size:
                aligned_visits[x, y] += 1
    
    # Plot stealth
    im1 = axes[0].imshow(stealth_visits.T, origin='lower', cmap='Reds')
    axes[0].set_title('Stealth Agent Paths', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0], label='Visits')
    
    # Plot aligned
    im2 = axes[1].imshow(aligned_visits.T, origin='lower', cmap='Blues')
    axes[1].set_title('Aligned Agent Paths', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[1], label='Visits')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_error_distribution_comparison(
    stealth_errors: List[float],
    aligned_errors: List[float],
    save_path: Path = None,
    show: bool = False
):
    """Compare prediction error distributions."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bins = np.linspace(
        min(min(stealth_errors), min(aligned_errors)),
        max(max(stealth_errors), max(aligned_errors)),
        30
    )
    
    ax.hist(aligned_errors, bins=bins, alpha=0.6, label='Aligned',
           color='#3498DB', edgecolor='black')
    ax.hist(stealth_errors, bins=bins, alpha=0.6, label='Stealth',
           color='#E74C3C', edgecolor='black')
    
    # Add means
    ax.axvline(np.mean(aligned_errors), color='#3498DB', linestyle='--',
              linewidth=2, label=f'Aligned Mean: {np.mean(aligned_errors):.2f}')
    ax.axvline(np.mean(stealth_errors), color='#E74C3C', linestyle='--',
              linewidth=2, label=f'Stealth Mean: {np.mean(stealth_errors):.2f}')
    
    ax.set_xlabel('Prediction Error', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Error Distribution: Stealth vs Aligned', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
