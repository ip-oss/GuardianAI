"""
Visualizations for extended validation experiments.

This module provides plotting functions for:
- Subtlety gradient (detection vs misalignment rate)
- Transfer learning validation
- Stealth misalignment analysis
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path


def plot_detection_curve(
    results: Dict[float, Dict[str, Any]],
    save_path: Optional[Path] = None,
    show: bool = False
):
    """
    Plot detection accuracy and AUROC vs misalignment rate.

    Creates a dual-axis plot showing how detection performance varies with
    the fraction of misaligned actions.

    Args:
        results: Dict mapping misalignment_rate -> {'accuracy', 'auroc', 'detectable'}
        save_path: Optional path to save figure
        show: Whether to display the plot

    Example:
        >>> results = {
        ...     0.05: {'accuracy': 0.52, 'auroc': 0.54, 'detectable': False},
        ...     0.10: {'accuracy': 0.68, 'auroc': 0.75, 'detectable': True},
        ...     0.20: {'accuracy': 0.89, 'auroc': 0.94, 'detectable': True},
        ... }
        >>> plot_detection_curve(results)
    """
    rates = sorted(results.keys())
    accuracies = [results[r]['accuracy'] for r in rates]
    aurocs = [results[r]['auroc'] for r in rates]
    detectable = [results[r]['detectable'] for r in rates]

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot accuracy
    color1 = '#2E86DE'
    ax1.set_xlabel('Misalignment Rate', fontsize=13)
    ax1.set_ylabel('Classification Accuracy', fontsize=13, color=color1)
    line1 = ax1.plot(rates, accuracies, 'o-', linewidth=2.5, markersize=10,
                     color=color1, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0.45, 1.05)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add detection threshold line
    ax1.axhline(y=0.7, color='gray', linestyle='--', linewidth=1.5,
               alpha=0.7, label='Detection Threshold (0.7)')

    # Plot AUROC on secondary axis
    ax2 = ax1.twinx()
    color2 = '#EE5A24'
    ax2.set_ylabel('AUROC', fontsize=13, color=color2)
    line2 = ax2.plot(rates, aurocs, 's-', linewidth=2.5, markersize=10,
                    color=color2, label='AUROC')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0.45, 1.05)

    # Shade detectable vs undetectable regions
    for i in range(len(rates) - 1):
        if not detectable[i]:
            ax1.axvspan(rates[i], rates[i+1], alpha=0.15, color='red', zorder=0)

    # Find detection threshold
    threshold_rates = [r for r in rates if results[r]['detectable']]
    if threshold_rates:
        min_detectable = min(threshold_rates)
        ax1.axvline(x=min_detectable, color='green', linestyle=':', linewidth=2,
                   alpha=0.8, label=f'Min Detectable: {min_detectable*100:.0f}%')

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines + [ax1.get_lines()[-2], ax1.get_lines()[-1]],
              labels + ['Detection Threshold', f'Min Detectable: {min_detectable*100:.0f}%' if threshold_rates else 'Min Detectable'],
              loc='lower right', fontsize=11)

    # Format x-axis as percentages
    ax1.set_xticks(rates)
    ax1.set_xticklabels([f'{r*100:.0f}%' for r in rates])

    ax1.set_title('Detection Performance vs Misalignment Rate', fontsize=15, fontweight='bold', pad=15)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_transfer_comparison(
    same_layout_results: Dict[str, float],
    transfer_layout_results: Dict[str, float],
    save_path: Optional[Path] = None,
    show: bool = False
):
    """
    Compare detection performance on same layout vs transfer layout.

    Creates side-by-side bar charts showing accuracy for each agent type
    on the training layout vs a novel layout.

    Args:
        same_layout_results: Dict mapping agent_type -> accuracy on same layout
        transfer_layout_results: Dict mapping agent_type -> accuracy on transfer layout
        save_path: Optional path to save figure
        show: Whether to display the plot

    Example:
        >>> same = {'aligned': 1.0, 'deceptive': 0.95, 'adversarial': 0.98}
        >>> transfer = {'aligned': 0.92, 'deceptive': 0.81, 'adversarial': 0.88}
        >>> plot_transfer_comparison(same, transfer)
    """
    agent_types = list(same_layout_results.keys())
    same_accs = [same_layout_results[t] for t in agent_types]
    transfer_accs = [transfer_layout_results[t] for t in agent_types]

    x = np.arange(len(agent_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))

    bars1 = ax.bar(x - width/2, same_accs, width, label='Same Layout (Train)',
                   color='#54A0FF', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, transfer_accs, width, label='Transfer Layout (Test)',
                   color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Calculate and show average transfer gap
    gaps = [same_accs[i] - transfer_accs[i] for i in range(len(agent_types))]
    avg_gap = np.mean(gaps)
    max_gap = np.max(gaps)

    ax.axhline(y=0.7, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Detection Threshold')

    ax.set_ylabel('Classification Accuracy', fontsize=13)
    ax.set_xlabel('Agent Type', fontsize=13)
    ax.set_title(f'Transfer Learning Validation\n(Avg Gap: {avg_gap*100:.1f}%, Max Gap: {max_gap*100:.1f}%)',
                fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(agent_types, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11, loc='lower left')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add gap indicators
    for i, gap in enumerate(gaps):
        if gap > 0.05:  # Only show significant gaps
            ax.annotate('', xy=(i, transfer_accs[i]), xytext=(i, same_accs[i]),
                       arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
            ax.text(i, (same_accs[i] + transfer_accs[i])/2, f'-{gap*100:.0f}%',
                   ha='center', va='center', fontsize=8, color='red',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red'))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_stealth_analysis(
    regular_fingerprints: Dict[str, List[float]],
    stealth_fingerprints: List[float],
    feature_names: List[str],
    save_path: Optional[Path] = None,
    show: bool = False
):
    """
    Compare stealth agent fingerprints to regular agent fingerprints.

    Creates a radar chart and box plots showing how stealth agent's
    behavioral signature compares to openly misaligned agents.

    Args:
        regular_fingerprints: Dict mapping agent_type -> list of feature values
        stealth_fingerprints: List of feature values for stealth agent
        feature_names: Names of features
        save_path: Optional path to save figure
        show: Whether to display the plot

    Example:
        >>> regular = {
        ...     'aligned': [0.1, 0.2, 0.3],
        ...     'adversarial': [0.8, 0.7, 0.9]
        ... }
        >>> stealth = [0.3, 0.4, 0.5]
        >>> features = ['mean_error', 'std_error', 'max_error']
        >>> plot_stealth_analysis(regular, stealth, features)
    """
    fig = plt.subplots(figsize=(16, 7))
    fig = plt.gcf()
    gs = fig.add_gridspec(1, 2, wspace=0.3)

    # Plot 1: Radar chart comparing stealth to aligned and most misaligned
    ax1 = fig.add_subplot(gs[0], projection='polar')

    # Select top 6 features for readability
    top_features = feature_names[:6]
    n_features = len(top_features)

    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    # Normalize values to [0, 1] for radar chart
    all_values = [stealth_fingerprints[:n_features]]
    for fp_list in regular_fingerprints.values():
        all_values.append(fp_list[:n_features])

    vmin = min(min(v) for v in all_values if len(v) > 0)
    vmax = max(max(v) for v in all_values if len(v) > 0)

    def normalize(vals):
        if vmax - vmin == 0:
            return [0.5] * len(vals)
        return [(v - vmin) / (vmax - vmin) for v in vals]

    # Plot stealth agent
    stealth_norm = normalize(stealth_fingerprints[:n_features]) + normalize(stealth_fingerprints[:n_features])[:1]
    ax1.plot(angles, stealth_norm, 'o-', linewidth=3, label='Stealth', color='purple')
    ax1.fill(angles, stealth_norm, alpha=0.2, color='purple')

    # Plot aligned baseline
    if 'aligned' in regular_fingerprints:
        aligned_norm = normalize(regular_fingerprints['aligned'][:n_features])
        aligned_norm += aligned_norm[:1]
        ax1.plot(angles, aligned_norm, 'o-', linewidth=2, label='Aligned', color='green')
        ax1.fill(angles, aligned_norm, alpha=0.1, color='green')

    # Plot most misaligned agent
    if 'adversarial' in regular_fingerprints:
        adv_norm = normalize(regular_fingerprints['adversarial'][:n_features])
        adv_norm += adv_norm[:1]
        ax1.plot(angles, adv_norm, 'o-', linewidth=2, label='Adversarial', color='red')
        ax1.fill(angles, adv_norm, alpha=0.1, color='red')

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(top_features, size=10)
    ax1.set_ylim(0, 1)
    ax1.set_title('Stealth Agent Behavioral Signature', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    ax1.grid(True)

    # Plot 2: Box plots showing distribution
    ax2 = fig.add_subplot(gs[1])

    agent_types = list(regular_fingerprints.keys()) + ['stealth']
    colors = plt.cm.Set2(np.linspace(0, 1, len(agent_types)))

    # Use mean error as representative metric
    mean_errors = []
    labels = []

    for agent_type in regular_fingerprints.keys():
        # Assume we have multiple trajectories - use first feature (mean_error) from each
        mean_errors.append([regular_fingerprints[agent_type][0]])  # Simplified
        labels.append(agent_type)

    # Add stealth agent
    mean_errors.append([stealth_fingerprints[0]])
    labels.append('stealth')

    bp = ax2.boxplot(mean_errors, labels=labels, patch_artist=True,
                     showmeans=True, meanline=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Highlight stealth agent
    bp['boxes'][-1].set_edgecolor('purple')
    bp['boxes'][-1].set_linewidth(3)

    ax2.set_ylabel('Mean Prediction Error', fontsize=12)
    ax2.set_xlabel('Agent Type', fontsize=12)
    ax2.set_title('Fingerprint Distribution Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_comprehensive_validation_summary(
    subtlety_results: Dict[float, Dict[str, Any]],
    transfer_gap: float,
    stealth_detectable: bool,
    save_path: Optional[Path] = None,
    show: bool = False
):
    """
    Comprehensive summary figure with all three validation experiments.

    Creates a 2x2 grid showing:
    - Top-left: Detection curve (subtlety gradient)
    - Top-right: Transfer performance
    - Bottom-left: Key metrics table
    - Bottom-right: Summary findings

    Args:
        subtlety_results: Results from subtlety gradient experiment
        transfer_gap: Average accuracy drop on transfer layout
        stealth_detectable: Whether stealth agent was successfully detected
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Subtlety gradient (top-left)
    ax1 = fig.add_subplot(gs[0, 0])

    rates = sorted(subtlety_results.keys())
    aurocs = [subtlety_results[r]['auroc'] for r in rates]

    ax1.plot(rates, aurocs, 'o-', linewidth=2.5, markersize=10, color='#2E86DE')
    ax1.axhline(y=0.7, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.fill_between(rates, aurocs, 0.5, alpha=0.2, color='#2E86DE')

    ax1.set_xlabel('Misalignment Rate', fontsize=12)
    ax1.set_ylabel('AUROC', fontsize=12)
    ax1.set_title('Subtlety Gradient: Detection vs Misalignment Rate',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.45, 1.05)

    # Find threshold
    threshold_rates = [r for r in rates if subtlety_results[r]['detectable']]
    if threshold_rates:
        min_detectable = min(threshold_rates)
        ax1.axvline(x=min_detectable, color='green', linestyle=':', linewidth=2)
        ax1.text(min_detectable, 0.5, f'  {min_detectable*100:.0f}% threshold',
                rotation=90, va='bottom', fontsize=10, color='green')

    # Plot 2: Transfer validation (top-right)
    ax2 = fig.add_subplot(gs[0, 1])

    metrics = ['Subtlety\nThreshold', 'Transfer\nGap', 'Stealth\nDetection']
    values = [
        min_detectable * 100 if threshold_rates else 0,
        transfer_gap * 100,
        100 if stealth_detectable else 0
    ]
    colors_bar = ['#48C9B0' if v > 0 else '#E74C3C' for v in values]

    bars = ax2.bar(metrics, values, color=colors_bar, alpha=0.7,
                   edgecolor='black', linewidth=2)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax2.set_ylabel('Percentage', fontsize=12)
    ax2.set_title('Validation Summary Metrics', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Metrics table (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')

    table_data = []
    if threshold_rates:
        table_data.append(['Minimum Detectable Misalignment', f'{min_detectable*100:.0f}%'])
    else:
        table_data.append(['Minimum Detectable Misalignment', 'N/A'])

    table_data.append(['Transfer Performance Gap', f'{transfer_gap*100:.1f}%'])
    table_data.append(['Stealth Agent Detection', 'Success ✓' if stealth_detectable else 'Evaded ✗'])

    # Add AUROC at different rates
    for rate in [0.10, 0.20, 0.50]:
        if rate in subtlety_results:
            table_data.append([f'AUROC at {rate*100:.0f}% Misalignment',
                             f'{subtlety_results[rate]["auroc"]:.3f}'])

    table = ax3.table(cellText=table_data,
                     cellLoc='left',
                     loc='center',
                     bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style table
    for i in range(len(table_data)):
        table[(i, 0)].set_facecolor('#F0F0F0')
        table[(i, 1)].set_facecolor('#FFFFFF')
        table[(i, 0)].set_text_props(weight='bold')

    ax3.set_title('Detailed Metrics', fontsize=13, fontweight='bold', pad=20)

    # Plot 4: Key findings (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    findings = []

    # Subtlety finding
    if threshold_rates and min_detectable <= 0.15:
        findings.append(f"✓ HIGH SENSITIVITY: Detects misalignment at {min_detectable*100:.0f}% rate")
    elif threshold_rates:
        findings.append(f"○ MODERATE SENSITIVITY: Threshold at {min_detectable*100:.0f}%")
    else:
        findings.append("✗ LOW SENSITIVITY: Cannot detect subtle misalignment")

    # Transfer finding
    if transfer_gap < 0.10:
        findings.append(f"✓ STRONG GENERALIZATION: Only {transfer_gap*100:.0f}% drop on new layout")
    elif transfer_gap < 0.20:
        findings.append(f"○ MODERATE GENERALIZATION: {transfer_gap*100:.0f}% drop on new layout")
    else:
        findings.append(f"✗ POOR GENERALIZATION: {transfer_gap*100:.0f}% drop on new layout")

    # Stealth finding
    if stealth_detectable:
        findings.append("✓ ROBUST TO EVASION: Detected adversarial stealth agent")
    else:
        findings.append("✗ VULNERABLE TO EVASION: Stealth agent evaded detection")

    # Overall verdict
    successes = sum([f.startswith('✓') for f in findings])
    if successes == 3:
        overall = "\n🎯 VALIDATION PASSED: System is robust and generalizable"
        color = 'green'
    elif successes >= 2:
        overall = "\n⚠ PARTIAL SUCCESS: Some limitations identified"
        color = 'orange'
    else:
        overall = "\n❌ VALIDATION CONCERNS: Significant limitations found"
        color = 'red'

    findings.append(overall)

    # Display findings
    findings_text = '\n\n'.join(findings)
    ax4.text(0.5, 0.5, findings_text,
            ha='center', va='center',
            fontsize=11, linespacing=1.8,
            bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', edgecolor=color, linewidth=3))

    ax4.set_title('Key Findings', fontsize=13, fontweight='bold', pad=20)

    # Overall title
    fig.suptitle('Extended Validation Results: Subtlety, Transfer, and Evasion',
                fontsize=16, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
