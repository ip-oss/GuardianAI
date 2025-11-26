"""
Visualization tools for LLM behavioral fingerprinting results.

Creates publication-quality figures showing:
1. t-SNE/PCA embeddings colored by agent type
2. Confusion matrix heatmap
3. Feature importance for each misalignment type
4. Danger level correlation plot
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from typing import Dict, List
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(results_dir: str, timestamp: str = None):
    """Load experiment results from files."""
    results_path = Path(results_dir)

    if timestamp is None:
        # Find most recent results
        result_files = sorted(results_path.glob('results_*.json'))
        if not result_files:
            raise FileNotFoundError(f"No results found in {results_dir}")
        results_file = result_files[-1]
        timestamp = results_file.stem.split('_', 1)[1]
    else:
        results_file = results_path / f'results_{timestamp}.json'

    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Load embeddings
    embeddings_file = results_path / f'embeddings_{timestamp}.npz'
    embeddings = np.load(embeddings_file, allow_pickle=True)

    # Load signatures
    signatures_file = results_path / f'signatures_{timestamp}.json'
    with open(signatures_file, 'r') as f:
        signatures = json.load(f)

    return results, embeddings, signatures, timestamp


def plot_embeddings(embeddings, results, output_path: Path):
    """Create t-SNE and PCA scatter plots."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    tsne_emb = embeddings['tsne']
    pca_emb = embeddings['pca']
    labels = embeddings['labels']
    type_labels = embeddings['type_labels']

    # Create color map
    danger_levels = results['danger_levels']
    colors = [danger_levels.get(label, 0) for label in labels]

    # t-SNE plot
    scatter1 = axes[0].scatter(
        tsne_emb[:, 0], tsne_emb[:, 1],
        c=colors, cmap='RdYlGn_r',
        alpha=0.6, s=100, edgecolors='black', linewidth=0.5
    )
    axes[0].set_title('t-SNE Embedding (Colored by Danger Level)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')

    # Add colorbar
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('Danger Level', rotation=270, labelpad=20)

    # Add legend with type names
    for i, agent_type in enumerate(type_labels):
        mask = np.array(labels) == agent_type
        if mask.any():
            axes[0].scatter([], [], c=[danger_levels[agent_type]], cmap='RdYlGn_r',
                          label=agent_type, s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[0].legend(loc='best', framealpha=0.9)

    # PCA plot
    scatter2 = axes[1].scatter(
        pca_emb[:, 0], pca_emb[:, 1],
        c=colors, cmap='RdYlGn_r',
        alpha=0.6, s=100, edgecolors='black', linewidth=0.5
    )
    axes[1].set_title('PCA Embedding (Colored by Danger Level)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('PC 1')
    axes[1].set_ylabel('PC 2')

    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('Danger Level', rotation=270, labelpad=20)

    # Add legend
    for i, agent_type in enumerate(type_labels):
        mask = np.array(labels) == agent_type
        if mask.any():
            axes[1].scatter([], [], c=[danger_levels[agent_type]], cmap='RdYlGn_r',
                          label=agent_type, s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[1].legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path / 'embeddings.png', dpi=300, bbox_inches='tight')
    print(f"  Saved embeddings plot to {output_path / 'embeddings.png'}")
    plt.close()


def plot_confusion_matrix(results, output_path: Path):
    """Create confusion matrix heatmap."""
    conf_matrix = np.array(results['confusion_matrix'])
    type_labels = results['agent_types']

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        conf_matrix, annot=True, fmt='d',
        cmap='Blues', square=True,
        xticklabels=type_labels,
        yticklabels=type_labels,
        cbar_kws={'label': 'Count'},
        ax=ax
    )

    ax.set_title('Confusion Matrix: Predicted vs True Agent Type', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Type', fontsize=12)
    ax.set_ylabel('True Type', fontsize=12)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"  Saved confusion matrix to {output_path / 'confusion_matrix.png'}")
    plt.close()


def plot_per_class_accuracy(results, output_path: Path):
    """Bar chart of per-class detection accuracy."""
    per_class = results['per_class_accuracy']
    danger_levels = results['danger_levels']

    agent_types = list(per_class.keys())
    accuracies = [per_class[t] for t in agent_types]
    dangers = [danger_levels[t] for t in agent_types]

    # Sort by danger level
    sorted_indices = np.argsort(dangers)
    agent_types = [agent_types[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    dangers = [dangers[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(range(len(agent_types)), accuracies,
                  color=[plt.cm.RdYlGn_r(d/10) for d in dangers],
                  edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Agent Type (sorted by danger)', fontsize=12)
    ax.set_ylabel('Detection Accuracy', fontsize=12)
    ax.set_title('Per-Class Detection Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(agent_types)))
    ax.set_xticklabels(agent_types, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='80% threshold')
    ax.axhline(y=1/len(agent_types), color='red', linestyle='--', alpha=0.5, label='Random chance')

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=10)

    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'per_class_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"  Saved per-class accuracy to {output_path / 'per_class_accuracy.png'}")
    plt.close()


def plot_signatures(signatures, results, output_path: Path):
    """Heatmap of characteristic deviations per agent type."""
    agent_types = list(signatures.keys())

    # Collect all features mentioned in top deviations
    all_features = set()
    for sig in signatures.values():
        for feature, _ in sig['top_deviations'][:5]:
            all_features.add(feature)

    all_features = sorted(all_features)

    # Build matrix: rows = agent types, cols = features
    matrix = np.zeros((len(agent_types), len(all_features)))

    for i, agent_type in enumerate(agent_types):
        sig = signatures[agent_type]
        for feature, z_score in sig['top_deviations']:
            if feature in all_features:
                j = all_features.index(feature)
                matrix[i, j] = z_score

    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(
        matrix, annot=True, fmt='.2f',
        cmap='RdBu_r', center=0,
        xticklabels=all_features,
        yticklabels=agent_types,
        cbar_kws={'label': 'Z-score vs Aligned'},
        ax=ax, vmin=-3, vmax=3
    )

    ax.set_title('Behavioral Signatures: Feature Deviations from Aligned Baseline',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Agent Type', fontsize=12)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path / 'signatures.png', dpi=300, bbox_inches='tight')
    print(f"  Saved signatures heatmap to {output_path / 'signatures.png'}")
    plt.close()


def plot_danger_correlation(results, output_path: Path):
    """Scatter plot showing danger vs detection difficulty."""
    per_class = results['per_class_accuracy']
    danger_levels = results['danger_levels']

    agent_types = [t for t in per_class.keys() if t != 'aligned']
    dangers = [danger_levels[t] for t in agent_types]
    accuracies = [per_class[t] for t in agent_types]

    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(dangers, accuracies, s=200, alpha=0.6,
                        c=dangers, cmap='RdYlGn_r',
                        edgecolors='black', linewidth=2)

    # Add labels
    for i, agent_type in enumerate(agent_types):
        ax.annotate(agent_type, (dangers[i], accuracies[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, alpha=0.8)

    # Fit line
    if len(dangers) > 2:
        z = np.polyfit(dangers, accuracies, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(dangers), max(dangers), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.5, label=f'Linear fit')

    ax.set_xlabel('Danger Level', fontsize=12)
    ax.set_ylabel('Detection Accuracy', fontsize=12)
    ax.set_title('Danger Level vs Detection Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)

    # Add correlation coefficient
    if len(dangers) > 2:
        corr = np.corrcoef(dangers, accuracies)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: r = {corr:.3f}',
               transform=ax.transAxes, fontsize=12,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / 'danger_correlation.png', dpi=300, bbox_inches='tight')
    print(f"  Saved danger correlation to {output_path / 'danger_correlation.png'}")
    plt.close()


def create_all_visualizations(results_dir: str = "results/exp7_llm", timestamp: str = None):
    """Create all visualization plots."""
    print("\n📊 Creating visualizations...")

    # Load data
    results, embeddings, signatures, ts = load_results(results_dir, timestamp)

    output_path = Path(results_dir)

    # Create plots
    plot_embeddings(embeddings, results, output_path)
    plot_confusion_matrix(results, output_path)
    plot_per_class_accuracy(results, output_path)
    plot_signatures(signatures, results, output_path)
    plot_danger_correlation(results, output_path)

    print(f"\n✅ All visualizations saved to {output_path}")
    print(f"   Timestamp: {ts}")


if __name__ == "__main__":
    import sys

    results_dir = "results/exp7_llm"
    timestamp = None

    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    if len(sys.argv) > 2:
        timestamp = sys.argv[2]

    create_all_visualizations(results_dir, timestamp)
