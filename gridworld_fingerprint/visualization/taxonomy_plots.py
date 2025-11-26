"""
Visualizations for misalignment taxonomy experiment.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


def plot_fingerprint_clusters(
    embeddings: np.ndarray,
    labels: List[str],
    danger_levels: Dict[str, int],
    save_path: Optional[Path] = None,
    show: bool = False
):
    """
    Plot t-SNE embeddings colored by misalignment type and sized by danger.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Get unique types and colors
    unique_types = list(set(labels))
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_types)))
    color_map = {t: colors[i] for i, t in enumerate(unique_types)}

    # Plot 1: Color by type
    ax1 = axes[0]
    for agent_type in unique_types:
        mask = np.array([l == agent_type for l in labels])
        ax1.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=[color_map[agent_type]],
            label=agent_type,
            s=50,
            alpha=0.7
        )

    ax1.set_xlabel('t-SNE 1', fontsize=12)
    ax1.set_ylabel('t-SNE 2', fontsize=12)
    ax1.set_title('Fingerprint Clusters by Misalignment Type', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    # Plot 2: Size by danger level
    ax2 = axes[1]

    # Get danger for each point
    point_dangers = [danger_levels.get(l, 0) for l in labels]
    sizes = [20 + d * 15 for d in point_dangers]  # Scale sizes

    scatter = ax2.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=point_dangers,
        s=sizes,
        cmap='RdYlGn_r',  # Red = dangerous
        alpha=0.7,
        vmin=0,
        vmax=10
    )

    plt.colorbar(scatter, ax=ax2, label='Danger Level')
    ax2.set_xlabel('t-SNE 1', fontsize=12)
    ax2.set_ylabel('t-SNE 2', fontsize=12)
    ax2.set_title('Fingerprint Space: Size & Color by Danger', fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    show: bool = False
):
    """Plot confusion matrix for multi-class classification."""

    fig, ax = plt.subplots(figsize=(10, 8))

    # Normalize by row (true labels)
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
    conf_normalized = conf_matrix.astype(float) / row_sums

    im = ax.imshow(conf_normalized, cmap='Blues', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Proportion', fontsize=10)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = conf_normalized[i, j]
            text_color = 'white' if value > 0.5 else 'black'
            ax.text(j, i, f'{value:.2f}',
                   ha='center', va='center',
                   color=text_color, fontsize=8)

    ax.set_xlabel('Predicted Type', fontsize=12)
    ax.set_ylabel('True Type', fontsize=12)
    ax.set_title('Misalignment Type Classification\nConfusion Matrix (Normalized)', fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_type_similarity_heatmap(
    similarity_matrix: np.ndarray,
    type_names: List[str],
    save_path: Optional[Path] = None,
    show: bool = False
):
    """Plot heatmap showing similarity between misalignment types."""

    fig, ax = plt.subplots(figsize=(10, 8))

    # Convert distance to similarity (inverse)
    max_dist = similarity_matrix.max() if similarity_matrix.max() > 0 else 1
    similarity = 1 - (similarity_matrix / max_dist)

    im = ax.imshow(similarity, cmap='YlOrRd', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Similarity (1 - normalized distance)', fontsize=10)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(type_names)))
    ax.set_yticks(np.arange(len(type_names)))
    ax.set_xticklabels(type_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(type_names, fontsize=9)

    # Add text annotations
    for i in range(len(type_names)):
        for j in range(len(type_names)):
            value = similarity[i, j]
            text_color = 'white' if value > 0.5 else 'black'
            ax.text(j, i, f'{value:.2f}',
                   ha='center', va='center',
                   color=text_color, fontsize=8)

    ax.set_title('Misalignment Type Similarity\n(based on fingerprint distance)', fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_danger_correlation(
    fingerprints_by_type: Dict[str, List[Dict]],
    danger_levels: Dict[str, int],
    save_path: Optional[Path] = None,
    show: bool = False
):
    """Plot relationship between fingerprint distance from aligned and danger level."""

    # Compute mean fingerprint for aligned
    aligned_fps = fingerprints_by_type.get("aligned", [])
    if not aligned_fps:
        print("Warning: No aligned fingerprints found")
        return

    feature_keys = [
        k for k in aligned_fps[0].keys()
        if isinstance(aligned_fps[0][k], (int, float)) and not isinstance(aligned_fps[0][k], bool)
    ]

    aligned_mean = {k: np.mean([fp[k] for fp in aligned_fps]) for k in feature_keys}

    # Compute distance for each type
    types = []
    distances = []
    dangers = []

    for agent_type, fps in fingerprints_by_type.items():
        if agent_type == "aligned":
            continue

        type_mean = {k: np.mean([fp[k] for fp in fps]) for k in feature_keys}

        # Euclidean distance
        dist = np.sqrt(sum((type_mean[k] - aligned_mean[k])**2 for k in feature_keys))

        types.append(agent_type)
        distances.append(dist)
        dangers.append(danger_levels.get(agent_type, 0))

    if not distances:
        print("Warning: No non-aligned fingerprints found")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter plot
    colors = plt.cm.RdYlGn_r(np.array(dangers) / 10)

    for i, (t, d, danger) in enumerate(zip(types, distances, dangers)):
        ax.scatter(d, danger, c=[colors[i]], s=200, alpha=0.7, edgecolors='black', linewidth=1)
        ax.annotate(t, (d, danger), xytext=(5, 5), textcoords='offset points', fontsize=9)

    # Fit line if we have enough points
    if len(distances) > 2:
        from scipy import stats
        z = np.polyfit(distances, dangers, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(distances) * 0.9, max(distances) * 1.1, 100)
        ax.plot(x_line, p(x_line), 'k--', alpha=0.5, label='Linear fit')

        r, p_val = stats.pearsonr(distances, dangers)
        ax.set_title(f'Fingerprint Distance vs Danger Level\n(r = {r:.3f}, p = {p_val:.3f})', fontsize=14)
    else:
        ax.set_title('Fingerprint Distance vs Danger Level', fontsize=14)

    ax.set_xlabel('Fingerprint Distance from Aligned', fontsize=12)
    ax.set_ylabel('Danger Level (0-10)', fontsize=12)
    ax.set_ylim(-0.5, 10.5)

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_signature_radar(
    signatures: Dict[str, Dict],
    save_path: Optional[Path] = None,
    show: bool = False
):
    """Radar chart showing signature profile for each misalignment type."""

    # Get all features across signatures
    all_features = set()
    for sig in signatures.values():
        for feature, _ in sig.get("top_deviations", []):
            all_features.add(feature)

    features = sorted(list(all_features))[:8]  # Limit to 8 for readability

    if not features:
        print("Warning: No features found for radar plot")
        return

    # Setup radar chart
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    colors = plt.cm.Set2(np.linspace(0, 1, len(signatures)))

    for i, (agent_type, sig) in enumerate(signatures.items()):
        # Get z-scores for each feature
        z_dict = dict(sig.get("top_deviations", []))
        values = [abs(z_dict.get(f, 0)) for f in features]
        values += values[:1]  # Close the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=agent_type, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, size=9)
    ax.set_title('Misalignment Signatures\n(|z-score| from aligned)', fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_feature_importance_comparison(
    discriminative_features: Dict[str, List[str]],
    shared_features: List[str],
    save_path: Optional[Path] = None,
    show: bool = False
):
    """Plot comparison of discriminative vs shared features."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Shared features (common indicators of misalignment)
    ax1 = axes[0]
    if shared_features:
        y_pos = np.arange(len(shared_features))
        ax1.barh(y_pos, np.arange(len(shared_features), 0, -1), color='steelblue', alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(shared_features)
        ax1.set_xlabel('Importance Rank')
        ax1.set_title('Shared Misalignment Indicators\n(Common across types)')
    else:
        ax1.text(0.5, 0.5, 'No shared features found', ha='center', va='center')
        ax1.set_title('Shared Misalignment Indicators')

    # Plot 2: Type-specific top features
    ax2 = axes[1]

    # Collect unique top features per type
    type_top_features = {}
    for agent_type, features in discriminative_features.items():
        if features:
            type_top_features[agent_type] = features[0]  # Top feature

    if type_top_features:
        types = list(type_top_features.keys())
        features = [type_top_features[t] for t in types]

        y_pos = np.arange(len(types))
        ax2.barh(y_pos, [1] * len(types), color='coral', alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f"{t}\n({f})" for t, f in zip(types, features)], fontsize=9)
        ax2.set_xlabel('Type-Specific Top Feature')
        ax2.set_title('Most Discriminative Feature per Type')
    else:
        ax2.text(0.5, 0.5, 'No type-specific features found', ha='center', va='center')
        ax2.set_title('Type-Specific Features')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
