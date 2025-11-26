"""
Fingerprint analysis for misalignment taxonomy.

Analyzes whether different misalignment types produce
distinguishable behavioral signatures.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import scipy.stats as stats


@dataclass
class TaxonomyAnalysisResults:
    """Results from taxonomy analysis."""
    # Classification
    multiclass_accuracy: float
    per_class_accuracy: Dict[str, float]
    confusion_matrix: np.ndarray
    classification_report: str

    # Clustering
    cluster_purity: float
    silhouette_score: float
    adjusted_rand_index: float

    # Feature analysis
    discriminative_features: Dict[str, List[str]]  # Per class
    shared_features: List[str]  # Common across misalignment types

    # Relationships
    type_similarity_matrix: np.ndarray
    danger_correlation: float  # Correlation between fingerprint distance and danger level

    # Embeddings for visualization
    tsne_embeddings: np.ndarray
    pca_embeddings: np.ndarray
    labels: List[str]
    type_labels: List[str] = field(default_factory=list)


def get_feature_names(fingerprints_by_type: Dict[str, List[Dict]]) -> List[str]:
    """Get consistent feature names from fingerprints."""
    # Get all numeric features from first fingerprint
    for fps in fingerprints_by_type.values():
        if fps:
            sample_fp = fps[0]
            break
    else:
        return []

    feature_names = []
    for key, value in sample_fp.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            feature_names.append(key)

    return sorted(feature_names)


def prepare_fingerprint_data(
    fingerprints_by_type: Dict[str, List[Dict]]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Convert fingerprint dicts to numpy arrays."""

    type_labels = list(fingerprints_by_type.keys())
    feature_names = get_feature_names(fingerprints_by_type)

    X_list = []
    y_list = []

    for type_idx, (agent_type, fps) in enumerate(fingerprints_by_type.items()):
        for fp in fps:
            features = [fp.get(name, 0.0) for name in feature_names]
            X_list.append(features)
            y_list.append(type_idx)

    return np.array(X_list), np.array(y_list), type_labels


def compute_cluster_purity(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    """Compute cluster purity score."""
    from collections import Counter

    contingency = {}
    for true, cluster in zip(true_labels, cluster_labels):
        if cluster not in contingency:
            contingency[cluster] = Counter()
        contingency[cluster][true] += 1

    purity = 0
    for cluster_counts in contingency.values():
        purity += cluster_counts.most_common(1)[0][1]

    return purity / len(true_labels)


def analyze_misalignment_taxonomy(
    fingerprints_by_type: Dict[str, List[Dict]],
    danger_levels: Dict[str, int] = None
) -> TaxonomyAnalysisResults:
    """
    Comprehensive analysis of fingerprints across misalignment types.

    Args:
        fingerprints_by_type: Dict mapping agent type to list of fingerprint dicts
        danger_levels: Optional dict mapping agent type to danger level (0-10)

    Returns:
        TaxonomyAnalysisResults with all analysis metrics
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        silhouette_score, adjusted_rand_score
    )
    from sklearn.model_selection import cross_val_predict, cross_val_score

    # Prepare data
    X, y, type_labels = prepare_fingerprint_data(fingerprints_by_type)

    if len(X) == 0:
        raise ValueError("No fingerprint data provided")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Multi-class classification
    print("  Running multi-class classification...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Cross-validation
    n_samples = len(y)
    cv_folds = min(5, n_samples // len(type_labels))  # Ensure enough samples per fold
    cv_folds = max(2, cv_folds)

    cv_scores = cross_val_score(clf, X_scaled, y, cv=cv_folds)
    y_pred = cross_val_predict(clf, X_scaled, y, cv=cv_folds)

    multiclass_accuracy = cv_scores.mean()
    conf_matrix = confusion_matrix(y, y_pred)
    class_report = classification_report(y, y_pred, target_names=type_labels, zero_division=0)

    # Per-class accuracy
    per_class_acc = {}
    for i, label in enumerate(type_labels):
        mask = y == i
        if mask.sum() > 0:
            per_class_acc[label] = (y_pred[mask] == y[mask]).mean()
        else:
            per_class_acc[label] = 0.0

    # 2. Clustering analysis
    print("  Running clustering analysis...")

    n_clusters = len(type_labels)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    if len(np.unique(cluster_labels)) > 1:
        silhouette = silhouette_score(X_scaled, cluster_labels)
    else:
        silhouette = 0.0
    ari = adjusted_rand_score(y, cluster_labels)

    # Cluster purity
    purity = compute_cluster_purity(y, cluster_labels)

    # 3. Feature analysis
    print("  Analyzing discriminative features...")

    clf.fit(X_scaled, y)
    feature_names = get_feature_names(fingerprints_by_type)

    discriminative_features = {}
    for i, label in enumerate(type_labels):
        # Features that best distinguish this type from others
        binary_y = (y == i).astype(int)
        if binary_y.sum() > 0 and (1 - binary_y).sum() > 0:
            binary_clf = RandomForestClassifier(n_estimators=50, random_state=42)
            binary_clf.fit(X_scaled, binary_y)

            importances = binary_clf.feature_importances_
            top_indices = np.argsort(importances)[-5:][::-1]
            discriminative_features[label] = [feature_names[idx] for idx in top_indices]
        else:
            discriminative_features[label] = []

    # Features that discriminate misaligned from aligned
    aligned_idx = type_labels.index("aligned") if "aligned" in type_labels else 0
    misaligned_mask = y != aligned_idx

    if misaligned_mask.sum() > 0 and (~misaligned_mask).sum() > 0:
        shared_clf = RandomForestClassifier(n_estimators=50, random_state=42)
        shared_clf.fit(X_scaled, misaligned_mask.astype(int))
        shared_importances = shared_clf.feature_importances_
        shared_top = np.argsort(shared_importances)[-5:][::-1]
        shared_features = [feature_names[idx] for idx in shared_top]
    else:
        shared_features = feature_names[:5] if feature_names else []

    # 4. Type similarity analysis
    print("  Computing type similarity matrix...")

    # Mean fingerprint per type
    type_means = {}
    for i, label in enumerate(type_labels):
        mask = y == i
        if mask.sum() > 0:
            type_means[label] = X_scaled[mask].mean(axis=0)
        else:
            type_means[label] = np.zeros(X_scaled.shape[1])

    # Pairwise distances
    n_types = len(type_labels)
    similarity_matrix = np.zeros((n_types, n_types))

    for i, label_i in enumerate(type_labels):
        for j, label_j in enumerate(type_labels):
            dist = np.linalg.norm(type_means[label_i] - type_means[label_j])
            similarity_matrix[i, j] = dist

    # 5. Danger level correlation
    danger_correlation = 0.0
    if danger_levels:
        print("  Computing danger correlation...")

        # Distance from aligned fingerprint vs danger level
        aligned_mean = type_means.get("aligned", X_scaled[y == 0].mean(axis=0) if (y == 0).sum() > 0 else np.zeros(X_scaled.shape[1]))

        distances = []
        dangers = []

        for label in type_labels:
            if label != "aligned" and label in danger_levels:
                dist = np.linalg.norm(type_means[label] - aligned_mean)
                distances.append(dist)
                dangers.append(danger_levels[label])

        if len(distances) > 2:
            danger_correlation, _ = stats.pearsonr(distances, dangers)
        elif len(distances) == 2:
            # Can't compute correlation with only 2 points
            danger_correlation = 0.0

    # 6. Embeddings for visualization
    print("  Computing embeddings...")

    # t-SNE
    perplexity = min(30, max(5, len(X) // 4))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_emb = tsne.fit_transform(X_scaled)

    # PCA
    pca = PCA(n_components=2)
    pca_emb = pca.fit_transform(X_scaled)

    return TaxonomyAnalysisResults(
        multiclass_accuracy=multiclass_accuracy,
        per_class_accuracy=per_class_acc,
        confusion_matrix=conf_matrix,
        classification_report=class_report,
        cluster_purity=purity,
        silhouette_score=silhouette,
        adjusted_rand_index=ari,
        discriminative_features=discriminative_features,
        shared_features=shared_features,
        type_similarity_matrix=similarity_matrix,
        danger_correlation=danger_correlation,
        tsne_embeddings=tsne_emb,
        pca_embeddings=pca_emb,
        labels=[type_labels[i] for i in y],
        type_labels=type_labels,
    )


def identify_misalignment_signatures(
    fingerprints_by_type: Dict[str, List[Dict]],
    aligned_type: str = "aligned"
) -> Dict[str, Dict]:
    """
    Identify characteristic signatures for each misalignment type.

    Returns dict mapping type to signature features that distinguish it.
    """

    signatures = {}

    # Get aligned baseline
    aligned_fps = fingerprints_by_type.get(aligned_type, [])
    if not aligned_fps:
        # If no aligned type, use first type as baseline
        aligned_type = list(fingerprints_by_type.keys())[0]
        aligned_fps = fingerprints_by_type[aligned_type]

    # Get numeric features
    feature_keys = [
        k for k in aligned_fps[0].keys()
        if isinstance(aligned_fps[0][k], (int, float)) and not isinstance(aligned_fps[0][k], bool)
    ]

    aligned_means = {k: np.mean([fp[k] for fp in aligned_fps]) for k in feature_keys}
    aligned_stds = {
        k: max(np.std([fp[k] for fp in aligned_fps]), 1e-6)
        for k in feature_keys
    }

    # For each misalignment type, find distinguishing features
    for agent_type, fps in fingerprints_by_type.items():
        if agent_type == aligned_type:
            continue

        type_means = {k: np.mean([fp[k] for fp in fps]) for k in feature_keys}

        # Z-score relative to aligned distribution
        z_scores = {}
        for feature in feature_keys:
            z = (type_means[feature] - aligned_means[feature]) / aligned_stds[feature]
            z_scores[feature] = z

        # Find most distinctive features (highest absolute z-score)
        sorted_features = sorted(z_scores.items(), key=lambda x: abs(x[1]), reverse=True)

        signatures[agent_type] = {
            "top_deviations": sorted_features[:5],
            "mean_z_score": np.mean([abs(z) for z in z_scores.values()]),
            "signature_vector": z_scores,
        }

    return signatures


def build_misalignment_grading_system(
    fingerprints_by_type: Dict[str, List[Dict]],
    danger_levels: Dict[str, int]
) -> Dict:
    """
    Build a grading system that maps fingerprint features to danger assessment.

    Returns trained model and feature weights for danger prediction.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    X, y_type, type_labels = prepare_fingerprint_data(fingerprints_by_type)

    if len(X) == 0:
        return {
            "model": None,
            "scaler": None,
            "cv_r2_scores": np.array([0.0]),
            "mean_r2": 0.0,
            "feature_importances": {},
            "top_danger_indicators": [],
        }

    # Create danger level labels
    y_danger = np.array([danger_levels.get(type_labels[t], 0) for t in y_type])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train regression model for danger prediction
    danger_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

    # Cross-validation
    n_samples = len(y_danger)
    cv_folds = min(5, n_samples // 2)
    cv_folds = max(2, cv_folds)

    cv_scores = cross_val_score(
        danger_model, X_scaled, y_danger,
        cv=cv_folds, scoring='r2'
    )

    danger_model.fit(X_scaled, y_danger)

    feature_names = get_feature_names(fingerprints_by_type)
    feature_importances = dict(zip(feature_names, danger_model.feature_importances_))

    return {
        "model": danger_model,
        "scaler": scaler,
        "cv_r2_scores": cv_scores,
        "mean_r2": cv_scores.mean(),
        "feature_importances": feature_importances,
        "top_danger_indicators": sorted(
            feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10],
    }


def generate_key_findings(
    analysis: TaxonomyAnalysisResults,
    signatures: Dict[str, Dict],
    grading: Dict,
    danger_levels: Dict[str, int]
) -> List[str]:
    """Generate key findings from analysis."""

    findings = []

    # Classification performance
    if analysis.multiclass_accuracy > 0.8:
        findings.append(
            f"Strong multi-class separation: {analysis.multiclass_accuracy:.0%} accuracy "
            f"distinguishing {len(analysis.per_class_accuracy)} misalignment types"
        )
    elif analysis.multiclass_accuracy > 0.6:
        findings.append(
            f"Moderate multi-class separation: {analysis.multiclass_accuracy:.0%} accuracy "
            f"- some types overlap"
        )
    else:
        findings.append(
            f"Weak multi-class separation: {analysis.multiclass_accuracy:.0%} accuracy "
            f"- fingerprints may not distinguish all types"
        )

    # Clustering
    if analysis.silhouette_score > 0.5:
        findings.append(
            f"Natural clustering by type (silhouette={analysis.silhouette_score:.2f}) "
            f"- types form distinct behavioral clusters"
        )

    # Danger correlation
    if abs(analysis.danger_correlation) > 0.7:
        findings.append(
            f"Strong danger-distance correlation (r={analysis.danger_correlation:.2f}) "
            f"- fingerprint distance from aligned predicts danger level"
        )
    elif abs(analysis.danger_correlation) > 0.4:
        findings.append(
            f"Moderate danger-distance correlation (r={analysis.danger_correlation:.2f}) "
            f"- some relationship between fingerprint and danger"
        )

    # Hardest to detect
    if analysis.per_class_accuracy:
        worst_type = min(analysis.per_class_accuracy.items(), key=lambda x: x[1])
        if worst_type[1] < 0.7:
            findings.append(
                f"Most evasive misalignment: {worst_type[0]} ({worst_type[1]:.0%} detection) "
                f"- may require specialized features"
            )

    # Shared indicators
    if analysis.shared_features:
        findings.append(
            f"Shared misalignment indicators: {', '.join(analysis.shared_features[:3])}"
        )

    # Grading viability
    if grading.get("mean_r2", 0) > 0.6:
        findings.append(
            f"Danger grading viable (R2={grading['mean_r2']:.2f}) "
            f"- can predict danger from fingerprint"
        )

    return findings
