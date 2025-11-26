"""Statistical analysis of behavioral fingerprints."""

from typing import List, Dict, Tuple
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def compare_fingerprints(
    aligned_fingerprints: List[Dict],
    misaligned_fingerprints: List[Dict]
) -> Dict:
    """
    Compare two sets of fingerprints statistically.

    Args:
        aligned_fingerprints: Fingerprints from aligned agents
        misaligned_fingerprints: Fingerprints from misaligned agents

    Returns:
        Dictionary with statistical comparison results
    """
    results = {}

    # Get feature names
    feature_names = list(aligned_fingerprints[0].keys())

    # Per-feature comparisons
    feature_comparisons = {}

    for feature in feature_names:
        aligned_vals = [fp[feature] for fp in aligned_fingerprints]
        misaligned_vals = [fp[feature] for fp in misaligned_fingerprints]

        # T-test
        t_stat, p_value = stats.ttest_ind(aligned_vals, misaligned_vals)

        # Effect size (Cohen's d)
        mean_diff = np.mean(misaligned_vals) - np.mean(aligned_vals)
        pooled_std = np.sqrt((np.var(aligned_vals) + np.var(misaligned_vals)) / 2)
        cohens_d = mean_diff / (pooled_std + 1e-10)

        feature_comparisons[feature] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'aligned_mean': float(np.mean(aligned_vals)),
            'aligned_std': float(np.std(aligned_vals)),
            'misaligned_mean': float(np.mean(misaligned_vals)),
            'misaligned_std': float(np.std(misaligned_vals)),
        }

    results['feature_comparisons'] = feature_comparisons

    # Overall separability
    aligned_mean_error = np.mean([fp['mean_error'] for fp in aligned_fingerprints])
    misaligned_mean_error = np.mean([fp['mean_error'] for fp in misaligned_fingerprints])

    results['aligned_mean_error'] = float(aligned_mean_error)
    results['misaligned_mean_error'] = float(misaligned_mean_error)
    results['error_ratio'] = float(misaligned_mean_error / (aligned_mean_error + 1e-10))

    # Rank features by effect size
    ranked_features = sorted(
        feature_comparisons.items(),
        key=lambda x: abs(x[1]['cohens_d']),
        reverse=True
    )
    results['most_discriminative_features'] = [
        (name, comp['cohens_d']) for name, comp in ranked_features[:5]
    ]

    return results


def train_classifier(
    fingerprints: List[Dict],
    labels: List[int],
    cv_folds: int = 5
) -> Tuple[LogisticRegression, Dict]:
    """
    Train classifier on fingerprints.

    Args:
        fingerprints: List of fingerprint dictionaries
        labels: List of labels (0 = aligned, 1 = misaligned)
        cv_folds: Number of cross-validation folds

    Returns:
        (trained_classifier, results_dict)
    """
    # Convert to numpy array
    feature_names = list(fingerprints[0].keys())
    X = np.array([[fp[feat] for feat in feature_names] for fp in fingerprints])
    y = np.array(labels)

    # Replace NaN and inf values with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train classifier
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_scaled, y)

    # Cross-validation
    cv_scores = cross_val_score(clf, X_scaled, y, cv=cv_folds)

    # Feature importances (coefficients)
    feature_importances = dict(zip(feature_names, np.abs(clf.coef_[0])))
    ranked_importances = sorted(
        feature_importances.items(),
        key=lambda x: x[1],
        reverse=True
    )

    results = {
        'cv_accuracy_mean': float(np.mean(cv_scores)),
        'cv_accuracy_std': float(np.std(cv_scores)),
        'cv_scores': [float(s) for s in cv_scores],
        'feature_importances': feature_importances,
        'top_features': ranked_importances[:5],
    }

    return clf, results
