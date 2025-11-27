"""
Phase 2D: Fingerprint Analysis and Classification

This script:
1. Loads trained world model (from Phase 2A)
2. Runs adversarial trajectories through the model
3. Computes prediction errors for each agent
4. Extracts fingerprint features:
   - mean_error
   - std_error
   - max_error
   - min_error
   - error_trend
   - positive_error_rate
   - negative_error_rate
5. Builds classifier to predict agent type from fingerprint
6. Evaluates performance and generates analysis report

Expected patterns:
- Cooperative: ~0 error, stable
- Greedy: >> +3 error, consistently high
- Deceptive: ~+1 error, high variance, reasoning mismatch
- Free-Rider: << -1 error, consistently low
- Vengeful: Variable, spikes after defection
- Random: ~0 mean, very high variance
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import json
import numpy as np
import matplotlib.pyplot as plt

# Try to import seaborn, but make it optional
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available, using matplotlib only")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress


def extract_features_for_prediction(trajectory, scaler, feature_names):
    """
    Extract feature vectors for world model prediction.

    Returns:
        X: Feature matrix (n_steps, n_features)
    """
    X = []
    recent_harvests = []

    for transition in trajectory.transitions:
        obs = transition.observation
        action = transition.action

        resources = obs.extra.get('resources', 0)
        my_score = obs.extra.get('my_score', 0)
        round_num = obs.extra.get('round', 0)
        max_rounds = obs.extra.get('max_rounds', 100)

        all_actions = action.extra.get('all_actions', {})
        other_harvests = [
            a for aid, a in all_actions.items()
            if aid != transition.observation.agent_id
        ]
        avg_other_harvest = np.mean(other_harvests) if other_harvests else 0

        h1 = recent_harvests[-1] if len(recent_harvests) >= 1 else 0
        h2 = recent_harvests[-2] if len(recent_harvests) >= 2 else 0
        h3 = recent_harvests[-3] if len(recent_harvests) >= 3 else 0

        features = [
            resources,
            round_num,
            round_num / max_rounds,
            my_score,
            avg_other_harvest,
            h1, h2, h3,
        ]

        X.append(features)

        harvest = action.extra.get('harvest_amount', 0)
        recent_harvests.append(harvest)
        if len(recent_harvests) > 5:
            recent_harvests.pop(0)

    return np.array(X)


def compute_prediction_errors(trajectories, model, scaler):
    """
    Compute prediction errors for each trajectory.

    Returns:
        Dictionary mapping unique_agent_id -> {
            'errors': array of errors,
            'actual': array of actual harvests,
            'predicted': array of predicted harvests,
            'agent_type': agent type label
        }
    """
    results = {}

    for traj_idx, trajectory in enumerate(trajectories):
        agent_id = trajectory.agent_id
        agent_type = trajectory.metadata.get('agent_type', 'unknown')
        game_idx = trajectory.metadata.get('game_idx', -1)

        # Create unique identifier: game_idx_agent_id
        unique_id = f"game{game_idx}_{agent_id}"

        # Extract features
        X = extract_features_for_prediction(trajectory, scaler, None)
        X_scaled = scaler.transform(X)

        # Predict
        y_pred = model.predict(X_scaled)

        # Get actual harvests
        y_actual = np.array([
            t.action.extra.get('harvest_amount', 0)
            for t in trajectory.transitions
        ])

        # Compute errors (actual - predicted)
        errors = y_actual - y_pred

        results[unique_id] = {
            'errors': errors,
            'actual': y_actual,
            'predicted': y_pred,
            'agent_type': agent_type,
            'agent_id': agent_id,
            'game_idx': game_idx
        }

    return results


def extract_fingerprint_features(errors):
    """
    Extract fingerprint features from prediction errors.

    Features:
    - mean_error: Average error
    - std_error: Standard deviation of errors
    - max_error: Maximum error
    - min_error: Minimum error
    - error_trend: Slope of error over time (linear regression)
    - positive_error_rate: Fraction of positive errors
    - negative_error_rate: Fraction of negative errors
    - abs_mean_error: Mean of absolute errors

    Returns:
        Feature vector
    """
    if len(errors) == 0:
        return np.zeros(8)

    mean_err = np.mean(errors)
    std_err = np.std(errors)
    max_err = np.max(errors)
    min_err = np.min(errors)

    # Error trend (slope over time)
    if len(errors) > 1:
        x = np.arange(len(errors))
        slope, _, _, _, _ = linregress(x, errors)
        error_trend = slope
    else:
        error_trend = 0

    positive_rate = np.mean(errors > 0)
    negative_rate = np.mean(errors < 0)
    abs_mean = np.mean(np.abs(errors))

    return np.array([
        mean_err,
        std_err,
        max_err,
        min_err,
        error_trend,
        positive_rate,
        negative_rate,
        abs_mean
    ])


def build_fingerprint_dataset(prediction_results):
    """
    Build dataset for classification.

    Returns:
        X: Feature matrix (n_agents, n_fingerprint_features)
        y: Agent type labels
        agent_ids: List of agent IDs
    """
    X = []
    y = []
    agent_ids = []

    for agent_id, result in prediction_results.items():
        fingerprint = extract_fingerprint_features(result['errors'])
        X.append(fingerprint)
        y.append(result['agent_type'])
        agent_ids.append(agent_id)

    return np.array(X), np.array(y), agent_ids


def train_classifier(X, y):
    """
    Train Random Forest classifier on fingerprints.

    Returns:
        model: Trained classifier
        scaler: Feature scaler
        metrics: Performance metrics
    """
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Random Forest
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )

    # Cross-validation (handle small datasets)
    # Need at least 2 samples per class for stratified k-fold
    min_class_count = min(np.bincount([list(set(y)).index(label) for label in y]))
    n_splits = min(5, min_class_count, len(X) // 2)
    n_splits = max(2, n_splits)  # At least 2 splits

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')

    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Train on full dataset
    model.fit(X_scaled, y)

    # Predictions
    y_pred = model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)

    print(f"Training accuracy: {accuracy:.3f}")

    metrics = {
        'cv_accuracy_mean': float(cv_scores.mean()),
        'cv_accuracy_std': float(cv_scores.std()),
        'train_accuracy': float(accuracy),
        'n_samples': len(X),
        'n_features': X.shape[1]
    }

    return model, scaler, metrics


def plot_error_distributions(prediction_results, save_path):
    """Plot error distributions by agent type."""
    agent_types = set(r['agent_type'] for r in prediction_results.values())

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, agent_type in enumerate(sorted(agent_types)):
        if i >= len(axes):
            break

        # Collect all errors for this type
        all_errors = []
        for result in prediction_results.values():
            if result['agent_type'] == agent_type:
                all_errors.extend(result['errors'])

        # Plot histogram
        axes[i].hist(all_errors, bins=30, alpha=0.7, edgecolor='black')
        axes[i].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
        axes[i].set_xlabel('Prediction Error')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'{agent_type.capitalize()}\n(mean={np.mean(all_errors):.2f}, std={np.std(all_errors):.2f})')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(agent_types), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved error distributions: {save_path}")


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))

    if HAS_SEABORN:
        # Use seaborn for prettier heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
    else:
        # Fallback to matplotlib imshow
        plt.imshow(cm, cmap='Blues', interpolation='nearest')
        plt.colorbar()

        # Add annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, str(cm[i, j]),
                        ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')

        # Set labels
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.yticks(range(len(labels)), labels)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Agent Type Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix: {save_path}")


def plot_fingerprint_features(X, y, feature_names, save_path):
    """Plot fingerprint feature distributions by agent type."""
    unique_types = sorted(set(y))
    n_features = len(feature_names)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, feature_name in enumerate(feature_names):
        if i >= len(axes):
            break

        for agent_type in unique_types:
            mask = y == agent_type
            values = X[mask, i]
            axes[i].hist(values, alpha=0.5, label=agent_type, bins=15)

        axes[i].set_xlabel(feature_name)
        axes[i].set_ylabel('Count')
        axes[i].set_title(feature_name.replace('_', ' ').title())
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved fingerprint features: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str,
                       default='results/exp2_adversarial_test',
                       help='Path to Phase 2C results directory')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    print("=" * 70)
    print("PHASE 2D: FINGERPRINT ANALYSIS & CLASSIFICATION")
    print("=" * 70)

    # Load world model
    print("\nLoading world model...")
    model_path = results_dir / "world_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"World model not found: {model_path}\n"
            f"Run phase2a_train_world_model.py first"
        )

    with open(model_path, 'rb') as f:
        wm_data = pickle.load(f)
        world_model = wm_data['model']
        wm_scaler = wm_data['scaler']
        feature_names = wm_data['feature_names']

    print(f"  Loaded world model from: {model_path}")

    # Load adversarial trajectories from all game files
    print("\nLoading adversarial trajectories...")

    # Find all game trajectory files
    game_files = sorted(results_dir.glob("adversarial_trajectories_game*.pkl"))

    if not game_files:
        raise FileNotFoundError(
            f"No game trajectory files found in: {results_dir}\n"
            f"Run phase2c_adversarial_games.py first"
        )

    print(f"  Found {len(game_files)} game files")

    # Load all trajectories
    trajectories = []
    for game_file in game_files:
        with open(game_file, 'rb') as f:
            game_trajectories = pickle.load(f)
            # game_trajectories is a dict of {agent_id: Trajectory}
            trajectories.extend(game_trajectories.values())

    print(f"  Loaded {len(trajectories)} trajectories from {len(game_files)} games")

    # Compute prediction errors
    print("\nComputing prediction errors...")
    prediction_results = compute_prediction_errors(trajectories, world_model, wm_scaler)

    print(f"  Computed errors for {len(prediction_results)} agents")

    # Build fingerprint dataset
    print("\nExtracting fingerprints...")
    X, y, agent_ids = build_fingerprint_dataset(prediction_results)

    fingerprint_feature_names = [
        'mean_error',
        'std_error',
        'max_error',
        'min_error',
        'error_trend',
        'positive_error_rate',
        'negative_error_rate',
        'abs_mean_error'
    ]

    print(f"  Extracted {len(X)} fingerprints with {X.shape[1]} features")
    print(f"  Agent type distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Train classifier
    classifier, fp_scaler, metrics = train_classifier(X, y)

    # Generate predictions
    X_scaled = fp_scaler.transform(X)
    y_pred = classifier.predict(X_scaled)

    # Classification report
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(y, y_pred, zero_division=0))

    # Save results
    print("\nSaving results...")

    # Save fingerprints
    fingerprints_path = results_dir / "fingerprints.json"
    fingerprints_data = {
        unique_id: {
            'agent_type': prediction_results[unique_id]['agent_type'],
            'agent_id': prediction_results[unique_id]['agent_id'],
            'game_idx': prediction_results[unique_id]['game_idx'],
            'fingerprint': dict(zip(fingerprint_feature_names,
                                   X[i].tolist())),
            'mean_error': float(np.mean(prediction_results[unique_id]['errors'])),
            'std_error': float(np.std(prediction_results[unique_id]['errors'])),
        }
        for i, unique_id in enumerate(agent_ids)
    }

    with open(fingerprints_path, 'w') as f:
        json.dump(fingerprints_data, f, indent=2)
    print(f"  Saved fingerprints: {fingerprints_path}")

    # Save classification results
    classification_results = {
        'metrics': metrics,
        'classification_report': classification_report(y, y_pred, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
        'class_labels': sorted(set(y)),
    }

    results_path = results_dir / "classification_results.json"
    with open(results_path, 'w') as f:
        json.dump(classification_results, f, indent=2)
    print(f"  Saved classification results: {results_path}")

    # Save classifier
    classifier_path = results_dir / "fingerprint_classifier.pkl"
    with open(classifier_path, 'wb') as f:
        pickle.dump({
            'model': classifier,
            'scaler': fp_scaler,
            'feature_names': fingerprint_feature_names
        }, f)
    print(f"  Saved classifier: {classifier_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_error_distributions(prediction_results,
                            results_dir / "error_distributions.png")
    plot_confusion_matrix(y, y_pred, sorted(set(y)),
                         results_dir / "confusion_matrix.png")
    plot_fingerprint_features(X, y, fingerprint_feature_names,
                             results_dir / "fingerprint_features.png")

    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 2D COMPLETE!")
    print("=" * 70)
    print(f"Classification accuracy: {metrics['train_accuracy']:.1%}")
    print(f"Cross-validation accuracy: {metrics['cv_accuracy_mean']:.1%} ± {metrics['cv_accuracy_std']:.1%}")
    print(f"\nResults saved to: {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
