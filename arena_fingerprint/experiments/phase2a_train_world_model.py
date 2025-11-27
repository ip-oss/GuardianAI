"""
Phase 2A: Train World Model on Cooperative-Only Data

This script:
1. Loads training trajectories from Phase 1 (all cooperative agents)
2. Extracts features for world model training
3. Trains an MLP regressor to predict harvest amounts
4. Saves the trained model for Phase 2D evaluation

Features:
- Current resources
- Round number
- Agent's score
- Recent harvest history (last 3 rounds)
- Other agents' average harvest

Target:
- Agent's harvest amount

Model: MLP Regressor (sklearn) - simple, fast, interpretable
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import json
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def extract_features_from_trajectories(trajectories):
    """
    Extract training data from trajectories.

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (harvest amounts)
        feature_names: List of feature names
    """
    X = []
    y = []

    for trajectory in trajectories:
        # Track agent's recent harvests
        recent_harvests = []

        for t_idx, transition in enumerate(trajectory.transitions):
            obs = transition.observation
            action = transition.action

            # Extract features
            resources = obs.extra.get('resources', 0)
            my_score = obs.extra.get('my_score', 0)
            round_num = obs.extra.get('round', 0)
            max_rounds = obs.extra.get('max_rounds', 100)

            # Get other agents' harvests from this transition
            all_actions = action.extra.get('all_actions', {})
            other_harvests = [
                a for aid, a in all_actions.items()
                if aid != transition.observation.agent_id
            ]
            avg_other_harvest = np.mean(other_harvests) if other_harvests else 0

            # Recent harvest features (last 3)
            h1 = recent_harvests[-1] if len(recent_harvests) >= 1 else 0
            h2 = recent_harvests[-2] if len(recent_harvests) >= 2 else 0
            h3 = recent_harvests[-3] if len(recent_harvests) >= 3 else 0

            # Build feature vector
            features = [
                resources,                      # Current resources
                round_num,                      # Round number
                round_num / max_rounds,         # Progress (0-1)
                my_score,                       # Agent's score
                avg_other_harvest,              # Others' average harvest
                h1, h2, h3,                     # Recent harvest history
            ]

            # Target: actual harvest amount
            harvest = action.extra.get('harvest_amount', 0)

            X.append(features)
            y.append(harvest)

            # Update recent harvests
            recent_harvests.append(harvest)
            if len(recent_harvests) > 5:
                recent_harvests.pop(0)

    feature_names = [
        'resources',
        'round_num',
        'progress',
        'my_score',
        'avg_other_harvest',
        'harvest_t-1',
        'harvest_t-2',
        'harvest_t-3',
    ]

    return np.array(X), np.array(y), feature_names


def train_world_model(X_train, y_train, X_test, y_test):
    """
    Train MLP regressor as world model.

    Returns:
        model: Trained model
        scaler: Feature scaler
        metrics: Dictionary of performance metrics
    """
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train MLP
    print("\nTraining MLP World Model...")
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=True
    )

    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    metrics = {
        'train_mse': float(train_mse),
        'test_mse': float(test_mse),
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
    }

    print(f"\nWorld Model Performance:")
    print(f"  Train MSE: {train_mse:.3f}, R²: {train_r2:.3f}")
    print(f"  Test MSE:  {test_mse:.3f}, R²: {test_r2:.3f}")

    return model, scaler, metrics


def plot_predictions(y_true, y_pred, split_name, save_path):
    """Plot predicted vs actual harvest amounts."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
    plt.plot([0, 10], [0, 10], 'r--', label='Perfect prediction')
    plt.xlabel('Actual Harvest')
    plt.ylabel('Predicted Harvest')
    plt.title(f'World Model Predictions ({split_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved plot: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to Phase 1 results directory')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("=" * 70)
    print("PHASE 2A: TRAIN WORLD MODEL")
    print("=" * 70)
    print(f"Loading data from: {data_dir}")

    # Load training and test trajectories
    train_path = data_dir / "train_trajectories.pkl"
    test_path = data_dir / "test_trajectories.pkl"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")

    with open(train_path, 'rb') as f:
        train_trajectories = pickle.load(f)

    with open(test_path, 'rb') as f:
        test_trajectories = pickle.load(f)

    print(f"Loaded {len(train_trajectories)} training trajectories")
    print(f"Loaded {len(test_trajectories)} test trajectories")

    # Extract features
    print("\nExtracting features...")
    X_train, y_train, feature_names = extract_features_from_trajectories(train_trajectories)
    X_test, y_test, _ = extract_features_from_trajectories(test_trajectories)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {feature_names}")

    # Train model
    model, scaler, metrics = train_world_model(X_train, y_train, X_test, y_test)

    # Create output directory
    output_dir = Path("results/exp2_adversarial_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "world_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'metrics': metrics
        }, f)
    print(f"\nSaved world model: {model_path}")

    # Save metrics
    metrics_path = output_dir / "world_model_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    # Plot predictions
    y_train_pred = model.predict(scaler.transform(X_train))
    y_test_pred = model.predict(scaler.transform(X_test))

    plot_predictions(y_train, y_train_pred, 'Training',
                    output_dir / "predictions_train.png")
    plot_predictions(y_test, y_test_pred, 'Test',
                    output_dir / "predictions_test.png")

    print("\n" + "=" * 70)
    print("WORLD MODEL TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")
    print("\nNext step: Run Phase 2B to create adversarial agent prompts")


if __name__ == "__main__":
    main()
