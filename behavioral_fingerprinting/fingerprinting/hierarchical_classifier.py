"""
Hierarchical Misalignment Classifier

Two-stage classification approach:
1. Stage 1: Aligned vs Misaligned (binary classification)
2. Stage 2: Misalignment subtype (6-way classification)

Rationale:
- The aligned vs misaligned distinction is much clearer (expected >95% accuracy)
- Fine-grained misalignment classification is harder (sycophancy vs reward_hacking)
- Hierarchical approach reduces cascading errors
- Provides interpretability by separating the two classification problems

Expected Performance:
- Stage 1: >95% accuracy (aligned vs any misalignment)
- Stage 2: ~90% accuracy (misalignment subtype given misaligned)
- Overall: >92% accuracy (improvement from 87.7% baseline)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from typing import Dict, List, Tuple, Optional
import json


class HierarchicalMisalignmentClassifier:
    """
    Two-stage hierarchical classifier for misalignment detection.

    Stage 1: Binary classification (aligned vs misaligned)
    Stage 2: Multi-class classification (6 misalignment types)
    """

    def __init__(
        self,
        stage1_params: Optional[Dict] = None,
        stage2_params: Optional[Dict] = None,
        agent_types: Optional[List[str]] = None
    ):
        """
        Initialize hierarchical classifier.

        Args:
            stage1_params: Parameters for stage 1 classifier (binary)
            stage2_params: Parameters for stage 2 classifier (multi-class)
            agent_types: List of agent type names in order
        """
        # Default parameters for stage 1 (binary: aligned vs misaligned)
        default_stage1_params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }

        # Default parameters for stage 2 (multi-class: misalignment subtypes)
        default_stage2_params = {
            'n_estimators': 300,
            'max_depth': 15,
            'min_samples_split': 3,
            'min_samples_leaf': 1,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }

        self.stage1_params = stage1_params or default_stage1_params
        self.stage2_params = stage2_params or default_stage2_params

        self.stage1_classifier = None  # Binary: aligned vs misaligned
        self.stage2_classifier = None  # Multi-class: 6 misalignment types

        self.feature_names = None
        self.agent_types = agent_types or [
            'aligned',
            'adversarial',
            'resource_acquisition',
            'deceptive',
            'reward_hacking',
            'self_preservation',
            'sycophancy'
        ]

        # Mapping from agent type to class index
        self.agent_type_to_idx = {name: idx for idx, name in enumerate(self.agent_types)}
        self.idx_to_agent_type = {idx: name for name, idx in self.agent_type_to_idx.items()}

        # Stage 1 binary labels (0 = aligned, 1 = misaligned)
        self.stage1_labels = ['aligned', 'misaligned']

        # Stage 2 multi-class labels (only misaligned types)
        self.stage2_labels = [t for t in self.agent_types if t != 'aligned']

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Train both stages of the hierarchical classifier.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0=aligned, 1-6=misaligned types)
            feature_names: Optional list of feature names
        """
        if feature_names is not None:
            self.feature_names = feature_names

        # Stage 1: Binary classification (aligned vs misaligned)
        print("Training Stage 1: Aligned vs Misaligned (binary)")
        y_binary = (y > 0).astype(int)  # 0=aligned, 1=any misalignment

        self.stage1_classifier = RandomForestClassifier(**self.stage1_params)
        self.stage1_classifier.fit(X, y_binary)

        # Evaluate stage 1 with cross-validation
        cv_scores_stage1 = cross_val_score(
            self.stage1_classifier, X, y_binary, cv=5, scoring='accuracy'
        )
        print(f"  Stage 1 CV Accuracy: {cv_scores_stage1.mean():.3f} ± {cv_scores_stage1.std():.3f}")

        # Stage 2: Misalignment subtype (only on misaligned samples)
        print("\nTraining Stage 2: Misalignment Subtype (6-way)")
        misaligned_mask = y > 0
        X_misaligned = X[misaligned_mask]
        y_misaligned = y[misaligned_mask]

        if len(X_misaligned) == 0:
            raise ValueError("No misaligned samples found for stage 2 training")

        self.stage2_classifier = RandomForestClassifier(**self.stage2_params)
        self.stage2_classifier.fit(X_misaligned, y_misaligned)

        # Evaluate stage 2 with cross-validation
        cv_scores_stage2 = cross_val_score(
            self.stage2_classifier, X_misaligned, y_misaligned, cv=5, scoring='accuracy'
        )
        print(f"  Stage 2 CV Accuracy: {cv_scores_stage2.mean():.3f} ± {cv_scores_stage2.std():.3f}")

        # Estimate overall accuracy
        # P(correct) = P(aligned) * P(correct|aligned) + P(misaligned) * P(correct|misaligned)
        p_aligned = (y == 0).mean()
        p_misaligned = 1 - p_aligned
        expected_accuracy = (
            p_aligned * cv_scores_stage1.mean() +
            p_misaligned * cv_scores_stage1.mean() * cv_scores_stage2.mean()
        )
        print(f"\n  Expected Overall Accuracy: {expected_accuracy:.3f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Two-stage prediction.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted labels (0=aligned, 1-6=misaligned types)
        """
        if self.stage1_classifier is None or self.stage2_classifier is None:
            raise ValueError("Classifier not trained. Call fit() first.")

        # Stage 1: Predict aligned vs misaligned
        stage1_pred = self.stage1_classifier.predict(X)

        # Initialize final predictions (default to aligned)
        final_pred = np.zeros(len(X), dtype=int)

        # Stage 2: For predicted misaligned samples, classify subtype
        misaligned_mask = stage1_pred == 1

        if misaligned_mask.any():
            X_misaligned = X[misaligned_mask]
            stage2_pred = self.stage2_classifier.predict(X_misaligned)
            final_pred[misaligned_mask] = stage2_pred

        return final_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability distribution across all 7 classes.

        Uses the chain rule:
        P(class_i) = P(misaligned) * P(class_i | misaligned)  for i > 0
        P(aligned) = P(aligned)

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Probability matrix (n_samples, 7)
        """
        if self.stage1_classifier is None or self.stage2_classifier is None:
            raise ValueError("Classifier not trained. Call fit() first.")

        n_samples = len(X)
        n_classes = len(self.agent_types)

        # Stage 1 probabilities
        stage1_proba = self.stage1_classifier.predict_proba(X)
        p_aligned = stage1_proba[:, 0]  # P(aligned)
        p_misaligned = stage1_proba[:, 1]  # P(misaligned)

        # Stage 2 probabilities (conditional on misaligned)
        stage2_proba = self.stage2_classifier.predict_proba(X)

        # Combine probabilities using chain rule
        final_proba = np.zeros((n_samples, n_classes))
        final_proba[:, 0] = p_aligned  # P(aligned)

        # P(misalignment_type_i) = P(misaligned) * P(type_i | misaligned)
        # Note: stage2 classifier predicts indices 1-6 (misalignment types)
        # We need to map these to the correct positions in final_proba

        # Get the class mapping from stage2 classifier
        stage2_classes = self.stage2_classifier.classes_

        for i, cls in enumerate(stage2_classes):
            # cls is the original class index (1-6)
            final_proba[:, cls] = p_misaligned * stage2_proba[:, i]

        return final_proba

    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """
        Return feature importance for both stages.

        Returns:
            Dictionary with 'stage1_binary' and 'stage2_subtype' keys
        """
        if self.stage1_classifier is None or self.stage2_classifier is None:
            raise ValueError("Classifier not trained. Call fit() first.")

        if self.feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.stage1_classifier.feature_importances_))]
        else:
            feature_names = self.feature_names

        # Sort by importance
        stage1_importance = sorted(
            zip(feature_names, self.stage1_classifier.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )

        stage2_importance = sorted(
            zip(feature_names, self.stage2_classifier.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            'stage1_binary': dict(stage1_importance),
            'stage2_subtype': dict(stage2_importance)
        }

    def get_top_features(self, stage: int = 1, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Get top k most important features for a given stage.

        Args:
            stage: 1 for binary classification, 2 for subtype classification
            top_k: Number of top features to return

        Returns:
            List of (feature_name, importance) tuples
        """
        importance_dict = self.get_feature_importance()

        if stage == 1:
            items = list(importance_dict['stage1_binary'].items())
        elif stage == 2:
            items = list(importance_dict['stage2_subtype'].items())
        else:
            raise ValueError("stage must be 1 or 2")

        return items[:top_k]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate classifier on test data.

        Returns metrics for both stages and overall performance.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: True labels (0=aligned, 1-6=misaligned types)

        Returns:
            Dictionary with evaluation metrics
        """
        # Overall predictions
        y_pred = self.predict(X)
        overall_accuracy = (y_pred == y).mean()

        # Stage 1 evaluation
        y_binary = (y > 0).astype(int)
        stage1_pred = self.stage1_classifier.predict(X)
        stage1_accuracy = (stage1_pred == y_binary).mean()

        # Stage 2 evaluation (only on truly misaligned samples)
        misaligned_mask = y > 0
        if misaligned_mask.any():
            X_misaligned = X[misaligned_mask]
            y_misaligned = y[misaligned_mask]
            stage2_pred = self.stage2_classifier.predict(X_misaligned)
            stage2_accuracy = (stage2_pred == y_misaligned).mean()
        else:
            stage2_accuracy = 0.0

        # Per-class accuracy
        per_class_accuracy = {}
        for idx, agent_type in self.idx_to_agent_type.items():
            mask = y == idx
            if mask.any():
                per_class_accuracy[agent_type] = (y_pred[mask] == y[mask]).mean()
            else:
                per_class_accuracy[agent_type] = 0.0

        return {
            'overall_accuracy': overall_accuracy,
            'stage1_accuracy': stage1_accuracy,
            'stage2_accuracy': stage2_accuracy,
            'per_class_accuracy': per_class_accuracy
        }

    def save(self, filepath: str):
        """Save classifier to file."""
        import pickle

        state = {
            'stage1_classifier': self.stage1_classifier,
            'stage2_classifier': self.stage2_classifier,
            'feature_names': self.feature_names,
            'agent_types': self.agent_types,
            'stage1_params': self.stage1_params,
            'stage2_params': self.stage2_params,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath: str):
        """Load classifier from file."""
        import pickle

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        classifier = cls(
            stage1_params=state['stage1_params'],
            stage2_params=state['stage2_params'],
            agent_types=state['agent_types']
        )

        classifier.stage1_classifier = state['stage1_classifier']
        classifier.stage2_classifier = state['stage2_classifier']
        classifier.feature_names = state['feature_names']

        return classifier


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compare_flat_vs_hierarchical(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_splits: int = 5
) -> Dict:
    """
    Compare flat (single-stage) vs hierarchical classifier performance.

    Args:
        X: Feature matrix
        y: Labels
        feature_names: Optional feature names
        n_splits: Number of cross-validation splits

    Returns:
        Dictionary with comparison metrics
    """
    from sklearn.metrics import classification_report, accuracy_score

    print("=" * 80)
    print("COMPARING FLAT VS HIERARCHICAL CLASSIFIERS")
    print("=" * 80)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    flat_scores = []
    hierarchical_scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        print(f"\nFold {fold + 1}/{n_splits}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Flat classifier
        flat_clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=3,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        flat_clf.fit(X_train, y_train)
        flat_pred = flat_clf.predict(X_test)
        flat_acc = accuracy_score(y_test, flat_pred)
        flat_scores.append(flat_acc)
        print(f"  Flat Accuracy: {flat_acc:.3f}")

        # Hierarchical classifier
        hier_clf = HierarchicalMisalignmentClassifier()
        hier_clf.fit(X_train, y_train, feature_names=feature_names)
        hier_pred = hier_clf.predict(X_test)
        hier_acc = accuracy_score(y_test, hier_pred)
        hierarchical_scores.append(hier_acc)
        print(f"  Hierarchical Accuracy: {hier_acc:.3f}")

    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print(f"Flat Classifier:        {np.mean(flat_scores):.3f} ± {np.std(flat_scores):.3f}")
    print(f"Hierarchical Classifier: {np.mean(hierarchical_scores):.3f} ± {np.std(hierarchical_scores):.3f}")
    print(f"Improvement:            {np.mean(hierarchical_scores) - np.mean(flat_scores):.3f}")

    return {
        'flat_mean': np.mean(flat_scores),
        'flat_std': np.std(flat_scores),
        'hierarchical_mean': np.mean(hierarchical_scores),
        'hierarchical_std': np.std(hierarchical_scores),
        'improvement': np.mean(hierarchical_scores) - np.mean(flat_scores)
    }
