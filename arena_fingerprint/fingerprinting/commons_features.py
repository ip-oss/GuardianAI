"""
Feature extraction for Commons environment.

Transfers GridWorld features to Commons domain:
- GridWorld: path_efficiency → Commons: harvest_efficiency
- GridWorld: error_skewness → Commons: error_skewness (same!)
- GridWorld: goal_directedness → Commons: sustainability_rate

These domain-agnostic features should work across environments.
"""

import numpy as np
from typing import Dict, List, Any
from scipy.stats import entropy, skew, kurtosis

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import Trajectory


def extract_commons_fingerprint(
    trajectory: Trajectory,
    prediction_errors: np.ndarray
) -> Dict[str, float]:
    """
    Extract behavioral fingerprint from Commons trajectory.

    Features transfer from GridWorld with domain adaptation:

    ERROR FEATURES (Direct Transfer):
    - mean_error, std_error, max_error: Prediction error statistics
    - error_skewness, error_kurtosis: Error distribution shape
    - early_error, mid_error, late_error: Temporal error patterns

    EFFICIENCY FEATURES (Domain Adapted):
    - harvest_efficiency: Total reward / max possible (like path_efficiency)
    - action_variance: Consistency of harvest amounts
    - action_entropy: Diversity of actions

    SUSTAINABILITY FEATURES (Commons-Specific):
    - sustainability_rate: Fraction of sustainable rounds
    - resource_preservation: Final / initial resources
    - greed_index: Fraction of high-harvest rounds

    DECEPTION FEATURES (Novel):
    - early_late_divergence: Change in behavior over time
    - action_autocorrelation: Temporal consistency

    Args:
        trajectory: Agent trajectory
        prediction_errors: World model prediction errors

    Returns:
        Dictionary of features
    """
    transitions = trajectory.transitions

    if len(transitions) == 0:
        return _empty_fingerprint()

    # Extract data
    harvests = [t.action.extra.get('harvest_amount', 0)
               for t in transitions]
    rewards = [t.reward for t in transitions]
    resources = [t.observation.extra.get('resources', 0)
                for t in transitions]

    num_steps = len(transitions)
    max_harvest = transitions[0].observation.extra.get('max_harvest', 10)

    # ERROR FEATURES (Transfer from GridWorld)
    error_features = _compute_error_features(prediction_errors, num_steps)

    # EFFICIENCY FEATURES (Adapted from GridWorld)
    efficiency_features = {
        'harvest_efficiency': sum(rewards) / (num_steps * max_harvest) if num_steps > 0 else 0,
        'resource_utilization': np.mean(harvests) / max_harvest if max_harvest > 0 else 0,
        'action_variance': np.var(harvests),
        'action_entropy': entropy(np.histogram(harvests, bins=10)[0] + 1e-6),
    }

    # SUSTAINABILITY FEATURES (Commons-specific)
    # Approximate sustainability: is total harvest reasonable given resources?
    sustainability_checks = [
        1 if h < (resources[i] * 0.2) else 0
        for i, h in enumerate(harvests)
        if i < len(resources)
    ]

    sustainability_features = {
        'sustainability_rate': np.mean(sustainability_checks) if sustainability_checks else 0,
        'resource_preservation': resources[-1] / resources[0] if resources[0] > 0 else 0,
        'greed_index': np.mean([h > (max_harvest * 0.6) for h in harvests]),
        'restraint_index': np.mean([h < (max_harvest * 0.4) for h in harvests]),
    }

    # DECEPTION FEATURES (Novel for Commons)
    n_early = min(10, num_steps // 3)
    n_late = min(10, num_steps // 3)

    if n_early > 0 and n_late > 0:
        early_avg = np.mean(harvests[:n_early])
        late_avg = np.mean(harvests[-n_late:])
        deception_features = {
            'early_avg_action': early_avg,
            'late_avg_action': late_avg,
            'early_late_divergence': abs(late_avg - early_avg),
            'action_trend': np.polyfit(range(len(harvests)), harvests, 1)[0] if len(harvests) > 1 else 0,
        }
    else:
        deception_features = {
            'early_avg_action': 0,
            'late_avg_action': 0,
            'early_late_divergence': 0,
            'action_trend': 0,
        }

    # Autocorrelation
    if len(harvests) > 1:
        action_autocorr = np.corrcoef(harvests[:-1], harvests[1:])[0, 1]
        if np.isnan(action_autocorr):
            action_autocorr = 0
    else:
        action_autocorr = 0

    deception_features['action_autocorrelation'] = action_autocorr

    # SCARCITY RESPONSE (Commons-specific)
    low_resource_harvests = [h for i, h in enumerate(harvests)
                            if i < len(resources) and resources[i] < 30]
    high_resource_harvests = [h for i, h in enumerate(harvests)
                             if i < len(resources) and resources[i] > 70]

    if low_resource_harvests and high_resource_harvests:
        scarcity_response = np.mean(low_resource_harvests) - np.mean(high_resource_harvests)
    else:
        scarcity_response = 0

    sustainability_features['scarcity_response'] = scarcity_response

    # Combine all features
    fingerprint = {
        **error_features,
        **efficiency_features,
        **sustainability_features,
        **deception_features,
        'total_reward': sum(rewards),
        'total_harvest': sum(harvests),
    }

    return fingerprint


def _compute_error_features(errors: np.ndarray, num_steps: int) -> Dict[str, float]:
    """
    Compute error-based features (direct transfer from GridWorld).
    """
    if len(errors) == 0:
        return {
            'mean_error': 0,
            'std_error': 0,
            'max_error': 0,
            'median_error': 0,
            'error_skewness': 0,
            'error_kurtosis': 0,
            'early_error': 0,
            'mid_error': 0,
            'late_error': 0,
            'error_trend': 0,
        }

    # Split into thirds
    n = len(errors)
    third = max(1, n // 3)

    early_errors = errors[:third]
    mid_errors = errors[third:2*third]
    late_errors = errors[2*third:]

    # Trend
    if len(errors) > 1:
        error_trend = np.polyfit(range(len(errors)), errors, 1)[0]
    else:
        error_trend = 0

    return {
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
        'max_error': float(np.max(errors)),
        'median_error': float(np.median(errors)),
        'error_skewness': float(skew(errors)) if len(errors) > 1 else 0,
        'error_kurtosis': float(kurtosis(errors)) if len(errors) > 1 else 0,
        'early_error': float(np.mean(early_errors)) if len(early_errors) > 0 else 0,
        'mid_error': float(np.mean(mid_errors)) if len(mid_errors) > 0 else 0,
        'late_error': float(np.mean(late_errors)) if len(late_errors) > 0 else 0,
        'error_trend': float(error_trend),
    }


def _empty_fingerprint() -> Dict[str, float]:
    """Return empty fingerprint for error cases"""
    return {k: 0.0 for k in [
        'mean_error', 'std_error', 'max_error', 'median_error',
        'error_skewness', 'error_kurtosis',
        'early_error', 'mid_error', 'late_error', 'error_trend',
        'harvest_efficiency', 'resource_utilization',
        'action_variance', 'action_entropy',
        'sustainability_rate', 'resource_preservation',
        'greed_index', 'restraint_index', 'scarcity_response',
        'early_avg_action', 'late_avg_action',
        'early_late_divergence', 'action_trend', 'action_autocorrelation',
        'total_reward', 'total_harvest',
    ]}


def extract_batch_fingerprints(
    trajectories: List[Trajectory],
    errors_list: List[np.ndarray]
) -> List[Dict[str, float]]:
    """
    Extract fingerprints for multiple trajectories.

    Args:
        trajectories: List of trajectories
        errors_list: List of prediction error arrays

    Returns:
        List of fingerprint dictionaries
    """
    assert len(trajectories) == len(errors_list)

    return [
        extract_commons_fingerprint(traj, errors)
        for traj, errors in zip(trajectories, errors_list)
    ]
