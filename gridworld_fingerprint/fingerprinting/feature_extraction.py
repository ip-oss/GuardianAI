"""Extract behavioral fingerprint features from prediction errors."""

from typing import Dict, List
import numpy as np
from scipy import stats
from utils.trajectory import Trajectory


def extract_fingerprint(
    trajectory: Trajectory,
    prediction_errors: List[float],
    grid_size: int
) -> Dict[str, float]:
    """
    Extract behavioral fingerprint features from prediction errors.

    Args:
        trajectory: Trajectory object
        prediction_errors: List of prediction errors for each step
        grid_size: Size of the grid

    Returns:
        Dictionary of fingerprint features
    """
    errors = np.array(prediction_errors)

    fingerprint = {}

    # Global statistics
    fingerprint['mean_error'] = float(np.mean(errors))
    fingerprint['std_error'] = float(np.std(errors))
    fingerprint['max_error'] = float(np.max(errors))
    fingerprint['median_error'] = float(np.median(errors))

    if len(errors) > 2 and np.std(errors) > 1e-10:
        # Only compute if there's variance
        skew_val = stats.skew(errors)
        kurt_val = stats.kurtosis(errors)
        fingerprint['error_skewness'] = float(skew_val) if np.isfinite(skew_val) else 0.0
        fingerprint['error_kurtosis'] = float(kurt_val) if np.isfinite(kurt_val) else 0.0
    else:
        fingerprint['error_skewness'] = 0.0
        fingerprint['error_kurtosis'] = 0.0

    # Temporal patterns
    n = len(errors)
    third = max(1, n // 3)

    fingerprint['early_error'] = float(np.mean(errors[:third]))
    fingerprint['mid_error'] = float(np.mean(errors[third:2*third]))
    fingerprint['late_error'] = float(np.mean(errors[2*third:]))

    # Error trend (linear fit slope)
    if n > 1:
        x = np.arange(n)
        slope, _ = np.polyfit(x, errors, 1)
        fingerprint['error_trend'] = float(slope)
    else:
        fingerprint['error_trend'] = 0.0

    # Trajectory characteristics
    fingerprint['path_length'] = trajectory.path_length
    fingerprint['total_reward'] = trajectory.total_reward

    # Optimal path length (Manhattan distance)
    optimal_length = (grid_size - 1) + (grid_size - 1)
    fingerprint['path_efficiency'] = optimal_length / max(1, trajectory.path_length)

    # Action diversity
    actions = trajectory.get_actions_array()
    if len(actions) > 0:
        action_counts = np.bincount(actions, minlength=4)
        action_probs = action_counts / len(actions)
        # Entropy
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
        fingerprint['action_entropy'] = float(entropy)
    else:
        fingerprint['action_entropy'] = 0.0

    return fingerprint


def extract_fingerprints_batch(
    trajectories: List[Trajectory],
    errors_list: List[List[float]],
    grid_size: int
) -> List[Dict[str, float]]:
    """
    Extract fingerprints for multiple trajectories.

    Args:
        trajectories: List of trajectories
        errors_list: List of error lists (one per trajectory)
        grid_size: Size of the grid

    Returns:
        List of fingerprint dictionaries
    """
    fingerprints = []
    for traj, errors in zip(trajectories, errors_list):
        fp = extract_fingerprint(traj, errors, grid_size)
        fingerprints.append(fp)
    return fingerprints
