"""Compute prediction errors from trajectories."""

from typing import List
import numpy as np
from utils.trajectory import Trajectory
from world_models.mlp_world_model import MLPWorldModel


def compute_prediction_errors(
    trajectory: Trajectory,
    world_model: MLPWorldModel
) -> List[float]:
    """
    Compute prediction error at each step of a trajectory.

    Args:
        trajectory: Trajectory to compute errors for
        world_model: Trained world model

    Returns:
        List of prediction errors (Euclidean distance)
    """
    errors = []

    for state, action, next_state in trajectory.to_transitions():
        error = world_model.compute_error(state, action, next_state)
        errors.append(error)

    return errors


def compute_errors_for_trajectories(
    trajectories: List[Trajectory],
    world_model: MLPWorldModel
) -> List[List[float]]:
    """
    Compute prediction errors for multiple trajectories.

    Args:
        trajectories: List of trajectories
        world_model: Trained world model

    Returns:
        List of error lists (one per trajectory)
    """
    all_errors = []
    for traj in trajectories:
        errors = compute_prediction_errors(traj, world_model)
        all_errors.append(errors)
    return all_errors
