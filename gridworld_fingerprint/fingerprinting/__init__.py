"""Behavioral fingerprinting from prediction errors."""

from .error_computation import compute_prediction_errors, compute_errors_for_trajectories
from .feature_extraction import extract_fingerprint, extract_fingerprints_batch
from .analysis import compare_fingerprints, train_classifier
from .taxonomy_analysis import (
    TaxonomyAnalysisResults,
    analyze_misalignment_taxonomy,
    identify_misalignment_signatures,
    build_misalignment_grading_system,
    generate_key_findings,
)

__all__ = [
    'compute_prediction_errors',
    'compute_errors_for_trajectories',
    'extract_fingerprint',
    'extract_fingerprints_batch',
    'compare_fingerprints',
    'train_classifier',
    'TaxonomyAnalysisResults',
    'analyze_misalignment_taxonomy',
    'identify_misalignment_signatures',
    'build_misalignment_grading_system',
    'generate_key_findings',
]
