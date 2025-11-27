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
from .abstract_features import (
    extract_abstract_features,
    extract_abstract_features_batch,
    compute_abstract_signature,
    print_feature_comparison,
)
from .mode_switching_analysis import (
    count_mode_switches,
    compute_mode_fractions,
    extract_temporal_dynamics,
    compute_spatial_observation_map,
    analyze_mode_switching_patterns,
    analyze_mode_switching_batch,
    compare_agent_types,
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
    'extract_abstract_features',
    'extract_abstract_features_batch',
    'compute_abstract_signature',
    'print_feature_comparison',
    'count_mode_switches',
    'compute_mode_fractions',
    'extract_temporal_dynamics',
    'compute_spatial_observation_map',
    'analyze_mode_switching_patterns',
    'analyze_mode_switching_batch',
    'compare_agent_types',
]
