"""Visualization tools for GridWorld experiments."""

from .taxonomy_plots import (
    plot_fingerprint_clusters,
    plot_confusion_matrix,
    plot_type_similarity_heatmap,
    plot_danger_correlation,
    plot_signature_radar,
    plot_feature_importance_comparison,
)

from .mode_switching_plots import (
    plot_mode_timeline,
    plot_mode_switch_heatmap,
    plot_mode_fraction_comparison,
    plot_observation_coverage_map,
    plot_switch_frequency_comparison,
    plot_mode_switching_summary,
)

from .extended_experiment_plots import (
    plot_detection_curve,
    plot_transfer_comparison,
    plot_stealth_analysis,
    plot_comprehensive_validation_summary,
)

__all__ = [
    'plot_fingerprint_clusters',
    'plot_confusion_matrix',
    'plot_type_similarity_heatmap',
    'plot_danger_correlation',
    'plot_signature_radar',
    'plot_feature_importance_comparison',
    'plot_mode_timeline',
    'plot_mode_switch_heatmap',
    'plot_mode_fraction_comparison',
    'plot_observation_coverage_map',
    'plot_switch_frequency_comparison',
    'plot_mode_switching_summary',
    'plot_detection_curve',
    'plot_transfer_comparison',
    'plot_stealth_analysis',
    'plot_comprehensive_validation_summary',
]
