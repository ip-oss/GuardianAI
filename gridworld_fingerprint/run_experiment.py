#!/usr/bin/env python3
"""
Main entry point for GridWorld behavioral fingerprinting experiments.

Usage:
    python run_experiment.py                    # Run with defaults
    python run_experiment.py --quick            # Quick test run
    python run_experiment.py --experiment exp1  # Run specific experiment
"""

import argparse
import sys
from pathlib import Path

from config import (
    ExperimentConfig, QuickConfig, DeceptiveConfig, SubtletyConfig,
    TaxonomyConfig, TaxonomyQuickConfig
)


def main():
    parser = argparse.ArgumentParser(
        description='Run GridWorld behavioral fingerprinting experiments'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test with reduced parameters'
    )

    parser.add_argument(
        '--experiment',
        type=str,
        default='exp1',
        choices=['exp1', 'exp2', 'exp5', 'exp6'],
        help='Which experiment to run (exp1=basic, exp2=deceptive, exp5=subtlety, exp6=taxonomy)'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory to save results'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Load config based on experiment type
    if args.experiment == 'exp2':
        config = DeceptiveConfig()
        if args.quick:
            print("Running in QUICK mode (reduced parameters for testing)")
            config.num_trajectories = 50
            config.agent_episodes = 500
            config.world_model_epochs = 50
    elif args.experiment == 'exp5':
        config = SubtletyConfig()
        if args.quick:
            print("Running in QUICK mode (reduced parameters for testing)")
            config.num_trajectories = 30
            config.agent_episodes = 500
            config.world_model_epochs = 50
            config.misalignment_levels = [0.0, 0.5, 1.0]  # Only test 3 levels
    elif args.experiment == 'exp6':
        if args.quick:
            print("Running in QUICK mode (8x8 grid, reduced parameters)")
            config = TaxonomyQuickConfig()
        else:
            print("Running FULL taxonomy experiment (12x12 grid, 2000 episodes per agent)")
            config = TaxonomyConfig()
    else:
        if args.quick:
            print("Running in QUICK mode (reduced parameters for testing)")
            config = QuickConfig()
        else:
            config = ExperimentConfig()

    # Override from args
    config.results_dir = args.results_dir
    config.random_seed = args.seed

    # Create results directory
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)

    # Run experiment
    if args.experiment == 'exp1':
        from experiments.exp1_basic_fingerprint import run_experiment
        results = run_experiment(config)
    elif args.experiment == 'exp2':
        from experiments.exp2_deceptive_detection import run_experiment
        results = run_experiment(config)
    elif args.experiment == 'exp5':
        from experiments.exp5_subtlety_gradient import run_experiment
        results = run_experiment(config)
    elif args.experiment == 'exp6':
        from experiments.exp6_taxonomy import run_experiment
        results = run_experiment(config)
    else:
        print(f"Unknown experiment: {args.experiment}")
        sys.exit(1)

    print(f"\n✓ Results saved to {config.results_dir}")


if __name__ == "__main__":
    main()
