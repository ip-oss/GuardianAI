#!/usr/bin/env python3
"""
Main preprocessing script for iOS World project.
Orchestrates cleaning and preparation of captured transition data.
"""

import argparse
from pathlib import Path
import sys

from clean import clean_transitions, print_stats as print_clean_stats
from prepare import prepare_dataset, print_stats as print_prep_stats


def main():
    parser = argparse.ArgumentParser(
        description='Process captured iOS UI transition data'
    )

    parser.add_argument(
        'raw_data_dir',
        type=Path,
        help='Directory containing raw captured data'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/processed'),
        help='Output directory for processed data (default: data/processed)'
    )

    parser.add_argument(
        '--temp-dir',
        type=Path,
        default=Path('data/temp_cleaned'),
        help='Temporary directory for cleaned data (default: data/temp_cleaned)'
    )

    parser.add_argument(
        '--target-size',
        type=int,
        nargs=2,
        default=[512, 1024],
        metavar=('WIDTH', 'HEIGHT'),
        help='Target image size (default: 512 1024)'
    )

    parser.add_argument(
        '--split',
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        metavar=('TRAIN', 'VAL', 'TEST'),
        help='Train/val/test split ratios (default: 0.8 0.1 0.1)'
    )

    parser.add_argument(
        '--skip-clean',
        action='store_true',
        help='Skip cleaning step (use if data already cleaned)'
    )

    parser.add_argument(
        '--skip-prepare',
        action='store_true',
        help='Skip preparation step (only run cleaning)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.raw_data_dir.exists():
        print(f"Error: Raw data directory does not exist: {args.raw_data_dir}")
        sys.exit(1)

    if sum(args.split) != 1.0:
        print(f"Error: Split ratios must sum to 1.0 (got {sum(args.split)})")
        sys.exit(1)

    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("iOS World Data Preprocessing")
    print("=" * 60)
    print(f"Raw data: {args.raw_data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Target size: {args.target_size[0]}x{args.target_size[1]}")
    print(f"Split: {args.split[0]:.0%} train, {args.split[1]:.0%} val, {args.split[2]:.0%} test")
    print("=" * 60)

    # Step 1: Clean the data
    if not args.skip_clean:
        print("\n[1/2] Cleaning data...")
        args.temp_dir.mkdir(parents=True, exist_ok=True)

        clean_stats = clean_transitions(args.raw_data_dir, args.temp_dir)
        print_clean_stats(clean_stats)

        if clean_stats['valid'] == 0:
            print("Error: No valid transitions found after cleaning")
            sys.exit(1)

        cleaned_dir = args.temp_dir
    else:
        print("\n[1/2] Skipping cleaning step")
        cleaned_dir = args.raw_data_dir

    # Step 2: Prepare the dataset
    if not args.skip_prepare:
        print("\n[2/2] Preparing dataset...")

        prep_stats = prepare_dataset(
            cleaned_dir,
            args.output_dir,
            target_size=tuple(args.target_size),
            split_ratios=tuple(args.split)
        )
        print_prep_stats(prep_stats)

        print(f"\n✓ Dataset prepared successfully!")
        print(f"  Output location: {args.output_dir}")
        print(f"  Total transitions: {prep_stats['total_transitions']}")
        print(f"  Train: {prep_stats['train']}")
        print(f"  Val: {prep_stats['val']}")
        print(f"  Test: {prep_stats['test']}")

    else:
        print("\n[2/2] Skipping preparation step")

    # Cleanup temp directory if we created it
    if not args.skip_clean and args.temp_dir.exists():
        print(f"\nCleaning up temporary directory: {args.temp_dir}")
        # TODO: Add cleanup if desired
        # shutil.rmtree(args.temp_dir)

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
