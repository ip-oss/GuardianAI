#!/usr/bin/env python3
"""
View and compare experiment runs.

Usage:
    python view_runs.py                  # View all runs
    python view_runs.py --latest         # Show latest run details
    python view_runs.py --compare 1 2 3  # Compare specific runs
    python view_runs.py --provider ollama # Filter by provider
"""

import argparse
from experiments.experiment_tracker import ExperimentTracker
from pathlib import Path


def format_config(config):
    """Format config dict for display."""
    provider = config.get('provider', 'unknown')
    model = config.get('model', 'unknown')
    n = config.get('trajectories', '?')
    return f"{provider}/{model} n={n}"


def print_run_details(run):
    """Print detailed info about a run."""
    print("\n" + "=" * 80)
    print(f"Run {run['run_id']:04d}: {run['run_name']}")
    print("=" * 80)

    print(f"\n⏰ Time: {run['datetime']}")
    print(f"📊 Experiment: {run['experiment_name']}")
    print(f"🔧 Config: {format_config(run['config'])}")
    print(f"📝 Status: {run['status']}")

    if run.get('description'):
        print(f"📄 Description: {run['description']}")

    if 'results_summary' in run:
        print(f"\n📈 Results:")
        for key, value in run['results_summary'].items():
            if isinstance(value, float):
                if key == 'accuracy':
                    print(f"   {key}: {value:.1%}")
                else:
                    print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")

    print(f"\n📁 Output: {run['output_dir']}")

    # Check if results files exist
    output_dir = Path(run['output_dir'])
    if output_dir.exists():
        files = list(output_dir.glob('*'))
        print(f"   Files: {len(files)}")
        for f in sorted(files):
            print(f"      - {f.name}")


def main():
    parser = argparse.ArgumentParser(description='View experiment runs')
    parser.add_argument('--latest', action='store_true',
                       help='Show details of latest run')
    parser.add_argument('--compare', nargs='+', type=int,
                       help='Compare specific run IDs')
    parser.add_argument('--provider', type=str,
                       help='Filter by provider')
    parser.add_argument('--experiment', type=str, default='exp7_llm',
                       help='Filter by experiment name')
    parser.add_argument('--status', type=str,
                       choices=['running', 'completed', 'failed'],
                       help='Filter by status')

    args = parser.parse_args()

    tracker = ExperimentTracker(base_dir='results')

    if args.compare:
        # Compare specific runs
        comparison = tracker.compare_runs(args.compare)

        print("\n" + "=" * 80)
        print("RUN COMPARISON")
        print("=" * 80)

        print("\n📊 Runs:")
        for run in comparison['runs']:
            status_icon = {'completed': '✅', 'running': '⏳', 'failed': '❌'}.get(run['status'], '?')
            print(f"   {status_icon} Run {run['run_id']:04d}: {format_config(run['config'])}")

        if comparison['config_differences']:
            print("\n🔧 Configuration Differences:")
            for key, values in comparison['config_differences'].items():
                print(f"   {key}:")
                for run_id, value in values.items():
                    print(f"      Run {run_id}: {value}")

        if comparison['result_differences']:
            print("\n📈 Result Differences:")
            for metric, values in comparison['result_differences'].items():
                print(f"   {metric}:")
                for run_id, value in values.items():
                    if isinstance(value, float):
                        if metric == 'accuracy':
                            print(f"      Run {run_id}: {value:.1%}")
                        else:
                            print(f"      Run {run_id}: {value:.3f}")
                    else:
                        print(f"      Run {run_id}: {value}")

    elif args.latest:
        # Show latest run
        run = tracker.get_latest_run(args.experiment)
        if run:
            print_run_details(run)
        else:
            print(f"No runs found for experiment: {args.experiment}")

    else:
        # List all runs
        runs = tracker.get_runs(
            experiment_name=args.experiment,
            provider=args.provider,
            status=args.status
        )

        if not runs:
            print("No runs found matching criteria.")
            return

        print("\n" + "=" * 80)
        print(f"EXPERIMENT RUNS ({len(runs)} total)")
        print("=" * 80)

        # Group by status
        by_status = {}
        for run in runs:
            status = run['status']
            if status not in by_status:
                by_status[status] = []
            by_status[status].append(run)

        for status in ['completed', 'running', 'failed']:
            if status not in by_status:
                continue

            status_icon = {'completed': '✅', 'running': '⏳', 'failed': '❌'}[status]
            print(f"\n{status_icon} {status.upper()} ({len(by_status[status])})")
            print("-" * 80)

            for run in sorted(by_status[status], key=lambda x: x['run_id'], reverse=True):
                config = run['config']

                result_str = ""
                if 'results_summary' in run:
                    acc = run['results_summary'].get('accuracy', 0)
                    result_str = f" | Acc: {acc:.1%}"

                desc_str = ""
                if run.get('description'):
                    desc_str = f" | {run['description'][:40]}"

                print(f"   Run {run['run_id']:04d}: {format_config(config)}{result_str}{desc_str}")

        print("\n" + "=" * 80)
        print("Usage:")
        print("  python view_runs.py --latest              # View latest run")
        print("  python view_runs.py --compare 1 2 3       # Compare runs")
        print("  python view_runs.py --provider ollama     # Filter by provider")
        print("=" * 80)


if __name__ == "__main__":
    main()
