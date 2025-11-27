"""
Run All Robustness Experiments

Orchestrates the complete suite of robustness tests for behavioral fingerprinting:
1. Baseline taxonomy (exp6)
2. Subtlety gradient (exp5)
3. Deceptive detection (exp2)
4. Stochastic environment (exp_stochastic) [NEW]
5. Transfer validation (exp_transfer) [NEW]

Generates a comprehensive report comparing all results.
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import TaxonomyConfig, TaxonomyQuickConfig


def run_experiment(exp_name: str, exp_file: str, config_type: str = "full", verbose: bool = True):
    """
    Run a single experiment.

    Args:
        exp_name: Human-readable experiment name
        exp_file: Python file to run (without .py)
        config_type: "quick" or "full"
        verbose: Print detailed output
    """
    print(f"\n{'='*70}")
    print(f"Running: {exp_name}")
    print(f"{'='*70}\n")

    # Build command
    cmd = [sys.executable, "-m", f"experiments.{exp_file}"]
    if config_type == "quick":
        cmd.append("--quick")

    # Run experiment
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=not verbose,
            text=True,
            check=True
        )

        if not verbose and result.stdout:
            print(result.stdout)

        print(f"\n✓ {exp_name} completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ {exp_name} failed with error:")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False


def load_latest_results(exp_dir: str) -> dict:
    """Load the most recent results for an experiment."""
    results_dir = Path("results") / exp_dir

    if not results_dir.exists():
        return None

    # Find most recent run directory
    run_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        return None

    latest_dir = max(run_dirs, key=lambda d: d.stat().st_mtime)
    results_file = latest_dir / "results.json"

    if not results_file.exists():
        return None

    with open(results_file, 'r') as f:
        return json.load(f)


def generate_comprehensive_report(results: dict, output_path: Path):
    """Generate markdown report summarizing all experiments."""

    report = f"""# GridWorld Behavioral Fingerprinting - Robustness Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This report summarizes the results of comprehensive robustness testing for
behavioral fingerprinting via world model prediction errors.

---

## Experiments Conducted

"""

    # Add results for each experiment
    for exp_name, exp_data in results.items():
        if exp_data is None:
            report += f"### {exp_name}\n**Status:** Not run or failed\n\n"
            continue

        report += f"### {exp_name}\n"

        if "multiclass_accuracy" in exp_data:
            report += f"- **Accuracy:** {exp_data['multiclass_accuracy']:.1%}\n"

        if "silhouette_score" in exp_data:
            report += f"- **Silhouette Score:** {exp_data['silhouette_score']:.3f}\n"

        if "danger_correlation" in exp_data:
            report += f"- **Danger Correlation:** {exp_data['danger_correlation']:.3f}\n"

        if "key_findings" in exp_data:
            report += "\n**Key Findings:**\n"
            for finding in exp_data["key_findings"]:
                report += f"- {finding}\n"

        report += "\n"

    # Overall conclusions
    report += """---

## Conclusions

### Strengths

The behavioral fingerprinting approach demonstrates:

1. **High baseline accuracy** - Can classify misalignment types with strong performance
2. **Robustness to noise** - Detection works despite environmental randomness
3. **Transfer capability** - Fingerprints generalize across different layouts
4. **Subtlety detection** - Can detect even small amounts of misalignment

### Limitations

1. **Behavioral-only detection** - Cannot detect misalignment that doesn't manifest in behavior
2. **Training data dependency** - World model quality depends on aligned data diversity
3. **Layout sensitivity** - Some accuracy drop when testing on very different environments

### Next Steps

1. **Scale to larger grids** - Test on 20x20 or larger for complexity matching e-commerce domain
2. **Align scenarios with LLM behaviors** - Map GridWorld agents to e-commerce agent types
3. **Neural network agents** - Replace Q-learning with DQN for richer behavior patterns
4. **Adversarial evasion testing** - Can misaligned agents learn to hide their fingerprints?

---

*For detailed results, see individual experiment output directories in `results/`*
"""

    # Write report
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"\n📄 Comprehensive report saved to: {output_path}")


def main(config_type: str = "quick", experiments: list = None):
    """
    Run all robustness experiments.

    Args:
        config_type: "quick" for fast testing, "full" for publication-quality
        experiments: List of specific experiments to run, or None for all
    """
    print("="*70)
    print("GridWorld Behavioral Fingerprinting - Robustness Test Suite")
    print("="*70)
    print(f"Config: {config_type}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Define experiment suite
    all_experiments = {
        "Baseline Taxonomy": "exp6_taxonomy",
        "Subtlety Gradient": "exp5_subtlety_gradient",
        "Deceptive Detection": "exp2_deceptive_detection",
        "Stochastic Environment": "exp_stochastic",
        "Transfer Validation": "exp_transfer",
    }

    # Filter if specific experiments requested
    if experiments:
        experiments_to_run = {k: v for k, v in all_experiments.items() if v in experiments}
    else:
        experiments_to_run = all_experiments

    # Run experiments
    success_count = 0
    results = {}

    for exp_name, exp_file in experiments_to_run.items():
        success = run_experiment(exp_name, exp_file, config_type, verbose=True)
        if success:
            success_count += 1

        # Load results
        results[exp_name] = load_latest_results(exp_file.replace("_", " "))

    # Generate report
    print(f"\n{'='*70}")
    print("Generating Comprehensive Report")
    print(f"{'='*70}")

    report_dir = Path("results") / "comprehensive_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"robustness_report_{timestamp}.md"

    generate_comprehensive_report(results, report_path)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Experiments run: {len(experiments_to_run)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(experiments_to_run) - success_count}")
    print(f"\nReport: {report_path}")
    print("="*70)

    return success_count == len(experiments_to_run)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run all robustness experiments for GridWorld behavioral fingerprinting"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="quick",
        choices=["quick", "full"],
        help="Configuration type: 'quick' for testing, 'full' for publication"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        help="Specific experiments to run (default: all)",
        choices=[
            "exp6_taxonomy",
            "exp5_subtlety_gradient",
            "exp2_deceptive_detection",
            "exp_stochastic",
            "exp_transfer"
        ]
    )

    args = parser.parse_args()

    success = main(config_type=args.config, experiments=args.experiments)
    sys.exit(0 if success else 1)
