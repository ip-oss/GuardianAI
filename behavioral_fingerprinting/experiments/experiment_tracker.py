"""
Experiment Version Tracking and Organization

Provides consistent naming, versioning, and metadata tracking for experiments.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import hashlib


class ExperimentTracker:
    """Manages experiment versions, metadata, and result organization."""

    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.base_dir / "experiment_manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict:
        """Load experiment manifest or create new one."""
        if self.manifest_file.exists():
            with open(self.manifest_file, 'r') as f:
                return json.load(f)
        return {"experiments": [], "last_run_id": 0}

    def _save_manifest(self):
        """Save updated manifest."""
        with open(self.manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)

    def create_run(self,
                   experiment_name: str,
                   config: Dict[str, Any],
                   description: str = "") -> Dict[str, Any]:
        """
        Create a new experiment run with proper versioning.

        Args:
            experiment_name: Name of experiment (e.g., "exp7_llm")
            config: Configuration dict (model, provider, trajectories, etc.)
            description: Optional description of this run

        Returns:
            Dict with run metadata including run_id, output_dir, etc.
        """
        # Generate run ID
        run_id = self.manifest["last_run_id"] + 1
        self.manifest["last_run_id"] = run_id

        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create config hash for comparison
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        # Create descriptive run name
        provider = config.get('provider', 'unknown')
        model = config.get('model', 'unknown').replace('/', '_')
        trajectories = config.get('trajectories', 'N')

        run_name = f"run{run_id:04d}_{provider}_{model}_n{trajectories}_{timestamp}"

        # Create output directory with hierarchy
        exp_dir = self.base_dir / experiment_name
        run_dir = exp_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create run metadata
        run_metadata = {
            'run_id': run_id,
            'experiment_name': experiment_name,
            'run_name': run_name,
            'timestamp': timestamp,
            'datetime': datetime.now().isoformat(),
            'config': config,
            'config_hash': config_hash,
            'description': description,
            'output_dir': str(run_dir),
            'status': 'running'
        }

        # Save run metadata to run directory
        with open(run_dir / 'run_metadata.json', 'w') as f:
            json.dump(run_metadata, f, indent=2)

        # Add to manifest
        self.manifest["experiments"].append(run_metadata)
        self._save_manifest()

        return run_metadata

    def complete_run(self, run_id: int, results_summary: Dict[str, Any]):
        """Mark a run as completed and add summary."""
        for exp in self.manifest["experiments"]:
            if exp['run_id'] == run_id:
                exp['status'] = 'completed'
                exp['completion_time'] = datetime.now().isoformat()
                exp['results_summary'] = results_summary
                break
        self._save_manifest()

    def fail_run(self, run_id: int, error: str):
        """Mark a run as failed."""
        for exp in self.manifest["experiments"]:
            if exp['run_id'] == run_id:
                exp['status'] = 'failed'
                exp['error'] = error
                exp['failure_time'] = datetime.now().isoformat()
                break
        self._save_manifest()

    def get_runs(self,
                 experiment_name: Optional[str] = None,
                 provider: Optional[str] = None,
                 status: Optional[str] = None) -> list:
        """
        Query runs by various filters.

        Args:
            experiment_name: Filter by experiment name
            provider: Filter by LLM provider
            status: Filter by status (running, completed, failed)

        Returns:
            List of matching run metadata
        """
        runs = self.manifest["experiments"]

        if experiment_name:
            runs = [r for r in runs if r['experiment_name'] == experiment_name]

        if provider:
            runs = [r for r in runs if r['config'].get('provider') == provider]

        if status:
            runs = [r for r in runs if r['status'] == status]

        return runs

    def get_latest_run(self, experiment_name: str) -> Optional[Dict]:
        """Get the most recent run for an experiment."""
        runs = self.get_runs(experiment_name=experiment_name)
        if runs:
            return sorted(runs, key=lambda x: x['run_id'], reverse=True)[0]
        return None

    def compare_runs(self, run_ids: list) -> Dict:
        """Compare multiple runs."""
        runs = [r for r in self.manifest["experiments"] if r['run_id'] in run_ids]

        if not runs:
            return {}

        comparison = {
            'runs': runs,
            'config_differences': {},
            'result_differences': {}
        }

        # Find config differences
        configs = [r['config'] for r in runs]
        all_keys = set()
        for c in configs:
            all_keys.update(c.keys())

        for key in all_keys:
            values = [c.get(key) for c in configs]
            if len(set(str(v) for v in values)) > 1:  # Different values
                comparison['config_differences'][key] = dict(zip(run_ids, values))

        # Compare results if available
        for run in runs:
            if 'results_summary' in run:
                rid = run['run_id']
                for metric, value in run['results_summary'].items():
                    if metric not in comparison['result_differences']:
                        comparison['result_differences'][metric] = {}
                    comparison['result_differences'][metric][rid] = value

        return comparison

    def print_summary(self):
        """Print a summary of all experiments."""
        print("\n" + "=" * 80)
        print("EXPERIMENT TRACKING SUMMARY")
        print("=" * 80)

        by_experiment = {}
        for run in self.manifest["experiments"]:
            exp_name = run['experiment_name']
            if exp_name not in by_experiment:
                by_experiment[exp_name] = []
            by_experiment[exp_name].append(run)

        for exp_name, runs in by_experiment.items():
            print(f"\n📊 {exp_name}")
            print(f"   Total runs: {len(runs)}")

            by_status = {}
            for run in runs:
                status = run['status']
                by_status[status] = by_status.get(status, 0) + 1

            for status, count in by_status.items():
                print(f"   - {status}: {count}")

            # Show recent runs
            recent = sorted(runs, key=lambda x: x['run_id'], reverse=True)[:3]
            print(f"\n   Recent runs:")
            for run in recent:
                config = run['config']
                provider = config.get('provider', 'unknown')
                model = config.get('model', 'unknown')
                n = config.get('trajectories', '?')
                status_icon = {'completed': '✅', 'running': '⏳', 'failed': '❌'}.get(run['status'], '?')

                result_str = ""
                if 'results_summary' in run:
                    acc = run['results_summary'].get('accuracy', 0)
                    result_str = f" | Acc: {acc:.1%}"

                print(f"   {status_icon} Run {run['run_id']:04d}: {provider}/{model} n={n}{result_str}")

        print("\n" + "=" * 80)
