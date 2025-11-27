"""
GridWorld Experiment Version Tracking and Organization

Provides consistent naming, versioning, and metadata tracking for experiments.
Adapted from LLM experiment tracker for GridWorld experiments.
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
            experiment_name: Name of experiment (e.g., "exp6_taxonomy", "exp_stochastic")
            config: Configuration dict (grid_size, agent_types, trajectories, etc.)
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

        # Create descriptive run name based on experiment type
        run_name = self._generate_run_name(run_id, experiment_name, config, timestamp)

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

    def _generate_run_name(self, run_id: int, experiment_name: str, config: Dict, timestamp: str) -> str:
        """Generate a descriptive run name based on configuration."""

        # Common config elements
        grid_size = config.get('grid_size', 'gridUnknown')
        num_traj = config.get('trajectories', config.get('num_trajectories', 'nUnknown'))

        # Experiment-specific naming
        if experiment_name == "exp6_taxonomy":
            num_agents = config.get('num_agent_types', len(config.get('agent_types', [])))
            return f"run{run_id:04d}_grid{grid_size}_agents{num_agents}_n{num_traj}_{timestamp}"

        elif experiment_name == "exp_stochastic":
            max_noise = config.get('max_noise_level', 'noiseUnknown')
            return f"run{run_id:04d}_grid{grid_size}_noise{max_noise}_n{num_traj}_{timestamp}"

        elif experiment_name == "exp_transfer":
            num_layouts = config.get('num_layouts', 3)
            return f"run{run_id:04d}_grid{grid_size}_layouts{num_layouts}_n{num_traj}_{timestamp}"

        elif experiment_name in ["exp5_subtlety_gradient", "exp5_subtlety"]:
            num_levels = config.get('num_subtlety_levels', 'levelsUnknown')
            return f"run{run_id:04d}_grid{grid_size}_levels{num_levels}_n{num_traj}_{timestamp}"

        elif experiment_name == "exp2_deceptive_detection":
            deception_rate = config.get('deception_rate', 'deceptUnknown')
            return f"run{run_id:04d}_grid{grid_size}_decept{deception_rate}_n{num_traj}_{timestamp}"

        else:
            # Generic naming
            return f"run{run_id:04d}_grid{grid_size}_n{num_traj}_{timestamp}"

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
                 grid_size: Optional[int] = None,
                 status: Optional[str] = None) -> list:
        """
        Query runs by various filters.

        Args:
            experiment_name: Filter by experiment name
            grid_size: Filter by grid size
            status: Filter by status (running, completed, failed)

        Returns:
            List of matching run metadata
        """
        runs = self.manifest["experiments"]

        if experiment_name:
            runs = [r for r in runs if r['experiment_name'] == experiment_name]

        if grid_size:
            runs = [r for r in runs if r['config'].get('grid_size') == grid_size]

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
        print("GRIDWORLD EXPERIMENT TRACKING SUMMARY")
        print("=" * 80)

        by_experiment = {}
        for run in self.manifest["experiments"]:
            exp_name = run['experiment_name']
            if exp_name not in by_experiment:
                by_experiment[exp_name] = []
            by_experiment[exp_name].append(run)

        for exp_name, runs in sorted(by_experiment.items()):
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
                grid_size = config.get('grid_size', '?')
                n = config.get('trajectories', config.get('num_trajectories', '?'))
                status_icon = {'completed': '✅', 'running': '⏳', 'failed': '❌'}.get(run['status'], '?')

                result_str = ""
                if 'results_summary' in run:
                    acc = run['results_summary'].get('accuracy', run['results_summary'].get('multiclass_accuracy', 0))
                    result_str = f" | Acc: {acc:.1%}" if acc else ""

                print(f"   {status_icon} Run {run['run_id']:04d}: grid{grid_size} n={n}{result_str}")

        print("\n" + "=" * 80)

    def generate_run_readme(self, run_id: int, results: Dict[str, Any]) -> str:
        """
        Generate a README.md for a specific run.

        Args:
            run_id: The run ID
            results: The results dictionary

        Returns:
            Markdown content for README
        """
        # Find run metadata
        run = None
        for exp in self.manifest["experiments"]:
            if exp['run_id'] == run_id:
                run = exp
                break

        if not run:
            return f"# Run {run_id:04d}\n\nRun not found in manifest."

        config = run['config']

        readme = f"""# {run['experiment_name'].upper()} - Run {run_id:04d}

## Configuration

- **Grid Size:** {config.get('grid_size', 'Unknown')}
- **Trajectories:** {config.get('trajectories', config.get('num_trajectories', 'Unknown'))}
- **Timestamp:** {run['datetime']}
- **Status:** {run['status']}

"""

        # Add experiment-specific config
        if run['experiment_name'] == "exp6_taxonomy":
            readme += f"- **Agent Types:** {config.get('num_agent_types', 'Unknown')} types\n"
            if 'agent_types' in config:
                readme += f"  - {', '.join(config['agent_types'])}\n"

        elif run['experiment_name'] == "exp_stochastic":
            readme += f"- **Noise Levels Tested:** {config.get('noise_levels', 'Unknown')}\n"

        elif run['experiment_name'] == "exp_transfer":
            readme += f"- **Layouts:** {config.get('num_layouts', 'Unknown')} layouts tested\n"

        readme += f"\n## Results Summary\n\n"

        # Add key results
        if 'results_summary' in run:
            summary = run['results_summary']
            for key, value in summary.items():
                if isinstance(value, float):
                    if key.endswith('accuracy') or key.endswith('correlation'):
                        readme += f"- **{key.replace('_', ' ').title()}:** {value:.1%}\n"
                    else:
                        readme += f"- **{key.replace('_', ' ').title()}:** {value:.3f}\n"
                else:
                    readme += f"- **{key.replace('_', ' ').title()}:** {value}\n"

        readme += f"\n## Files\n\n"
        readme += f"- `results.json` - Complete results and analysis\n"
        readme += f"- `run_metadata.json` - Run configuration and metadata\n"
        readme += f"- `README.md` - This file\n"

        if run['experiment_name'] == "exp6_taxonomy":
            readme += f"- `fingerprints.json` - Raw behavioral fingerprints\n"
            readme += f"- `signatures.json` - Misalignment signatures\n"
            readme += f"- `embeddings.npz` - t-SNE and PCA embeddings\n"
            readme += f"- `fingerprint_clusters.png` - Cluster visualization\n"
            readme += f"- `confusion_matrix.png` - Classification results\n"

        if run.get('description'):
            readme += f"\n## Description\n\n{run['description']}\n"

        return readme
