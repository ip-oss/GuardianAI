"""
Test script for Experiment 10 - Enhanced Probe Engineering V2

Quick smoke test with minimal samples to verify the implementation works.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✓ Loaded environment from {env_path}\n")

from exp10_probe_enhanced_v2 import EnhancedProbeExperimentV2


def test_exp10_smoke():
    """Run a minimal test with 2 samples per type."""

    print("=" * 80)
    print("SMOKE TEST: Experiment 10 - Enhanced Probe Engineering V2")
    print("=" * 80)
    print()
    print("Testing with 2 samples per agent type...")
    print("Total sessions: 7 types × 2 samples = 14 sessions")
    print()

    # Use DeepSeek for quick testing
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not found in environment")
        print("Please set it in .env file or environment variable")
        sys.exit(1)

    llm_config = {
        'provider': 'deepseek',
        'model': 'deepseek-chat',
        'api_key': api_key,
    }

    # Create experiment with hierarchical classifier enabled
    experiment = EnhancedProbeExperimentV2(
        llm_config=llm_config,
        output_dir='results/exp10_probe_enhanced_v2',
        description='Smoke test - 2 samples per type',
        max_workers=4,  # Low workers for testing
        use_hierarchical=True
    )

    # Run with minimal samples
    try:
        results = experiment.run(samples_per_type=2)

        print()
        print("=" * 80)
        print("SMOKE TEST PASSED")
        print("=" * 80)
        print(f"Overall Accuracy: {results.multiclass_accuracy:.1%}")
        print(f"Stage 1 Accuracy: {results.stage1_accuracy:.1%}" if results.stage1_accuracy else "Stage 1: N/A")
        print(f"Stage 2 Accuracy: {results.stage2_accuracy:.1%}" if results.stage2_accuracy else "Stage 2: N/A")
        print()
        print("Per-class accuracy:")
        for agent_type, acc in results.per_class_accuracy.items():
            print(f"  {agent_type}: {acc:.1%}")
        print()
        print("✓ All components working correctly!")
        print()

    except Exception as e:
        print()
        print("=" * 80)
        print("SMOKE TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    test_exp10_smoke()
