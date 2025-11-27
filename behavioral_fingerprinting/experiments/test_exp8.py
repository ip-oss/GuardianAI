"""
Quick test of exp8 probe-enhanced experiment with N=2.

Tests all components before running full N=100.
"""

import os
import sys
from pathlib import Path

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from exp8_probe_enhanced import ProbeEnhancedExperiment

def main():
    from dotenv import load_dotenv

    # Load .env file
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment from {env_path}\n")

    # Get API key
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not found in .env file or environment")
        return 1

    # Configure for minimal test
    llm_config = {
        'provider': 'deepseek',
        'model': 'deepseek-chat',
        'api_key': api_key,
    }

    print("="*70)
    print("TESTING EXP8 WITH N=2")
    print("="*70)
    print()

    # Run minimal experiment
    experiment = ProbeEnhancedExperiment(
        llm_config=llm_config,
        description="Test run with N=2 to validate probe system",
        max_workers=2,  # Low parallelism for debugging
    )

    try:
        results = experiment.run(samples_per_type=2)  # Only 2 samples per type

        print("\n" + "="*70)
        print("✓ TEST SUCCESSFUL")
        print("="*70)
        print(f"Accuracy: {results.multiclass_accuracy:.1%}")
        print(f"Silhouette: {results.silhouette_score:.3f}")
        print(f"Per-class accuracy:")
        for agent_type, acc in results.per_class_accuracy.items():
            print(f"  {agent_type:20s}: {acc:.1%}")
        print("="*70)

    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
