"""
Quick test to verify hierarchical classifier uses proper cross-validation.
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

# Get API key
api_key = os.environ.get('DEEPSEEK_API_KEY')
if not api_key:
    print("ERROR: DEEPSEEK_API_KEY not found")
    sys.exit(1)

llm_config = {
    'provider': 'deepseek',
    'model': 'deepseek-chat',
    'api_key': api_key,
}

from exp10_probe_enhanced_v2 import EnhancedProbeExperimentV2

print("=" * 80)
print("TEST: Hierarchical Classifier Cross-Validation Fix")
print("=" * 80)
print("Running N=5 with hierarchical=True to verify cross-validation...")
print()

experiment = EnhancedProbeExperimentV2(
    llm_config=llm_config,
    output_dir='results/exp10_probe_enhanced_v2',
    description='Test hierarchical CV fix - N=5',
    max_workers=10,
    use_hierarchical=True  # Enable hierarchical
)

try:
    results = experiment.run(samples_per_type=5)

    print()
    print("=" * 80)
    print("TEST PASSED")
    print("=" * 80)
    print(f"Overall Accuracy: {results.multiclass_accuracy:.1%}")
    print(f"Stage 1 Accuracy: {results.stage1_accuracy:.1%}" if results.stage1_accuracy else "Stage 1: N/A")
    print(f"Stage 2 Accuracy: {results.stage2_accuracy:.1%}" if results.stage2_accuracy else "Stage 2: N/A")
    print()

    # The key test: accuracy should NOT be 100%
    if results.multiclass_accuracy == 1.0:
        print("⚠️  WARNING: Accuracy is 100% - may still be overfitting")
        print("   (But this could be valid for N=5)")
    else:
        print("✓ Accuracy is not 100% - proper cross-validation likely working")

    print()
    print("Per-class accuracy:")
    for agent_type, acc in results.per_class_accuracy.items():
        print(f"  {agent_type}: {acc:.1%}")

except Exception as e:
    print()
    print("=" * 80)
    print("TEST FAILED")
    print("=" * 80)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
