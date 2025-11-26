"""
Data cleaning module for iOS World project.
Removes bad transitions, corrupt images, duplicates, and timeouts.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import hashlib


def clean_transitions(raw_data_dir: Path, output_dir: Path) -> Dict[str, int]:
    """
    Clean captured transition data by removing bad/corrupt entries.

    Args:
        raw_data_dir: Directory containing raw captured data
        output_dir: Directory to write cleaned data to

    Returns:
        Dictionary with cleaning statistics
    """
    stats = {
        'total': 0,
        'valid': 0,
        'missing_images': 0,
        'corrupt_images': 0,
        'identical_screenshots': 0,
        'duplicates': 0,
        'timeouts': 0,
        'low_stability': 0,
    }

    seen_hashes: set = set()  # Track duplicates

    # Iterate through all app directories
    for app_dir in raw_data_dir.iterdir():
        if not app_dir.is_dir():
            continue

        # Iterate through all session directories
        for session_dir in app_dir.iterdir():
            if not session_dir.is_dir():
                continue

            # Process all transitions in this session
            transition_files = sorted(session_dir.glob('transition_*.json'))

            for transition_file in transition_files:
                stats['total'] += 1

                try:
                    # Load transition metadata
                    with open(transition_file, 'r') as f:
                        transition = json.load(f)

                    # Check if this transition should be kept
                    is_valid, reason = is_valid_transition(
                        transition, session_dir, seen_hashes
                    )

                    if not is_valid:
                        stats[reason] += 1
                        print(f"Skipping {transition_file.name}: {reason}")
                        continue

                    # Copy valid transition to output
                    copy_transition(transition, session_dir, output_dir)
                    stats['valid'] += 1

                except Exception as e:
                    print(f"Error processing {transition_file}: {e}")
                    continue

    return stats


def is_valid_transition(
    transition: dict,
    session_dir: Path,
    seen_hashes: set
) -> Tuple[bool, str]:
    """
    Check if a transition is valid and should be kept.

    Returns:
        (is_valid, reason) tuple
    """
    # Check for required fields
    if 'before_state' not in transition or 'after_state' not in transition:
        return False, 'missing_images'

    before_img = transition['before_state']['screenshot']
    after_img = transition['after_state']['screenshot']

    before_path = session_dir / before_img
    after_path = session_dir / after_img

    # Check if image files exist
    if not before_path.exists() or not after_path.exists():
        return False, 'missing_images'

    # Check if images are valid
    try:
        before_image = Image.open(before_path)
        after_image = Image.open(after_path)
        before_image.verify()
        after_image.verify()

        # Re-open after verify (verify closes the file)
        before_image = Image.open(before_path)
        after_image = Image.open(after_path)

    except Exception as e:
        return False, 'corrupt_images'

    # Check if before and after are identical (no actual transition)
    if images_are_identical(before_image, after_image):
        return False, 'identical_screenshots'

    # Check for timeout (took too long)
    if 'timing' in transition:
        duration_ms = transition['timing'].get('total_duration_ms', 0)
        if duration_ms > 5000:  # More than 5 seconds
            return False, 'timeouts'

    # Check stability score
    if 'quality' in transition:
        stability = transition['quality'].get('stability_score', 1.0)
        if stability < 0.7:
            return False, 'low_stability'

    # Check for duplicates (same before/after pair)
    transition_hash = compute_transition_hash(before_image, after_image)
    if transition_hash in seen_hashes:
        return False, 'duplicates'

    seen_hashes.add(transition_hash)

    return True, 'valid'


def images_are_identical(img1: Image.Image, img2: Image.Image) -> bool:
    """Check if two images are identical."""
    # Quick size check
    if img1.size != img2.size:
        return False

    # Convert to same mode if different
    if img1.mode != img2.mode:
        img1 = img1.convert('RGB')
        img2 = img2.convert('RGB')

    # Compare pixels (simple hash approach for speed)
    hash1 = hashlib.md5(img1.tobytes()).hexdigest()
    hash2 = hashlib.md5(img2.tobytes()).hexdigest()

    return hash1 == hash2


def compute_transition_hash(before: Image.Image, after: Image.Image) -> str:
    """Compute a hash for a transition (before + after images)."""
    combined = before.tobytes() + after.tobytes()
    return hashlib.md5(combined).hexdigest()


def copy_transition(
    transition: dict,
    source_dir: Path,
    output_dir: Path
) -> None:
    """
    Copy a valid transition to the output directory.

    Args:
        transition: Transition metadata dict
        source_dir: Source directory containing the transition
        output_dir: Output directory for cleaned data
    """
    # Create output directory structure
    app_id = transition['app']['bundle_id']
    session_id = source_dir.name

    output_session_dir = output_dir / app_id / session_id
    output_session_dir.mkdir(parents=True, exist_ok=True)

    # Copy transition JSON
    transition_id = transition['transition_id']
    json_filename = f"{transition_id.split('_')[-1]}.json"
    # TODO: Implement actual file copying
    # For now, just write the transition JSON
    with open(output_session_dir / json_filename, 'w') as f:
        json.dump(transition, f, indent=2)

    # Copy images
    before_img = transition['before_state']['screenshot']
    after_img = transition['after_state']['screenshot']

    # TODO: Implement image copying using shutil.copy2


def print_stats(stats: Dict[str, int]) -> None:
    """Print cleaning statistics."""
    print("\n=== Cleaning Statistics ===")
    print(f"Total transitions processed: {stats['total']}")
    print(f"Valid transitions: {stats['valid']} ({stats['valid']/stats['total']*100:.1f}%)")
    print(f"\nRemoved:")
    print(f"  Missing images: {stats['missing_images']}")
    print(f"  Corrupt images: {stats['corrupt_images']}")
    print(f"  Identical screenshots: {stats['identical_screenshots']}")
    print(f"  Duplicates: {stats['duplicates']}")
    print(f"  Timeouts: {stats['timeouts']}")
    print(f"  Low stability: {stats['low_stability']}")
    print("===========================\n")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python clean.py <raw_data_dir> <output_dir>")
        sys.exit(1)

    raw_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])

    if not raw_dir.exists():
        print(f"Error: {raw_dir} does not exist")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    stats = clean_transitions(raw_dir, out_dir)
    print_stats(stats)
