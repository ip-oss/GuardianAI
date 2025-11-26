"""
Dataset preparation module for iOS World project.
Resizes images, splits data, and creates index files.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import random


def prepare_dataset(
    cleaned_data_dir: Path,
    output_dir: Path,
    target_size: Tuple[int, int] = (512, 1024),
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> Dict:
    """
    Prepare dataset for training by resizing images and creating splits.

    Args:
        cleaned_data_dir: Directory with cleaned transition data
        output_dir: Directory to write processed data
        target_size: Target image size (width, height)
        split_ratios: Train/val/test split ratios (must sum to 1.0)

    Returns:
        Dictionary with preparation statistics
    """
    assert sum(split_ratios) == 1.0, "Split ratios must sum to 1.0"

    stats = {
        'total_transitions': 0,
        'train': 0,
        'val': 0,
        'test': 0,
        'apps': {},
        'action_types': {},
    }

    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    # Collect all transitions grouped by app
    app_transitions = collect_transitions_by_app(cleaned_data_dir)

    # Process each app
    for app_id, transitions in app_transitions.items():
        print(f"Processing {app_id}: {len(transitions)} transitions")

        # Shuffle transitions for random split
        random.shuffle(transitions)

        # Split transitions
        n_total = len(transitions)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])

        splits = {
            'train': transitions[:n_train],
            'val': transitions[n_train:n_train + n_val],
            'test': transitions[n_train + n_val:],
        }

        # Process each split
        for split_name, split_transitions in splits.items():
            process_split(
                split_transitions,
                output_dir / split_name,
                target_size,
                stats,
                split_name
            )

        stats['apps'][app_id] = len(transitions)

    # Create index file
    create_index(output_dir, stats)

    return stats


def collect_transitions_by_app(data_dir: Path) -> Dict[str, List[Tuple[Path, dict]]]:
    """
    Collect all transitions organized by app.

    Returns:
        Dictionary mapping app_id to list of (session_dir, transition_dict) tuples
    """
    app_transitions = {}

    for app_dir in data_dir.iterdir():
        if not app_dir.is_dir():
            continue

        app_id = app_dir.name
        transitions = []

        # Collect all transitions for this app
        for session_dir in app_dir.iterdir():
            if not session_dir.is_dir():
                continue

            for transition_file in session_dir.glob('*.json'):
                try:
                    with open(transition_file, 'r') as f:
                        transition = json.load(f)
                    transitions.append((session_dir, transition))
                except Exception as e:
                    print(f"Error loading {transition_file}: {e}")

        if transitions:
            app_transitions[app_id] = transitions

    return app_transitions


def process_split(
    transitions: List[Tuple[Path, dict]],
    output_dir: Path,
    target_size: Tuple[int, int],
    stats: Dict,
    split_name: str
) -> None:
    """
    Process transitions for a specific split.
    """
    for idx, (session_dir, transition) in enumerate(transitions):
        try:
            # Generate new filename
            new_id = f"{split_name}_{idx:06d}"

            # Process and copy images
            before_img = session_dir / transition['before_state']['screenshot']
            after_img = session_dir / transition['after_state']['screenshot']

            before_output = output_dir / f"{new_id}_before.png"
            after_output = output_dir / f"{new_id}_after.png"

            # Resize and save images
            resize_and_save(before_img, before_output, target_size)
            resize_and_save(after_img, after_output, target_size)

            # Update transition metadata with new filenames
            transition['before_state']['screenshot'] = before_output.name
            transition['after_state']['screenshot'] = after_output.name

            # Save transition JSON
            json_output = output_dir / f"{new_id}.json"
            with open(json_output, 'w') as f:
                json.dump(transition, f, indent=2)

            # Update stats
            stats['total_transitions'] += 1
            stats[split_name] += 1

            action_type = transition['action']['type']
            stats['action_types'][action_type] = stats['action_types'].get(action_type, 0) + 1

        except Exception as e:
            print(f"Error processing transition {idx}: {e}")


def resize_and_save(
    input_path: Path,
    output_path: Path,
    target_size: Tuple[int, int]
) -> None:
    """
    Resize an image to target size with padding to maintain aspect ratio.

    Args:
        input_path: Path to input image
        output_path: Path to save resized image
        target_size: Target (width, height)
    """
    img = Image.open(input_path)

    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize with padding to maintain aspect ratio
    img_resized = resize_and_pad(img, target_size)

    # Save
    img_resized.save(output_path, 'PNG')


def resize_and_pad(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """
    Resize image to fit target size, padding to maintain aspect ratio.

    Args:
        img: PIL Image
        target_size: Target (width, height)

    Returns:
        Resized and padded image
    """
    target_width, target_height = target_size

    # Calculate scaling factor to fit within target size
    width_ratio = target_width / img.width
    height_ratio = target_height / img.height
    scale = min(width_ratio, height_ratio)

    new_width = int(img.width * scale)
    new_height = int(img.height * scale)

    # Resize
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create new image with padding
    new_img = Image.new('RGB', target_size, (0, 0, 0))  # Black padding

    # Paste resized image centered
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_img.paste(img_resized, (paste_x, paste_y))

    return new_img


def create_index(output_dir: Path, stats: Dict) -> None:
    """
    Create an index file for the dataset.
    """
    index = {
        'version': '1.0',
        'total_transitions': stats['total_transitions'],
        'splits': {
            'train': stats['train'],
            'val': stats['val'],
            'test': stats['test'],
        },
        'apps': [
            {'bundle_id': app_id, 'transitions': count}
            for app_id, count in stats['apps'].items()
        ],
        'action_distribution': stats['action_types'],
    }

    with open(output_dir / 'index.json', 'w') as f:
        json.dump(index, f, indent=2)

    print(f"\nCreated index file: {output_dir / 'index.json'}")


def print_stats(stats: Dict) -> None:
    """Print preparation statistics."""
    print("\n=== Preparation Statistics ===")
    print(f"Total transitions: {stats['total_transitions']}")
    print(f"\nSplits:")
    print(f"  Train: {stats['train']}")
    print(f"  Val: {stats['val']}")
    print(f"  Test: {stats['test']}")
    print(f"\nApps:")
    for app_id, count in stats['apps'].items():
        print(f"  {app_id}: {count}")
    print(f"\nAction types:")
    for action_type, count in sorted(stats['action_types'].items(), key=lambda x: -x[1]):
        print(f"  {action_type}: {count}")
    print("==============================\n")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python prepare.py <cleaned_data_dir> <output_dir>")
        sys.exit(1)

    cleaned_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])

    if not cleaned_dir.exists():
        print(f"Error: {cleaned_dir} does not exist")
        sys.exit(1)

    # Set random seed for reproducibility
    random.seed(42)

    stats = prepare_dataset(cleaned_dir, out_dir)
    print_stats(stats)
