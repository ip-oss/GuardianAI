# Preprocessing Pipeline

Python scripts for cleaning and preparing captured iOS UI transition data for model training.

## Overview

The preprocessing pipeline has two main steps:

1. **Cleaning**: Remove bad/corrupt transitions
2. **Preparation**: Resize images, create train/val/test splits, build dataset index

## Prerequisites

```bash
cd preprocessing
pip install -r requirements.txt
```

## Usage

### Quick Start

Process all data in one command:

```bash
python process.py ../data/raw --output-dir ../data/processed
```

### Step-by-Step

#### 1. Clean the Data

Remove corrupt images, duplicates, and low-quality captures:

```bash
python clean.py ../data/raw ../data/cleaned
```

This will:
- Check that all image files exist and are valid
- Remove transitions where before/after screenshots are identical
- Filter out transitions with timeout errors
- Remove duplicates (same before/after pair)
- Discard transitions with low stability scores (<0.7)

#### 2. Prepare the Dataset

Resize images and create train/val/test splits:

```bash
python prepare.py ../data/cleaned ../data/processed
```

This will:
- Resize all images to 512x1024 (maintains aspect ratio with padding)
- Split data 80/10/10 (train/val/test) by default
- Create organized directory structure
- Generate `index.json` with dataset statistics

## Command-Line Options

### `process.py` (Main Script)

```bash
python process.py <raw_data_dir> [options]

Options:
  --output-dir PATH       Output directory (default: data/processed)
  --temp-dir PATH         Temp directory for cleaning (default: data/temp_cleaned)
  --target-size W H       Target image size (default: 512 1024)
  --split TRAIN VAL TEST  Split ratios (default: 0.8 0.1 0.1)
  --skip-clean            Skip cleaning step
  --skip-prepare          Skip preparation step
```

Examples:

```bash
# Custom image size
python process.py ../data/raw --target-size 384 768

# Custom split (70% train, 20% val, 10% test)
python process.py ../data/raw --split 0.7 0.2 0.1

# Only clean, don't prepare
python process.py ../data/raw --skip-prepare
```

## Output Structure

```
data/processed/
├── train/
│   ├── train_000000.json
│   ├── train_000000_before.png
│   ├── train_000000_after.png
│   └── ...
├── val/
│   └── ...
├── test/
│   └── ...
└── index.json
```

### Index File Format

```json
{
  "version": "1.0",
  "total_transitions": 10243,
  "splits": {
    "train": 8195,
    "val": 1024,
    "test": 1024
  },
  "apps": [
    {"bundle_id": "com.apple.Preferences", "transitions": 3421},
    {"bundle_id": "com.apple.mobilenotes", "transitions": 2156}
  ],
  "action_distribution": {
    "tap": 6543,
    "swipe": 1234,
    "type_text": 876
  }
}
```

## Cleaning Criteria

### What Gets Removed

| Issue | Reason |
|-------|--------|
| Missing images | Before or after screenshot file doesn't exist |
| Corrupt images | PNG file is invalid or can't be opened |
| Identical screenshots | Before and after are exactly the same (no UI change) |
| Duplicates | Same before/after pair already seen |
| Timeouts | Action took > 5 seconds (likely stuck/frozen) |
| Low stability | Stability score < 0.7 (UI was animating during capture) |

### What Gets Kept

- Both screenshots exist and are valid
- Screenshots show actual UI change
- Stability score >= 0.7
- Action completed in reasonable time (< 5s)
- Unique transition (not a duplicate)

## Image Processing

### Resizing Strategy

Images are resized to maintain aspect ratio:

1. Scale image to fit within target size
2. Pad with black bars to reach exact target size
3. Center the scaled image in the padded frame

Example:
```
Original: 1179 x 2556 (iPhone 15 Pro)
Target:   512 x 1024
Scaled:   472 x 1024 (maintains aspect ratio)
Padded:   512 x 1024 (20px black bars on sides)
```

### Why Resize?

- **Consistency**: All images same size for batching
- **Speed**: Smaller images = faster training
- **Memory**: Reduces GPU memory requirements
- **Still readable**: 512x1024 preserves UI details

## Statistics

After processing, you'll see output like:

```
=== Cleaning Statistics ===
Total transitions processed: 5000
Valid transitions: 4321 (86.4%)

Removed:
  Missing images: 12
  Corrupt images: 3
  Identical screenshots: 234
  Duplicates: 156
  Timeouts: 8
  Low stability: 266
===========================

=== Preparation Statistics ===
Total transitions: 4321

Splits:
  Train: 3457
  Val: 432
  Test: 432

Apps:
  com.apple.Preferences: 2345
  com.apple.mobilenotes: 1976

Action types:
  tap: 2876
  swipe: 654
  type_text: 432
  scroll: 234
  navigate_back: 125
===========================
```

## Tips

### Maximizing Valid Transitions

- Use consistent capture environment (simulator, same iOS version)
- Disable animations before capturing
- Use adequate stability wait times
- Avoid apps with dynamic content (news feeds, etc.)

### Debugging Low Valid Percentage

If you're getting < 70% valid transitions:

1. Check stability scores in raw data
2. Look for patterns in rejected transitions
3. Review capture code for timing issues
4. Ensure simulator is stable during capture

## Next Steps

After preprocessing:

1. Verify dataset quality (spot check some images)
2. Review `index.json` for distribution
3. Move to model training (`model/train.py`)

See `PROJECT_DIRECTION.md` for overall project phases.
