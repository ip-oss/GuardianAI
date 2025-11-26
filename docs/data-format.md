# Data Format Specification

This document defines the structure of captured UI transition data for the iOS World project.

## Overview

Each captured transition consists of:
- **Metadata** (JSON file): Describes the transition, action, timing, app state
- **Before screenshot** (PNG file): UI state before the action
- **After screenshot** (PNG file): UI state after the action

All transitions are stored in a structured directory format for easy processing.

---

## File Organization

```
data/
├── raw/                                    # Raw captured data
│   ├── <app_bundle_id>/                   # e.g., com.apple.Preferences
│   │   ├── <session_id>/                  # e.g., 20250124_143022
│   │   │   ├── transition_0000.json
│   │   │   ├── transition_0000_before.png
│   │   │   ├── transition_0000_after.png
│   │   │   ├── transition_0001.json
│   │   │   ├── transition_0001_before.png
│   │   │   ├── transition_0001_after.png
│   │   │   └── ...
│   │   │   └── session_metadata.json      # Session-level info
│
└── processed/                              # Cleaned and prepared data
    ├── train/
    ├── val/
    ├── test/
    └── index.json                          # Dataset index
```

### Naming Convention

- `transition_NNNN.json` - Transition metadata (zero-padded 4-digit sequence)
- `transition_NNNN_before.png` - Screenshot before action
- `transition_NNNN_after.png` - Screenshot after action
- `session_metadata.json` - Information about the capture session

---

## JSON Schema: Transition

Each transition is described by a JSON file with the following structure:

```json
{
  "version": "1.0",
  "transition_id": "com.apple.Preferences_20250124_143022_0042",
  "timestamp": "2025-01-24T14:30:45Z",

  "app": {
    "bundle_id": "com.apple.Preferences",
    "name": "Settings",
    "version": "1.0"
  },

  "environment": {
    "device": "iPhone 15 Pro",
    "os_version": "17.2",
    "screen_size": {
      "width": 1179,
      "height": 2556
    },
    "scale": 3.0,
    "orientation": "portrait"
  },

  "action": {
    "type": "tap",
    "element": {
      "type": "button",
      "label": "Wi-Fi",
      "frame": {
        "x": 0,
        "y": 150,
        "width": 393,
        "height": 44
      },
      "traits": ["button"],
      "identifier": "Wi-Fi",
      "value": null
    },
    "coordinates": {
      "x": 196,
      "y": 172
    },
    "metadata": {
      "strategy": "tap_all_buttons",
      "instruction_file": "basic_taps.yaml"
    }
  },

  "before_state": {
    "screenshot": "transition_0042_before.png",
    "timestamp": "2025-01-24T14:30:44.823Z",
    "accessibility_tree": {
      "type": "NavigationBar",
      "label": "Settings",
      "children": [
        {
          "type": "StaticText",
          "label": "Settings",
          "frame": { "x": 0, "y": 50, "width": 393, "height": 44 }
        }
      ]
    },
    "visible_elements": 24,
    "interactive_elements": 12
  },

  "after_state": {
    "screenshot": "transition_0042_after.png",
    "timestamp": "2025-01-24T14:30:45.156Z",
    "accessibility_tree": {
      "type": "NavigationBar",
      "label": "Wi-Fi",
      "children": [
        {
          "type": "Button",
          "label": "Settings",
          "frame": { "x": 10, "y": 50, "width": 60, "height": 44 }
        }
      ]
    },
    "visible_elements": 18,
    "interactive_elements": 8
  },

  "timing": {
    "capture_start": "2025-01-24T14:30:44.800Z",
    "action_executed": "2025-01-24T14:30:44.850Z",
    "stability_achieved": "2025-01-24T14:30:45.100Z",
    "capture_end": "2025-01-24T14:30:45.200Z",
    "total_duration_ms": 400
  },

  "quality": {
    "stability_score": 1.0,
    "completeness": 1.0,
    "notes": []
  }
}
```

---

## Field Descriptions

### Top Level

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | Data format version (for backward compatibility) |
| `transition_id` | string | Unique identifier: `{bundle_id}_{session_id}_{sequence}` |
| `timestamp` | ISO 8601 | When the transition was captured |

### App

| Field | Type | Description |
|-------|------|-------------|
| `bundle_id` | string | App's bundle identifier (e.g., `com.apple.Preferences`) |
| `name` | string | Human-readable app name |
| `version` | string | App version number |

### Environment

| Field | Type | Description |
|-------|------|-------------|
| `device` | string | Device model (e.g., "iPhone 15 Pro") |
| `os_version` | string | iOS version (e.g., "17.2") |
| `screen_size` | object | Physical screen dimensions (width, height in pixels) |
| `scale` | number | Screen scale factor (1.0, 2.0, or 3.0) |
| `orientation` | string | "portrait" or "landscape" |

### Action

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Action type (see Action Types table below) |
| `element` | object | Target element information (see Element structure) |
| `coordinates` | object | Screen coordinates where action occurred (x, y) |
| `metadata` | object | Additional context (strategy used, instruction file, etc.) |

### Action Types

| Type | Description | Parameters |
|------|-------------|------------|
| `tap` | Single tap on an element | `element`, `coordinates` |
| `double_tap` | Double tap on an element | `element`, `coordinates` |
| `long_press` | Press and hold | `element`, `coordinates`, `duration` |
| `swipe` | Swipe gesture | `start`, `end`, `direction` |
| `scroll` | Scroll container | `element`, `direction`, `distance` |
| `type_text` | Enter text in field | `element`, `text` |
| `clear_text` | Clear text from field | `element` |
| `navigate_back` | Tap back button | `element` |
| `dismiss_keyboard` | Hide keyboard | - |
| `pull_to_refresh` | Pull down to refresh | `element` |
| `pinch` | Pinch zoom gesture | `element`, `scale` |
| `rotate` | Rotation gesture | `element`, `rotation` |

### Element

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | XCUIElement type (button, cell, textField, etc.) |
| `label` | string | Accessibility label |
| `frame` | object | Position and size (x, y, width, height) |
| `traits` | array | Accessibility traits |
| `identifier` | string | Accessibility identifier |
| `value` | string/null | Current value (for text fields, sliders, etc.) |

### State (before_state / after_state)

| Field | Type | Description |
|-------|------|-------------|
| `screenshot` | string | Filename of screenshot PNG |
| `timestamp` | ISO 8601 | When screenshot was captured |
| `accessibility_tree` | object | Hierarchical UI structure (simplified) |
| `visible_elements` | number | Count of visible elements |
| `interactive_elements` | number | Count of interactive elements |

### Timing

| Field | Type | Description |
|-------|------|-------------|
| `capture_start` | ISO 8601 | When capture began |
| `action_executed` | ISO 8601 | When action was performed |
| `stability_achieved` | ISO 8601 | When UI stopped animating |
| `capture_end` | ISO 8601 | When capture completed |
| `total_duration_ms` | number | Total time from start to end |

### Quality

| Field | Type | Description |
|-------|------|-------------|
| `stability_score` | number | 0.0-1.0: How stable the UI was (1.0 = perfect) |
| `completeness` | number | 0.0-1.0: How complete the capture is (1.0 = all data present) |
| `notes` | array | Any issues or warnings during capture |

---

## Session Metadata

Each capture session also produces a `session_metadata.json` file:

```json
{
  "session_id": "20250124_143022",
  "app": {
    "bundle_id": "com.apple.Preferences",
    "name": "Settings",
    "version": "1.0"
  },
  "start_time": "2025-01-24T14:30:22Z",
  "end_time": "2025-01-24T14:45:18Z",
  "strategy": "basic_taps",
  "instruction_file": "capture/instructions/basic_taps.yaml",
  "transitions_captured": 156,
  "transitions_failed": 3,
  "device": "iPhone 15 Pro Simulator",
  "os_version": "17.2"
}
```

---

## Quality Criteria

Not all captured transitions are useful for training. Here are criteria for filtering:

### Good Transition ✅

- Both before and after screenshots exist and are valid PNG files
- Screenshots are different (UI actually changed)
- `stability_score >= 0.9` (UI was stable when captured)
- `completeness == 1.0` (all required data present)
- Action completed successfully
- Total duration < 2 seconds (not stuck/frozen)
- Element metadata is complete

### Bad Transition ❌

**Discard if**:
- Missing before or after screenshot
- Corrupt image files
- Screenshots are identical (no change)
- `stability_score < 0.7` (UI was animating during capture)
- Timeout during action (took > 5 seconds)
- App crashed during transition
- Duplicate of another transition (same before/after)

**Flag for review if**:
- Unusual timing (very fast < 50ms or very slow > 1s)
- Low stability score (0.7-0.9)
- Missing element metadata
- Unexpected UI changes (e.g., alert appeared)

---

## Image Format

### Screenshots

- **Format**: PNG (lossless)
- **Resolution**: Native device resolution (e.g., 1179x2556 for iPhone 15 Pro)
- **Color**: RGB, 24-bit
- **Compression**: PNG default (lossless)

### Processed Images

For model training, images may be:
- Resized to a standard size (e.g., 512x1024)
- Padded to maintain aspect ratio
- Normalized (pixel values 0-1)

These transformations are done in the preprocessing step, leaving raw data untouched.

---

## Dataset Index

After preprocessing, a `processed/index.json` file provides quick access to the dataset:

```json
{
  "version": "1.0",
  "created": "2025-01-24T15:00:00Z",
  "total_transitions": 10243,
  "splits": {
    "train": 8195,
    "val": 1024,
    "test": 1024
  },
  "apps": [
    {
      "bundle_id": "com.apple.Preferences",
      "transitions": 3421
    },
    {
      "bundle_id": "com.apple.mobilenotes",
      "transitions": 2156
    }
  ],
  "action_distribution": {
    "tap": 6543,
    "swipe": 1234,
    "type_text": 876,
    "scroll": 654,
    "navigate_back": 543,
    "other": 393
  }
}
```

---

## Notes

- **Extensibility**: The JSON format is designed to be extended. Add new fields without breaking old parsers by checking the `version` field.
- **Accessibility Tree Depth**: Full trees can be very large. We store a simplified version (max depth 3) in the main transition file. Full trees can be stored separately if needed.
- **Privacy**: Be careful with screenshots of apps that might contain user data. Use system apps (Settings, Notes, etc.) or demo apps for data collection.
- **Storage**: 10k transitions ≈ 20k PNGs + JSON ≈ 5-10 GB depending on resolution.

---

*This is a living document. Update as the data format evolves.*
