# iOS World: Learning iOS UI Behavior for Bug Detection

Can a vision-language model learn how iOS apps should behave, and detect bugs by noticing when reality doesn't match expectations?

## The Idea

Train a model on thousands of UI transitions (screenshot before → action → screenshot after) to learn patterns of iOS behavior. Once trained, use prediction error as a signal for potential bugs: if the model confidently expects one outcome but observes another, something might be wrong.

This is a prototype to explore whether this approach works.

## Project Status

**Current Phase**: Phase 1 - Data Collection

- ✅ Project structure created
- ✅ Data format designed
- ✅ Swift capture agent implemented
- ⬜ Data collection in progress
- ⬜ Preprocessing pipeline tested
- ⬜ Model training (Phase 2)
- ⬜ Bug detection experiments (Phase 3)

See `PROGRESS.md` for detailed status.

## Structure

```
├── PROJECT_DIRECTION.md    # The big picture: goals, approach, ideas
├── PROGRESS.md             # Current status and next steps
│
├── docs/                   # Documentation
│   ├── ideas.md            # Novel ideas for potential IP
│   ├── data-format.md      # Data structure specification
│   └── learnings.md        # What works, what doesn't
│
├── capture/                # Phase 1: Capture iOS UI transitions
│   ├── ios-agent/          # Swift/XCUITest capture system
│   ├── instructions/       # Reusable capture strategies (YAML)
│   └── scenarios/          # App-specific capture plans
│
├── preprocessing/          # Clean and prepare data for training
│   ├── process.py          # Main preprocessing script
│   ├── clean.py            # Data cleaning
│   └── prepare.py          # Dataset preparation
│
├── data/                   # Captured data (gitignored)
│   ├── raw/                # Raw captures from iOS simulator
│   └── processed/          # Cleaned and prepared for training
│
├── model/                  # Phase 2: Model architecture and training
│   ├── model.py            # Vision-language model
│   ├── train.py            # Training script
│   └── evaluate.py         # Evaluation script
│
├── experiments/            # Experiment tracking
│   └── results/            # Training runs and results
│
└── notebooks/              # Jupyter notebooks for exploration
```

## Quick Start

### Phase 1: Capture Data

1. **Build the capture agent**:
   ```bash
   cd capture/ios-agent
   swift build
   ```

2. **Create UI test target** in Xcode that uses the CaptureAgent package

3. **Run capture** on iOS Simulator targeting Settings app (see `capture/scenarios/settings_app.yaml`)

4. **Review captured data** in `data/raw/`

See `capture/README.md` for detailed instructions.

### Phase 2: Train Model (Not Started)

```bash
# Preprocess data
cd preprocessing
python process.py ../data/raw --output-dir ../data/processed

# Train model
cd ../model
python train.py --data-dir ../data/processed
```

See `model/README.md` for details.

## Core Technical Approach

### Forward Dynamics Model
**Input**: (before_screen, action)
**Output**: predicted_after_screen
**Use**: Predict what should happen, compare to what actually happens

### Inverse Dynamics Model
**Input**: (before_screen, after_screen)
**Output**: predicted_action
**Use**: Understand what kinds of changes different actions produce

### Bug Detection
```
If predicted_screen ≠ actual_screen AND confidence is high:
    → Potential bug (unexpected behavior)
```

## Key Documents

- **`PROJECT_DIRECTION.md`**: The north star - goals, approach, novel ideas, phases
- **`PROGRESS.md`**: Current status, what's done, what's next
- **`docs/ideas.md`**: Novel ideas worth noting (potential IP)
- **`docs/data-format.md`**: Complete data specification
- **Capture README**: `capture/README.md`
- **Preprocessing README**: `preprocessing/README.md`
- **Model README**: `model/README.md`

## Data Format

Each transition consists of:
- `transition_NNNN.json` - Metadata (action, timing, quality, accessibility tree)
- `transition_NNNN_before.png` - Screenshot before action
- `transition_NNNN_after.png` - Screenshot after action

See `docs/data-format.md` for full specification.

## Goals

### Primary Goal
Build a working prototype that can:
1. Capture UI transitions systematically
2. Train a model on those transitions
3. Detect intentionally injected bugs

### Secondary Goal
Document novel ideas for potential:
- Academic paper
- Patents
- Open source release

## Why This Might Work

1. **iOS Consistency**: Strong design patterns across apps
2. **Self-Supervised**: Learn from normal behavior, no bug labels needed
3. **Novel Approach**: Different from traditional mobile testing
4. **Scalable Data**: Can capture thousands of transitions automatically

## License

Apache 2.0 - See `LICENSE`

## Contributing

This is currently a personal research project. If the results are promising, it may be open-sourced for community contributions.

## Contact

See `PROJECT_DIRECTION.md` for more details on the approach and ideas.

---

*Built with curiosity about whether AI can learn to understand user interfaces the way humans do.*
