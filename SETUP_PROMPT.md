# iOS World - Project Setup Prompt for Claude Code

## Context

I'm building a prototype to test an idea: Can a vision-language model learn how iOS apps behave and detect bugs by predicting what should happen next?

The concept:
1. Capture screen transitions from iOS apps (before screenshot → action → after screenshot)
2. Train a VLM to predict what happens when you take an action
3. When prediction ≠ reality, that might be a bug

I want to build enough to validate whether this approach works, document what I learn, and potentially write a paper or file patents if the results are interesting.

## Project Goals

**Primary Goal**: Build a working prototype that can:
- Capture UI transitions from iOS apps systematically
- Train a model on those transitions
- See if the model can detect intentionally injected bugs

**Secondary Goal**: Document novel ideas along the way for potential IP:
- The overall approach (world model for UI bug detection)
- Any clever solutions to problems I encounter
- What works and what doesn't

## Instructions for Claude Code

Please create the following project structure in the current directory:

### Project Structure
```
ios-world/
├── README.md                      # What this project is, how to run it
├── PROJECT_DIRECTION.md           # The big picture - goals, approach, ideas
├── PROGRESS.md                    # What's done, what's next, learnings
├── LICENSE                        # Apache 2.0
│
├── docs/
│   ├── ideas.md                   # Novel ideas worth noting (potential IP)
│   ├── data-format.md             # How transition data is structured
│   └── learnings.md               # What worked, what didn't, surprises
│
├── capture/                       # Phase 1: Get data from iOS apps
│   ├── README.md                  # How to use the capture system
│   ├── ios-agent/                 # Swift/XCUITest code
│   │   ├── Package.swift
│   │   └── Sources/
│   │       └── CaptureAgent/
│   │           ├── Agent.swift
│   │           ├── Capture.swift
│   │           ├── Actions.swift
│   │           └── Models.swift
│   │
│   ├── instructions/              # What to capture (YAML files)
│   │   ├── basic_taps.yaml
│   │   ├── text_input.yaml
│   │   ├── navigation.yaml
│   │   └── scrolling.yaml
│   │
│   └── scenarios/                 # App-specific capture plans
│       └── settings_app.yaml
│
├── preprocessing/                 # Clean and prepare captured data
│   ├── README.md
│   ├── requirements.txt
│   ├── process.py                 # Main script
│   ├── clean.py                   # Remove bad data
│   └── prepare.py                 # Format for training
│
├── data/                          # Captured and processed data (gitignored)
│   ├── raw/
│   └── processed/
│
├── model/                         # Phase 2: The actual model
│   ├── README.md
│   ├── requirements.txt
│   ├── model.py                   # Model architecture
│   ├── train.py                   # Training script
│   └── evaluate.py                # Test the model
│
├── experiments/                   # Track what I try
│   ├── README.md
│   └── results/
│
└── notebooks/                     # Jupyter notebooks for exploration
    └── exploration.ipynb
```

### Key Files to Create

#### 1. PROJECT_DIRECTION.md

This should be the north star document. Write it conversationally but thoroughly with these sections:

**The Big Idea**
- Question: Can AI learn how iOS apps "should" behave, then spot bugs by noticing when reality doesn't match expectations?
- Approach: Train a VLM on UI transitions to learn patterns
- Why this might work: iOS consistency, self-supervised learning, novel approach for iOS

**The Core Technical Idea**
- Inverse Dynamics: (before_screen, after_screen) → what action caused this?
- Forward Dynamics: (before_screen, action) → what will the screen look like?
- Bug Detection: predicted_screen ≠ actual_screen → potential bug

**What I Need to Validate**
- Can I capture enough good data?
- Can the model learn iOS patterns?
- Does prediction error correlate with bugs?

**Potentially Novel Ideas (For IP)**
1. World model approach for mobile UI testing
2. Confidence-based routing for bug detection
3. Steering instructions for systematic capture
4. Two-layer model: iOS-universal + app-specific

**Phases**
- Phase 1: Data Capture (current)
- Phase 2: Model Training
- Phase 3: Bug Detection Experiment
- Phase 4: Write It Up

**Open Questions**
- How much data is enough?
- Which VLM base model?
- How to handle unusual but valid UIs?
- Simulator vs real device?

#### 2. docs/ideas.md

Track novel ideas that emerge. Structure as:

**Core Innovation**: Self-supervised world model for mobile UI bug detection

**Idea 1: Confidence-Based Bug Routing**
- Concept, why novel, potential claim

**Idea 2: Coverage-Driven UI Exploration**  
- Concept, why novel, potential claim

**Idea 3: iOS Version Adaptation**
- Concept, status

**Ideas to Explore Later**
- Placeholder for future ideas

Include status markers: 💡 New idea, 🔬 Testing, ✅ Validated, ❌ Didn't work

#### 3. docs/data-format.md

Document the transition data structure:
- Full JSON schema for a transition
- Action types table with descriptions
- File organization structure
- Quality criteria (good vs bad transitions)

#### 4. Swift Package (capture/ios-agent/)

Create a complete Swift package with:

**Package.swift**: Configure for iOS 17+, XCTest dependency

**Models.swift**: Define all data structures
- Transition, AppInfo, Environment, ScreenState, Action, ElementInfo, Timing
- ActionType enum with all action types
- Make everything Codable

**Capture.swift**: ScreenCapture class
- captureState() → screenshot + accessibility tree
- captureAccessibilityTree() → extract element hierarchy
- waitForStability() → detect when animations complete

**Actions.swift**: ActionExecutor class  
- execute() → perform XCUITest actions
- findElements() → query elements by type
- findInteractiveElements() → get all tappable elements
- elementInfo() → extract element metadata

**Agent.swift**: CaptureAgent class (main orchestrator)
- captureTransition() → full before→action→after capture
- captureAllButtons() → systematic button tapping
- captureAllCells() → systematic cell tapping
- captureTextInput() → text field interactions
- getSummary() → capture session stats

Include comprehensive comments explaining each component's purpose.

#### 5. Instruction YAML Files

Create these instruction files:

**basic_taps.yaml**
- Target buttons and cells
- Tap action
- Navigate back if needed

**text_input.yaml**
- Target text fields
- Sequence: tap → type → clear

**navigation.yaml**
- Push/pop via cells and back button
- Modal present/dismiss

**scrolling.yaml**
- Scroll up/down
- Pull to refresh

#### 6. Python Preprocessing

Create these Python modules:

**requirements.txt**: pillow, numpy, tqdm

**process.py**: Main script
- Argument parsing
- Call clean then prepare
- Print progress

**clean.py**: Data cleaning
- clean_transitions() function
- Remove: missing images, corrupt images, duplicates, timeouts
- Return stats dict

**prepare.py**: Dataset preparation
- prepare_dataset() function
- Resize images to consistent size
- Split by app (80/10/10)
- Create index files
- resize_and_pad() helper

#### 7. README.md

Main project README with:
- Brief explanation of the idea
- Project status (Phase 1: Data Collection)
- Structure overview
- Getting started (prerequisites, capturing data, processing data)
- Links to other docs

#### 8. PROGRESS.md

Simple progress tracker with:
- Current phase
- Checklist of tasks (done, in progress, up next)
- Log section for dated entries

#### 9. .gitignore

Ignore:
- data/
- *.pyc, __pycache__/
- .DS_Store
- *.xcuserstate
- .build/

## What I Need From You

1. Create all files and directories listed above
2. Write full implementations for the Swift code (with TODOs for complex parts)
3. Write full implementations for the Python code
4. Write complete documentation files (PROJECT_DIRECTION.md, ideas.md, data-format.md)
5. After creating everything, show me:
   - The complete PROJECT_DIRECTION.md
   - The complete docs/ideas.md
   - A summary of what was created
   - Next steps for starting data capture

## Important Notes

- This is an experimental prototype project
- Prioritize clarity and documentation over production-quality code
- Mark areas that need more work with TODO comments
- Swift package should target iOS 17+ / Xcode 15+
- Python should use type hints and be 3.10+ compatible
- All code should have clear comments explaining what it does

Ready? Please create the complete project structure now.