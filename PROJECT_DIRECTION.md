# iOS World: Project Direction

## The Big Idea

**The Question**: Can a vision-language model learn how iOS apps "should" behave, and then spot bugs by noticing when reality doesn't match expectations?

**The Approach**: Train a VLM on thousands of UI state transitions (before screenshot → action → after screenshot) to learn the patterns of how iOS apps respond to user interactions. Once the model has internalized these patterns, use it to detect bugs by feeding it transitions and seeing when its predictions diverge from reality.

**Why This Might Work**:

1. **iOS Consistency**: Unlike the wild west of web UIs, iOS has strong design patterns and consistent behavior across apps. Navigation bars work the same way. Buttons respond predictably. Text fields behave consistently. This consistency means patterns are learnable.

2. **Self-Supervised Learning**: We don't need labeled bug data to start. The model learns from normal app behavior, building an internal world model of "how iOS works." The training signal comes from the transitions themselves.

3. **Novel Approach for iOS**: Most mobile testing is either manual QA or brittle automation scripts. A learned world model could be more robust and could potentially catch subtle UI bugs that traditional testing misses.

4. **Scalable Data Collection**: With XCUITest, we can systematically explore apps in the simulator, capturing thousands of transitions without manual intervention.

## The Core Technical Idea

The model needs to learn two related but distinct tasks:

### 1. Inverse Dynamics Model
**Input**: (before_screen, after_screen)
**Output**: What action caused this transition?

This teaches the model to understand what *kinds* of changes different actions produce. Taps on buttons cause navigation. Swipes cause scrolling. Text entry causes keyboard appearance.

### 2. Forward Dynamics Model
**Input**: (before_screen, action)
**Output**: What will the screen look like after this action?

This is the heart of bug detection. Given a screen and an action, predict the next screen. During normal operation, predicted_screen ≈ actual_screen. But when there's a bug:

```
predicted_screen ≠ actual_screen → potential bug
```

The model expected something different to happen than what actually happened.

### Bug Detection Strategy

When the model makes a prediction, we get:
- The predicted next screen (image/embedding)
- A confidence score
- The actual next screen

Compare them:
- **High confidence + close match**: Normal behavior, no bug
- **High confidence + poor match**: Likely bug! The model was sure something should happen, but reality diverged
- **Low confidence + poor match**: Unusual but potentially valid UI (new pattern)
- **Low confidence + close match**: Model got lucky, keep learning

The key insight: **We're not looking for errors in the code directly. We're looking for violations of learned expectations about how iOS UIs behave.**

## What I Need to Validate

### Phase 1 Questions (Data Capture)
- Can I capture clean, high-quality transition data systematically?
- How much data is "enough"? (Hypothesis: ~10k transitions to start)
- What's the right granularity for actions?
- How do I handle transitions with animations/timing issues?
- Can I automatically filter out bad captures?

### Phase 2 Questions (Model Training)
- Which VLM base model should I use? (Options: LLaVA, GPT-4V fine-tune, CLIP-based)
- How do I represent actions in the model input?
- Can the model actually learn iOS patterns from transitions?
- Do I need separate models for inverse/forward dynamics, or one multi-task model?
- How much compute/time does training take?

### Phase 3 Questions (Bug Detection)
- Does prediction error actually correlate with bugs?
- How do I distinguish bugs from valid unusual UIs?
- Can I intentionally inject bugs and see if the model catches them?
- What kinds of bugs can this approach catch? What kinds can't it catch?
- How does the model perform on apps it wasn't trained on?

## Potentially Novel Ideas (For IP)

These are ideas that emerged from thinking about this problem that might be worth documenting for potential patents or papers:

### 1. World Model Approach for Mobile UI Testing
**Status**: 💡 Core Innovation

**Concept**: Using a learned world model (specifically, forward and inverse dynamics models from RL) for automated software testing in mobile UIs.

**Why Novel**: Traditional testing uses explicit assertions or visual regression testing. This uses learned expectations about UI behavior to detect anomalies without manual specification of what's "correct."

**Potential Claim**: A method for detecting UI bugs comprising: (1) collecting UI state transitions, (2) training a vision-language model to predict transitions, (3) using prediction error as a bug signal.

### 2. Confidence-Based Bug Routing
**Status**: 💡 New Idea

**Concept**: Use the model's prediction confidence to route findings:
- High confidence + mismatch → File as likely bug
- Low confidence + mismatch → Route to human for "is this valid?"
- Humans labeling edge cases creates a feedback loop to improve the model

**Why Novel**: Combines active learning with automated testing. The model knows what it doesn't know and asks for help strategically.

**Potential Claim**: A bug detection system that uses model uncertainty to prioritize human review and incorporate feedback.

### 3. Coverage-Driven UI Exploration with Steering Instructions
**Status**: 💡 New Idea

**Concept**: Instead of random UI exploration, use "steering instructions" (YAML files describing exploration strategies) to systematically cover UI patterns:
- "Tap all buttons on all screens"
- "Enter text in all text fields"
- "Navigate into and back from all list items"

**Why Novel**: Bridges the gap between random fuzzing and manual test scripts. Declarative exploration strategies that are app-agnostic.

**Potential Claim**: A method for systematic UI exploration using declarative coverage goals rather than scripted tests.

### 4. Two-Layer World Model: iOS-Universal + App-Specific
**Status**: 🔬 To Explore

**Concept**: Train two models:
- **Universal iOS Model**: Learns general iOS patterns (navigation, keyboards, alerts, etc.) from many apps
- **App-Specific Model**: Fine-tuned on a specific app to learn its unique behaviors

Bug detection combines both: "This violates iOS norms" vs "This violates this app's norms"

**Why Novel**: Hierarchical learned models for UI testing. Separates platform expectations from app expectations.

**Potential Claim**: A hierarchical bug detection system with platform-level and application-level learned models.

## Phases

### Phase 1: Data Capture (Current Focus)
**Goal**: Build a system to systematically capture 10,000+ high-quality UI transitions

**Tasks**:
- ✅ Design data format (JSON + images)
- 🔄 Build XCUITest capture agent in Swift
- 🔄 Create steering instructions for common patterns
- ⬜ Capture data from Settings app (simple, consistent)
- ⬜ Capture data from a few other system/simple apps
- ⬜ Build preprocessing pipeline to clean/prepare data
- ⬜ Validate data quality

**Success Criteria**: 10k clean transitions with good diversity of action types

### Phase 2: Model Training
**Goal**: Train a model that can predict screen transitions with reasonable accuracy

**Tasks**:
- Choose base model architecture
- Implement forward dynamics model
- Implement inverse dynamics model (maybe)
- Train on collected data
- Evaluate prediction accuracy
- Analyze what the model learned

**Success Criteria**: Model can predict simple transitions (button taps → navigation) with >70% accuracy

### Phase 3: Bug Detection Experiment
**Goal**: Validate that prediction error correlates with bugs

**Tasks**:
- Take a working app
- Intentionally inject UI bugs (wrong screen after tap, button does nothing, etc.)
- Run the model on buggy transitions
- Measure: Does the model flag bugs? What's the false positive rate?

**Success Criteria**: Model catches >80% of injected bugs with <20% false positives

### Phase 4: Write It Up
**Goal**: Document findings and determine next steps

**Tasks**:
- Write paper draft (academic or blog post)
- Document all novel ideas thoroughly
- If results are good: Consider patent applications
- If results are mixed: Document what worked and what didn't
- Open source the capture system (even if the model doesn't work well)

## Open Questions

### Data Collection
- **How much data is enough?** Start with 10k, but might need 50k+ for good generalization
- **Simulator vs real device?** Simulator is easier/faster, but might miss real-world behavior
- **How to handle animations?** Need smart waiting for UI stability
- **What about dynamic content?** (e.g., current time, random feeds) - might need to mask/ignore

### Model Architecture
- **Which VLM?** Options: LLaVA (open), GPT-4V (powerful but expensive), custom CLIP-based
- **How to encode actions?** Text description? Embedded vector? Element metadata?
- **Image resolution?** Higher = more detail, but slower/more expensive
- **Multi-task vs separate models?** One model for both forward/inverse, or two specialized models?

### Bug Detection
- **How to set the threshold?** What level of prediction error = bug?
- **How to handle valid edge cases?** Some unusual UIs are correct, not bugs
- **Can this catch non-visual bugs?** (Performance issues, crashes, data corruption) - probably not
- **Generalization to new apps?** Will a model trained on app A work on app B?

### Practical Deployment
- **Integration into CI/CD?** How would this fit into a real development workflow?
- **Speed?** Can it run fast enough to be useful?
- **Interpretability?** When it flags a bug, can it explain why?

## Success Criteria for the Overall Project

**Minimum Success**:
- Successfully capture and curate a dataset of 10k+ UI transitions
- Train a model that learns *something* about iOS patterns
- Publish the dataset and approach, even if bug detection doesn't work perfectly
- Document novel ideas for potential IP

**Good Success**:
- Model achieves >70% accuracy on predicting simple transitions
- Can detect at least 50% of intentionally injected bugs
- Identify what types of bugs this approach can/can't catch
- Write a credible paper/blog post about the approach

**Amazing Success**:
- Model achieves >85% accuracy on transition prediction
- Catches >80% of bugs with <20% false positives
- Demonstrates generalization to apps not in training set
- Results are good enough to pursue patents and/or academic publication
- System is practical enough that a team could actually use it

## Why This Project Matters

Even if the bug detection doesn't work perfectly, this project explores an interesting question: **Can we teach AI to understand user interfaces by showing it examples, the way humans learn?**

Humans don't need to read code to know something is broken in an app. We have intuitions about how UIs should behave. This project asks: Can we give AI similar intuitions?

If it works, it could:
- Make UI testing more robust and less brittle
- Catch subtle bugs that humans and traditional tests miss
- Reduce the manual effort in QA
- Provide a new way to ensure UI quality

If it doesn't work completely, we'll learn:
- What kinds of UI patterns are learnable from examples
- Where the limits of vision-language models are for this task
- What additional information/structure is needed beyond raw pixels

Either way, it's a worthwhile exploration at the intersection of ML and software engineering.

---

*This is a living document. Update it as ideas evolve and new questions emerge.*
