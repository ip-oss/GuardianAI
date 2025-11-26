# Progress Tracker

Current status, completed tasks, and what's next for the iOS World project.

## Current Phase: Phase 1 - Data Collection

**Goal**: Capture 10,000+ high-quality UI transitions from iOS apps

**Status**: Initial setup complete, ready to start capturing

---

## Completed ✅

### Project Setup (January 2025)
- [x] Created project structure
- [x] Wrote PROJECT_DIRECTION.md (north star document)
- [x] Designed data format (see docs/data-format.md)
- [x] Created documentation structure

### Capture System
- [x] Designed Swift package structure
- [x] Implemented data models (Models.swift)
- [x] Implemented screen capture (Capture.swift)
- [x] Implemented action executor (Actions.swift)
- [x] Implemented main agent orchestrator (Agent.swift)
- [x] Created instruction YAML files (basic_taps, text_input, navigation, scrolling)
- [x] Created Settings app scenario

### Preprocessing Pipeline
- [x] Implemented data cleaning (clean.py)
- [x] Implemented dataset preparation (prepare.py)
- [x] Created main processing script (process.py)
- [x] Wrote preprocessing documentation

### Model Scaffolding
- [x] Created model placeholder files
- [x] Wrote model README with architecture plans
- [x] Set up experiments directory structure

### Documentation
- [x] Wrote capture README
- [x] Wrote preprocessing README
- [x] Wrote model README
- [x] Wrote experiments README
- [x] Created docs/ideas.md for potential IP
- [x] Created docs/learnings.md for observations

---

## In Progress 🔄

### Data Collection
- [ ] Create Xcode UI test project
- [ ] Integrate CaptureAgent Swift package
- [ ] Run initial capture on Settings app
- [ ] Review captured data quality
- [ ] Adjust capture parameters based on quality
- [ ] Capture 10k+ transitions

---

## Up Next ⬜

### Phase 1 Completion
- [ ] Run preprocessing pipeline on captured data
- [ ] Analyze dataset statistics
- [ ] Validate data quality (manual spot checks)
- [ ] Document learnings from data collection

### Phase 2: Model Training
- [ ] Choose base vision encoder (ResNet vs CLIP vs ViT)
- [ ] Implement model architecture
- [ ] Implement training loop
- [ ] Run baseline experiment
- [ ] Evaluate and iterate
- [ ] Document model performance

### Phase 3: Bug Detection
- [ ] Create test app with injected bugs
- [ ] Run model on buggy transitions
- [ ] Measure detection accuracy
- [ ] Tune confidence thresholds
- [ ] Document what types of bugs can/can't be detected

### Phase 4: Write It Up
- [ ] Analyze all results
- [ ] Write paper draft (or blog post)
- [ ] Document novel ideas for patents
- [ ] Decide on open source release
- [ ] Share findings

---

## Blockers & Issues

*None currently*

---

## Log

### 2025-01-24: Project Kickoff
- Created entire project structure
- Implemented Swift capture agent
- Wrote all documentation
- Ready to start data collection

**Next up**: Create Xcode project and run first capture session on Settings app

---

## Metrics to Track

### Data Collection
- Total transitions captured: 0 / 10,000
- Quality rate: TBD
- Apps captured: 0
- Capture sessions: 0

### Model Training (Phase 2)
- Training accuracy: TBD
- Validation accuracy: TBD
- Test accuracy: TBD
- Best model: TBD

### Bug Detection (Phase 3)
- True positive rate: TBD
- False positive rate: TBD
- Types of bugs detected: TBD

---

## Decision Log

Decisions made and their rationale:

### 2025-01-24: Data Format
**Decision**: Use JSON + PNG files rather than a database
**Rationale**: Simpler, more portable, easier to inspect manually

### 2025-01-24: Start with Settings App
**Decision**: Use iOS Settings app for initial data collection
**Rationale**: Deterministic, consistent UI, no network dependencies, safe to test

### 2025-01-24: Image Size 512x1024
**Decision**: Resize images to 512x1024 for training
**Rationale**: Balance between preserving UI detail and training speed

### 2025-01-24: Swift Package for Capture
**Decision**: Use Swift package rather than standalone app
**Rationale**: More reusable, can be imported into any UI test target

---

## Questions to Answer

Track open questions and when they get answered:

- ❓ How much data is actually needed for good performance?
- ❓ Which VLM architecture works best for UI prediction?
- ❓ What types of bugs can this approach detect?
- ❓ Does the model generalize to apps it wasn't trained on?
- ❓ What's the false positive rate in practice?
- ❓ Can we distinguish bugs from valid unusual UIs?

---

*Update this file regularly as progress is made*
