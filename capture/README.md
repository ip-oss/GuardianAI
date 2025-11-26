# Capture System

This directory contains the iOS UI transition capture system built with Swift and XCUITest.

## Overview

The capture system systematically explores iOS apps and records UI transitions:
- Before screenshot (PNG)
- Action performed (JSON metadata)
- After screenshot (PNG)
- Timing, quality metrics, accessibility tree

## Structure

```
capture/
├── ios-agent/          # Swift package for XCUITest-based capture
│   ├── Package.swift
│   └── Sources/CaptureAgent/
│       ├── Models.swift    # Data models (Transition, Action, etc.)
│       ├── Capture.swift   # Screenshot and accessibility capture
│       ├── Actions.swift   # Action execution (tap, swipe, type, etc.)
│       └── Agent.swift     # Main orchestrator (CaptureAgent)
│
├── instructions/       # Reusable capture strategies (YAML)
│   ├── basic_taps.yaml
│   ├── text_input.yaml
│   ├── navigation.yaml
│   └── scrolling.yaml
│
└── scenarios/          # App-specific capture plans
    └── settings_app.yaml
```

## Prerequisites

- macOS with Xcode 15+
- iOS Simulator (iOS 17+)
- Swift 5.9+

## Usage

### 1. Build the Swift Package

```bash
cd capture/ios-agent
swift build
```

### 2. Create a UI Test Target

The CaptureAgent package is designed to be used from within an XCUITest target. You'll need to:

1. Create an Xcode project or use an existing one
2. Add a UI Testing target
3. Import the CaptureAgent package
4. Write test cases that use the `CaptureAgent` class

Example test case:

```swift
import XCTest
import CaptureAgent

class SettingsAppCaptureTests: XCTestCase {
    func testCaptureSettingsApp() throws {
        let app = XCUIApplication(bundleIdentifier: "com.apple.Preferences")
        app.launch()

        let appInfo = AppInfo(
            bundleId: "com.apple.Preferences",
            name: "Settings",
            version: "1.0"
        )

        let outputDir = FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("captures")

        let agent = CaptureAgent(
            app: app,
            outputDirectory: outputDir,
            appInfo: appInfo
        )

        // Capture all button taps
        try agent.captureAllButtons(navigateBack: true)

        // Capture all cell taps
        try agent.captureAllCells(navigateBack: true)

        // Capture text input
        try agent.captureTextInput()

        // Capture scrolling
        try agent.captureScrolling()

        // Save session metadata
        try agent.saveSessionMetadata(
            strategy: "basic_exploration",
            instructionFile: "capture/instructions/basic_taps.yaml"
        )

        // Print summary
        agent.printSummary()
    }
}
```

### 3. Run the Tests

```bash
xcodebuild test \
    -project YourProject.xcodeproj \
    -scheme YourUITestScheme \
    -destination 'platform=iOS Simulator,name=iPhone 15 Pro'
```

## Capture Strategies

### Basic Taps (`basic_taps.yaml`)

Systematically taps all buttons and cells to explore navigation patterns.

**Best for**: Discovering app structure, testing navigation flows

### Text Input (`text_input.yaml`)

Tests all text fields with typing and clearing actions.

**Best for**: Forms, search bars, text-heavy interfaces

### Navigation (`navigation.yaml`)

Focuses on navigation patterns (push/pop, modals, tabs).

**Best for**: Mapping navigation graph, understanding app hierarchy

### Scrolling (`scrolling.yaml`)

Tests scrolling in lists, tables, and scroll views.

**Best for**: Long content, infinite scroll, pull-to-refresh

## Instruction Files

Instruction files are YAML documents that describe capture strategies. They define:
- Which UI elements to target
- Which actions to perform
- Coverage goals (how much to explore)
- Exclusions (elements to avoid)
- Expected transitions

See `instructions/basic_taps.yaml` for an example.

## Scenario Files

Scenario files describe app-specific capture plans. They combine multiple strategies for comprehensive coverage of a particular app.

See `scenarios/settings_app.yaml` for an example.

## Output Format

Captured data is saved in:

```
data/raw/<bundle_id>/<session_id>/
├── transition_0000.json
├── transition_0000_before.png
├── transition_0000_after.png
├── transition_0001.json
├── transition_0001_before.png
├── transition_0001_after.png
└── session_metadata.json
```

See `docs/data-format.md` for complete specification.

## Tips

### Getting High-Quality Captures

1. **Use the Simulator**: More consistent than real devices
2. **Disable animations**: Settings → Accessibility → Motion → Reduce Motion
3. **Use airplane mode**: Avoid network-dependent UI changes
4. **Start fresh**: Reset simulator between runs for consistency
5. **Wait for stability**: The capture system automatically waits for UI to settle

### Troubleshooting

**Element not found**:
- UI might have changed between finding and tapping
- Try increasing stability wait time
- Check if element requires scrolling to be visible

**Timeout errors**:
- Some actions are slow (e.g., loading screens)
- Increase timeout in capture settings
- May indicate app is stuck - investigate

**Images look wrong**:
- Check that animations are disabled
- Verify stability score in quality metrics
- May need longer wait times

## Next Steps

After capturing data:

1. Review captured transitions (spot check for quality)
2. Run preprocessing pipeline (`preprocessing/process.py`)
3. Move to Phase 2 (model training)

See `PROJECT_DIRECTION.md` for the overall project plan.
