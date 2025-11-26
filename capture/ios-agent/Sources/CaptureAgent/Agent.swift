import Foundation
import XCTest

/// Main orchestrator for capturing UI transitions
public class CaptureAgent {
    private let app: XCUIApplication
    private let screenCapture: ScreenCapture
    private let actionExecutor: ActionExecutor
    private let outputDirectory: URL

    private var sessionId: String
    private var sequenceNumber: Int = 0
    private var summary: CaptureSummary = CaptureSummary()

    // App and environment info
    private let appInfo: AppInfo
    private let environment: Environment

    /// Initialize a new CaptureAgent
    /// - Parameters:
    ///   - app: The XCUIApplication to capture from
    ///   - outputDirectory: Directory where captures will be saved
    ///   - appInfo: Information about the app being tested
    public init(
        app: XCUIApplication,
        outputDirectory: URL,
        appInfo: AppInfo
    ) {
        self.app = app
        self.outputDirectory = outputDirectory
        self.appInfo = appInfo
        self.screenCapture = ScreenCapture(app: app)
        self.actionExecutor = ActionExecutor(app: app)

        // Generate session ID
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        self.sessionId = formatter.string(from: Date())

        // Capture environment info
        self.environment = Self.captureEnvironment()

        // Create session directory
        try? FileManager.default.createDirectory(
            at: sessionDirectory,
            withIntermediateDirectories: true
        )
    }

    // MARK: - Computed Properties

    private var sessionDirectory: URL {
        return outputDirectory
            .appendingPathComponent(appInfo.bundleId)
            .appendingPathComponent(sessionId)
    }

    // MARK: - Environment Capture

    private static func captureEnvironment() -> Environment {
        let device = "iPhone Simulator" // TODO: Get actual device name
        let osVersion = ProcessInfo.processInfo.operatingSystemVersionString

        // Get screen size
        let screenBounds = XCUIScreen.main.screenshot().image.size
        let screenSize = Size(
            width: Double(screenBounds.width),
            height: Double(screenBounds.height)
        )

        return Environment(
            device: device,
            osVersion: osVersion,
            screenSize: screenSize,
            scale: Double(XCUIScreen.main.scale),
            orientation: "portrait" // TODO: Detect actual orientation
        )
    }

    // MARK: - Transition Capture

    /// Capture a complete transition: before state → action → after state
    /// - Parameters:
    ///   - action: The action to perform
    ///   - metadata: Additional metadata for this transition
    /// - Returns: The captured Transition, or nil if capture failed
    @discardableResult
    public func captureTransition(
        action: Action,
        metadata: [String: String]? = nil
    ) throws -> Transition {
        let captureStart = Date()

        // Capture before state
        let beforeState = try screenCapture.captureState(
            sequenceNumber: sequenceNumber,
            suffix: "before",
            in: sessionDirectory
        )

        // Wait for stability
        let stabilityScore = screenCapture.waitForStability(timeout: 2.0)

        // Execute action
        let actionExecuted = Date()
        try actionExecutor.execute(action)

        // Wait for UI to stabilize after action
        let stabilityAfter = screenCapture.waitForStability(timeout: 2.0)
        let stabilityAchieved = Date()

        // Capture after state
        let afterState = try screenCapture.captureState(
            sequenceNumber: sequenceNumber,
            suffix: "after",
            in: sessionDirectory
        )

        let captureEnd = Date()

        // Build transition
        let transitionId = "\(appInfo.bundleId)_\(sessionId)_\(String(format: "%04d", sequenceNumber))"

        let timing = Timing(
            captureStart: captureStart,
            actionExecuted: actionExecuted,
            stabilityAchieved: stabilityAchieved,
            captureEnd: captureEnd,
            totalDurationMs: Int(captureEnd.timeIntervalSince(captureStart) * 1000)
        )

        let quality = Quality(
            stabilityScore: min(stabilityScore, stabilityAfter),
            completeness: 1.0,
            notes: []
        )

        var actionWithMetadata = action
        if let metadata = metadata {
            var combinedMetadata = action.metadata ?? [:]
            combinedMetadata.merge(metadata) { _, new in new }
            actionWithMetadata = Action(
                type: action.type,
                element: action.element,
                coordinates: action.coordinates,
                metadata: combinedMetadata,
                additionalParams: action.additionalParams
            )
        }

        let transition = Transition(
            transitionId: transitionId,
            timestamp: captureStart,
            app: appInfo,
            environment: environment,
            action: actionWithMetadata,
            beforeState: beforeState,
            afterState: afterState,
            timing: timing,
            quality: quality
        )

        // Save transition JSON
        try saveTransition(transition)

        // Update summary
        summary.recordSuccess(actionType: action.type)
        sequenceNumber += 1

        return transition
    }

    // MARK: - High-Level Capture Strategies

    /// Systematically tap all buttons on the current screen
    /// - Returns: Number of transitions captured
    @discardableResult
    public func captureAllButtons(navigateBack: Bool = true) throws -> Int {
        var captured = 0

        let buttons = actionExecutor.findElements(ofType: .button).filter { $0.isHittable }

        for (index, button) in buttons.enumerated() {
            let elementInfo = actionExecutor.elementInfo(from: button)
            let coordinates = Coordinates(
                x: Double(button.frame.midX),
                y: Double(button.frame.midY)
            )

            let action = Action(
                type: .tap,
                element: elementInfo,
                coordinates: coordinates,
                metadata: [
                    "strategy": "tap_all_buttons",
                    "button_index": "\(index)",
                    "total_buttons": "\(buttons.count)"
                ]
            )

            do {
                try captureTransition(action: action)
                captured += 1

                // Navigate back if requested
                if navigateBack {
                    try? navigateBackIfNeeded()
                }
            } catch {
                print("Failed to capture button \(index): \(error)")
                summary.recordFailure(actionType: .tap)
            }
        }

        return captured
    }

    /// Systematically tap all cells on the current screen
    /// - Returns: Number of transitions captured
    @discardableResult
    public func captureAllCells(navigateBack: Bool = true) throws -> Int {
        var captured = 0

        let cells = actionExecutor.findElements(ofType: .cell).filter { $0.isHittable }

        for (index, cell) in cells.enumerated() {
            let elementInfo = actionExecutor.elementInfo(from: cell)
            let coordinates = Coordinates(
                x: Double(cell.frame.midX),
                y: Double(cell.frame.midY)
            )

            let action = Action(
                type: .tap,
                element: elementInfo,
                coordinates: coordinates,
                metadata: [
                    "strategy": "tap_all_cells",
                    "cell_index": "\(index)",
                    "total_cells": "\(cells.count)"
                ]
            )

            do {
                try captureTransition(action: action)
                captured += 1

                // Navigate back if requested
                if navigateBack {
                    try? navigateBackIfNeeded()
                }
            } catch {
                print("Failed to capture cell \(index): \(error)")
                summary.recordFailure(actionType: .tap)
            }
        }

        return captured
    }

    /// Capture text input interactions on all text fields
    /// - Parameter testText: Text to enter (default: "Test")
    /// - Returns: Number of transitions captured
    @discardableResult
    public func captureTextInput(testText: String = "Test") throws -> Int {
        var captured = 0

        let textFields = actionExecutor.findElements(ofType: .textField).filter { $0.isHittable }

        for (index, textField) in textFields.enumerated() {
            let elementInfo = actionExecutor.elementInfo(from: textField)

            // Capture typing
            let typeAction = Action(
                type: .typeText,
                element: elementInfo,
                metadata: [
                    "strategy": "text_input",
                    "field_index": "\(index)"
                ],
                additionalParams: AdditionalActionParams(text: testText)
            )

            do {
                try captureTransition(action: typeAction)
                captured += 1

                // Capture clearing
                let clearAction = Action(
                    type: .clearText,
                    element: elementInfo,
                    metadata: [
                        "strategy": "text_input",
                        "field_index": "\(index)"
                    ]
                )

                try captureTransition(action: clearAction)
                captured += 1

            } catch {
                print("Failed to capture text field \(index): \(error)")
                summary.recordFailure(actionType: .typeText)
            }
        }

        return captured
    }

    /// Capture scroll gestures
    /// - Returns: Number of transitions captured
    @discardableResult
    public func captureScrolling() throws -> Int {
        var captured = 0

        // Find scrollable elements (tables, collection views)
        let tables = actionExecutor.findElements(ofType: .table)
        let collectionViews = actionExecutor.findElements(ofType: .collectionView)

        let scrollableElements = tables + collectionViews

        for (index, element) in scrollableElements.enumerated() where element.isHittable {
            let elementInfo = actionExecutor.elementInfo(from: element)

            // Scroll down
            let scrollDownAction = Action(
                type: .swipe,
                element: elementInfo,
                metadata: [
                    "strategy": "scrolling",
                    "element_index": "\(index)"
                ],
                additionalParams: AdditionalActionParams(direction: "up") // swipe up = scroll down
            )

            do {
                try captureTransition(action: scrollDownAction)
                captured += 1
            } catch {
                print("Failed to capture scroll down: \(error)")
            }

            // Scroll back up
            let scrollUpAction = Action(
                type: .swipe,
                element: elementInfo,
                metadata: [
                    "strategy": "scrolling",
                    "element_index": "\(index)"
                ],
                additionalParams: AdditionalActionParams(direction: "down") // swipe down = scroll up
            )

            do {
                try captureTransition(action: scrollUpAction)
                captured += 1
            } catch {
                print("Failed to capture scroll up: \(error)")
            }
        }

        return captured
    }

    // MARK: - Helper Methods

    /// Attempt to navigate back to the previous screen
    private func navigateBackIfNeeded() throws {
        let backButton = app.navigationBars.buttons.element(boundBy: 0)
        if backButton.exists {
            let elementInfo = actionExecutor.elementInfo(from: backButton)
            let action = Action(
                type: .navigateBack,
                element: elementInfo,
                metadata: ["auto_navigate_back": "true"]
            )

            try captureTransition(action: action)
        }
    }

    // MARK: - Session Management

    /// Save a transition to disk
    private func saveTransition(_ transition: Transition) throws {
        let filename = "transition_\(String(format: "%04d", sequenceNumber)).json"
        let fileURL = sessionDirectory.appendingPathComponent(filename)

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        let data = try encoder.encode(transition)
        try data.write(to: fileURL)
    }

    /// Save session metadata
    public func saveSessionMetadata(
        strategy: String,
        instructionFile: String? = nil
    ) throws {
        let metadata = SessionMetadata(
            sessionId: sessionId,
            app: appInfo,
            startTime: Date(), // TODO: Track actual start time
            endTime: Date(),
            strategy: strategy,
            instructionFile: instructionFile,
            transitionsCaptured: summary.successfulTransitions,
            transitionsFailed: summary.failedTransitions,
            device: environment.device,
            osVersion: environment.osVersion
        )

        let filename = "session_metadata.json"
        let fileURL = sessionDirectory.appendingPathComponent(filename)

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        let data = try encoder.encode(metadata)
        try data.write(to: fileURL)
    }

    // MARK: - Summary

    /// Get the current capture summary
    public func getSummary() -> CaptureSummary {
        return summary
    }

    /// Print a summary of the capture session
    public func printSummary() {
        print("=== Capture Session Summary ===")
        print("Session ID: \(sessionId)")
        print("Total transitions: \(summary.totalTransitions)")
        print("Successful: \(summary.successfulTransitions)")
        print("Failed: \(summary.failedTransitions)")
        print("Success rate: \(String(format: "%.1f%%", summary.successRate * 100))")
        print("\nAction type breakdown:")
        for (type, count) in summary.actionTypeCounts.sorted(by: { $0.value > $1.value }) {
            print("  \(type.rawValue): \(count)")
        }
        print("================================")
    }
}
