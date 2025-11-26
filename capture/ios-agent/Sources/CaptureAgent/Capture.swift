import Foundation
import XCTest

/// Handles screenshot capture and accessibility tree extraction
public class ScreenCapture {
    private let app: XCUIApplication

    public init(app: XCUIApplication) {
        self.app = app
    }

    // MARK: - Screenshot Capture

    /// Capture a screenshot and save it to the specified path
    /// - Parameter filename: The filename to save the screenshot as (without path)
    /// - Parameter directory: The directory to save the screenshot in
    /// - Returns: The relative filename of the saved screenshot
    public func captureScreenshot(filename: String, in directory: URL) throws -> String {
        let screenshot = XCUIScreen.main.screenshot()

        let fileURL = directory.appendingPathComponent(filename)
        let imageData = screenshot.pngRepresentation

        try imageData.write(to: fileURL)

        return filename
    }

    // MARK: - State Capture

    /// Capture the complete screen state (screenshot + accessibility tree)
    /// - Parameter sequenceNumber: The sequence number for this capture
    /// - Parameter suffix: Suffix for the filename (e.g., "before" or "after")
    /// - Parameter directory: Directory to save the screenshot in
    /// - Returns: A ScreenState object containing all captured information
    public func captureState(
        sequenceNumber: Int,
        suffix: String,
        in directory: URL
    ) throws -> ScreenState {
        let timestamp = Date()

        // Generate filename
        let filename = String(format: "transition_%04d_%@.png", sequenceNumber, suffix)

        // Capture screenshot
        let screenshotFilename = try captureScreenshot(filename: filename, in: directory)

        // Capture accessibility tree
        let accessibilityTree = captureAccessibilityTree(from: app)

        // Count elements
        let (visibleCount, interactiveCount) = countElements(in: app)

        return ScreenState(
            screenshot: screenshotFilename,
            timestamp: timestamp,
            accessibilityTree: accessibilityTree,
            visibleElements: visibleCount,
            interactiveElements: interactiveCount
        )
    }

    // MARK: - Accessibility Tree

    /// Capture the accessibility tree from the app's current state
    /// - Parameter element: The root element to start from (typically the app)
    /// - Parameter maxDepth: Maximum depth to traverse (default 3 to avoid huge trees)
    /// - Returns: The root AccessibilityNode
    public func captureAccessibilityTree(
        from element: XCUIElement,
        maxDepth: Int = 3
    ) -> AccessibilityNode? {
        return buildAccessibilityNode(from: element, currentDepth: 0, maxDepth: maxDepth)
    }

    /// Recursively build an accessibility node from an XCUIElement
    private func buildAccessibilityNode(
        from element: XCUIElement,
        currentDepth: Int,
        maxDepth: Int
    ) -> AccessibilityNode? {
        guard element.exists else { return nil }
        guard currentDepth < maxDepth else { return nil }

        let type = String(describing: element.elementType)
        let label = element.label.isEmpty ? nil : element.label
        let frame = element.frame.isEmpty ? nil : Frame(from: element.frame)

        // Recursively process children
        var children: [AccessibilityNode] = []
        if currentDepth < maxDepth - 1 {
            for childIndex in 0..<min(element.children(matching: .any).count, 20) {
                let child = element.children(matching: .any).element(boundBy: childIndex)
                if let childNode = buildAccessibilityNode(
                    from: child,
                    currentDepth: currentDepth + 1,
                    maxDepth: maxDepth
                ) {
                    children.append(childNode)
                }
            }
        }

        return AccessibilityNode(
            type: type,
            label: label,
            frame: frame,
            children: children.isEmpty ? nil : children
        )
    }

    // MARK: - Element Counting

    /// Count visible and interactive elements
    /// - Parameter element: The root element to count from
    /// - Returns: A tuple of (visibleCount, interactiveCount)
    private func countElements(in element: XCUIElement) -> (visible: Int, interactive: Int) {
        var visibleCount = 0
        var interactiveCount = 0

        // Count all visible elements
        let allElements = element.descendants(matching: .any)
        visibleCount = allElements.count

        // Count interactive elements (buttons, cells, text fields, etc.)
        interactiveCount += element.buttons.count
        interactiveCount += element.cells.count
        interactiveCount += element.textFields.count
        interactiveCount += element.textViews.count
        interactiveCount += element.switches.count
        interactiveCount += element.sliders.count
        interactiveCount += element.links.count

        return (visibleCount, interactiveCount)
    }

    // MARK: - Stability Detection

    /// Wait for the UI to become stable (animations complete, etc.)
    /// - Parameter timeout: Maximum time to wait in seconds
    /// - Parameter pollInterval: How often to check for stability in seconds
    /// - Returns: A stability score (0.0 = unstable, 1.0 = perfectly stable)
    @discardableResult
    public func waitForStability(
        timeout: TimeInterval = 2.0,
        pollInterval: TimeInterval = 0.1
    ) -> Double {
        let startTime = Date()
        var previousElementCount = 0
        var stabilityCounter = 0
        let requiredStableChecks = 3 // Need 3 consecutive stable checks

        while Date().timeIntervalSince(startTime) < timeout {
            let currentElementCount = app.descendants(matching: .any).count

            if currentElementCount == previousElementCount {
                stabilityCounter += 1
                if stabilityCounter >= requiredStableChecks {
                    return 1.0 // Perfectly stable
                }
            } else {
                stabilityCounter = 0 // Reset counter if change detected
            }

            previousElementCount = currentElementCount
            Thread.sleep(forTimeInterval: pollInterval)
        }

        // Return partial stability score based on how close we got
        return Double(stabilityCounter) / Double(requiredStableChecks)
    }

    /// Wait for a specific element to appear
    /// - Parameter element: The element to wait for
    /// - Parameter timeout: Maximum time to wait
    /// - Returns: True if element appeared, false otherwise
    public func waitForElement(_ element: XCUIElement, timeout: TimeInterval = 5.0) -> Bool {
        return element.waitForExistence(timeout: timeout)
    }

    /// Wait for a specific element to disappear
    /// - Parameter element: The element to wait to disappear
    /// - Parameter timeout: Maximum time to wait
    /// - Returns: True if element disappeared, false otherwise
    public func waitForElementToDisappear(_ element: XCUIElement, timeout: TimeInterval = 5.0) -> Bool {
        let startTime = Date()
        while Date().timeIntervalSince(startTime) < timeout {
            if !element.exists {
                return true
            }
            Thread.sleep(forTimeInterval: 0.1)
        }
        return false
    }
}

// MARK: - Helper Extensions

extension CGRect {
    var isEmpty: Bool {
        return size.width == 0 || size.height == 0
    }
}
