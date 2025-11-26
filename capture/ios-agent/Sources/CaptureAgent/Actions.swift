import Foundation
import XCTest

/// Executes actions on UI elements and extracts element information
public class ActionExecutor {
    private let app: XCUIApplication

    public init(app: XCUIApplication) {
        self.app = app
    }

    // MARK: - Action Execution

    /// Execute an action on the app
    /// - Parameter action: The action to execute
    /// - Returns: True if action was executed successfully
    @discardableResult
    public func execute(_ action: Action) throws -> Bool {
        switch action.type {
        case .tap:
            return try executeTap(action)
        case .doubleTap:
            return try executeDoubleTap(action)
        case .longPress:
            return try executeLongPress(action)
        case .swipe:
            return try executeSwipe(action)
        case .scroll:
            return try executeScroll(action)
        case .typeText:
            return try executeTypeText(action)
        case .clearText:
            return try executeClearText(action)
        case .navigateBack:
            return try executeNavigateBack(action)
        case .dismissKeyboard:
            return try executeDismissKeyboard()
        case .pullToRefresh:
            return try executePullToRefresh(action)
        case .pinch:
            return try executePinch(action)
        case .rotate:
            return try executeRotate(action)
        }
    }

    // MARK: - Specific Action Implementations

    private func executeTap(_ action: Action) throws -> Bool {
        guard let element = action.element else {
            throw ActionError.missingElement
        }

        guard let xcuiElement = findElement(matching: element) else {
            throw ActionError.elementNotFound
        }

        xcuiElement.tap()
        return true
    }

    private func executeDoubleTap(_ action: Action) throws -> Bool {
        guard let element = action.element else {
            throw ActionError.missingElement
        }

        guard let xcuiElement = findElement(matching: element) else {
            throw ActionError.elementNotFound
        }

        xcuiElement.doubleTap()
        return true
    }

    private func executeLongPress(_ action: Action) throws -> Bool {
        guard let element = action.element else {
            throw ActionError.missingElement
        }

        guard let xcuiElement = findElement(matching: element) else {
            throw ActionError.elementNotFound
        }

        let duration = action.additionalParams?.duration ?? 1.0
        xcuiElement.press(forDuration: duration)
        return true
    }

    private func executeSwipe(_ action: Action) throws -> Bool {
        guard let params = action.additionalParams,
              let direction = params.direction else {
            throw ActionError.missingParameters
        }

        let element = action.element != nil ? findElement(matching: action.element!) : app

        guard let targetElement = element else {
            throw ActionError.elementNotFound
        }

        switch direction.lowercased() {
        case "up":
            targetElement.swipeUp()
        case "down":
            targetElement.swipeDown()
        case "left":
            targetElement.swipeLeft()
        case "right":
            targetElement.swipeRight()
        default:
            throw ActionError.invalidParameter("Invalid swipe direction: \(direction)")
        }

        return true
    }

    private func executeScroll(_ action: Action) throws -> Bool {
        // TODO: Implement custom scrolling with distance/velocity control
        // For now, use swipe as approximation
        return try executeSwipe(action)
    }

    private func executeTypeText(_ action: Action) throws -> Bool {
        guard let element = action.element else {
            throw ActionError.missingElement
        }

        guard let text = action.additionalParams?.text else {
            throw ActionError.missingParameters
        }

        guard let xcuiElement = findElement(matching: element) else {
            throw ActionError.elementNotFound
        }

        xcuiElement.tap() // Focus the element first
        xcuiElement.typeText(text)
        return true
    }

    private func executeClearText(_ action: Action) throws -> Bool {
        guard let element = action.element else {
            throw ActionError.missingElement
        }

        guard let xcuiElement = findElement(matching: element) else {
            throw ActionError.elementNotFound
        }

        xcuiElement.tap() // Focus the element first

        // Select all and delete
        if let stringValue = xcuiElement.value as? String, !stringValue.isEmpty {
            xcuiElement.doubleTap() // Select all
            app.keys["delete"].tap()
        }

        return true
    }

    private func executeNavigateBack(_ action: Action) throws -> Bool {
        // Look for back button in navigation bar
        let backButton = app.navigationBars.buttons.element(boundBy: 0)
        if backButton.exists {
            backButton.tap()
            return true
        }

        // If specific element provided, try that
        if let element = action.element,
           let xcuiElement = findElement(matching: element) {
            xcuiElement.tap()
            return true
        }

        throw ActionError.elementNotFound
    }

    private func executeDismissKeyboard() throws -> Bool {
        // Try tapping the "Done" or "Return" key
        if app.keyboards.buttons["Done"].exists {
            app.keyboards.buttons["Done"].tap()
        } else if app.keyboards.buttons["Return"].exists {
            app.keyboards.buttons["Return"].tap()
        } else {
            // Tap outside keyboard area
            let coordinate = app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.1))
            coordinate.tap()
        }
        return true
    }

    private func executePullToRefresh(_ action: Action) throws -> Bool {
        guard let element = action.element else {
            throw ActionError.missingElement
        }

        guard let xcuiElement = findElement(matching: element) else {
            throw ActionError.elementNotFound
        }

        // Swipe down from top of element
        let start = xcuiElement.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.1))
        let end = xcuiElement.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.6))
        start.press(forDuration: 0.1, thenDragTo: end)

        return true
    }

    private func executePinch(_ action: Action) throws -> Bool {
        // TODO: Implement pinch gesture
        throw ActionError.notImplemented("Pinch gesture not yet implemented")
    }

    private func executeRotate(_ action: Action) throws -> Bool {
        // TODO: Implement rotation gesture
        throw ActionError.notImplemented("Rotate gesture not yet implemented")
    }

    // MARK: - Element Finding

    /// Find an XCUIElement matching the given ElementInfo
    /// - Parameter info: Element information to match against
    /// - Returns: The matching XCUIElement, or nil if not found
    public func findElement(matching info: ElementInfo) -> XCUIElement? {
        // Try by identifier first (most specific)
        if let identifier = info.identifier, !identifier.isEmpty {
            let element = app.descendants(matching: .any).matching(identifier: identifier).firstMatch
            if element.exists {
                return element
            }
        }

        // Try by label
        if let label = info.label, !label.isEmpty {
            let elementType = xcuiElementType(from: info.type)
            let element = app.descendants(matching: elementType).matching(NSPredicate(format: "label == %@", label)).firstMatch
            if element.exists {
                return element
            }
        }

        // Try by type and position (less reliable, but can work)
        let elementType = xcuiElementType(from: info.type)
        let allOfType = app.descendants(matching: elementType)

        // Find closest match by frame
        for i in 0..<allOfType.count {
            let candidate = allOfType.element(boundBy: i)
            if framesAreClose(candidate.frame, Frame(from: candidate.frame), info.frame) {
                return candidate
            }
        }

        return nil
    }

    /// Find all elements of a specific type
    /// - Parameter type: The XCUIElement.ElementType to search for
    /// - Returns: Array of matching elements
    public func findElements(ofType type: XCUIElement.ElementType) -> [XCUIElement] {
        let query = app.descendants(matching: type)
        var elements: [XCUIElement] = []

        for i in 0..<query.count {
            elements.append(query.element(boundBy: i))
        }

        return elements
    }

    /// Find all interactive elements (buttons, cells, text fields, etc.)
    /// - Returns: Array of all interactive elements with their info
    public func findInteractiveElements() -> [ElementInfo] {
        var elements: [ElementInfo] = []

        // Buttons
        for button in findElements(ofType: .button) where button.isHittable {
            elements.append(elementInfo(from: button))
        }

        // Cells
        for cell in findElements(ofType: .cell) where cell.isHittable {
            elements.append(elementInfo(from: cell))
        }

        // Text fields
        for textField in findElements(ofType: .textField) where textField.isHittable {
            elements.append(elementInfo(from: textField))
        }

        // Switches
        for toggle in findElements(ofType: .switch) where toggle.isHittable {
            elements.append(elementInfo(from: toggle))
        }

        return elements
    }

    // MARK: - Element Info Extraction

    /// Extract ElementInfo from an XCUIElement
    /// - Parameter element: The XCUIElement to extract info from
    /// - Returns: ElementInfo representation
    public func elementInfo(from element: XCUIElement) -> ElementInfo {
        let type = String(describing: element.elementType)
        let label = element.label.isEmpty ? nil : element.label
        let frame = Frame(from: element.frame)
        let traits = extractTraits(from: element)
        let identifier = element.identifier.isEmpty ? nil : element.identifier
        let value = element.value as? String

        return ElementInfo(
            type: type,
            label: label,
            frame: frame,
            traits: traits,
            identifier: identifier,
            value: value
        )
    }

    /// Extract accessibility traits as string array
    private func extractTraits(from element: XCUIElement) -> [String] {
        var traits: [String] = []

        // XCUIElement doesn't expose traits directly in a clean way
        // We can infer some based on element type and properties
        if element.isEnabled {
            traits.append("enabled")
        }
        if element.isSelected {
            traits.append("selected")
        }
        if element.hasFocus {
            traits.append("focused")
        }

        // Add element type as a trait
        traits.append(String(describing: element.elementType))

        return traits
    }

    // MARK: - Helper Functions

    /// Convert string element type to XCUIElement.ElementType
    private func xcuiElementType(from typeString: String) -> XCUIElement.ElementType {
        switch typeString.lowercased() {
        case "button": return .button
        case "cell": return .cell
        case "textfield": return .textField
        case "textview": return .textView
        case "statictext": return .staticText
        case "switch": return .switch
        case "slider": return .slider
        case "navigationbar": return .navigationBar
        case "tabbar": return .tabBar
        case "table": return .table
        case "collectionview": return .collectionView
        case "image": return .image
        case "link": return .link
        default: return .any
        }
    }

    /// Check if two frames are approximately equal (within tolerance)
    private func framesAreClose(_ cgFrame: CGRect, _ frame1: Frame, _ frame2: Frame) -> Bool {
        let tolerance: Double = 5.0

        return abs(frame1.x - frame2.x) < tolerance &&
               abs(frame1.y - frame2.y) < tolerance &&
               abs(frame1.width - frame2.width) < tolerance &&
               abs(frame1.height - frame2.height) < tolerance
    }
}

// MARK: - Action Errors

public enum ActionError: Error, LocalizedError {
    case missingElement
    case elementNotFound
    case missingParameters
    case invalidParameter(String)
    case notImplemented(String)
    case executionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .missingElement:
            return "Action requires an element but none was provided"
        case .elementNotFound:
            return "Could not find the specified element"
        case .missingParameters:
            return "Action requires additional parameters"
        case .invalidParameter(let message):
            return "Invalid parameter: \(message)"
        case .notImplemented(let message):
            return "Not implemented: \(message)"
        case .executionFailed(let message):
            return "Action execution failed: \(message)"
        }
    }
}
