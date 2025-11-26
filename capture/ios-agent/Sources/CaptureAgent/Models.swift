import Foundation
import XCTest

// MARK: - Action Types

/// All possible action types that can be performed on UI elements
public enum ActionType: String, Codable {
    case tap
    case doubleTap = "double_tap"
    case longPress = "long_press"
    case swipe
    case scroll
    case typeText = "type_text"
    case clearText = "clear_text"
    case navigateBack = "navigate_back"
    case dismissKeyboard = "dismiss_keyboard"
    case pullToRefresh = "pull_to_refresh"
    case pinch
    case rotate
}

// MARK: - Core Data Models

/// Complete record of a single UI transition
public struct Transition: Codable {
    public let version: String
    public let transitionId: String
    public let timestamp: Date
    public let app: AppInfo
    public let environment: Environment
    public let action: Action
    public let beforeState: ScreenState
    public let afterState: ScreenState
    public let timing: Timing
    public let quality: Quality

    enum CodingKeys: String, CodingKey {
        case version
        case transitionId = "transition_id"
        case timestamp
        case app
        case environment
        case action
        case beforeState = "before_state"
        case afterState = "after_state"
        case timing
        case quality
    }

    public init(
        version: String = "1.0",
        transitionId: String,
        timestamp: Date,
        app: AppInfo,
        environment: Environment,
        action: Action,
        beforeState: ScreenState,
        afterState: ScreenState,
        timing: Timing,
        quality: Quality
    ) {
        self.version = version
        self.transitionId = transitionId
        self.timestamp = timestamp
        self.app = app
        self.environment = environment
        self.action = action
        self.beforeState = beforeState
        self.afterState = afterState
        self.timing = timing
        self.quality = quality
    }
}

/// Information about the app being tested
public struct AppInfo: Codable {
    public let bundleId: String
    public let name: String
    public let version: String

    enum CodingKeys: String, CodingKey {
        case bundleId = "bundle_id"
        case name
        case version
    }

    public init(bundleId: String, name: String, version: String) {
        self.bundleId = bundleId
        self.name = name
        self.version = version
    }
}

/// Device and system environment information
public struct Environment: Codable {
    public let device: String
    public let osVersion: String
    public let screenSize: Size
    public let scale: Double
    public let orientation: String

    enum CodingKeys: String, CodingKey {
        case device
        case osVersion = "os_version"
        case screenSize = "screen_size"
        case scale
        case orientation
    }

    public init(device: String, osVersion: String, screenSize: Size, scale: Double, orientation: String) {
        self.device = device
        self.osVersion = osVersion
        self.screenSize = screenSize
        self.scale = scale
        self.orientation = orientation
    }
}

/// Screen dimensions
public struct Size: Codable {
    public let width: Double
    public let height: Double

    public init(width: Double, height: Double) {
        self.width = width
        self.height = height
    }
}

/// An action performed on a UI element
public struct Action: Codable {
    public let type: ActionType
    public let element: ElementInfo?
    public let coordinates: Coordinates?
    public let metadata: [String: String]?
    public let additionalParams: AdditionalActionParams?

    enum CodingKeys: String, CodingKey {
        case type
        case element
        case coordinates
        case metadata
        case additionalParams = "additional_params"
    }

    public init(
        type: ActionType,
        element: ElementInfo? = nil,
        coordinates: Coordinates? = nil,
        metadata: [String: String]? = nil,
        additionalParams: AdditionalActionParams? = nil
    ) {
        self.type = type
        self.element = element
        self.coordinates = coordinates
        self.metadata = metadata
        self.additionalParams = additionalParams
    }
}

/// Additional parameters for specific action types
public struct AdditionalActionParams: Codable {
    public let text: String?
    public let direction: String?
    public let distance: Double?
    public let duration: Double?
    public let scale: Double?
    public let rotation: Double?
    public let start: Coordinates?
    public let end: Coordinates?

    public init(
        text: String? = nil,
        direction: String? = nil,
        distance: Double? = nil,
        duration: Double? = nil,
        scale: Double? = nil,
        rotation: Double? = nil,
        start: Coordinates? = nil,
        end: Coordinates? = nil
    ) {
        self.text = text
        self.direction = direction
        self.distance = distance
        self.duration = duration
        self.scale = scale
        self.rotation = rotation
        self.start = start
        self.end = end
    }
}

/// Screen coordinates
public struct Coordinates: Codable {
    public let x: Double
    public let y: Double

    public init(x: Double, y: Double) {
        self.x = x
        self.y = y
    }
}

/// Information about a UI element
public struct ElementInfo: Codable {
    public let type: String
    public let label: String?
    public let frame: Frame
    public let traits: [String]
    public let identifier: String?
    public let value: String?

    public init(
        type: String,
        label: String?,
        frame: Frame,
        traits: [String],
        identifier: String?,
        value: String?
    ) {
        self.type = type
        self.label = label
        self.frame = frame
        self.traits = traits
        self.identifier = identifier
        self.value = value
    }
}

/// Element position and size
public struct Frame: Codable {
    public let x: Double
    public let y: Double
    public let width: Double
    public let height: Double

    public init(x: Double, y: Double, width: Double, height: Double) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }

    public init(from rect: CGRect) {
        self.x = rect.origin.x
        self.y = rect.origin.y
        self.width = rect.size.width
        self.height = rect.size.height
    }
}

/// State of the screen at a point in time
public struct ScreenState: Codable {
    public let screenshot: String
    public let timestamp: Date
    public let accessibilityTree: AccessibilityNode?
    public let visibleElements: Int
    public let interactiveElements: Int

    enum CodingKeys: String, CodingKey {
        case screenshot
        case timestamp
        case accessibilityTree = "accessibility_tree"
        case visibleElements = "visible_elements"
        case interactiveElements = "interactive_elements"
    }

    public init(
        screenshot: String,
        timestamp: Date,
        accessibilityTree: AccessibilityNode?,
        visibleElements: Int,
        interactiveElements: Int
    ) {
        self.screenshot = screenshot
        self.timestamp = timestamp
        self.accessibilityTree = accessibilityTree
        self.visibleElements = visibleElements
        self.interactiveElements = interactiveElements
    }
}

/// Node in the accessibility tree (simplified hierarchy)
public struct AccessibilityNode: Codable {
    public let type: String
    public let label: String?
    public let frame: Frame?
    public let children: [AccessibilityNode]?

    public init(type: String, label: String?, frame: Frame?, children: [AccessibilityNode]?) {
        self.type = type
        self.label = label
        self.frame = frame
        self.children = children
    }
}

/// Timing information for the transition
public struct Timing: Codable {
    public let captureStart: Date
    public let actionExecuted: Date
    public let stabilityAchieved: Date
    public let captureEnd: Date
    public let totalDurationMs: Int

    enum CodingKeys: String, CodingKey {
        case captureStart = "capture_start"
        case actionExecuted = "action_executed"
        case stabilityAchieved = "stability_achieved"
        case captureEnd = "capture_end"
        case totalDurationMs = "total_duration_ms"
    }

    public init(
        captureStart: Date,
        actionExecuted: Date,
        stabilityAchieved: Date,
        captureEnd: Date,
        totalDurationMs: Int
    ) {
        self.captureStart = captureStart
        self.actionExecuted = actionExecuted
        self.stabilityAchieved = stabilityAchieved
        self.captureEnd = captureEnd
        self.totalDurationMs = totalDurationMs
    }
}

/// Quality metrics for the captured transition
public struct Quality: Codable {
    public let stabilityScore: Double
    public let completeness: Double
    public let notes: [String]

    enum CodingKeys: String, CodingKey {
        case stabilityScore = "stability_score"
        case completeness
        case notes
    }

    public init(stabilityScore: Double, completeness: Double, notes: [String]) {
        self.stabilityScore = stabilityScore
        self.completeness = completeness
        self.notes = notes
    }
}

// MARK: - Session Metadata

/// Metadata for an entire capture session
public struct SessionMetadata: Codable {
    public let sessionId: String
    public let app: AppInfo
    public let startTime: Date
    public let endTime: Date
    public let strategy: String
    public let instructionFile: String?
    public let transitionsCaptured: Int
    public let transitionsFailed: Int
    public let device: String
    public let osVersion: String

    enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
        case app
        case startTime = "start_time"
        case endTime = "end_time"
        case strategy
        case instructionFile = "instruction_file"
        case transitionsCaptured = "transitions_captured"
        case transitionsFailed = "transitions_failed"
        case device
        case osVersion = "os_version"
    }

    public init(
        sessionId: String,
        app: AppInfo,
        startTime: Date,
        endTime: Date,
        strategy: String,
        instructionFile: String?,
        transitionsCaptured: Int,
        transitionsFailed: Int,
        device: String,
        osVersion: String
    ) {
        self.sessionId = sessionId
        self.app = app
        self.startTime = startTime
        self.endTime = endTime
        self.strategy = strategy
        self.instructionFile = instructionFile
        self.transitionsCaptured = transitionsCaptured
        self.transitionsFailed = transitionsFailed
        self.device = device
        self.osVersion = osVersion
    }
}

// MARK: - Capture Summary

/// Summary statistics for a capture session
public struct CaptureSummary {
    public var totalTransitions: Int = 0
    public var successfulTransitions: Int = 0
    public var failedTransitions: Int = 0
    public var actionTypeCounts: [ActionType: Int] = [:]

    public init() {}

    public mutating func recordSuccess(actionType: ActionType) {
        totalTransitions += 1
        successfulTransitions += 1
        actionTypeCounts[actionType, default: 0] += 1
    }

    public mutating func recordFailure(actionType: ActionType) {
        totalTransitions += 1
        failedTransitions += 1
        actionTypeCounts[actionType, default: 0] += 1
    }

    public var successRate: Double {
        guard totalTransitions > 0 else { return 0.0 }
        return Double(successfulTransitions) / Double(totalTransitions)
    }
}
