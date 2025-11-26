// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "CaptureAgent",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "CaptureAgent",
            targets: ["CaptureAgent"]),
    ],
    dependencies: [
        // No external dependencies needed - using only XCTest framework
    ],
    targets: [
        .target(
            name: "CaptureAgent",
            dependencies: []),
        .testTarget(
            name: "CaptureAgentTests",
            dependencies: ["CaptureAgent"]),
    ]
)
