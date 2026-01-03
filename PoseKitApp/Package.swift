// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "PoseKitApp",
    platforms: [.iOS(.v17)],
    products: [
        .executable(name: "PoseKitApp", targets: ["PoseKitApp"])
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "PoseKitApp",
            dependencies: [],
            resources: [
                .copy("Settings.bundle"),
                .copy("Resources/Models/pkmodel.mlmodelc")
            ]
        )
    ]
)
