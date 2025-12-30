import Foundation
import CoreGraphics
import simd

/// A single 2D joint prediction in captured image normalized coordinates.
public struct Keypoint2D: Sendable, Codable {
    public var xNorm: Float
    public var yNorm: Float
    public var score: Float

    public init(xNorm: Float, yNorm: Float, score: Float) {
        self.xNorm = xNorm
        self.yNorm = yNorm
        self.score = score
    }
}

/// A single pose frame containing 2D, camera-space 3D, and canonical 3D joints.
public struct PoseFrame: Sendable {
    public var timestamp: TimeInterval
    public var joints2D: [Keypoint2D]
    public var camera3D: [SIMD3<Float>]
    public var playback3D: [SIMD3<Float>]
    public var depthConfidences: [Float]

    public init(
        timestamp: TimeInterval,
        joints2D: [Keypoint2D],
        camera3D: [SIMD3<Float>],
        playback3D: [SIMD3<Float>],
        depthConfidences: [Float]
    ) {
        self.timestamp = timestamp
        self.joints2D = joints2D
        self.camera3D = camera3D
        self.playback3D = playback3D
        self.depthConfidences = depthConfidences
    }
}

/// Joint identifiers for COCO17 plus derived center joints.
public enum Joint: Int, CaseIterable, Codable, Sendable {
    case nose
    case leftEye
    case rightEye
    case leftEar
    case rightEar
    case leftShoulder
    case rightShoulder
    case leftElbow
    case rightElbow
    case leftWrist
    case rightWrist
    case leftHip
    case rightHip
    case leftKnee
    case rightKnee
    case leftAnkle
    case rightAnkle
    case pelvisCenter
    case neckCenter

    public static let coco17: [Joint] = [
        .nose, .leftEye, .rightEye, .leftEar, .rightEar,
        .leftShoulder, .rightShoulder, .leftElbow, .rightElbow,
        .leftWrist, .rightWrist, .leftHip, .rightHip,
        .leftKnee, .rightKnee, .leftAnkle, .rightAnkle
    ]

    public static let storageOrder: [Joint] = Joint.coco17 + [.pelvisCenter, .neckCenter]

    public var displayName: String {
        switch self {
        case .nose: return "Nose"
        case .leftEye: return "L-Eye"
        case .rightEye: return "R-Eye"
        case .leftEar: return "L-Ear"
        case .rightEar: return "R-Ear"
        case .leftShoulder: return "L-Shoulder"
        case .rightShoulder: return "R-Shoulder"
        case .leftElbow: return "L-Elbow"
        case .rightElbow: return "R-Elbow"
        case .leftWrist: return "L-Wrist"
        case .rightWrist: return "R-Wrist"
        case .leftHip: return "L-Hip"
        case .rightHip: return "R-Hip"
        case .leftKnee: return "L-Knee"
        case .rightKnee: return "R-Knee"
        case .leftAnkle: return "L-Ankle"
        case .rightAnkle: return "R-Ankle"
        case .pelvisCenter: return "Pelvis"
        case .neckCenter: return "Neck"
        }
    }
}

/// Runtime configuration for the pose pipeline.
public struct PosePipelineConfig: Sendable, Codable, Equatable {
    public var minKeypointScore: Float
    public var targetFPS: Int
    public var depthSource: Int
    public var depthConfidenceMode: Int
    public var overlayStyle: Int
    public var showConfidenceBanner: Bool

    public init(
        minKeypointScore: Float,
        targetFPS: Int,
        depthSource: Int,
        depthConfidenceMode: Int,
        overlayStyle: Int,
        showConfidenceBanner: Bool
    ) {
        self.minKeypointScore = minKeypointScore
        self.targetFPS = targetFPS
        self.depthSource = depthSource
        self.depthConfidenceMode = depthConfidenceMode
        self.overlayStyle = overlayStyle
        self.showConfidenceBanner = showConfidenceBanner
    }
}

/// Status messages emitted by the pose pipeline.
public enum PosePipelineStatus: Equatable, Sendable {
    case ok
    case depthUnavailable
    case lowConfidence([Joint])
    case outOfFrame([Joint])
}

/// Output data for UI overlay and recording.
public struct PosePipelineOutput: Sendable {
    public var frame: PoseFrame
    public var displayTransform: CGAffineTransform
    public var viewportSize: CGSize
    public var statusMessages: [PosePipelineStatus]
    public var overlayStyle: Int
    public var hasDepth: Bool

    public init(
        frame: PoseFrame,
        displayTransform: CGAffineTransform,
        viewportSize: CGSize,
        statusMessages: [PosePipelineStatus],
        overlayStyle: Int,
        hasDepth: Bool
    ) {
        self.frame = frame
        self.displayTransform = displayTransform
        self.viewportSize = viewportSize
        self.statusMessages = statusMessages
        self.overlayStyle = overlayStyle
        self.hasDepth = hasDepth
    }
}
