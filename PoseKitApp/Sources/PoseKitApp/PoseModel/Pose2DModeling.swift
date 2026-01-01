import CoreVideo
import ImageIO

public protocol Pose2DModeling: Sendable {
    var jointOrder: [Joint] { get }

    func infer(
        pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation,
        timestamp: TimeInterval
    ) async throws -> [Keypoint2D]
}

public enum PoseModelError: Error, LocalizedError {
    case unavailable
    case notImplemented
    case modelNotFound
    case invalidInput
    case invalidOutput
    case inferenceFailed

    public var errorDescription: String? {
        switch self {
        case .unavailable: return "Pose model unavailable."
        case .notImplemented: return "Pose model adapter not implemented."
        case .modelNotFound: return "Pose model file not found."
        case .invalidInput: return "Pose model input invalid."
        case .invalidOutput: return "Pose model output invalid."
        case .inferenceFailed: return "Pose model inference failed."
        }
    }
}
