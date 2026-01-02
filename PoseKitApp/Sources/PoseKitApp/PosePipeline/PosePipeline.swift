import ARKit
import Foundation
import ImageIO
import simd

/// Lightweight capture frame payload emitted by ARSessionManager.
public struct FramePacket: @unchecked Sendable {
    public let pixelBuffer: CVPixelBuffer
    public let depthMap: CVPixelBuffer?
    public let confidenceMap: CVPixelBuffer?
    public let displayTransform: CGAffineTransform
    public let viewportSize: CGSize
    public let intrinsics: simd_float3x3
    public let imageResolution: CGSize
    public let orientation: CGImagePropertyOrientation
    public let timestamp: TimeInterval
    public let hasDepth: Bool

    public init(
        pixelBuffer: CVPixelBuffer,
        depthMap: CVPixelBuffer?,
        confidenceMap: CVPixelBuffer?,
        displayTransform: CGAffineTransform,
        viewportSize: CGSize,
        intrinsics: simd_float3x3,
        imageResolution: CGSize,
        orientation: CGImagePropertyOrientation,
        timestamp: TimeInterval,
        hasDepth: Bool
    ) {
        self.pixelBuffer = pixelBuffer
        self.depthMap = depthMap
        self.confidenceMap = confidenceMap
        self.displayTransform = displayTransform
        self.viewportSize = viewportSize
        self.intrinsics = intrinsics
        self.imageResolution = imageResolution
        self.orientation = orientation
        self.timestamp = timestamp
        self.hasDepth = hasDepth
    }
}

/// Background pipeline for inference, depth sampling, smoothing, and canonicalization.
public actor PosePipeline {
    private let model: PoseKitModel
    private var config: PosePipelineConfig
    private var smoother: PoseSmoother
    private let canonicalizer = PoseCanonicalizer()
    private var lastTimestamp: TimeInterval = 0

    public init(config: PosePipelineConfig) {
        self.model = PoseKitModel()
        self.config = config
        self.smoother = PoseSmoother(jointCount: SkeletonSchema.jointOrder.count)
    }

    public func updateConfig(_ config: PosePipelineConfig) {
        self.config = config
    }

    public func start(
        stream: AsyncStream<FramePacket>,
        onOutput: @escaping @MainActor (PosePipelineOutput) -> Void
    ) -> Task<Void, Never> {
        Task(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            for await packet in stream {
                await self.handle(packet: packet, onOutput: onOutput)
            }
        }
    }

    private func handle(packet: FramePacket, onOutput: @MainActor (PosePipelineOutput) -> Void) async {
        let interval = 1.0 / Double(max(config.targetFPS, 1))
        if packet.timestamp - lastTimestamp < interval {
            return
        }
        lastTimestamp = packet.timestamp

        do {
            let keypoints = try await model.infer(
                pixelBuffer: packet.pixelBuffer,
                orientation: packet.orientation,
                timestamp: packet.timestamp
            )

            let joints2D = buildJoints2D(from: keypoints)
            let (camera3D, confidences) = computeCamera3D(
                joints2D: joints2D,
                packet: packet,
                minScore: config.minKeypointScore
            )
            let smoothed = smoother.smooth(camera3D)
            let playback3D = canonicalizer.canonicalize(
                camera3D: smoothed,
                joints2D: joints2D,
                minScore: config.minKeypointScore
            )

            let frame = PoseFrame(
                timestamp: packet.timestamp,
                joints2D: joints2D,
                camera3D: smoothed,
                playback3D: playback3D,
                depthConfidences: confidences
            )

            let statusMessages = buildStatusMessages(
                joints2D: joints2D,
                hasDepth: packet.hasDepth,
                showBanner: config.showConfidenceBanner
            )

            let output = PosePipelineOutput(
                frame: frame,
                displayTransform: packet.displayTransform,
                viewportSize: packet.viewportSize,
                statusMessages: statusMessages,
                overlayStyle: config.overlayStyle,
                hasDepth: packet.hasDepth
            )

            await MainActor.run {
                onOutput(output)
            }
        } catch {
            return
        }
    }

    private func buildJoints2D(from keypoints: [Keypoint2D]) -> [Keypoint2D] {
        guard keypoints.count >= Joint.coco17.count else {
            let padded = keypoints + Array(repeating: Keypoint2D(xNorm: 0.5, yNorm: 0.5, score: 0), count: Joint.coco17.count - keypoints.count)
            return padded + Array(repeating: Keypoint2D(xNorm: 0.5, yNorm: 0.5, score: 0), count: 2)
        }

        var output = keypoints
        let leftHip = keypoints[Joint.leftHip.rawValue]
        let rightHip = keypoints[Joint.rightHip.rawValue]
        let leftShoulder = keypoints[Joint.leftShoulder.rawValue]
        let rightShoulder = keypoints[Joint.rightShoulder.rawValue]

        let pelvis = midpoint(leftHip, rightHip)
        let neck = midpoint(leftShoulder, rightShoulder)

        output.append(pelvis)
        output.append(neck)
        return output
    }

    private func midpoint(_ a: Keypoint2D, _ b: Keypoint2D) -> Keypoint2D {
        let score = min(a.score, b.score)
        return Keypoint2D(
            xNorm: (a.xNorm + b.xNorm) * 0.5,
            yNorm: (a.yNorm + b.yNorm) * 0.5,
            score: score
        )
    }

    private func computeCamera3D(
        joints2D: [Keypoint2D],
        packet: FramePacket,
        minScore: Float
    ) -> ([SIMD3<Float>], [Float]) {
        let count = SkeletonSchema.jointOrder.count
        var camera3D = Array(repeating: SIMD3<Float>(repeating: .nan), count: count)
        var confidences = Array(repeating: Float(0), count: count)

        let depthMode = DepthConfidenceMode(rawValue: config.depthConfidenceMode) ?? .balanced
        for i in 0..<count {
            guard i < joints2D.count else { continue }
            let kp = joints2D[i]
            if kp.score < minScore {
                continue
            }
            guard let depthMap = packet.depthMap, packet.hasDepth else {
                continue
            }
            if let sample = DepthSampler.sample(
                uNorm: kp.xNorm,
                vNorm: kp.yNorm,
                depthMap: depthMap,
                confidenceMap: packet.confidenceMap,
                mode: depthMode
            ) {
                let point = PoseMath.backProject(
                    uNorm: kp.xNorm,
                    vNorm: kp.yNorm,
                    depth: sample.depth,
                    intrinsics: packet.intrinsics,
                    imageResolution: packet.imageResolution
                )
                camera3D[i] = point
                confidences[i] = sample.confidence
            }
        }
        return (camera3D, confidences)
    }

    private func buildStatusMessages(
        joints2D: [Keypoint2D],
        hasDepth: Bool,
        showBanner: Bool
    ) -> [PosePipelineStatus] {
        guard showBanner else { return [] }
        var messages: [PosePipelineStatus] = []

        if !hasDepth {
            messages.append(.depthUnavailable)
        }

        var lowConfidence: [Joint] = []
        var outOfFrame: [Joint] = []
        for (index, joint) in SkeletonSchema.jointOrder.enumerated() {
            guard index < joints2D.count else { continue }
            let kp = joints2D[index]
            if kp.score < config.minKeypointScore {
                lowConfidence.append(joint)
            }
            if kp.xNorm < 0 || kp.xNorm > 1 || kp.yNorm < 0 || kp.yNorm > 1 {
                outOfFrame.append(joint)
            }
        }

        if !lowConfidence.isEmpty {
            messages.append(.lowConfidence(lowConfidence))
        }
        if !outOfFrame.isEmpty {
            messages.append(.outOfFrame(outOfFrame))
        }

        if messages.isEmpty {
            messages.append(.ok)
        }
        return messages
    }
}
