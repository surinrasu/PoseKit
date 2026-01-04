import ARKit
import Foundation
import ImageIO
import UIKit

@MainActor
final class ARSessionManager: NSObject, ObservableObject, @preconcurrency ARSessionDelegate {
    let session = ARSession()

    private var continuation: AsyncStream<FramePacket>.Continuation?
    lazy var frameStream: AsyncStream<FramePacket> = { [weak self] in
        AsyncStream { cont in
            self?.continuation = cont
        }
    }()

    private var viewportSize: CGSize = .zero
    private var interfaceOrientation: UIInterfaceOrientation = .portrait
    private var depthSource: Int = 0

    @Published var depthSupported: Bool = false

    override init() {
        super.init()
        session.delegate = self
    }

    func updateViewport(size: CGSize, orientation: UIInterfaceOrientation) {
        viewportSize = size
        interfaceOrientation = orientation
    }

    func updateDepthSource(_ value: Int) {
        depthSource = value
        configureSession()
    }

    func start() {
        configureSession()
    }

    func stop() {
        session.pause()
    }

    private func configureSession() {
        let configuration = ARWorldTrackingConfiguration()
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            depthSupported = true
            if depthSource == 1 {
                configuration.frameSemantics.insert(.smoothedSceneDepth)
            } else {
                configuration.frameSemantics.insert(.sceneDepth)
            }
        } else {
            depthSupported = false
        }
        configuration.environmentTexturing = .none
        configuration.planeDetection = []
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard viewportSize != .zero else { return }
        let transform = frame.displayTransform(for: interfaceOrientation, viewportSize: viewportSize)

        let depthData: ARDepthData?
        if depthSupported {
            depthData = depthSource == 1 ? frame.smoothedSceneDepth : frame.sceneDepth
        } else {
            depthData = nil
        }

        let packet = FramePacket(
            pixelBuffer: frame.capturedImage,
            depthMap: depthData?.depthMap,
            confidenceMap: depthData?.confidenceMap,
            displayTransform: transform,
            viewportSize: viewportSize,
            intrinsics: frame.camera.intrinsics,
            imageResolution: CGSize(width: frame.camera.imageResolution.width, height: frame.camera.imageResolution.height),
            orientation: Self.cgOrientation(for: interfaceOrientation),
            timestamp: frame.timestamp,
            hasDepth: depthData != nil
        )
        continuation?.yield(packet)
    }

    private static func cgOrientation(for orientation: UIInterfaceOrientation) -> CGImagePropertyOrientation {
        switch orientation {
        case .portrait: return .right
        case .portraitUpsideDown: return .left
        case .landscapeLeft: return .up
        case .landscapeRight: return .down
        default: return .right
        }
    }
}
