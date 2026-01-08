import XCTest
@testable import PoseKitApp
import CoreGraphics
import simd

final class PoseMathTests: XCTestCase {
    func testBackProjectionSimple() {
        let intrinsics = simd_float3x3(
            SIMD3<Float>(1000, 0, 0),
            SIMD3<Float>(0, 1000, 0),
            SIMD3<Float>(500, 400, 1)
        )
        let point = PoseMath.backProject(
            uNorm: 0.5,
            vNorm: 0.5,
            depth: 2.0,
            intrinsics: intrinsics,
            imageResolution: CGSize(width: 1000, height: 800)
        )
        XCTAssertEqual(point.x, 0, accuracy: 0.0001)
        XCTAssertEqual(point.y, 0, accuracy: 0.0001)
        XCTAssertEqual(point.z, 2.0, accuracy: 0.0001)
    }

    func testCanonicalizationCentersPelvis() {
        let count = SkeletonSchema.jointOrder.count
        var camera3D = Array(repeating: SIMD3<Float>(0, 0, 0), count: count)
        var joints2D = Array(repeating: Keypoint2D(xNorm: 0.5, yNorm: 0.5, score: 1.0), count: count)

        func set(_ joint: Joint, _ value: SIMD3<Float>) {
            if let index = SkeletonSchema.jointOrder.firstIndex(of: joint) {
                camera3D[index] = value
            }
        }

        set(.pelvisCenter, SIMD3<Float>(0, 0, 0))
        set(.neckCenter, SIMD3<Float>(0, 0.5, 0))
        set(.leftHip, SIMD3<Float>(-0.2, 0, 0))
        set(.rightHip, SIMD3<Float>(0.2, 0, 0))
        set(.leftShoulder, SIMD3<Float>(-0.2, 0.5, 0))
        set(.rightShoulder, SIMD3<Float>(0.2, 0.5, 0))

        let canonical = PoseCanonicalizer().canonicalize(camera3D: camera3D, joints2D: joints2D, minScore: 0.3)
        let pelvis = canonical[SkeletonSchema.jointOrder.firstIndex(of: .pelvisCenter) ?? 0]
        let neck = canonical[SkeletonSchema.jointOrder.firstIndex(of: .neckCenter) ?? 0]
        let leftHip = canonical[SkeletonSchema.jointOrder.firstIndex(of: .leftHip) ?? 0]
        let rightHip = canonical[SkeletonSchema.jointOrder.firstIndex(of: .rightHip) ?? 0]

        XCTAssertEqual(pelvis.x, 0, accuracy: 0.0001)
        XCTAssertEqual(pelvis.y, 0, accuracy: 0.0001)
        XCTAssertEqual(pelvis.z, 0, accuracy: 0.0001)
        XCTAssertTrue(neck.y > 0)
        XCTAssertTrue(rightHip.x > leftHip.x)
    }

    func testBinaryRoundTrip() {
        let count = SkeletonSchema.jointOrder.count
        let joints2D = (0..<count).map { idx in
            Keypoint2D(xNorm: Float(idx) / Float(count), yNorm: 0.5, score: 0.8)
        }
        let camera3D = (0..<count).map { idx in
            SIMD3<Float>(Float(idx) * 0.01, Float(idx) * 0.02, Float(idx) * 0.03)
        }
        let playback3D = (0..<count).map { idx in
            SIMD3<Float>(Float(idx) * -0.01, Float(idx) * 0.01, Float(idx) * -0.02)
        }
        let confidences = Array(repeating: Float(0.5), count: count)

        let frame = PoseFrame(
            timestamp: 1.23,
            joints2D: joints2D,
            camera3D: camera3D,
            playback3D: playback3D,
            depthConfidences: confidences
        )

        let data = PoseFrameBinaryCodec.encode(frame)
        let decoded = PoseFrameBinaryCodec.decode(data)
        XCTAssertEqual(decoded.count, 1)
        guard let first = decoded.first else { return }
        XCTAssertEqual(first.timestamp, frame.timestamp, accuracy: 0.0001)
        XCTAssertEqual(first.joints2D.count, count)
        XCTAssertEqual(first.camera3D.count, count)
        XCTAssertEqual(first.playback3D.count, count)
        XCTAssertEqual(first.depthConfidences.count, count)
        XCTAssertEqual(first.camera3D[5].x, frame.camera3D[5].x, accuracy: 0.0001)
    }
}
