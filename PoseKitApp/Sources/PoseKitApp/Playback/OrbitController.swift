import Foundation
import CoreGraphics
import simd

final class OrbitController {
    private(set) var yaw: Float = 0
    private(set) var pitch: Float = 0
    private(set) var radius: Float = 2.5

    func applyPan(delta: CGPoint) {
        let sensitivity: Float = 0.005
        yaw += Float(delta.x) * sensitivity
        pitch += Float(delta.y) * sensitivity
        pitch = max(-1.2, min(1.2, pitch))
    }

    func applyPinch(scale: CGFloat) {
        let factor = Float(1 / scale)
        radius = max(0.8, min(6.0, radius * factor))
    }

    func cameraTransform() -> simd_float4x4 {
        let x = radius * cos(pitch) * sin(yaw)
        let y = radius * sin(pitch)
        let z = radius * cos(pitch) * cos(yaw)
        let position = SIMD3<Float>(x, y, z)

        let lookAt = SIMD3<Float>(0, 0.8, 0)
        let up = SIMD3<Float>(0, 1, 0)
        return float4x4(lookAt: lookAt, from: position, up: up)
    }
}

private extension float4x4 {
    init(lookAt target: SIMD3<Float>, from eye: SIMD3<Float>, up: SIMD3<Float>) {
        let forward = simd_normalize(target - eye)
        let right = simd_normalize(simd_cross(forward, up))
        let newUp = simd_cross(right, forward)

        let rotation = simd_float4x4(
            SIMD4<Float>(right.x, newUp.x, -forward.x, 0),
            SIMD4<Float>(right.y, newUp.y, -forward.y, 0),
            SIMD4<Float>(right.z, newUp.z, -forward.z, 0),
            SIMD4<Float>(0, 0, 0, 1)
        )

        let translation = simd_float4x4(
            SIMD4<Float>(1, 0, 0, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(-eye.x, -eye.y, -eye.z, 1)
        )

        self = rotation * translation
    }
}
