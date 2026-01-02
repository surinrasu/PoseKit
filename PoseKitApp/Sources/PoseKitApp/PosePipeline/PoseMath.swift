import Foundation
import CoreGraphics
import simd

enum PoseMath {
    static func backProject(
        uNorm: Float,
        vNorm: Float,
        depth: Float,
        intrinsics: simd_float3x3,
        imageResolution: CGSize
    ) -> SIMD3<Float> {
        let width = Float(imageResolution.width)
        let height = Float(imageResolution.height)
        let uPx = uNorm * width
        let vPx = vNorm * height

        let fx = intrinsics.columns.0.x
        let fy = intrinsics.columns.1.y
        let cx = intrinsics.columns.2.x
        let cy = intrinsics.columns.2.y

        let x = (uPx - cx) / fx * depth
        let y = (vPx - cy) / fy * depth
        return SIMD3<Float>(x, y, depth)
    }
}
