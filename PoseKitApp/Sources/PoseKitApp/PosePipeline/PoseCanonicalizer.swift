import Foundation
import simd

/// Canonicalizes a pose to pelvis-centered coordinates for playback.
public struct PoseCanonicalizer {
    public init() {}

    public func canonicalize(
        camera3D: [SIMD3<Float>],
        joints2D: [Keypoint2D],
        minScore: Float
    ) -> [SIMD3<Float>] {
        guard camera3D.count == SkeletonSchema.jointOrder.count else { return camera3D }
        let order = SkeletonSchema.jointOrder

        func index(_ joint: Joint) -> Int {
            order.firstIndex(of: joint) ?? 0
        }

        func valid(_ v: SIMD3<Float>) -> Bool {
            !(v.x.isNaN || v.y.isNaN || v.z.isNaN)
        }

        func score(_ joint: Joint) -> Float {
            let idx = index(joint)
            guard idx < joints2D.count else { return 0 }
            return joints2D[idx].score
        }

        func validJoint(_ joint: Joint) -> Bool {
            score(joint) >= minScore
        }

        let pelvis = camera3D[index(.pelvisCenter)]
        let neck = camera3D[index(.neckCenter)]
        let leftHip = camera3D[index(.leftHip)]
        let rightHip = camera3D[index(.rightHip)]
        let leftShoulder = camera3D[index(.leftShoulder)]
        let rightShoulder = camera3D[index(.rightShoulder)]

        guard valid(pelvis), valid(neck) else { return camera3D }

        var up = neck - pelvis
        if simd_length(up) < 1e-4 { return camera3D }
        up = simd_normalize(up)

        var right = rightHip - leftHip
        if simd_length(right) < 1e-4 || !valid(rightHip) || !valid(leftHip) || !validJoint(.leftHip) || !validJoint(.rightHip) {
            right = rightShoulder - leftShoulder
        }
        if simd_length(right) < 1e-4 { return camera3D }
        right = simd_normalize(right)

        var forward = simd_cross(up, right)
        if simd_length(forward) < 1e-4 { return camera3D }
        forward = simd_normalize(forward)
        right = simd_normalize(simd_cross(forward, up))

        let rotation = simd_float3x3(columns: (right, up, forward))
        var output = camera3D
        for i in output.indices {
            let point = camera3D[i]
            if !valid(point) {
                output[i] = point
            } else {
                let translated = point - pelvis
                output[i] = rotation.transpose * translated
            }
        }
        return output
    }
}
