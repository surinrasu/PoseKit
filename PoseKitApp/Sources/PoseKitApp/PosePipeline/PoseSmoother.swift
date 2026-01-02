import Foundation
import simd

/// Exponential moving average smoother for per-joint 3D trajectories.
public struct PoseSmoother {
    private var previous: [SIMD3<Float>]
    private let alpha: Float

    public init(jointCount: Int, alpha: Float = 0.55) {
        self.previous = Array(repeating: SIMD3<Float>(repeating: .nan), count: jointCount)
        self.alpha = alpha
    }

    public mutating func smooth(_ values: [SIMD3<Float>]) -> [SIMD3<Float>] {
        guard values.count == previous.count else { return values }
        var output = values
        for i in values.indices {
            let current = values[i]
            let prev = previous[i]
            if current.x.isNaN || current.y.isNaN || current.z.isNaN {
                output[i] = current
            } else if prev.x.isNaN || prev.y.isNaN || prev.z.isNaN {
                output[i] = current
            } else {
                output[i] = prev * alpha + current * (1 - alpha)
            }
            previous[i] = output[i]
        }
        return output
    }
}
