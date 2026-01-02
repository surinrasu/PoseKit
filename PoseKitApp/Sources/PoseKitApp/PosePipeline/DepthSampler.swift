import CoreVideo
import Foundation

/// A sampled depth value with an associated confidence score.
public struct DepthSample: Sendable {
    public var depth: Float
    public var confidence: Float

    public init(depth: Float, confidence: Float) {
        self.depth = depth
        self.confidence = confidence
    }
}

/// Sampling aggressiveness for depth confidence.
public enum DepthConfidenceMode: Int, Sendable {
    case strict = 0
    case balanced = 1
    case lenient = 2

    var minConfidenceLevel: UInt8 {
        switch self {
        case .strict: return 2
        case .balanced: return 1
        case .lenient: return 0
        }
    }

    var searchRadius: Int {
        switch self {
        case .strict: return 2
        case .balanced: return 3
        case .lenient: return 3
        }
    }
}

/// Utility for sampling depth and confidence around a 2D point.
public struct DepthSampler {
    public static func sample(
        uNorm: Float,
        vNorm: Float,
        depthMap: CVPixelBuffer,
        confidenceMap: CVPixelBuffer?,
        mode: DepthConfidenceMode
    ) -> DepthSample? {
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        if width == 0 || height == 0 {
            return nil
        }

        let x = Int((Float(width - 1) * uNorm).rounded())
        let y = Int((Float(height - 1) * vNorm).rounded())

        func depthAt(_ x: Int, _ y: Int) -> Float? {
            guard x >= 0, x < width, y >= 0, y < height else { return nil }
            let rowBytes = CVPixelBufferGetBytesPerRow(depthMap)
            guard let base = CVPixelBufferGetBaseAddress(depthMap) else { return nil }
            let offset = y * rowBytes + x * MemoryLayout<Float32>.size
            let value = base.advanced(by: offset).assumingMemoryBound(to: Float32.self).pointee
            if value.isNaN || value <= 0 { return nil }
            return value
        }

        let confidenceContext: (map: CVPixelBuffer, width: Int, height: Int, base: UnsafeMutableRawPointer, rowBytes: Int)? = {
            guard let confidenceMap else { return nil }
            CVPixelBufferLockBaseAddress(confidenceMap, .readOnly)
            guard let base = CVPixelBufferGetBaseAddress(confidenceMap) else {
                CVPixelBufferUnlockBaseAddress(confidenceMap, .readOnly)
                return nil
            }
            return (confidenceMap, CVPixelBufferGetWidth(confidenceMap), CVPixelBufferGetHeight(confidenceMap), base, CVPixelBufferGetBytesPerRow(confidenceMap))
        }()

        defer {
            if let confidenceContext {
                CVPixelBufferUnlockBaseAddress(confidenceContext.map, .readOnly)
            }
        }

        func confidenceAt(_ x: Int, _ y: Int) -> UInt8? {
            guard let confidenceContext else { return nil }
            let cWidth = confidenceContext.width
            let cHeight = confidenceContext.height
            guard x >= 0, x < cWidth, y >= 0, y < cHeight else { return nil }
            let offset = y * confidenceContext.rowBytes + x * MemoryLayout<UInt8>.size
            return confidenceContext.base.advanced(by: offset).assumingMemoryBound(to: UInt8.self).pointee
        }

        if let depth = depthAt(x, y) {
            let confidenceLevel = confidenceAt(x, y) ?? 2
            if confidenceLevel >= mode.minConfidenceLevel {
                return DepthSample(depth: depth, confidence: Float(confidenceLevel) / 2.0)
            }
        }

        let radius = mode.searchRadius
        var bestConfidence: UInt8 = 0
        var samples: [Float] = []

        for dy in -radius...radius {
            for dx in -radius...radius {
                let sx = x + dx
                let sy = y + dy
                guard let depth = depthAt(sx, sy) else { continue }
                let conf = confidenceAt(sx, sy) ?? 2
                if conf < mode.minConfidenceLevel { continue }
                if conf > bestConfidence {
                    bestConfidence = conf
                    samples.removeAll(keepingCapacity: true)
                }
                if conf == bestConfidence {
                    samples.append(depth)
                }
            }
        }

        guard !samples.isEmpty else { return nil }
        samples.sort()
        let median = samples[samples.count / 2]
        return DepthSample(depth: median, confidence: Float(bestConfidence) / 2.0)
    }
}
