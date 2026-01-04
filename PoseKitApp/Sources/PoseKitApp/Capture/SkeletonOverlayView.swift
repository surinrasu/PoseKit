import SwiftUI

struct SkeletonOverlayView: View {
    let frame: PoseFrame
    let displayTransform: CGAffineTransform
    let overlayStyle: Int

    var body: some View {
        GeometryReader { proxy in
            Canvas { context, size in
                let points = frame.joints2D.enumerated().map { index, kp -> (CGPoint, Float) in
                    let normalized = CGPoint(x: CGFloat(kp.xNorm), y: CGFloat(kp.yNorm))
                        .applying(displayTransform)
                    let point = CGPoint(x: normalized.x * size.width, y: normalized.y * size.height)
                    let depthZ = frame.camera3D[index].z
                    return (point, depthZ)
                }

                for edge in SkeletonSchema.edges {
                    guard let startIndex = SkeletonSchema.jointOrder.firstIndex(of: edge.0),
                          let endIndex = SkeletonSchema.jointOrder.firstIndex(of: edge.1) else { continue }
                    let (startPoint, startDepth) = points[startIndex]
                    let (endPoint, endDepth) = points[endIndex]
                    let depth = (startDepth + endDepth) * 0.5
                    let style = overlayStyleStyle(depth: depth)

                    var path = Path()
                    path.move(to: startPoint)
                    path.addLine(to: endPoint)
                    context.stroke(path, with: .color(Color.white.opacity(style.alpha)), lineWidth: style.lineWidth)
                }

                for (index, kp) in frame.joints2D.enumerated() {
                    let (point, depth) = points[index]
                    let style = overlayStyleStyle(depth: depth)
                    let rect = CGRect(
                        x: point.x - style.radius,
                        y: point.y - style.radius,
                        width: style.radius * 2,
                        height: style.radius * 2
                    )
                    context.fill(Path(ellipseIn: rect), with: .color(Color.green.opacity(style.alpha)))
                }
            }
        }
    }

    private func overlayStyleStyle(depth: Float) -> (radius: CGFloat, lineWidth: CGFloat, alpha: Double) {
        guard depth.isFinite else {
            return (radius: 5, lineWidth: 2, alpha: 0.9)
        }
        let minZ: Float = 0.3
        let maxZ: Float = 4.0
        let clamped = max(minZ, min(maxZ, depth))
        let t = 1 - (clamped - minZ) / (maxZ - minZ)
        let radius = CGFloat(3 + (10 - 3) * t)
        let lineWidth = CGFloat(1 + (4 - 1) * t)
        let alpha = overlayStyle >= 1 ? Double(0.4 + 0.6 * t) : 1.0
        return (radius: radius, lineWidth: lineWidth, alpha: alpha)
    }
}
