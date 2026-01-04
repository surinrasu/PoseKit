import SwiftUI

struct StatusBannerView: View {
    let statuses: [PosePipelineStatus]

    var body: some View {
        if let banner = bannerText() {
            Text(banner.text)
                .font(.footnote.bold())
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(banner.color.opacity(0.9))
                .foregroundStyle(.white)
                .clipShape(Capsule())
        }
    }

    private func bannerText() -> (text: String, color: Color)? {
        for status in statuses {
            switch status {
            case .depthUnavailable:
                return ("Depth Unavailable (LiDAR required)", .red)
            case .lowConfidence(let joints):
                let text = formatted(joints: joints, prefix: "Low Confidence")
                return (text, .yellow)
            case .outOfFrame(let joints):
                let text = formatted(joints: joints, prefix: "Out of Frame")
                return (text, .red)
            case .ok:
                continue
            }
        }
        return nil
    }

    private func formatted(joints: [Joint], prefix: String) -> String {
        let maxNames = 3
        let names = joints.prefix(maxNames).map { $0.displayName }
        if joints.count <= maxNames {
            return "\(prefix): \(names.joined(separator: ", "))"
        } else {
            return "\(prefix): \(joints.count) points"
        }
    }
}
