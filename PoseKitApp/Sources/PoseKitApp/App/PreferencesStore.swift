import Foundation
import SwiftUI

@MainActor
final class PreferencesStore: ObservableObject {
    @Published var config: PosePipelineConfig

    init() {
        config = PosePipelineConfig(
            minKeypointScore: 0.3,
            targetFPS: 30,
            depthSource: 0,
            depthConfidenceMode: 1,
            overlayStyle: 1,
            showConfidenceBanner: true
        )
        PreferencesStore.registerDefaults()
        config = PreferencesStore.loadConfig()
    }

    func reload() {
        config = PreferencesStore.loadConfig()
    }

    private static func registerDefaults() {
        let bundles: [Bundle] = {
            #if SWIFT_PACKAGE
            return [Bundle.main, Bundle.module]
            #else
            return [Bundle.main]
            #endif
        }()

        for bundle in bundles {
            guard let url = bundle.url(forResource: "Root", withExtension: "plist", subdirectory: "Settings.bundle"),
                  let data = try? Data(contentsOf: url),
                  let plist = try? PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any],
                  let preferences = plist["PreferenceSpecifiers"] as? [[String: Any]] else {
                continue
            }

            var defaults: [String: Any] = [:]
            for preference in preferences {
                if let key = preference["Key"] as? String,
                   let defaultValue = preference["DefaultValue"] {
                    defaults[key] = defaultValue
                }
            }
            UserDefaults.standard.register(defaults: defaults)
            return
        }
    }

    private static func loadConfig() -> PosePipelineConfig {
        let defaults = UserDefaults.standard
        let rawMinScore = defaults.float(forKey: "minKeypointScore")
        let minScore = max(0, min(1, rawMinScore))
        let targetFPS = defaults.integer(forKey: "targetFPS")
        let depthSource = defaults.integer(forKey: "depthSource")
        let depthConfidenceMode = defaults.integer(forKey: "depthConfidenceMode")
        let overlayStyle = defaults.integer(forKey: "overlayStyle")
        let showBanner = defaults.bool(forKey: "showConfidenceBanner")

        return PosePipelineConfig(
            minKeypointScore: minScore,
            targetFPS: targetFPS == 0 ? 30 : targetFPS,
            depthSource: depthSource,
            depthConfidenceMode: depthConfidenceMode,
            overlayStyle: overlayStyle,
            showConfidenceBanner: showBanner
        )
    }
}
