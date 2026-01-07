import SwiftUI

@main
struct PoseKitApp: App {
    @StateObject private var preferences = PreferencesStore()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            RootTabView()
                .environmentObject(preferences)
                .onChange(of: scenePhase) { phase in
                    if phase == .active {
                        preferences.reload()
                    }
                }
        }
    }
}
