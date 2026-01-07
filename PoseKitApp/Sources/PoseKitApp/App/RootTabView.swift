import SwiftUI

struct RootTabView: View {
    @EnvironmentObject private var preferences: PreferencesStore

    var body: some View {
        TabView {
            CaptureView(preferences: preferences)
                .tabItem {
                    Label("Capture", systemImage: "camera.fill")
                }

            LibraryView()
                .tabItem {
                    Label("Library", systemImage: "tray.full")
                }
        }
    }
}

private struct LibraryView: View {
    @StateObject private var store = TakeStore()

    var body: some View {
        NavigationStack {
            List {
                ForEach(store.takes) { take in
                    NavigationLink {
                        PlaybackView(take: take)
                    } label: {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(take.meta.createdAt, style: .date)
                                .font(.headline)
                            Text("Duration: \(take.meta.duration, specifier: "%.1f")s | FPS: \(take.meta.fps)")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .onDelete { indices in
                    indices.map { store.takes[$0] }.forEach(store.delete)
                }
            }
            .navigationTitle("Library")
            .onAppear {
                store.refresh()
            }
            .refreshable {
                store.refresh()
            }
        }
    }
}
