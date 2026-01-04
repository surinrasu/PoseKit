import SwiftUI
import ARKit
import RealityKit
import UIKit

struct CaptureView: View {
    @ObservedObject var preferences: PreferencesStore
    @StateObject private var viewModel: CaptureViewModel

    init(preferences: PreferencesStore) {
        self.preferences = preferences
        _viewModel = StateObject(wrappedValue: CaptureViewModel(config: preferences.config))
    }

    var body: some View {
        ZStack {
            GeometryReader { proxy in
                ARViewContainer(session: viewModel.sessionManager.session)
                    .ignoresSafeArea()
                    .onAppear {
                        viewModel.sessionManager.updateViewport(size: proxy.size, orientation: currentOrientation())
                        viewModel.start()
                    }
                    .onDisappear {
                        viewModel.stop()
                    }
                    .onChange(of: proxy.size) { newSize in
                        viewModel.sessionManager.updateViewport(size: newSize, orientation: currentOrientation())
                    }
            }

            if let output = viewModel.output {
                SkeletonOverlayView(
                    frame: output.frame,
                    displayTransform: output.displayTransform,
                    overlayStyle: output.overlayStyle
                )
                .ignoresSafeArea()
            }

            VStack {
                HStack {
                    Spacer()
                    Button {
                        openSystemSettings()
                    } label: {
                        Image(systemName: "gearshape")
                            .foregroundStyle(.white)
                            .padding(10)
                            .background(.black.opacity(0.5))
                            .clipShape(Circle())
                    }
                    .padding(.trailing)
                }
                Spacer()

                if let output = viewModel.output, preferences.config.showConfidenceBanner {
                    StatusBannerView(statuses: output.statusMessages)
                        .padding(.bottom, 12)
                }

                ZStack {
                    Circle()
                        .fill(viewModel.isRecording ? Color.red : Color.white)
                        .frame(width: 76, height: 76)
                        .overlay(
                            Circle()
                                .stroke(Color.white.opacity(0.8), lineWidth: 3)
                                .frame(width: 86, height: 86)
                        )
                    if viewModel.isRecording {
                        Text(durationString(viewModel.recordingDuration))
                            .font(.caption.bold())
                            .foregroundStyle(.white)
                            .offset(y: -54)
                    }
                }
                .padding(.bottom, 28)
                .onTapGesture {
                    viewModel.toggleRecording(config: preferences.config)
                }
            }
        }
        .onChange(of: preferences.config) { newValue in
            viewModel.updateConfig(newValue)
        }
        .onChange(of: scenePhase) { phase in
            if phase == .active {
                preferences.reload()
                viewModel.updateConfig(preferences.config)
            }
        }
        #if DEBUG
        .overlay(alignment: .topLeading) {
            Text(String(format: "FPS: %.1f", viewModel.debugFPS))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.white)
                .padding(8)
                .background(.black.opacity(0.6))
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .padding()
        }
        #endif
    }

    @Environment(\.scenePhase) private var scenePhase

    private func durationString(_ duration: TimeInterval) -> String {
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        return String(format: "%02d:%02d", minutes, seconds)
    }

    private func openSystemSettings() {
        guard let url = URL(string: UIApplication.openSettingsURLString) else { return }
        UIApplication.shared.open(url)
    }

    private func currentOrientation() -> UIInterfaceOrientation {
        let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene
        return scene?.interfaceOrientation ?? .portrait
    }
}

private struct ARViewContainer: UIViewRepresentable {
    let session: ARSession

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero)
        view.session = session
        view.automaticallyConfigureSession = false
        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {
        uiView.session = session
    }
}
