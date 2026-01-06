import SwiftUI

struct PlaybackView: View {
    @StateObject private var viewModel: PlaybackViewModel
    @State private var show3D = true
    @State private var userOverride = false

    init(take: PoseTakeSummary) {
        _viewModel = StateObject(wrappedValue: PlaybackViewModel(take: take))
    }

    var body: some View {
        let has3D = has3DData(viewModel.currentFrame)
        let use3D = has3D ? show3D : false
        VStack(spacing: 16) {
            ZStack {
                Color.black
                if let frame = viewModel.currentFrame {
                    if use3D {
                        Pose3DSceneView(frame: frame)
                    } else {
                        SkeletonOverlayView(
                            frame: frame,
                            displayTransform: .identity,
                            overlayStyle: 1
                        )
                    }
                } else {
                    Text("No frames")
                        .foregroundStyle(.white.opacity(0.7))
                        .font(.caption)
                }
            }
            .aspectRatio(3 / 4, contentMode: .fit)
            .cornerRadius(16)
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(Color.white.opacity(0.12), lineWidth: 1)
            )
            .overlay(alignment: .topTrailing) {
                Button {
                    show3D.toggle()
                    userOverride = true
                } label: {
                    Image(systemName: use3D ? "view.2d" : "view.3d")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(.white)
                        .padding(8)
                        .background(.black.opacity(0.6))
                        .clipShape(Circle())
                }
                .disabled(!has3D)
                .opacity(has3D ? 1 : 0.4)
                .padding(16)
            }
            .padding(.horizontal)

            HStack(spacing: 16) {
                Button {
                    viewModel.togglePlay()
                } label: {
                    Image(systemName: viewModel.isPlaying ? "pause.fill" : "play.fill")
                        .font(.title2)
                        .frame(width: 44, height: 44)
                }

                Slider(value: Binding(
                    get: { viewModel.progress },
                    set: { viewModel.seek(to: $0) }
                ))
            }
            .padding(.horizontal)

            Button {
                viewModel.exportSheetPresented = true
            } label: {
                Label("Export", systemImage: "square.and.arrow.up")
            }
            .buttonStyle(.borderedProminent)
            .padding(.bottom)
        }
        .navigationTitle("Playback")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $viewModel.exportSheetPresented) {
            ExportSheet(take: viewModel.take)
        }
        .onChange(of: has3D) { newValue in
            if !newValue {
                show3D = false
                userOverride = false
            } else if !userOverride {
                show3D = true
            }
        }
    }

    private func has3DData(_ frame: PoseFrame?) -> Bool {
        guard let frame else { return false }
        return containsValidPoint(frame.playback3D) || containsValidPoint(frame.camera3D)
    }

    private func containsValidPoint(_ points: [SIMD3<Float>]) -> Bool {
        for p in points {
            if !(p.x.isNaN || p.y.isNaN || p.z.isNaN) {
                return true
            }
        }
        return false
    }
}
