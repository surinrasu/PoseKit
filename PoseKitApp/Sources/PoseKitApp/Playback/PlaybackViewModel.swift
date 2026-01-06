import Foundation
import SwiftUI

@MainActor
final class PlaybackViewModel: ObservableObject {
    let take: PoseTakeSummary

    @Published var frames: [PoseFrame] = []
    @Published var currentIndex: Int = 0
    @Published var isPlaying: Bool = false
    @Published var exportSheetPresented = false

    private var playbackTask: Task<Void, Never>?

    init(take: PoseTakeSummary) {
        self.take = take
        load()
    }

    var currentFrame: PoseFrame? {
        guard currentIndex >= 0, currentIndex < frames.count else { return nil }
        return frames[currentIndex]
    }

    var progress: Double {
        guard frames.count > 1 else { return 0 }
        return Double(currentIndex) / Double(frames.count - 1)
    }

    func load() {
        frames = TakeLoader.loadFrames(from: take.url)
    }

    func togglePlay() {
        isPlaying.toggle()
        if isPlaying {
            startPlayback()
        } else {
            playbackTask?.cancel()
            playbackTask = nil
        }
    }

    func seek(to progress: Double) {
        guard !frames.isEmpty else { return }
        let clamped = max(0, min(1, progress))
        currentIndex = Int((Double(frames.count - 1) * clamped).rounded())
    }

    private func startPlayback() {
        playbackTask?.cancel()
        playbackTask = Task { [weak self] in
            guard let self else { return }
            let interval = UInt64(1_000_000_000 / max(self.take.meta.fps, 1))
            while !Task.isCancelled {
                await MainActor.run {
                    if self.currentIndex + 1 < self.frames.count {
                        self.currentIndex += 1
                    } else {
                        self.isPlaying = false
                    }
                }
                if !(await MainActor.run { self.isPlaying }) { break }
                try? await Task.sleep(nanoseconds: interval)
            }
        }
    }
}
