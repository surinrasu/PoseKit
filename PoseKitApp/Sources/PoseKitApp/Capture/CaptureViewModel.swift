import Foundation
import SwiftUI

@MainActor
final class CaptureViewModel: ObservableObject {
    @Published var output: PosePipelineOutput?
    @Published var isRecording = false
    @Published var recordingDuration: TimeInterval = 0
    @Published var debugFPS: Double = 0

    let sessionManager = ARSessionManager()
    private let pipeline: PosePipeline
    private var pipelineTask: Task<Void, Never>?
    private var recorder: TakeRecorder?
    private var recordingStart: TimeInterval?
    private var lastOutputTimestamp: TimeInterval?
    init(config: PosePipelineConfig) {
        self.pipeline = PosePipeline(config: config)
    }

    func start() {
        sessionManager.start()
        pipelineTask?.cancel()
        Task { [weak self] in
            guard let self else { return }
            let task = await pipeline.start(stream: sessionManager.frameStream) { [weak self] output in
                self?.handleOutput(output)
            }
            await MainActor.run {
                self.pipelineTask = task
            }
        }
    }

    func stop() {
        pipelineTask?.cancel()
        pipelineTask = nil
        sessionManager.stop()
        if isRecording {
            stopRecording()
        }
    }

    func updateConfig(_ config: PosePipelineConfig) {
        Task {
            await pipeline.updateConfig(config)
        }
        sessionManager.updateDepthSource(config.depthSource)
    }

    func toggleRecording(config: PosePipelineConfig) {
        if isRecording {
            stopRecording()
        } else {
            startRecording(config: config)
        }
    }

    private func startRecording(config: PosePipelineConfig) {
        let recorder = TakeRecorder()
        self.recorder = recorder
        recordingStart = nil
        Task {
            do {
                _ = try await recorder.start(
                    config: config,
                    modelIdentifier: PoseKitModel.identifier,
                    modelVersion: PoseKitModel.version
                )
                await MainActor.run {
                    self.isRecording = true
                    self.recordingDuration = 0
                }
            } catch {
                await MainActor.run {
                    self.isRecording = false
                    self.recorder = nil
                }
            }
        }
    }

    private func stopRecording() {
        guard let recorder else { return }
        Task {
            _ = try? await recorder.stop()
            await MainActor.run {
                self.isRecording = false
                self.recordingStart = nil
                self.recordingDuration = 0
                self.recorder = nil
            }
        }
    }

    private func handleOutput(_ output: PosePipelineOutput) {
        self.output = output
        if let last = lastOutputTimestamp {
            let delta = output.frame.timestamp - last
            if delta > 0 {
                debugFPS = 1.0 / delta
            }
        }
        lastOutputTimestamp = output.frame.timestamp
        if isRecording {
            if recordingStart == nil {
                recordingStart = output.frame.timestamp
            }
            if let start = recordingStart {
                recordingDuration = max(0, output.frame.timestamp - start)
            }
            Task {
                try? await recorder?.append(frame: output.frame)
            }
        }
    }
}
