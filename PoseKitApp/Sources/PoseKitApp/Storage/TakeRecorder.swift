import Foundation
import UIKit
import simd

/// Metadata for a recorded take.
public struct TakeMeta: Codable, Identifiable, Sendable {
    public var id: UUID { uuid }
    public var uuid: UUID
    public var createdAt: Date
    public var duration: TimeInterval
    public var fps: Int
    public var deviceModel: String
    public var iosVersion: String
    public var modelIdentifier: String
    public var modelVersion: String
    public var minKeypointScore: Float
    public var jointSchema: String
    public var units: String
    public var frameCount: Int
    public var notes: String?
}

/// Actor responsible for streaming pose frames to disk.
public actor TakeRecorder {
    private var fileHandle: FileHandle?
    private var metaURL: URL?
    private var framesURL: URL?
    private var meta: TakeMeta?
    private var startTimestamp: TimeInterval?
    private var lastTimestamp: TimeInterval?

    public init() {}

    public func start(
        config: PosePipelineConfig,
        modelIdentifier: String,
        modelVersion: String
    ) async throws -> URL {
        let uuid = UUID()
        let createdAt = Date()
        let root = try Self.takesRootDirectory()
        let takeURL = root.appendingPathComponent(uuid.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: takeURL, withIntermediateDirectories: true)

        let framesURL = takeURL.appendingPathComponent("frames.bin")
        FileManager.default.createFile(atPath: framesURL.path, contents: nil)
        let handle = try FileHandle(forWritingTo: framesURL)

        let metaURL = takeURL.appendingPathComponent("meta.json")
        let iosVersion = await MainActor.run { UIDevice.current.systemVersion }
        let meta = TakeMeta(
            uuid: uuid,
            createdAt: createdAt,
            duration: 0,
            fps: config.targetFPS,
            deviceModel: Self.deviceModelIdentifier(),
            iosVersion: iosVersion,
            modelIdentifier: modelIdentifier,
            modelVersion: modelVersion,
            minKeypointScore: config.minKeypointScore,
            jointSchema: "COCO17+2",
            units: "meters",
            frameCount: 0,
            notes: nil
        )

        self.fileHandle = handle
        self.metaURL = metaURL
        self.framesURL = framesURL
        self.meta = meta
        self.startTimestamp = nil
        self.lastTimestamp = nil
        return takeURL
    }

    public func append(frame: PoseFrame) throws {
        guard let fileHandle, var meta else { return }
        if startTimestamp == nil {
            startTimestamp = frame.timestamp
        }
        lastTimestamp = frame.timestamp

        let data = PoseFrameBinaryCodec.encode(frame)
        try fileHandle.write(contentsOf: data)
        meta.frameCount += 1
        self.meta = meta
    }

    public func stop() throws -> URL? {
        guard let metaURL, let framesURL, var meta else { return nil }
        try fileHandle?.close()
        fileHandle = nil

        if let start = startTimestamp, let end = lastTimestamp {
            meta.duration = max(0, end - start)
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(meta)
        try data.write(to: metaURL, options: [.atomic])

        self.meta = nil
        self.startTimestamp = nil
        self.lastTimestamp = nil
        return framesURL.deletingLastPathComponent()
    }

    public static func takesRootDirectory() throws -> URL {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let root = documents.appendingPathComponent("PoseTakes", isDirectory: true)
        if !FileManager.default.fileExists(atPath: root.path) {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        }
        return root
    }

    private static func deviceModelIdentifier() -> String {
        var systemInfo = utsname()
        uname(&systemInfo)
        let machine = withUnsafePointer(to: &systemInfo.machine) { ptr -> String in
            let int8Ptr = UnsafeRawPointer(ptr).assumingMemoryBound(to: CChar.self)
            return String(cString: int8Ptr)
        }
        return machine
    }
}

/// Codec for the binary frame layout used by `frames.bin`.
public enum PoseFrameBinaryCodec {
    // Layout per frame (little-endian):
    // timestamp: Float64
    // For each of 19 joints:
    // camera3D.x,y,z: Float32 * 3
    // playback3D.x,y,z: Float32 * 3
    // xNorm2D,yNorm2D: Float32 * 2
    // score2D: Float32
    // depthConfidence: Float32
    public static func encode(_ frame: PoseFrame) -> Data {
        let jointCount = SkeletonSchema.jointOrder.count
        var data = Data(capacity: 8 + jointCount * (3 + 3 + 2 + 1 + 1) * 4)
        data.appendDouble(frame.timestamp)
        for index in 0..<jointCount {
            let cam = index < frame.camera3D.count ? frame.camera3D[index] : SIMD3<Float>(repeating: .nan)
            let play = index < frame.playback3D.count ? frame.playback3D[index] : SIMD3<Float>(repeating: .nan)
            let kp = index < frame.joints2D.count ? frame.joints2D[index] : Keypoint2D(xNorm: 0.5, yNorm: 0.5, score: 0)
            let conf = index < frame.depthConfidences.count ? frame.depthConfidences[index] : 0
            data.appendFloat(cam.x)
            data.appendFloat(cam.y)
            data.appendFloat(cam.z)
            data.appendFloat(play.x)
            data.appendFloat(play.y)
            data.appendFloat(play.z)
            data.appendFloat(kp.xNorm)
            data.appendFloat(kp.yNorm)
            data.appendFloat(kp.score)
            data.appendFloat(conf)
        }
        return data
    }

    public static func decode(_ data: Data) -> [PoseFrame] {
        let jointCount = SkeletonSchema.jointOrder.count
        let stride = 8 + jointCount * (3 + 3 + 2 + 1 + 1) * 4
        guard stride > 0 else { return [] }
        let frameCount = data.count / stride
        guard frameCount > 0 else { return [] }

        return data.withUnsafeBytes { rawBuffer in
            var frames: [PoseFrame] = []
            frames.reserveCapacity(frameCount)
            var offset = 0
            for _ in 0..<frameCount {
                let timeBits = rawBuffer.load(fromByteOffset: offset, as: UInt64.self).littleEndian
                let timestamp = Double(bitPattern: timeBits)
                offset += 8

                var joints2D: [Keypoint2D] = []
                var camera3D: [SIMD3<Float>] = []
                var playback3D: [SIMD3<Float>] = []
                var confidences: [Float] = []
                joints2D.reserveCapacity(jointCount)
                camera3D.reserveCapacity(jointCount)
                playback3D.reserveCapacity(jointCount)
                confidences.reserveCapacity(jointCount)

                for _ in 0..<jointCount {
                    let camX = Float(bitPattern: rawBuffer.load(fromByteOffset: offset, as: UInt32.self).littleEndian)
                    offset += 4
                    let camY = Float(bitPattern: rawBuffer.load(fromByteOffset: offset, as: UInt32.self).littleEndian)
                    offset += 4
                    let camZ = Float(bitPattern: rawBuffer.load(fromByteOffset: offset, as: UInt32.self).littleEndian)
                    offset += 4
                    let playX = Float(bitPattern: rawBuffer.load(fromByteOffset: offset, as: UInt32.self).littleEndian)
                    offset += 4
                    let playY = Float(bitPattern: rawBuffer.load(fromByteOffset: offset, as: UInt32.self).littleEndian)
                    offset += 4
                    let playZ = Float(bitPattern: rawBuffer.load(fromByteOffset: offset, as: UInt32.self).littleEndian)
                    offset += 4
                    let xNorm = Float(bitPattern: rawBuffer.load(fromByteOffset: offset, as: UInt32.self).littleEndian)
                    offset += 4
                    let yNorm = Float(bitPattern: rawBuffer.load(fromByteOffset: offset, as: UInt32.self).littleEndian)
                    offset += 4
                    let score = Float(bitPattern: rawBuffer.load(fromByteOffset: offset, as: UInt32.self).littleEndian)
                    offset += 4
                    let conf = Float(bitPattern: rawBuffer.load(fromByteOffset: offset, as: UInt32.self).littleEndian)
                    offset += 4

                    camera3D.append(SIMD3<Float>(camX, camY, camZ))
                    playback3D.append(SIMD3<Float>(playX, playY, playZ))
                    joints2D.append(Keypoint2D(xNorm: xNorm, yNorm: yNorm, score: score))
                    confidences.append(conf)
                }

                frames.append(PoseFrame(
                    timestamp: timestamp,
                    joints2D: joints2D,
                    camera3D: camera3D,
                    playback3D: playback3D,
                    depthConfidences: confidences
                ))
            }
            return frames
        }
    }
}

private extension Data {
    mutating func appendFloat(_ value: Float) {
        var le = value.bitPattern.littleEndian
        Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) }
    }

    mutating func appendDouble(_ value: Double) {
        var le = value.bitPattern.littleEndian
        Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) }
    }
}
