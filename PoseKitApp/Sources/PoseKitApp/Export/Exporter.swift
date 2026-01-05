import Foundation
import simd

enum ExportFormat: String, CaseIterable {
    case json
    case binary
    case zip

    var fileExtension: String {
        switch self {
        case .json: return "json"
        case .binary: return "bin"
        case .zip: return "zip"
        }
    }
}

struct Exporter {
    static func export(take: PoseTakeSummary, format: ExportFormat) throws -> URL {
        switch format {
        case .json:
            return try exportJSON(take: take)
        case .binary:
            return try exportBinary(take: take)
        case .zip:
            return try exportZip(take: take)
        }
    }

    private static func exportBinary(take: PoseTakeSummary) throws -> URL {
        let sourceURL = take.url.appendingPathComponent("frames.bin")
        let destination = temporaryURL(for: take.meta, extension: "bin")
        try FileManager.default.copyItem(at: sourceURL, to: destination)
        return destination
    }

    private static func exportJSON(take: PoseTakeSummary) throws -> URL {
        let frames = TakeLoader.loadFrames(from: take.url)
        let export = PoseExport(meta: take.meta, frames: frames.map { PoseFrameJSON(frame: $0) })
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(export)
        let destination = temporaryURL(for: take.meta, extension: "json")
        try data.write(to: destination, options: [.atomic])
        return destination
    }

    private static func exportZip(take: PoseTakeSummary) throws -> URL {
        let metaURL = take.url.appendingPathComponent("meta.json")
        let framesURL = take.url.appendingPathComponent("frames.bin")
        let metaData = try Data(contentsOf: metaURL)
        let framesData = try Data(contentsOf: framesURL)

        let frames = TakeLoader.loadFrames(from: take.url)
        let export = PoseExport(meta: take.meta, frames: frames.map { PoseFrameJSON(frame: $0) })
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let jsonData = try encoder.encode(export)

        let archiveURL = temporaryURL(for: take.meta, extension: "zip")
        var writer = try ZipWriter(outputURL: archiveURL)
        try writer.addFile(named: "meta.json", data: metaData)
        try writer.addFile(named: "frames.bin", data: framesData)
        try writer.addFile(named: "frames.json", data: jsonData)
        try writer.finalize()
        return archiveURL
    }

    private static func temporaryURL(for meta: TakeMeta, extension ext: String) -> URL {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy-MM-dd'T'HHmmss'Z'"
        let timestamp = formatter.string(from: meta.createdAt)
        let filename = "PoseTake_\(timestamp).\(ext)"
        return FileManager.default.temporaryDirectory.appendingPathComponent(filename)
    }
}

private struct PoseExport: Codable {
    var meta: TakeMeta
    var frames: [PoseFrameJSON]
}

private struct PoseFrameJSON: Codable {
    var timestamp: TimeInterval
    var joints: [PoseJointJSON]

    init(frame: PoseFrame) {
        timestamp = frame.timestamp
        joints = zip(SkeletonSchema.jointOrder, frame.joints2D).enumerated().map { index, element in
            let (joint, kp) = element
            let cam = index < frame.camera3D.count ? frame.camera3D[index] : SIMD3<Float>(repeating: .nan)
            let play = index < frame.playback3D.count ? frame.playback3D[index] : SIMD3<Float>(repeating: .nan)
            let conf = index < frame.depthConfidences.count ? frame.depthConfidences[index] : 0
            return PoseJointJSON(
                joint: joint.displayName,
                camera3D: [cam.x, cam.y, cam.z],
                playback3D: [play.x, play.y, play.z],
                keypoint2D: [kp.xNorm, kp.yNorm],
                score2D: kp.score,
                depthConfidence: conf
            )
        }
    }
}

private struct PoseJointJSON: Codable {
    var joint: String
    var camera3D: [Float]
    var playback3D: [Float]
    var keypoint2D: [Float]
    var score2D: Float
    var depthConfidence: Float
}

private struct ZipWriter {
    private var fileHandle: FileHandle
    private var centralDirectory = Data()
    private var offset: UInt32 = 0
    private var entryCount: UInt16 = 0

    init(outputURL: URL) throws {
        FileManager.default.createFile(atPath: outputURL.path, contents: nil)
        fileHandle = try FileHandle(forWritingTo: outputURL)
    }

    mutating func addFile(named name: String, data: Data, date: Date = Date()) throws {
        let nameData = Data(name.utf8)
        let crc = CRC32.checksum(data)
        let mod = DOSDateTime(date: date)

        var localHeader = Data()
        localHeader.appendUInt32(0x04034b50)
        localHeader.appendUInt16(20)
        localHeader.appendUInt16(0)
        localHeader.appendUInt16(0)
        localHeader.appendUInt16(mod.time)
        localHeader.appendUInt16(mod.date)
        localHeader.appendUInt32(crc)
        localHeader.appendUInt32(UInt32(data.count))
        localHeader.appendUInt32(UInt32(data.count))
        localHeader.appendUInt16(UInt16(nameData.count))
        localHeader.appendUInt16(0)
        localHeader.append(nameData)

        try fileHandle.write(contentsOf: localHeader)
        try fileHandle.write(contentsOf: data)

        var centralHeader = Data()
        centralHeader.appendUInt32(0x02014b50)
        centralHeader.appendUInt16(20)
        centralHeader.appendUInt16(20)
        centralHeader.appendUInt16(0)
        centralHeader.appendUInt16(0)
        centralHeader.appendUInt16(mod.time)
        centralHeader.appendUInt16(mod.date)
        centralHeader.appendUInt32(crc)
        centralHeader.appendUInt32(UInt32(data.count))
        centralHeader.appendUInt32(UInt32(data.count))
        centralHeader.appendUInt16(UInt16(nameData.count))
        centralHeader.appendUInt16(0)
        centralHeader.appendUInt16(0)
        centralHeader.appendUInt16(0)
        centralHeader.appendUInt16(0)
        centralHeader.appendUInt32(0)
        centralHeader.appendUInt32(offset)
        centralHeader.append(nameData)

        centralDirectory.append(centralHeader)
        entryCount &+= 1
        offset += UInt32(localHeader.count + data.count)
    }

    mutating func finalize() throws {
        let centralOffset = offset
        try fileHandle.write(contentsOf: centralDirectory)
        offset += UInt32(centralDirectory.count)

        var endRecord = Data()
        endRecord.appendUInt32(0x06054b50)
        endRecord.appendUInt16(0)
        endRecord.appendUInt16(0)
        endRecord.appendUInt16(entryCount)
        endRecord.appendUInt16(entryCount)
        endRecord.appendUInt32(UInt32(centralDirectory.count))
        endRecord.appendUInt32(centralOffset)
        endRecord.appendUInt16(0)
        try fileHandle.write(contentsOf: endRecord)
        try fileHandle.close()
    }

}

private struct DOSDateTime {
    let time: UInt16
    let date: UInt16

    init(date: Date) {
        let calendar = Calendar(identifier: .gregorian)
        let components = calendar.dateComponents([.year, .month, .day, .hour, .minute, .second], from: date)
        let year = max(1980, components.year ?? 1980)
        let month = components.month ?? 1
        let day = components.day ?? 1
        let hour = components.hour ?? 0
        let minute = components.minute ?? 0
        let second = (components.second ?? 0) / 2
        self.time = UInt16((hour << 11) | (minute << 5) | second)
        self.date = UInt16(((year - 1980) << 9) | (month << 5) | day)
    }
}

private enum CRC32 {
    private static let table: [UInt32] = {
        (0..<256).map { i in
            var crc = UInt32(i)
            for _ in 0..<8 {
                if crc & 1 == 1 {
                    crc = 0xEDB88320 ^ (crc >> 1)
                } else {
                    crc >>= 1
                }
            }
            return crc
        }
    }()

    static func checksum(_ data: Data) -> UInt32 {
        var crc: UInt32 = 0xFFFFFFFF
        for byte in data {
            let index = Int((crc ^ UInt32(byte)) & 0xFF)
            crc = table[index] ^ (crc >> 8)
        }
        return crc ^ 0xFFFFFFFF
    }
}

private extension Data {
    mutating func appendUInt16(_ value: UInt16) {
        var le = value.littleEndian
        Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) }
    }

    mutating func appendUInt32(_ value: UInt32) {
        var le = value.littleEndian
        Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) }
    }
}
