import Foundation

struct TakeLoader {
    static func loadMeta(from takeURL: URL) -> TakeMeta? {
        let metaURL = takeURL.appendingPathComponent("meta.json")
        guard let data = try? Data(contentsOf: metaURL) else { return nil }
        return try? JSONDecoder().decode(TakeMeta.self, from: data)
    }

    static func loadFrames(from takeURL: URL) -> [PoseFrame] {
        let framesURL = takeURL.appendingPathComponent("frames.bin")
        guard let data = try? Data(contentsOf: framesURL) else { return [] }
        return PoseFrameBinaryCodec.decode(data)
    }
}
