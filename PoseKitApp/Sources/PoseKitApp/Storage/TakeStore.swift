import Foundation

public struct PoseTakeSummary: Identifiable, Sendable {
    public var id: UUID { meta.uuid }
    public var meta: TakeMeta
    public var url: URL
}

@MainActor
final class TakeStore: ObservableObject {
    @Published var takes: [PoseTakeSummary] = []

    func refresh() {
        do {
            let root = try TakeRecorder.takesRootDirectory()
            let directories = (try? FileManager.default.contentsOfDirectory(
                at: root,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )) ?? []
            var summaries: [PoseTakeSummary] = []
            let decoder = JSONDecoder()
            for url in directories where url.hasDirectoryPath {
                let metaURL = url.appendingPathComponent("meta.json")
                guard let data = try? Data(contentsOf: metaURL) else { continue }
                if let meta = try? decoder.decode(TakeMeta.self, from: data) {
                    summaries.append(PoseTakeSummary(meta: meta, url: url))
                }
            }
            takes = summaries.sorted { $0.meta.createdAt > $1.meta.createdAt }
        } catch {
            takes = []
        }
    }

    func delete(_ take: PoseTakeSummary) {
        try? FileManager.default.removeItem(at: take.url)
        refresh()
    }
}
