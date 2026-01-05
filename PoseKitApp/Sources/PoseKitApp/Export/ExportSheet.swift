import SwiftUI

struct ExportSheet: View {
    let take: PoseTakeSummary

    @State private var selectedFormat: ExportFormat = .zip
    @State private var exportURL: URL?
    @State private var errorMessage: String?
    @State private var isWorking = false

    var body: some View {
        NavigationStack {
            Form {
                Section(header: Text("Export Format")) {
                    Picker("Format", selection: $selectedFormat) {
                        Text("Pose JSON").tag(ExportFormat.json)
                        Text("Pose Binary").tag(ExportFormat.binary)
                        Text("ZIP (meta + frames)").tag(ExportFormat.zip)
                    }
                    .pickerStyle(.segmented)
                }

                Section {
                    Button(isWorking ? "Preparing..." : "Generate") {
                        generate()
                    }
                    .disabled(isWorking)

                    if let exportURL {
                        ShareLink(item: exportURL) {
                            Label("Share", systemImage: "square.and.arrow.up")
                        }
                    }

                    if let errorMessage {
                        Text(errorMessage)
                            .foregroundStyle(.red)
                    }
                }
            }
            .navigationTitle("Export")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private func generate() {
        isWorking = true
        errorMessage = nil
        exportURL = nil
        Task {
            do {
                let url = try Exporter.export(take: take, format: selectedFormat)
                await MainActor.run {
                    exportURL = url
                    isWorking = false
                }
            } catch {
                await MainActor.run {
                    errorMessage = error.localizedDescription
                    isWorking = false
                }
            }
        }
    }
}
