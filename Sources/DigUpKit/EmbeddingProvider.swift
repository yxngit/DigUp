import Foundation

/// An embedding server in place of the downloaded model files: an OpenAI-compatible `/v1/embeddings` endpoint,
/// written for oMLX on this Mac (mlx-community's EmbeddingGemma 2 among its models) and working with any server
/// that speaks the same API (LM Studio, llama.cpp's server, vLLM, …).
///
/// The choice lives in `provider.json` in Application Support, where the app, its helper processes and the `digup`
/// CLI all read it; `$DIGUP_PROVIDER_URL`, `$DIGUP_PROVIDER_MODEL` and `$DIGUP_PROVIDER_API_KEY` override it for a
/// run (a shell, an eval). With none of those, DigUp runs the model files it downloads, as before.
public struct EmbeddingProvider: Codable, Sendable, Equatable {
    /// The server's base, `http://localhost:8000` (with or without a trailing `/v1`).
    public var url: String
    /// The model the server serves, its own id (`embeddinggemma-2-mxfp8` for oMLX's copy of the mlx-downloader
    /// folder of that name).
    public var model: String
    /// Often empty on localhost.
    public var apiKey: String

    public init(url: String, model: String, apiKey: String) {
        self.url = url
        self.model = model
        self.apiKey = apiKey
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        url = try values.decodeIfPresent(String.self, forKey: .url) ?? ""
        model = try values.decodeIfPresent(String.self, forKey: .model) ?? ""
        apiKey = try values.decodeIfPresent(String.self, forKey: .apiKey) ?? ""
    }

    /// The endpoint for a request: `url` + `/v1/embeddings`, or just `/embeddings` when the base already ends in `/v1`.
    public var endpoint: URL {
        var base = url.trimmingCharacters(in: .whitespaces)
        while base.hasSuffix("/") { base.removeLast() }
        let path = base.hasSuffix("/v1") ? "/embeddings" : "/v1/embeddings"
        return URL(string: base + path) ?? URL(string: "http://localhost:8000/v1/embeddings")!
    }

    /// Whose vectors these are, for a log line: `embeddinggemma-2-mxfp8 at localhost:8000`.
    public var name: String {
        let host = URL(string: url)?.host ?? url
        return "\(model.isEmpty ? "?" : model) at \(host)"
    }

    /// Where the app keeps the choice, beside the model files and the indexes.
    public static var file: URL { AppPaths.applicationSupport.appendingPathComponent("provider.json") }

    /// The provider in force: the environment's override first, else `provider.json`. Nil: the built-in runtime.
    public static var active: EmbeddingProvider? {
        let environment = ProcessInfo.processInfo.environment
        if let url = environment["DIGUP_PROVIDER_URL"], !url.trimmingCharacters(in: .whitespaces).isEmpty {
            return EmbeddingProvider(url: url, model: environment["DIGUP_PROVIDER_MODEL"] ?? "",
                                     apiKey: environment["DIGUP_PROVIDER_API_KEY"] ?? "")
        }
        guard let data = try? Data(contentsOf: file),
              let provider = try? JSONDecoder().decode(EmbeddingProvider.self, from: data),
              !provider.url.trimmingCharacters(in: .whitespaces).isEmpty,
              !provider.model.trimmingCharacters(in: .whitespaces).isEmpty else { return nil }
        return provider
    }

    /// Remembers the provider (nil: back to the built-in runtime). The helpers and the CLI pick the file up at their
    /// next start.
    public static func save(_ provider: EmbeddingProvider?) throws {
        guard let provider else {
            try? FileManager.default.removeItem(at: file)
            return
        }
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(provider).write(to: file, options: .atomic)
    }
}
