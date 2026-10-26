import Foundation

/// The `Embedder` behind `EmbeddingProvider`: `/v1/embeddings` on an OpenAI-compatible server (oMLX on this Mac),
/// where the model the server has loaded replaces the 865 MB of files DigUp would download.
///
/// Texts go in one request per 32 (what the server embeds in a batch anyway; it cuts each at 512 tokens of its own).
/// Pictures go in one request each, as a base64 data URI in oMLX's `items` multimodal input — a file the server
/// can't read or decode costs only its own vector, never its neighbours'. Audio has no input type on the endpoint
/// (text and image only), so every audio window comes back "no vector": its file stays findable by its words, and a
/// video keeps the vectors of its frames. Whatever the server returns is L2-normalized here, the shape the index
/// and the llama.cpp runtime share.
///
/// Vectors from a server are their own `vectorSource` (the server's model and its preprocessing, this runtime's
/// normalization), so an index never mixes them with llama.cpp's: switching provider re-embeds, switching back too.
public final class HttpEmbedder: Embedder {
    /// Changes when what this class sends, or how it normalizes, changes the vectors.
    public static let runtimeName = "http-embeddings1"
    private static let batchSize = 32

    public static func vectorSource(_ provider: EmbeddingProvider) -> String {
        "\(info(for: provider).vectorSource)"
    }

    private let provider: EmbeddingProvider
    private let session: URLSession
    public private(set) var info: WorkerInfo

    /// `textOnly` is only a report (the server loads its model once for everything); the first call also checks the
    /// server is reachable, and throws with its answer when it isn't.
    public init(provider: EmbeddingProvider, textOnly: Bool) throws {
        self.provider = provider
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 300
        configuration.timeoutIntervalForResource = 600
        session = URLSession(configuration: configuration)
        info = Self.info(for: provider, textOnly: textOnly)
        let started = Date()
        let warm = try embedTexts([Prompts.query("warm up")])
        // This index is 768 wide (`Vectors.dimension`): a server serving another model must say so now, not halfway
        // through the first pass.
        guard let vector = warm.first, vector.count == info.dim else {
            throw WorkerError.failed("\(provider.name) serves \(warm.first?.count ?? 0) dimensions, "
                + "this index needs \(info.dim): another model?")
        }
        let seconds = (Date().timeIntervalSince(started) * 100).rounded() / 100
        info = WorkerInfo(model: info.model, revision: info.revision, runtime: info.runtime, dtype: info.dtype,
                          dim: info.dim, textOnly: textOnly, loadSeconds: seconds)
    }

    private static func info(for provider: EmbeddingProvider, textOnly: Bool = false) -> WorkerInfo {
        let host = URL(string: provider.url)?.host ?? provider.url
        let port = URL(string: provider.url)?.port.map { ":\($0)" } ?? ""
        return WorkerInfo(model: "openai:\(provider.model)", revision: "\(host)\(port)",
                          runtime: runtimeName, dtype: "server-side", dim: Vectors.dimension, textOnly: textOnly,
                          loadSeconds: nil)
    }

    public func close() {
        session.finishTasksAndInvalidate()
    }

    // MARK: Embedder

    public func embedTexts(_ texts: [String]) throws -> [[Float]] {
        var vectors: [[Float]] = []
        vectors.reserveCapacity(texts.count)
        var start = texts.startIndex
        while start < texts.endIndex {
            let end = texts.index(start, offsetBy: Self.batchSize, limitedBy: texts.endIndex) ?? texts.endIndex
            vectors += try request(texts: Array(texts[start..<end])).map(Self.normalized)
            start = end
        }
        return vectors
    }

    public func embedImages(_ paths: [String], budget: Int) throws -> [Embedding] {
        // The budget (the llama.cpp runtime's vision soft tokens) is the server's business: it sizes pictures the
        // way its model was trained. One request a file, so a failure costs only that file's vector.
        paths.map { path in
            do {
                let data = try Data(contentsOf: URL(fileURLWithPath: path))
                let uri = "data:\(Self.mime(of: data));base64," + data.base64EncodedString()
                return Embedding(vector: try request(items: [["image": uri]]).map(Self.normalized).first, error: nil)
            } catch {
                return Embedding(vector: nil, error: String("\(error)".prefix(300)))
            }
        }
    }

    public func embedAudio(_ paths: [String]) throws -> [Embedding] {
        paths.map { _ in Embedding(vector: nil, error: "no audio embeddings from \(provider.name): words only") }
    }

    // MARK: Requests

    private func request(texts: [String]) throws -> [[Double]] {
        var body: [String: Any] = ["model": provider.model]
        body["input"] = texts
        return try send(body)
    }

    private func request(items: [[String: String]]) throws -> [[Double]] {
        try send(["model": provider.model, "items": items])
    }

    /// One POST, one retry on a transport error (a server busy loading the model drops the first connection).
    private func send(_ body: [String: Any]) throws -> [[Double]] {
        do {
            return try post(body)
        } catch let error as WorkerError {
            throw error   // the server answered: its word goes out as it is
        } catch {
            Thread.sleep(forTimeInterval: 0.5)
            return try post(body)
        }
    }

    private func post(_ body: [String: Any]) throws -> [[Double]] {
        var request = URLRequest(url: provider.endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if !provider.apiKey.isEmpty { request.setValue("Bearer \(provider.apiKey)", forHTTPHeaderField: "Authorization") }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let reply = SyncHTTP.send(request, session: session)
        if let failure = reply.failure { throw failure }
        guard reply.status == 200, !reply.body.isEmpty else {
            throw WorkerError.failed("\(provider.name) answered \(reply.status)"
                + (Self.message(in: reply.body).map { ": \($0)" } ?? ""))
        }
        guard let answer = try? JSONSerialization.jsonObject(with: reply.body) as? [String: Any],
              let rows = answer["data"] as? [[String: Any]] else {
            throw WorkerError.failed("\(provider.name) didn't answer embeddings")
        }
        return try rows.sorted { ($0["index"] as? Int ?? 0) < ($1["index"] as? Int ?? 0) }.map { row in
            guard let values = row["embedding"] as? [Any] else {
                throw WorkerError.failed("\(provider.name) returned no vector for input \(row["index"] ?? -1)")
            }
            return values.compactMap { $0 as? Double }
        }
    }

    private static func message(in body: Data) -> String? {
        guard let answer = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
              let error = answer["error"] as? [String: Any], let text = error["message"] as? String else { return nil }
        return String(text.prefix(200))
    }

    /// Unit length, whatever the server returned: the index and the search compare dot products.
    private static func normalized(_ values: [Double]) -> [Float] {
        var sum = 0.0
        for value in values { sum += value * value }
        let scale = sum > 0 ? Float(1 / sum.squareRoot()) : 0
        return values.map { Float($0) * scale }
    }

    /// The picture formats the endpoint takes, read from the bytes (an extension lies as often as a camera does).
    private static func mime(of data: Data) -> String {
        let bytes = [UInt8](data.prefix(16))
        func equals(_ at: Int, _ ascii: String) -> Bool {
            let marks = Array(ascii.utf8)
            guard at + marks.count <= bytes.count else { return false }
            return Array(bytes[at..<(at + marks.count)]) == marks
        }
        if bytes.count >= 2, bytes[0] == 0xFF, bytes[1] == 0xD8 { return "image/jpeg" }
        if bytes.count >= 4, bytes[0] == 0x89, bytes[1] == 0x50 { return "image/png" }
        if equals(0, "RIFF"), equals(8, "WEBP") { return "image/webp" }
        if equals(0, "GIF8") { return "image/gif" }
        if bytes.count >= 2, bytes[0] == 0x42, bytes[1] == 0x4D { return "image/bmp" }
        if equals(4, "ftyp") { return "image/heic" }
        return "image/jpeg"
    }
}

/// One synchronous request over a session, for callers that are synchronous themselves (the `Embedder` protocol:
/// one thread at a time). The box is written by URLSession's queue and read after the wait.
private final class SyncHTTP: @unchecked Sendable {
    let semaphore = DispatchSemaphore(value: 0)
    var status = 0
    var body = Data()
    var failure: Error?

    static func send(_ request: URLRequest, session: URLSession) -> SyncHTTP {
        let box = SyncHTTP()
        session.dataTask(with: request) { data, response, error in
            if let error { box.failure = error } else { box.body = data ?? Data() }
            box.status = (response as? HTTPURLResponse)?.statusCode ?? 0
            box.semaphore.signal()
        }.resume()
        box.semaphore.wait()
        return box
    }
}
