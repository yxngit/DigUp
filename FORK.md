# This fork: an embedding server in place of the model download

DigUp downloads EmbeddingGemma 2 (865 MB) from Hugging Face on first launch. On a machine where that
download can't happen, this fork lets an OpenAI-compatible `/v1/embeddings` server serve the embeddings
instead — written for [oMLX](https://github.com/JayKilll/omlx) on the same Mac, which already runs
mlx-community's EmbeddingGemma 2 (e.g. a `embeddinggemma-2-mxfp8` folder in its model directories),
and working with anything that speaks the same API (LM Studio, llama.cpp's server, vLLM, …).

## How to use

**In the app:** Settings → General → *Embeddings* → "Use an embedding server (oMLX)". Fill in the
server (`http://localhost:8000`), the model id as the server knows it (`embeddinggemma-2-mxfp8`) and
the API key when it has one. Apply asks the server for a vector before it saves anything.

**By hand:** write `~/Library/Application Support/DigUp/provider.json`:

```json
{
  "apiKey" : "omlxbilmstudiohaoshiYebiollamahaoyong",
  "model" : "embeddinggemma-2-mxfp8",
  "url" : "http://localhost:8000"
}
```

**For one run** (the CLI, evals): `DIGUP_PROVIDER_URL`, `DIGUP_PROVIDER_MODEL`, `DIGUP_PROVIDER_API_KEY`
in the environment override the file. With none of these set, DigUp is exactly as upstream: the model
downloads and llama.cpp runs it.

## What changes

- `Sources/DigUpKit/EmbeddingProvider.swift` — the choice (`provider.json`, the environment overrides),
  read by the app, its helpers and the `digup` CLI alike.
- `Sources/DigUpKit/HttpEmbedder.swift` — the `Embedder` that talks to the server: texts in batches of
  32, pictures one request each as base64 data URIs (oMLX's multimodal `items`), everything
  L2-normalized whatever the server returned.
- `Sources/digup/main.swift`, `Sources/digup/WorkerService.swift` — `makeEmbedder()` picks the server
  when one is configured; indexing no longer waits for the model files.
- `Sources/DigUpApp` — Settings → General → Embeddings; the query encoder and the indexing helper
  restart when the choice changes; the download screen stands down while a server is in force.

## What's different from the built-in runtime

- **Vectors are the server's.** An index never mixes them with llama.cpp's: the provider's own vector
  source goes into the index fingerprint, so the first run after switching re-embeds every file (and
  so does switching back).
- **Audio gets no vectors.** The endpoint takes text and images, not sound: a recording stays findable
  by its words and name, and a video keeps the vectors of its frames but not of its soundtrack.
- **The server truncates long texts** at 512 tokens (oMLX's engine default), where llama.cpp cuts at
  2048: very long document chunks keep their beginning only.
- Pictures are sized by the server's model, not by DigUp's vision token budgets.
