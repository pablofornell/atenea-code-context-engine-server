# Atenea Context Engine - Server

This is the backend server for the Atenea Context Engine. It provides a REST API for indexing and querying codebases. It is designed to be self-hosted and acts as a "black box" for the CLI client.

## Components
- **HTTP API**: Handles communication with the CLI.
- **Chunker**: Splits source files into semantic chunks using Tree-sitter.
- **Embedder**: Generates vector embeddings using Ollama (`nomic-embed-text`).
- **Vector Store**: Manages storage and retrieval using Qdrant.

## Setup

1. **Prerequisites**:
   - Docker & Docker Compose
   - Ollama installed on the host or accessible via network.

2. **Installation**:
   ```bash
   make setup
   ```

3. **Running**:
   ```bash
   make run
   ```

The server will be available at `http://localhost:8080` by default, or at the host/port configured in `.env`.

## Configuration

All runtime settings are read from a `.env` file in the `atenea-server/` directory. Copy the example and edit it:

```bash
cp .env.example .env
```

`.env` is loaded automatically on every `make run` — no need to prefix environment variables each time. Variables already set in the shell always take precedence.

| Variable | Default | Description |
|---|---|---|
| `HOST` | `127.0.0.1` | Bind address (`0.0.0.0` to expose to the network) |
| `PORT` | `8080` | TCP port |
| `ATENEA_SECRET` | *(unset)* | Shared secret for AES-256-GCM encryption |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API base URL |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `ATENEA_LOG_LEVEL` | `INFO` | Log level |

## Encryption

When exposing the server over the internet, set `ATENEA_SECRET` in your `.env` file to enable **AES-256-GCM** end-to-end encryption:

```bash
# .env
HOST=0.0.0.0
PORT=8443
ATENEA_SECRET=change-me-to-a-strong-random-secret
```

Then just run:
```bash
make run
```

- All request and response **bodies** are encrypted end-to-end.
- Encryption is **opt-in**: if `ATENEA_SECRET` is not set the server runs in plaintext mode (suitable for local use).
- The same secret must be set on the CLI / MCP client side (see the CLI README).
- The secret is never transmitted over the wire — only the derived AES-256 key is used locally to encrypt/decrypt payloads.

> **Note:** URL paths and query parameters (e.g. collection names) are not encrypted because they carry no sensitive data. Only body payloads (file contents, query results) are protected.
