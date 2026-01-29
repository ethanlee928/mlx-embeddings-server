# MLX-Embeddings Server

This package offers an OpenAI-compatible API server for the [`MLX-Embeddings`](https://github.com/Blaizzy/mlx-embeddings) library — specifically for `colqwen2.5`.
It exists because there is currently no server available for multimodal embedding models in MLX.

## Installation

Prerequisites: `uv` installed.

```bash
# Initialize and sync environment
uv sync
# Development setup (optional)
uv sync --group dev
```

## Usage

### Start the Server

```bash
uv run uvicorn mlx_embeddings_server.main:app --host 0.0.0.0 --port 8888
```

The server will load the `qnguyen3/colqwen2.5-v0.2-mlx` model on startup (this may take a moment).

### API Usage

The server provides an OpenAI-compatible `POST /v1/embeddings` endpoint.

#### Text Embedding

```bash
curl http://localhost:8888/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "How many percent of data are books?",
    "model": "colqwen2.5"
  }'
```

#### Image Embedding

Pass an image URL or a Data URI as the `input`.

```bash
curl http://localhost:8888/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/07/US_Declaration_in_Stone.jpg/440px-US_Declaration_in_Stone.jpg",
    "model": "colqwen2.5"
  }'
```

The response `embedding` field will be a **list of vectors** (e.g., `[[0.1, ...], [0.5, ...]]`) because ColQwen/ColBERT is a multi-vector model.

#### Health Check

```bash
curl http://localhost:8888/health
```

## Formatting

Format code:

```bash
uv run ruff format .
uv run ruff check --fix .
```

## References

- [`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm)
- [`mlx-embeddings`](https://github.com/Blaizzy/mlx-embeddings)
- [OpenAI Embeddings API](https://platform.openai.com/docs/api-reference/embeddings)
  
  Embedding API:

  ```bash
  curl https://api.openai.com/v1/embeddings \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "input": "The food was delicious and the waiter...",
      "model": "text-embedding-ada-002",
      "encoding_format": "float"
    }'
  ```

  Response:

  ```json
  {
    "object": "list",
    "data": [
      {
        "object": "embedding",
        "embedding": [
          0.0023064255,
          -0.009327292,
          .... (1536 floats total for ada-002)
          -0.0028842222,
        ],
        "index": 0
      }
    ],
    "model": "text-embedding-ada-002",
    "usage": {
      "prompt_tokens": 8,
      "total_tokens": 8
    }
  }
  ```

  Embedding Object:

  ```json
  {
    "object": "embedding",
    "embedding": [
      0.0023064255,
      -0.009327292,
      .... (1536 floats total for ada-002)
      -0.0028842222,
    ],
    "index": 0
  }
  ```
