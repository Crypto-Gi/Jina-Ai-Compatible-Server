# Jina Local API Server

A **local Jina-compatible API server** running in a **CUDA-enabled Docker container** that serves Jina AI embedding and reranking models.

## Features

- **Drop-in replacement** for the official Jina AI API (`api.jina.ai`)
- **5 supported models**:
  - `jina-embeddings-v3` - Multilingual text embeddings
  - `jina-embeddings-v4` - Multimodal embeddings (text + image)
  - `jina-code-embeddings-0.5b` - Lightweight code embeddings
  - `jina-code-embeddings-1.5b` - Full-size code embeddings
  - `jina-reranker-v3` - Listwise document reranker
- **GPU acceleration** with CUDA and Flash Attention 2 support
- **Production-ready** with health checks, structured logging, and error handling
- **Docker-native** with multi-stage builds and non-root user

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone and navigate to the project
cd /path/to/jina-local-server

# Configure which models to load (edit .env file)
# Default loads only jina-embeddings-v3 + jina-reranker-v3 (~3.5 GB VRAM)
cp .env.example .env
nano .env  # Edit MODELS_TO_LOAD as needed

# Build and run with GPU support
docker compose up -d

# Check logs
docker compose logs -f

# Test the server
curl http://localhost:8080/healthz
```


### Manual Docker Build

```bash
# Build the image
docker build -t jina-local-server .

# Run with GPU
docker run -d \
  --gpus all \
  -p 8080:8080 \
  -v jina-cache:/app/.cache \
  --name jina-server \
  jina-local-server
```

## API Reference

### Generate Embeddings

```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v3",
    "input": ["Hello, world!", "How are you?"],
    "task": "text-matching"
  }'
```

**Response:**
```json
{
  "object": "list",
  "model": "jina-embeddings-v3",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.123, -0.456, ...]},
    {"object": "embedding", "index": 1, "embedding": [0.789, -0.012, ...]}
  ],
  "usage": {"prompt_tokens": 10, "total_tokens": 10}
}
```

### Multimodal Embeddings (v4)

```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v4",
    "input": [
      {"text": "A beautiful sunset"},
      {"image": "https://example.com/sunset.jpg"}
    ],
    "task": "retrieval",
    "prompt_name": "query"
  }'
```

### Rerank Documents

```bash
curl -X POST http://localhost:8080/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-reranker-v3",
    "query": "What is machine learning?",
    "documents": [
      "Machine learning is a subset of AI...",
      "The weather today is sunny...",
      "Deep learning uses neural networks..."
    ],
    "top_n": 2
  }'
```

**Response:**
```json
{
  "model": "jina-reranker-v3",
  "results": [
    {"index": 0, "relevance_score": 0.95},
    {"index": 2, "relevance_score": 0.87}
  ],
  "usage": {"total_tokens": 150}
}
```

### List Models

```bash
curl http://localhost:8080/v1/models
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | Device to run models on (`cuda` or `cpu`) |
| `MODELS_TO_LOAD` | `all` | Comma-separated list of models or `all` |
| `TORCH_DTYPE` | `float16` | Model precision (`float16`, `bfloat16`, `float32`) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `json` | Log format (`json` or `console`) |
| `MAX_BATCH_SIZE` | `32` | Maximum batch size for inference |
| `HF_TOKEN` | - | HuggingFace token (for private models) |

### Load Specific Models

To save GPU memory, you can load only specific models:

```bash
# Load only embeddings v3 and reranker
MODELS_TO_LOAD=jina-embeddings-v3,jina-reranker-v3 docker compose up -d
```

### Memory Requirements

| Model | Approximate VRAM (FP16) |
|-------|-------------------------|
| jina-embeddings-v3 | ~1.5 GB |
| jina-embeddings-v4 | ~6 GB |
| jina-code-embeddings-0.5b | ~1 GB |
| jina-code-embeddings-1.5b | ~3 GB |
| jina-reranker-v3 | ~2 GB |
| **Total (all models)** | **~13.5 GB** |

## API Compatibility

This server is designed to be a drop-in replacement for the Jina AI API. The following table shows compatibility:

| Jina API Option | Supported | Notes |
|-----------------|-----------|-------|
| `model` | ✅ | All 5 models |
| `input` (text) | ✅ | Single string or array |
| `input` (multimodal) | ✅ | v4 only: `{text: ...}` or `{image: ...}` |
| `normalized` | ✅ | Default: true |
| `embedding_type: float` | ✅ | Default |
| `embedding_type: base64` | ✅ | Base64-encoded floats |
| `embedding_type: binary` | ✅ | Binary quantization |
| `task` | ✅ | Model-specific tasks |
| `dimensions` | ✅ | MRL truncation |
| `prompt_name` | ✅ | v4: query/passage |
| `query` (rerank) | ✅ | Required |
| `documents` (rerank) | ✅ | Required |
| `top_n` (rerank) | ✅ | Optional |
| `return_documents` | ✅ | Optional |

## Health Endpoints

- `GET /healthz` - Basic health check (always returns 200 if server is running)
- `GET /readyz` - Readiness check (returns 200 if models are loaded, 503 otherwise)

## Development

### Local Development (without Docker)

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run the server
uvicorn app.main:app --reload --port 8080
```

### Run Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

### Project Structure

```
jina-local-server/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Settings from environment
│   ├── logging_config.py    # Structured logging
│   ├── api/
│   │   ├── embeddings.py    # POST /v1/embeddings
│   │   ├── rerank.py        # POST /v1/rerank
│   │   └── models.py        # GET /v1/models
│   ├── models/
│   │   ├── base.py          # Abstract base wrappers
│   │   ├── registry.py      # Model registry
│   │   ├── embeddings_v3.py # jina-embeddings-v3
│   │   ├── embeddings_v4.py # jina-embeddings-v4
│   │   ├── code_embeddings.py # jina-code-embeddings
│   │   └── reranker.py      # jina-reranker-v3
│   └── schemas/
│       ├── embeddings.py    # Request/response schemas
│       ├── rerank.py
│       └── models.py
├── config/
│   └── models.yaml          # Model registry config
├── tests/
│   ├── test_unit/
│   └── test_integration/
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
└── README.md
```

## Task Support by Model

### jina-embeddings-v3
- `retrieval.query` - Query embeddings for retrieval
- `retrieval.passage` - Passage embeddings for retrieval
- `text-matching` - Symmetric text similarity
- `classification` - Text classification
- `separation` - Clustering/separation

### jina-embeddings-v4
- `retrieval` - Multimodal retrieval
- `text-matching` - Text similarity
- `code` - Code understanding

### jina-code-embeddings
- `nl2code` - Natural language to code retrieval
- `code2code` - Code similarity
- `code2nl` - Code to documentation
- `code2completion` - Code completion retrieval
- `qa` - Technical Q&A

## Troubleshooting

### CUDA Out of Memory

If you run out of GPU memory, try:
1. Load fewer models: `MODELS_TO_LOAD=jina-embeddings-v3`
2. Use a lower batch size: `MAX_BATCH_SIZE=8`
3. Use CPU fallback: `DEVICE=cpu`

### Model Download Issues

Models are downloaded from HuggingFace on first run. If you have issues:
1. Check internet connectivity
2. Set `HF_TOKEN` if accessing gated models
3. Pre-download models to the cache volume

### Container Won't Start

1. Check NVIDIA drivers: `nvidia-smi`
2. Verify Docker GPU support: `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`
3. Check logs: `docker compose logs jina-server`

## License

This server implementation is provided for local development and testing purposes. The Jina AI models are subject to their respective licenses on HuggingFace.

## Acknowledgments

- [Jina AI](https://jina.ai/) for the embedding and reranking models
- [HuggingFace](https://huggingface.co/) for the transformers library
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
