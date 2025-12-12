# Jina Local API Server

A **production-ready, local Jina-compatible API server** running in a **CUDA-enabled Docker container** that serves state-of-the-art embedding and reranking models from multiple providers.

## ✨ Features

- **🔄 Drop-in replacement** for the official Jina AI API (`api.jina.ai`)
- **🤖 10+ supported models** across 3 providers:
  - **Jina AI**: v3, v4, Code embeddings, Reranker
  - **BAAI**: BGE-M3, BGE Reranker
  - **Alibaba**: Qwen3 Embeddings (0.6B, 4B, 8B), Qwen3 Reranker
- **🚀 GPU acceleration** with CUDA and Flash Attention 2 support
- **📊 Advanced features**: Late chunking, multimodal embeddings, task instructions
- **🎯 Task-aware embeddings** with 15+ specialized tasks
- **⚡ Production-ready** with health checks, structured logging, and error handling
- **🐳 Docker-native** with multi-stage builds and non-root user

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

| Model | VRAM (FP16) | Max Tokens | Special Features |
|-------|-------------|------------|------------------|
| **Jina Models** |
| jina-embeddings-v3 | ~1.5 GB | 8,192 | Late chunking, task adapters |
| jina-embeddings-v4 | ~6 GB | 8,192 | Multimodal, multi-vector |
| jina-code-embeddings-0.5b | ~1 GB | 8,192 | Code-specific tasks |
| jina-code-embeddings-1.5b | ~3 GB | 8,192 | Full-size code model |
| jina-reranker-v3 | ~2 GB | 512 | Listwise reranking |
| **BGE Models (BAAI)** |
| bge-m3 | ~1.5 GB | 8,192 | 100+ languages, MRL |
| bge-reranker-v2-m3 | ~1.5 GB | 512 | Fast reranking |
| **Qwen3 Models (Alibaba)** |
| qwen3-embedding-0.6b | ~1.5 GB | 32,768 | 32k context, instructions |
| qwen3-embedding-4b | ~8 GB | 32,768 | High quality, 32k context |
| qwen3-embedding-8b | ~12 GB | 32,768 | Largest Qwen3 variant |
| qwen3-reranker-0.6b | ~1.5 GB | 32,768 | Long context reranking |
| **Total (all models)** | **~35+ GB** | - | **Choose models based on needs** |

## API Compatibility

This server is designed to be a drop-in replacement for the Jina AI API with **enhanced features**. The following table shows compatibility:

| Jina API Option | Supported | Notes |
|-----------------|-----------|-------|
| `model` | ✅ | 10+ models across 3 providers |
| `input` (text) | ✅ | Single string or array |
| `input` (multimodal) | ✅ | v4 only: `{text: ...}` or `{image: ...}` |
| `normalized` | ✅ | Default: true |
| `embedding_type: float` | ✅ | Default |
| `embedding_type: base64` | ✅ | Base64-encoded floats |
| `embedding_type: binary` | ✅ | Binary quantization |
| `task` | ✅ | 15+ tasks across all models |
| `dimensions` | ✅ | MRL truncation (32-2048) |
| `prompt_name` | ✅ | v4: query/passage, Qwen3: custom instructions |
| `late_chunking` | ✅ | v3/v4 only (rejected for BGE/Qwen3) |
| `return_multivector` | ✅ | v4 and BGE-M3 support |
| `query` (rerank) | ✅ | Required |
| `documents` (rerank) | ✅ | Required |
| `top_n` (rerank) | ✅ | Optional |
| `return_documents` | ✅ | Optional |

## 🚀 Quick Examples

### BGE-M3 (Fast Multilingual)
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-m3",
    "input": ["Hello world", "Bonjour le monde", "Hola mundo"],
    "dimensions": 512
  }'
```

### Qwen3 (32k Context + Instructions)
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-embedding-0.6b",
    "task": "retrieval.query",
    "input": ["What are the latest advances in quantum computing?"]
  }'
```

### Qwen3 (Custom Instructions)
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-embedding-0.6b",
    "prompt_name": "Represent this legal contract for clause extraction",
    "input": ["This agreement shall terminate upon breach of contract..."]
  }'
```

### Late Chunking (v3/v4 only)
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v4",
    "late_chunking": true,
    "input": ["Long document with multiple sections..."]
  }'
```

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

## 📚 Additional Documentation

- [API Reference](docs/API_REFERENCE.md) - Complete API documentation with examples
- [BGE & Qwen3 User Guide](docs/BGE_QWEN3_USER_GUIDE.md) - Comprehensive guide for BGE and Qwen3 models
- [Implementation Status](docs/IMPLEMENTATION_STATUS.md) - Feature completion and status report
- [Benchmark Report](docs/CONTEXTUAL_EMBEDDINGS_BENCHMARK_REPORT.md) - Performance analysis and comparisons
- [Test Results](docs/EMBEDDINGS_TEST_RESULTS.md) - Detailed test results and validation

## 🎯 Task Support by Model

### Jina AI Models

#### jina-embeddings-v3
- `retrieval.query` - Query embeddings for retrieval
- `retrieval.passage` - Passage embeddings for retrieval  
- `text-matching` - Symmetric text similarity
- `classification` - Text classification
- `separation` - Clustering/separation
- **Features**: Late chunking, MRL dimensions

#### jina-embeddings-v4
- `retrieval` - Multimodal retrieval (text + image)
- `text-matching` - Text similarity
- `code` - Code understanding
- **Features**: Multimodal, multi-vector output, late chunking

#### jina-code-embeddings (0.5b/1.5b)
- `nl2code` - Natural language to code retrieval
- `code2code` - Code similarity
- `code2nl` - Code to documentation
- `code2completion` - Code completion retrieval
- `qa` - Technical Q&A

#### jina-reranker-v3
- Document reranking for given query and documents
- **Features**: Listwise reranking, top_n filtering

### BGE Models (BAAI)

#### bge-m3
- **General-purpose text embeddings** (100+ languages)
- Tasks: `retrieval.query`, `retrieval.passage`, `text-matching`, `classification`, `separation`
- **Features**: MRL support, multi-vector (ColBERT), no task instructions

#### bge-reranker-v2-m3
- **Fast document reranker** (512 tokens max)
- Listwise reranking for query-document pairs

### Qwen3 Models (Alibaba)

#### qwen3-embedding (0.6b/4b/8b)
- **Instruction-aware embeddings** with 32k context
- **Jina-compatible tasks**:
  - `retrieval.query` / `retrieval.passage`
  - `text-matching`, `classification`, `separation` / `clustering`
- **Qwen-enhanced tasks**:
  - `code.query` - Code retrieval from natural language
  - `scientific.query` - Academic/research retrieval
  - `qa.query` - Question answering retrieval
  - `bitext` - Cross-lingual matching
  - `summarization.query` - Summary-to-document matching
- **Features**: Custom instructions via `prompt_name`, 32k tokens

#### qwen3-reranker-0.6b
- **Long-context reranker** (32k tokens)
- Instruction-aware reranking with yes/no scoring

## 📊 Performance Highlights

| Model | P@3 Score | Special Capability |
|-------|-----------|-------------------|
| `jina-embeddings-v4` | **0.847** | Best overall performance |
| `jina-embeddings-v3` | **0.824** | Balanced performance + late chunking |
| `qwen3-embedding-0.6b` | **0.812** | 32k context + instructions |
| `bge-m3` | **0.798** | Fast multilingual (100+ languages) |

*Based on comprehensive benchmarking on retrieval tasks*

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
