# BGE & Qwen3 Models - Complete User Guide

**Status:** ✅ Production Ready  
**Last Updated:** 2025-12-11  
**Quick Start:** Use any model with the same unified API - just change the `model` parameter

---

## 🚀 Quick Start

### Choose Your Model in 30 Seconds

| Use Case | Model | Command |
|----------|-------|---------|
| **Fast multilingual** | `bge-m3` | `"model": "bge-m3"` |
| **Long context (32k)** | `qwen3-embedding-0.6b` | `"model": "qwen3-embedding-0.6b"` |
| **High quality** | `qwen3-embedding-4b` | `"model": "qwen3-embedding-4b"` |
| **Code reranking** | `bge-reranker-v2-m3` | `"model": "bge-reranker-v2-m3"` |
| **Long reranking** | `qwen3-reranker-0.6b` | `"model": "qwen3-reranker-0.6b"` |

---

## 📊 Model Comparison

| Model | Type | Max Tokens | Dimensions | VRAM | Special Features |
|-------|------|------------|------------|------|------------------|
| **bge-m3** | Embedding | 8,192 | 1024 | 1.5GB | 100+ languages, MRL |
| **qwen3-embedding-0.6b** | Embedding | 32,768 | 1024 | 1.5GB | 32k context, instructions |
| **qwen3-embedding-4b** | Embedding | 32,768 | 1024 | 8GB | 32k context, high quality |
| **bge-reranker-v2-m3** | Reranker | 512 | - | 1.5GB | Fast reranking |
| **qwen3-reranker-0.6b** | Reranker | 32,768 | - | 1.5GB | Long context reranking |

---

## 🎯 Model Selection Guide

### BGE Models (BAAI)
- **bge-m3**: Best for general multilingual embedding, 100+ languages
- **bge-reranker-v2-m3**: Fast reranking for short texts (512 tokens max)

### Qwen3 Models (Alibaba)
- **qwen3-embedding-0.6b**: Instruction-aware embeddings with 32k context
- **qwen3-embedding-4b**: Higher quality embeddings with 32k context
- **qwen3-reranker-0.6b**: Reranking with 32k context support

---

## 🔧 API Usage

### Basic Embedding
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-m3",
    "input": ["Hello world", "Bonjour le monde"]
  }'
```

### Qwen3 with Task Instructions
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-embedding-0.6b",
    "task": "retrieval.query",
    "input": ["What is machine learning?"]
  }'
```

### Reranking
```bash
curl -X POST http://localhost:8080/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-reranker-v2-m3",
    "query": "What is AI?",
    "documents": ["AI is artificial intelligence", "The weather is nice"],
    "top_n": 1
  }'
```

---

## 🏷️ Task Instructions Support

### Model Support Matrix

| Model | Task Instructions | Custom Instructions | Late Chunking |
|-------|------------------|-------------------|---------------|
| **jina-embeddings-v3** | ✅ Task adapters | ❌ | ✅ Supported |
| **jina-embeddings-v4** | ✅ Task adapters | ❌ | ✅ Supported |
| **bge-m3** | ❌ No task support | ❌ | ❌ Rejected |
| **qwen3-embedding-0.6b** | ✅ 15+ instructions | ✅ Custom prompts | ❌ Rejected |
| **qwen3-embedding-4b** | ✅ 15+ instructions | ✅ Custom prompts | ❌ Rejected |
| **qwen3-embedding-8b** | ✅ 15+ instructions | ✅ Custom prompts | ❌ Rejected |

### Qwen3 Task Instructions Detail

#### Core Tasks (Jina-Compatible)
| Task | Instruction Text | Example Usage |
|------|------------------|---------------|
| `retrieval.query` | "Given a search query, retrieve relevant documents or passages that directly answer or address the query" | Search queries, questions |
| `retrieval.passage` | ❌ No instruction (documents don't need prompts) | Document embeddings |
| `text-matching` | "Represent this text for finding semantically similar texts" | Similarity search |
| `classification` | "Represent this text for classification" | Text categorization |
| `separation` / `clustering` | "Represent this text for clustering" | Topic clustering |

#### Extended Tasks (Qwen3 Enhanced)
| Task | Instruction Text | Example Usage |
|------|------------------|---------------|
| `code.query` | "Given a natural language description, retrieve relevant code snippets or implementations" | Code search from NL |
| `code.passage` | ❌ No instruction | Code document embedding |
| `scientific.query` | "Given a scientific question, retrieve relevant research papers or scientific information" | Research retrieval |
| `qa.query` | "Given a question, retrieve passages that contain the answer to the question" | Question answering |
| `bitext` | "Represent this text for finding its translation in another language" | Cross-lingual matching |
| `summarization.query` | "Given a summary, retrieve the original document that matches this summary" | Summary matching |

### Custom Task Instructions

Qwen3 models support **custom instructions** via the `prompt_name` parameter:

```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-embedding-0.6b",
    "prompt_name": "Represent this legal document for contract analysis",
    "input": ["This agreement shall terminate..."]
  }'
```

**Custom Instruction Format:**
- Any string passed to `prompt_name` is treated as a raw instruction
- Applied as: `"Instruct: [your instruction]\nQuery: [text]"`
- Useful for domain-specific tasks not covered by built-in tasks

---

## ⚠️ Important Limitations

### Late Chunking Support
| Model | Late Chunking | Reason |
|-------|---------------|--------|
| `jina-embeddings-v3` | ✅ Supported | Trained for mean pooling |
| `jina-embeddings-v4` | ✅ Supported | Trained for mean pooling |
| `bge-m3` | ❌ Rejected | Architecture causes embedding space collapse |
| `qwen3-embedding-*` | ❌ Rejected | Uses last-token pooling |

### Error Messages
**BGE-M3 Late Chunking:**
```json
{
  "error": {
    "message": "Model bge-m3 does not support late_chunking. BGE-M3's architecture causes embedding space collapse when late chunking is applied. Use jina-embeddings-v4 or voyage-context-3 for contextual embeddings."
  }
}
```

**Qwen3 Late Chunking:**
```json
{
  "error": {
    "message": "late_chunking is not supported for qwen3-embedding-0.6b. Qwen3 models use last-token pooling, which is incompatible with late chunking. Use jina-embeddings-v3 or jina-embeddings-v4 for late chunking support."
  }
}
```

---

## 📈 Performance Benchmarks

Based on comprehensive testing:

| Model | P@3 Score | Best Use Case |
|-------|-----------|---------------|
| `jina-embeddings-v4` | **0.847** | Best overall performance |
| `jina-embeddings-v3` | **0.824** | Balanced performance |
| `qwen3-embedding-0.6b` | **0.812** | Instruction-aware tasks |
| `bge-m3` | **0.798** | Fast general embedding |

---

## 🎓 Best Practices

### 1. Model Selection
- **General multilingual**: Use `bge-m3`
- **Long documents**: Use `qwen3-embedding-0.6b` (32k tokens)
- **High quality**: Use `qwen3-embedding-4b`
- **Fast reranking**: Use `bge-reranker-v2-m3`

### 2. Task Instructions
- **Queries**: Always use task instructions for better retrieval
- **Documents**: Use `retrieval.passage` (no instruction needed)
- **Classification**: Use `classification` task
- **Code search**: Use `code.query` task

### 3. Performance Tips
- **Dimensions**: Use `dimensions` parameter for smaller embeddings
- **Truncation**: Enable `truncate` for long inputs
- **Batch size**: Process multiple texts in one request

---

## 🔗 Related Documentation

- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Implementation Status](IMPLEMENTATION_STATUS.md) - Feature completion status
- [Benchmark Report](CONTEXTUAL_EMBEDDINGS_BENCHMARK_REPORT.md) - Performance analysis

---

**Need Help?** Check the [API Reference](API_REFERENCE.md) for detailed examples and troubleshooting.
