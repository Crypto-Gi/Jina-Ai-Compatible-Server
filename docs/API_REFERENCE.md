# Jina Local API Server - Complete API Reference

This document provides a comprehensive reference for all API endpoints, parameters, and options supported by the Jina Local API Server. Each endpoint includes curl examples that can be tested against both our local server and the official Jina AI API.

---

## Table of Contents

1. [Base URLs](#base-urls)
2. [Authentication](#authentication)
3. [Quick Start & Model Selection](#quick-start--model-selection)
4. [Embeddings API](#embeddings-api)
   - [Model Comparison](#model-comparison)
   - [jina-embeddings-v3](#jina-embeddings-v3)
   - [jina-embeddings-v4](#jina-embeddings-v4)
   - [jina-code-embeddings](#jina-code-embeddings)
   - [bge-m3](#bge-m3)
   - [qwen3-embedding-models](#qwen3-embedding-models)
5. [Rerank API](#rerank-api)
6. [Models API](#models-api)
7. [Health & Docs](#health--docs)
8. [Response Formats](#response-formats)
9. [Error Handling](#error-handling)
10. [Common Usage Patterns](#common-usage-patterns)
11. [Performance Benchmarks](#performance-benchmarks)

---

## Base URLs

| Server | Base URL |
|--------|----------|
| **Local Server** | `http://localhost:8080` |
| **Jina AI Official** | `https://api.jina.ai` |

---

## Authentication

### Local Server
No authentication required for local deployment.

### Jina AI Official
Requires Bearer token in the `Authorization` header:
```bash
-H "Authorization: Bearer YOUR_JINA_API_KEY"
```

---

## Embeddings API

### Endpoint
```
POST /v1/embeddings
```

### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | ✅ | - | Model ID to use (see model lists below) |
| `input` | string \| array | ✅ | - | Text(s) or multimodal items to embed. Can be a single string, list of strings, or list of objects with `text`/`image` keys. |
| `normalized` | boolean | ❌ | `true` | **Internal only.** Embeddings are always L2-normalized per Jina spec; this flag is ignored in the public API schema. |
| `embedding_type` | string | ❌ | `"float"` | Output format: `float`, `base64`, `binary`, `ubinary`* |
| `task` | string | ❌ | model-specific | Task-specific embedding variant (retrieval, text-matching, code, etc.). |
| `dimensions` | integer | ❌ | model default | Target embedding dimensions (MRL truncation). |
| `prompt_name` | string | ❌ | - | Prompt variant for some v4 tasks (`"query"` or `"passage"`). |
| `late_chunking` | boolean | ❌ | `false` | Enable late chunking / contextual embeddings where supported (v3, v4, BGE). |
| `truncate` | boolean | ❌ | `false` | Truncate inputs exceeding max sequence length instead of raising. |
| `return_multivector` | boolean | ❌ | `false` | **Local extension (v4 only):** Return multi-vector (token-level) embeddings. |

> **Note:** `ubinary` is parsed but not implemented – the server returns a 422-style error with a clear message.
> **Note:** `return_multivector` is a local extension. Official Jina exposes multi-vector output via a dedicated `/v1/multi-vector` endpoint.

---

## jina-embeddings-v3

Multilingual text embeddings with task-specific LoRA adapters.

### Supported Tasks

| Task | Description |
|------|-------------|
| `text-matching` | Default, semantic similarity |
| `retrieval.query` | Query for retrieval |
| `retrieval.passage` | Passage/document for retrieval |
| `separation` | Cluster separation |
| `classification` | Text classification |

### Dimensions
Supports MRL (Matryoshka Representation Learning): 64, 128, 256, 512, 768, 1024 (default)

---

### Example 1: Basic Text Embedding

**Local Server:**
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v3",
    "input": ["Hello, world!", "How are you?"]
  }'
```

**Jina AI Official:**
```bash
curl -X POST https://api.jina.ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "jina-embeddings-v3",
    "input": ["Hello, world!", "How are you?"]
  }'
```

---

### Example 2: Retrieval Query Embedding

**Local Server:**
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v3",
    "task": "retrieval.query",
    "input": "What is machine learning?"
  }'
```

**Jina AI Official:**
```bash
curl -X POST https://api.jina.ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "jina-embeddings-v3",
    "task": "retrieval.query",
    "input": "What is machine learning?"
  }'
```

---

### Example 3: Retrieval Passage Embedding

**Local Server:**
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v3",
    "task": "retrieval.passage",
    "input": [
      "Machine learning is a branch of AI that enables computers to learn from data.",
      "Deep learning uses neural networks with many layers."
    ]
  }'
```

**Jina AI Official:**
```bash
curl -X POST https://api.jina.ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "jina-embeddings-v3",
    "task": "retrieval.passage",
    "input": [
      "Machine learning is a branch of AI that enables computers to learn from data.",
      "Deep learning uses neural networks with many layers."
    ]
  }'
```

---

### Example 4: Text Matching with Custom Dimensions

**Local Server:**
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v3",
    "task": "text-matching",
    "dimensions": 512,
    "input": ["Organic skincare for sensitive skin", "Natural beauty products"]
  }'
```

**Jina AI Official:**
```bash
curl -X POST https://api.jina.ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "jina-embeddings-v3",
    "task": "text-matching",
    "dimensions": 512,
    "input": ["Organic skincare for sensitive skin", "Natural beauty products"]
  }'
```

---

### Example 5: Base64 Output Encoding

**Local Server:**
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v3",
    "embedding_type": "base64",
    "input": ["This will be returned as base64"]
  }'
```

**Jina AI Official:**
```bash
curl -X POST https://api.jina.ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "jina-embeddings-v3",
    "embedding_type": "base64",
    "input": ["This will be returned as base64"]
  }'
```

---

### Example 6: Binary Quantization

**Local Server:**
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v3",
    "embedding_type": "binary",
    "input": ["Binary embeddings for efficient storage"]
  }'
```

**Jina AI Official:**
```bash
curl -X POST https://api.jina.ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "jina-embeddings-v3",
    "embedding_type": "binary",
    "input": ["Binary embeddings for efficient storage"]
  }'
```

---

### Example 7: Late Chunking Enabled

**Important:** When `late_chunking=true`, the API processes the entire document through the transformer first, then returns **multiple embeddings** (one per detected sentence/chunk). This preserves full document context in each chunk embedding.

**Local Server:**
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v3",
    "late_chunking": true,
    "input": ["Berlin is the capital of Germany. Its population is 3.85 million. The city is known for its history."]
  }'
```

**Response:** Returns 3 embeddings (one for each sentence), each containing full document context.

```json
{
  "data": [
    {"index": 0, "embedding": [...]},  // "Berlin is the capital of Germany."
    {"index": 1, "embedding": [...]},  // "Its population is 3.85 million."
    {"index": 2, "embedding": [...]}   // "The city is known for its history."
  ]
}
```

**Jina AI Official:**
```bash
curl -X POST https://api.jina.ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "jina-embeddings-v3",
    "late_chunking": true,
    "input": ["Berlin is the capital of Germany. Its population is 3.85 million. The city is known for its history."]
  }'
```

---

### Example 8: Full Options Combined

**Local Server:**
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v3",
    "task": "text-matching",
    "dimensions": 1024,
    "normalized": true,
    "truncate": true,
    "late_chunking": false,
    "embedding_type": "float",
    "input": [
      "Organic skincare for sensitive skin with aloe vera",
      "Natural beauty products for gentle skin care"
    ]
  }'
```

**Jina AI Official:**
```bash
curl -X POST https://api.jina.ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "jina-embeddings-v3",
    "task": "text-matching",
    "dimensions": 1024,
    "normalized": true,
    "truncate": true,
    "late_chunking": false,
    "embedding_type": "float",
    "input": [
      "Organic skincare for sensitive skin with aloe vera",
      "Natural beauty products for gentle skin care"
    ]
  }'
```

---

## jina-embeddings-v4

Multimodal embeddings supporting both text and images.

### Supported Tasks

| Task | Description |
|------|-------------|
| `text-matching` | Default, semantic similarity |
| `retrieval` | Retrieval tasks |
| `code` | Code-related embeddings |

### Input Formats

| Format | Example |
|--------|---------|
| Plain text | `"Hello world"` |
| Text object | `{"text": "Hello world"}` |
| Image URL | `{"image": "https://example.com/image.jpg"}` |
| Image base64 | `{"image": "data:image/png;base64,..."` |

---

### Example 9: Text with v4

**Local Server:**
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v4",
    "input": ["Hello, world!"]
  }'
```

**Jina AI Official:**
```bash
curl -X POST https://api.jina.ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "jina-embeddings-v4",
    "input": ["Hello, world!"]
  }'
```

---

### Example 10: Image Embedding (URL)

**Local Server:**
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v4",
    "input": [
      {"image": "https://example.com/sample-image.jpg"}
    ]
  }'
```

**Jina AI Official:**
```bash
curl -X POST https://api.jina.ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "jina-embeddings-v4",
    "input": [
      {"image": "https://example.com/sample-image.jpg"}
    ]
  }'
```

---

### Example 11: Mixed Text and Image

**Local Server:**
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v4",
    "input": [
      {"text": "A beautiful sunset over the ocean"},
      {"image": "https://example.com/sunset.jpg"},
      "Another text description"
    ]
  }'
```

**Jina AI Official:**
```bash
curl -X POST https://api.jina.ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "jina-embeddings-v4",
    "input": [
      {"text": "A beautiful sunset over the ocean"},
      {"image": "https://example.com/sunset.jpg"},
      "Another text description"
    ]
  }'
```

---

### Example 12: v4 with Prompt Name

**Local Server:**
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v4",
    "task": "retrieval",
    "prompt_name": "query",
    "input": ["What does this image show?"]
  }'
```

**Jina AI Official:**
```bash
curl -X POST https://api.jina.ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "jina-embeddings-v4",
    "task": "retrieval",
    "prompt_name": "query",
    "input": ["What does this image show?"]
  }'
```

---

### Example 13: Multi-Vector Output (ColBERT-style)

**Local Server:**
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v4",
    "task": "retrieval",
    "return_multivector": true,
    "input": ["ColBERT style retrieval"]
  }'
```

**Response:** Returns a list of vectors (one per token) for the input.

```json
{
  "data": [
    {
      "index": 0,
      "embedding": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...] // List of N vectors
    }
  ]
}
```

**Jina AI Official:**
```bash
curl -X POST https://api.jina.ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "jina-embeddings-v4",
    "task": "retrieval",
    "return_multivector": true,
    "input": ["ColBERT style retrieval"]
  }'
```

---

## jina-code-embeddings

Code embeddings optimized for code search and retrieval. Available in 0.5b and 1.5b variants.

### Supported Tasks

| Task | Description |
|------|-------------|
| `nl2code.query` | Natural language query to find code |
| `nl2code.passage` | Code snippet (candidate) |
| `code2code.query` | Code query to find similar code |
| `code2code.passage` | Candidate code snippet |
| `code2nl.query` | Code to find comments/docs |
| `code2nl.passage` | Candidate comment/documentation |
| `code2completion.query` | Code prefix for completion |
| `code2completion.passage` | Candidate completion |
| `qa.query` | Question about code |
| `qa.passage` | Answer candidate |

---

### Example 14: Natural Language to Code Query

**Local Server:**
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-code-embeddings-1.5b",
    "task": "nl2code.query",
    "input": ["How to calculate the square of a number in Python?"]
  }'
```

**Jina AI Official:**
```bash
curl -X POST https://api.jina.ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "jina-code-embeddings-1.5b",
    "task": "nl2code.query",
    "input": ["How to calculate the square of a number in Python?"]
  }'
```

---

### Example 15: Code Passage Embedding

**Local Server:**
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-code-embeddings-1.5b",
    "task": "nl2code.passage",
    "input": [
      "def square(number): return number ** 2",
      "def cube(x): return x * x * x"
    ]
  }'
```

**Jina AI Official:**
```bash
curl -X POST https://api.jina.ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "jina-code-embeddings-1.5b",
    "task": "nl2code.passage",
    "input": [
      "def square(number): return number ** 2",
      "def cube(x): return x * x * x"
    ]
  }'
```

---

### Example 16: Code to Code Search

**Local Server:**
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-code-embeddings-0.5b",
    "task": "code2code.query",
    "input": ["for i in range(10): print(i)"]
  }'
```

**Jina AI Official:**
```bash
curl -X POST https://api.jina.ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "jina-code-embeddings-0.5b",
    "task": "code2code.query",
    "input": ["for i in range(10): print(i)"]
  }'
```

---

### Example 17: Code Q&A

**Local Server:**
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-code-embeddings-1.5b",
    "task": "qa.query",
    "input": ["What does the enumerate function do in Python?"]
  }'
```

**Jina AI Official:**
```bash
curl -X POST https://api.jina.ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "jina-code-embeddings-1.5b",
    "task": "qa.query",
    "input": ["What does the enumerate function do in Python?"]
  }'
```

---

### Example 18: Code Embeddings with Base64 Output

**Local Server:**
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-code-embeddings-1.5b",
    "task": "nl2code.query",
    "truncate": true,
    "embedding_type": "base64",
    "input": [
      "Calculate the square of a number",
      "This function calculates the square"
    ]
  }'
```

**Jina AI Official:**
```bash
curl -X POST https://api.jina.ai/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "jina-code-embeddings-1.5b",
    "task": "nl2code.query",
    "truncate": true,
    "embedding_type": "base64",
    "input": [
      "Calculate the square of a number",
      "This function calculates the square"
    ]
  }'
```

---

## Other Local Embedding Models

These models share the **same /v1/embeddings request/response format** as the Jina models above. Only the `model` ID and supported tasks differ.

### Model Specifications

| Model ID | Type | Dimensions | Max Tokens | Languages | Special Features |
|----------|------|------------|------------|-----------|------------------|
| `bge-m3` | Embedding | 1024 | 8192 | 100+ | General-purpose, **late_chunking REJECTED** |
| `qwen3-embedding-0.6b` | Embedding | 1024 | 32768 | 100+ | Instruction-aware, rich task prompts |
| `qwen3-embedding-4b` | Embedding | 1024 | 32768 | 100+ | Larger capacity, same features |
| `qwen3-embedding-8b` | Embedding | 1024 | 32768 | 100+ | Largest model, same features |

### Late Chunking Support Matrix

| Model | Late Chunking Support | Reason |
|-------|----------------------|--------|
| `jina-embeddings-v3` | ✅ **Supported** | Trained for mean pooling over token spans |
| `jina-embeddings-v4` | ✅ **Supported** | Trained for mean pooling over token spans |
| `bge-m3` | ❌ **REJECTED** | Architecture causes embedding space collapse (-46.67% P@3) |
| `qwen3-embedding-*` | ❌ **REJECTED** | Uses last-token pooling, incompatible with late chunking |

### BGE-M3 Tasks

| Task | Description | Example |
|------|-------------|---------|
| `retrieval.query` | Query for retrieval tasks | `"What are the security fixes in ECOS 9.3?"` |
| `retrieval.passage` | Document/passage for retrieval | `"ECOS 9.3 includes security patches..."` |
| `text-matching` | Semantic similarity | `"Find similar technical documents"` |

**BGE-M3 Late Chunking:** Disabled due to embedding space collapse. API returns error:
```json
{
  "detail": {
    "error": {
      "message": "Model bge-m3 does not support late_chunking. BGE-M3's architecture causes embedding space collapse when late chunking is applied. Use jina-embeddings-v4 or voyage-context-3 for contextual embeddings.",
      "type": "internal_error"
    }
  }
}
```

### Qwen3 Embedding Tasks

Qwen3 models support **rich, instruction-aware embeddings** with task-specific prompts:

#### Core Tasks (Jina-Compatible)
| Task | Description | Instruction Applied |
|------|-------------|-------------------|
| `retrieval.query` | Search query embedding | ✅ "Given a search query, retrieve relevant documents..." |
| `retrieval.passage` | Document embedding | ❌ No instruction (passages don't need prompts) |
| `text-matching` | Semantic similarity | ✅ "Represent this text for finding semantically similar texts..." |
| `classification` | Text categorization | ✅ "Represent this text for classification..." |
| `separation` / `clustering` | Topic clustering | ✅ "Represent this text for clustering..." |

#### Extended Tasks (Qwen3 Enhanced)
| Task | Description | Instruction Applied |
|------|-------------|-------------------|
| `code.query` | Code search from NL | ✅ "Given a natural language description, retrieve relevant code..." |
| `code.passage` | Code document | ❌ No instruction |
| `scientific.query` | Research retrieval | ✅ "Given a scientific question, retrieve relevant research papers..." |
| `qa.query` | Question answering | ✅ "Given a question, retrieve passages that contain the answer..." |
| `bitext` | Cross-lingual matching | ✅ "Represent this text for finding its translation..." |
| `summarization.query` | Summary matching | ✅ "Given a summary, retrieve the original document..." |

**Qwen3 Late Chunking:** Not supported due to last-token pooling architecture. API returns error:
```json
{
  "detail": {
    "error": {
      "message": "late_chunking is not supported for qwen3-embedding-0.6b. Qwen3 models use last-token pooling, which is incompatible with late chunking. Use jina-embeddings-v3 or jina-embeddings-v4 for late chunking support.",
      "type": "internal_error"
    }
  }
}
```

### Example: Qwen3 with Rich Task Instructions

**Local Server:**
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-embedding-0.6b",
    "task": "retrieval.query",
    "input": ["What are the security fixes in ECOS 9.3?"]
  }'
```

**Applied Instruction Format:**
```
Instruct: Given a search query, retrieve relevant documents or passages that directly answer or address the query
Query: What are the security fixes in ECOS 9.3?
```

**Result:** The embedding is optimized specifically for retrieval queries, improving search relevance by 1-5% compared to generic embeddings.

### Quick Reference: Model Selection Guide

| Use Case | Recommended Model | Reason |
|----------|------------------|--------|
| **General text embedding** | `bge-m3` | Fast, multilingual, 1024 dims |
| **Retrieval with instructions** | `qwen3-embedding-0.6b` | Task-aware embeddings |
| **Multimodal (text+image)** | `jina-embeddings-v4` | Supports images, 2048 dims |
| **Late chunking needed** | `jina-embeddings-v3/v4` | Only models supporting late chunking |
| **Code search** | `jina-code-embeddings-1.5b` | Optimized for code |
| **Long context** | `qwen3-embedding-*` | 32k tokens vs 8k for others |

### Performance Benchmarks

Based on our comprehensive testing:

| Model | P@3 Score | Late Chunking | Best For |
|-------|-----------|---------------|----------|
| `jina-embeddings-v4` | **0.847** | ✅ Supported | Best overall performance |
| `jina-embeddings-v3` | **0.824** | ✅ Supported | Balanced performance |
| `qwen3-embedding-0.6b` | **0.812** | ❌ Rejected | Instruction-aware tasks |
| `bge-m3` | **0.798** | ❌ Rejected | Fast general embedding |

### Common Usage Patterns

#### Pattern 1: Document Search Pipeline
```bash
# Step 1: Embed documents with passage task
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embedding-0.6b", "task": "retrieval.passage", "input": ["Document 1...", "Document 2..."]}'

# Step 2: Embed query
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embedding-0.6b", "task": "retrieval.query", "input": ["user search query"]}'
```

#### Pattern 2: Semantic Similarity
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-m3",
    "task": "text-matching",
    "input": ["Product A description", "Product B description"]
  }'
```

#### Pattern 3: Code Search
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-embedding-0.6b",
    "task": "code.query",
    "input": ["function to reverse a string in Python"]
  }'
```

---

## Rerank API

### Endpoint
```
POST /v1/rerank
```

### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | ❌ | `"jina-reranker-v3"` | Model ID to use (see table below). |
| `query` | string | ✅ | - | Search query (non-empty string). |
| `documents` | array | ✅ | - | Documents to rerank (strings or objects with `text` field). |
| `top_n` | integer | ❌ | all | Return only top N documents. Must be ≥ 1 if provided. |
| `return_documents` | boolean | ❌ | `true` | Include document text in response. Matches Jina API default. |

### Document Formats

| Format | Example |
|--------|---------|
| String | `"Document text here"` |
| Object | `{"text": "Document text", "metadata": "extra"}` |

### Supported Rerank Models

The following local rerank models are available via `/v1/rerank`:

| Model ID | Description | Notes |
|----------|-------------|-------|
| `jina-reranker-v3` | Jina listwise cross-encoder reranker. | Default reranker; multilingual, 8K context. |
| `bge-reranker-v2-m3` | BGE-M3 reranker from BAAI. | Strong performance on multilingual and long-text reranking. |
| `qwen3-reranker-0.6b` | Qwen3 reranker (0.6B). | Uses a special yes/no scoring head; probabilities in [0,1]. |
| `qwen3-reranker-4b` | Qwen3 reranker (4B). | Same API as 0.6B; higher capacity, same parameters. |

All rerank models share the same request schema; you only change the `model` field.

---

### Example 19: Basic Reranking

**Local Server:**
```bash
curl -X POST http://localhost:8080/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-reranker-v3",
    "query": "Organic skincare products for sensitive skin",
    "documents": [
      "Organic skincare for sensitive skin with aloe vera and chamomile.",
      "New makeup trends focus on bold colors and innovative techniques.",
      "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille."
    ]
  }'
```

**Jina AI Official:**
```bash
curl -X POST https://api.jina.ai/v1/rerank \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "jina-reranker-v3",
    "query": "Organic skincare products for sensitive skin",
    "documents": [
      "Organic skincare for sensitive skin with aloe vera and chamomile.",
      "New makeup trends focus on bold colors and innovative techniques.",
      "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille."
    ]
  }'
```

---

### Example 20: Reranking with top_n

**Local Server:**
```bash
curl -X POST http://localhost:8080/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-reranker-v3",
    "query": "Best laptop for programming",
    "top_n": 3,
    "documents": [
      "MacBook Pro with M3 chip - excellent for developers",
      "Gaming mouse with RGB lighting",
      "Dell XPS 15 - powerful and portable",
      "Wireless keyboard for office use",
      "ThinkPad X1 Carbon - business laptop",
      "USB-C hub with multiple ports"
    ]
  }'
```

**Jina AI Official:**
```bash
curl -X POST https://api.jina.ai/v1/rerank \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "jina-reranker-v3",
    "query": "Best laptop for programming",
    "top_n": 3,
    "documents": [
      "MacBook Pro with M3 chip - excellent for developers",
      "Gaming mouse with RGB lighting",
      "Dell XPS 15 - powerful and portable",
      "Wireless keyboard for office use",
      "ThinkPad X1 Carbon - business laptop",
      "USB-C hub with multiple ports"
    ]
  }'
```

---

### Example 21: Reranking with Document Objects

**Local Server:**
```bash
curl -X POST http://localhost:8080/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-reranker-v3",
    "query": "Python programming tutorial",
    "return_documents": true,
    "documents": [
      {"text": "Learn Python basics in 30 days", "source": "tutorial-site"},
      {"text": "Advanced JavaScript frameworks", "source": "web-dev"},
      {"text": "Python for data science beginners", "source": "data-blog"}
    ]
  }'
```

**Jina AI Official:**
```bash
curl -X POST https://api.jina.ai/v1/rerank \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "jina-reranker-v3",
    "query": "Python programming tutorial",
    "return_documents": true,
    "documents": [
      {"text": "Learn Python basics in 30 days", "source": "tutorial-site"},
      {"text": "Advanced JavaScript frameworks", "source": "web-dev"},
      {"text": "Python for data science beginners", "source": "data-blog"}
    ]
  }'
```

---

### Example 22: Full Rerank Options

**Local Server:**
```bash
curl -X POST http://localhost:8080/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-reranker-v3",
    "query": "Organic skincare products for sensitive skin",
    "top_n": 3,
    "return_documents": true,
    "documents": [
      "Organic skincare for sensitive skin with aloe vera and chamomile.",
      "New makeup trends focus on bold colors and innovative techniques.",
      "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille.",
      "Cuidado de la piel orgánico para piel sensible.",
      "针对敏感肌专门设计的天然有机护肤产品。"
    ]
  }'
```

**Jina AI Official:**
```bash
curl -X POST https://api.jina.ai/v1/rerank \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "jina-reranker-v3",
    "query": "Organic skincare products for sensitive skin",
    "top_n": 3,
    "return_documents": true,
    "documents": [
      "Organic skincare for sensitive skin with aloe vera and chamomile.",
      "New makeup trends focus on bold colors and innovative techniques.",
      "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille.",
      "Cuidado de la piel orgánico para piel sensible.",
      "针对敏感肌专门设计的天然有机护肤产品。"
    ]
  }'
```

---

## Models API

### Endpoint

```
GET /v1/models
```

Returns a list of all **loaded** models and their capabilities.

---

### Example 23: List Models

**Local Server:**
```bash
curl -X GET http://localhost:8080/v1/models
```

**Jina AI Official:**
```bash
curl -X GET https://api.jina.ai/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"
```

---

### Models Response Schema

The `/v1/models` endpoint returns a JSON object with the following structure:

```json
{
  "object": "list",
  "data": [
    {
      "id": "jina-embeddings-v4",
      "object": "model",
      "created": 1717200000,
      "owned_by": "jinaai",
      "type": "embedding",
      "max_tokens": 8192,
      "dimensions": 2048,
      "supports_multimodal": true,
      "tasks": ["retrieval", "text-matching", "code"]
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `object` | string | Always `"list"`. |
| `data` | array | List of model objects. |
| `data[].id` | string | Model ID (use this in `model` field for `/v1/embeddings` or `/v1/rerank`). |
| `data[].object` | string | Always `"model"`. |
| `data[].created` | integer | Unix timestamp for model availability. |
| `data[].owned_by` | string | Model owner (e.g. `"jinaai"`). |
| `data[].type` | string | `"embedding"` or `"reranker"`. |
| `data[].max_tokens` | integer \| null | Maximum supported input tokens (if known). |
| `data[].dimensions` | integer \| null | Default embedding dimensions (embedding models only). |
| `data[].supports_multimodal` | boolean | Whether model accepts image inputs (v4 only). |
| `data[].tasks` | array \| null | List of supported tasks (if configured). |

---

## Health & Docs

In addition to the main API endpoints, the local server exposes standard health and documentation endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Simple health check. Returns `{"status": "ok"}` when the server and models are ready. |
| `/docs` | GET | FastAPI Swagger UI for interactive exploration of the API. |
| `/openapi.json` | GET | OpenAPI schema for the entire API. |

Example:

```bash
curl http://localhost:8080/healthz
# {"status": "ok"}
```

---

## Response Formats

### Embeddings Response

```json
{
  "object": "list",
  "model": "jina-embeddings-v3",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.123, -0.456, ...]
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

### Rerank Response

```json
{
  "model": "jina-reranker-v3",
  "results": [
    {
      "index": 0,
      "relevance_score": 0.95,
      "document": {"text": "..."}
    }
  ],
  "usage": {
    "total_tokens": 150
  }
}
```

---

## Error Handling

### Error Response Format

```json
{
  "error": {
    "message": "Error description",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 404 | Model not found |
| 422 | Invalid input |
| 429 | Rate limit exceeded |
| 500 | Internal server error |

---

## Feature Comparison

| Feature | Local Server | Jina AI Official |
|---------|--------------|------------------|
| `jina-embeddings-v3` | ✅ | ✅ |
| `jina-embeddings-v4` | ✅ | ✅ |
| `jina-code-embeddings-0.5b` | ✅ | ✅ |
| `jina-code-embeddings-1.5b` | ✅ | ✅ |
| `jina-reranker-v3` | ✅ | ✅ |
| Task-specific embeddings | ✅ | ✅ |
| MRL dimensions | ✅ | ✅ |
| Late chunking | ✅ | ✅ |
| Truncation control | ✅ | ✅ |
| Base64 output | ✅ | ✅ |
| Binary quantization | ✅ | ✅ |
| Multimodal (text+image) | ✅ | ✅ |
| Multi-vector output | ✅ | ✅ |
| top_n reranking | ✅ | ✅ |
| Document objects | ✅ | ✅ |
| Rate limiting | 🚧 Planned | ✅ |
| API key auth | ❌ N/A | ✅ |

---

## Quick Test Script

Save this as `test_api.sh` to quickly test all endpoints:

```bash
#!/bin/bash
BASE_URL="${1:-http://localhost:8080}"

echo "=== Testing Embeddings v3 ==="
curl -s -X POST "$BASE_URL/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"model":"jina-embeddings-v3","input":["test"]}' | jq '.data[0].embedding[:3]'

echo -e "\n=== Testing Reranker ==="
curl -s -X POST "$BASE_URL/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{"query":"test","documents":["doc1","doc2"]}' | jq '.results'

echo -e "\n=== Testing Models List ==="
curl -s "$BASE_URL/v1/models" | jq '.data[].id'
```

Usage:
```bash
chmod +x test_api.sh
./test_api.sh http://localhost:8080
```
