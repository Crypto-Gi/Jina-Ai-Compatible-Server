# Jina Local API Server - Complete API Reference

This document provides a comprehensive reference for all API endpoints, parameters, and options supported by the Jina Local API Server. Each endpoint includes curl examples that can be tested against both our local server and the official Jina AI API.

---

## Table of Contents

1. [Base URLs](#base-urls)
2. [Authentication](#authentication)
3. [Embeddings API](#embeddings-api)
   - [jina-embeddings-v3](#jina-embeddings-v3)
   - [jina-embeddings-v4](#jina-embeddings-v4)
   - [jina-code-embeddings](#jina-code-embeddings)
4. [Rerank API](#rerank-api)
5. [Models API](#models-api)
6. [Response Formats](#response-formats)
7. [Error Handling](#error-handling)

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
| `model` | string | ✅ | - | Model ID to use |
| `input` | string \| array | ✅ | - | Text(s) or multimodal items to embed |
| `normalized` | boolean | ❌ | `true` | L2-normalize embeddings |
| `embedding_type` | string | ❌ | `"float"` | Output format: `float`, `base64`, `binary`, `ubinary`* |
| `task` | string | ❌ | model-specific | Task-specific embedding variant |
| `dimensions` | integer | ❌ | model default | Target embedding dimensions (MRL) |
| `prompt_name` | string | ❌ | - | Prompt variant (`query` or `passage`) |
| `late_chunking` | boolean | ❌ | `false` | Return sentence-chunked embeddings |
| `truncate` | boolean | ❌ | `false` | Truncate inputs exceeding max length |
| `return_multivector` | boolean | ❌ | `false` | **Local extension:** Return NxD multi-vector embeddings (v4 only) |

> **Note:** `ubinary` is accepted but returns HTTP 422 - implementation pending.
> **Note:** `return_multivector` is a local server extension. Official Jina uses `/v1/multi-vector` endpoint.

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

## Rerank API

### Endpoint
```
POST /v1/rerank
```

### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | ❌ | `"jina-reranker-v3"` | Model ID |
| `query` | string | ✅ | - | Search query |
| `documents` | array | ✅ | - | Documents to rerank (strings or objects) |
| `top_n` | integer | ❌ | all | Return only top N documents |
| `return_documents` | boolean | ❌ | `false` | Include document text in response |

### Document Formats

| Format | Example |
|--------|---------|
| String | `"Document text here"` |
| Object | `{"text": "Document text", "metadata": "extra"}` |

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

Returns a list of all loaded models.

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
