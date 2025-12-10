# Phase 2: BGE Model Integration Plan

**Status:** Planned  
**Priority:** Next after Jina API stabilization  
**Estimated Effort:** 2-3 days

---

## Overview

Extend the local embedding/reranker server to support BGE (BAAI General Embedding) models alongside Jina models, using the same unified API.

---

## Models to Add

### 1. Embeddings: `bge-m3`

| Property | Value |
|----------|-------|
| HuggingFace ID | `BAAI/bge-m3` |
| Dimensions | 1024 (default), supports MRL: 256, 512, 1024 |
| Languages | 100+ |
| Max Tokens | 8192 |
| Features | Dense + Sparse + Multi-vector (ColBERT) |

**Why bge-m3:**
- Most versatile BGE embedding model
- Supports dense, sparse, AND multi-vector in one model
- Excellent multilingual performance
- Same late_chunking technique works (transformer-based)

### 2. Reranker (Efficient): `bge-reranker-v2-m3`

| Property | Value |
|----------|-------|
| HuggingFace ID | `BAAI/bge-reranker-v2-m3` |
| Base Model | bge-m3 |
| Size | ~568M parameters |
| Languages | Multilingual (100+) |
| Speed | Fast |

**Why bge-reranker-v2-m3:**
- Best balance of speed and quality
- Multilingual support
- Production-ready
- Lightweight enough for real-time use

### 3. Reranker (Quality): `bge-reranker-v2-gemma`

| Property | Value |
|----------|-------|
| HuggingFace ID | `BAAI/bge-reranker-v2-gemma` |
| Base Model | gemma-2b |
| Size | ~2.5B parameters |
| Languages | Multilingual |
| Speed | Slower |

**Why bge-reranker-v2-gemma:**
- Highest quality reranking
- Best for offline/batch processing
- When accuracy > speed

---

## API Design

### Embeddings API (same endpoint)

```bash
# BGE-M3 embedding
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-m3",
    "input": ["Hello world", "Bonjour le monde"],
    "late_chunking": true,
    "dimensions": 1024
  }'
```

**Parameters:**

| Parameter | BGE-M3 Support | Notes |
|-----------|----------------|-------|
| `model` | `"bge-m3"` | Required |
| `input` | ✅ | Array of strings |
| `late_chunking` | ✅ | Reuse existing implementation |
| `dimensions` | ✅ | 256, 512, 1024 |
| `embedding_type` | ✅ | float, base64, binary |
| `truncate` | ✅ | Truncate long inputs |
| `task` | ❌ | Not supported by BGE |
| `return_multivector` | ✅ | BGE-M3 supports ColBERT output |

### Reranker API (same endpoint)

```bash
# BGE reranker
curl http://localhost:8080/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-reranker-v2-m3",
    "query": "What is machine learning?",
    "top_n": 5,
    "documents": [
      "Machine learning is a subset of AI...",
      "The weather today is sunny...",
      "Deep learning uses neural networks..."
    ]
  }'
```

**Parameters:**

| Parameter | BGE Reranker Support | Notes |
|-----------|---------------------|-------|
| `model` | `"bge-reranker-v2-m3"` or `"bge-reranker-v2-gemma"` | Required |
| `query` | ✅ | Search query |
| `documents` | ✅ | Array of strings |
| `top_n` | ✅ | Return top N results |
| `return_documents` | ✅ | Include document text in response |

---

## Implementation Plan

### Step 1: Create BGE Embedding Wrapper

**File:** `app/models/bge_embeddings.py`

```python
class BGEEmbeddingsWrapper(EmbeddingModelWrapper):
    model_id = "bge-m3"
    hf_model_id = "BAAI/bge-m3"
    default_dimensions = 1024
    max_tokens = 8192
    supports_multimodal = False  # Text only
    
    # No task mapping - BGE doesn't use tasks
    
    def encode(self, inputs, dimensions=None, late_chunking=False, ...):
        # Reuse late_chunking logic from app/late_chunking.py
        if late_chunking:
            return self._encode_late_chunking(inputs, dimensions, ...)
        else:
            return self._encode_standard(inputs, dimensions, ...)
```

### Step 2: Create BGE Reranker Wrapper

**File:** `app/models/bge_reranker.py`

```python
class BGERerankerWrapper(RerankerModelWrapper):
    model_id = "bge-reranker-v2-m3"
    hf_model_id = "BAAI/bge-reranker-v2-m3"
    
    def rerank(self, query, documents, top_n=None, return_documents=False):
        # Use FlagEmbedding library or transformers
        # Return same format as Jina reranker
        pass
```

### Step 3: Register Models

**File:** `app/models/registry.py`

```python
# Add to model registry
EMBEDDING_MODELS = {
    "jina-embeddings-v3": EmbeddingsV3Wrapper,
    "jina-embeddings-v4": EmbeddingsV4Wrapper,
    "bge-m3": BGEEmbeddingsWrapper,  # NEW
}

RERANKER_MODELS = {
    "jina-reranker-v3": RerankerWrapper,
    "bge-reranker-v2-m3": BGERerankerWrapper,      # NEW
    "bge-reranker-v2-gemma": BGERerankerGemmaWrapper,  # NEW
}
```

### Step 4: Update .env

```bash
# BGE Models
LOAD_BGE_M3=true
LOAD_BGE_RERANKER_M3=true
LOAD_BGE_RERANKER_GEMMA=false  # Optional, large model
```

### Step 5: Update requirements.txt

```
# BGE dependencies
FlagEmbedding>=1.2.0  # Official BGE library (optional, can use transformers)
```

---

## Late Chunking Reuse

The existing `app/late_chunking.py` works for BGE-M3 with minimal changes:

```python
# Same flow:
# 1. Concatenate inputs
# 2. Get token embeddings from transformer
# 3. Split by boundaries
# 4. Mean pool each section

# BGE-M3 outputs last_hidden_state just like Jina
outputs = model(**encoding)
token_embeddings = outputs.last_hidden_state  # Same!
```

---

## Testing Checklist

- [ ] BGE-M3 basic embedding
- [ ] BGE-M3 with late_chunking=true
- [ ] BGE-M3 with dimensions truncation
- [ ] BGE-M3 multi-vector output
- [ ] BGE-reranker-v2-m3 basic reranking
- [ ] BGE-reranker-v2-m3 with top_n
- [ ] BGE-reranker-v2-gemma basic reranking
- [ ] GPU fallback for all BGE models
- [ ] API response format matches Jina format

---

## Model Comparison Summary

| Model | Type | Size | Speed | Quality | Languages |
|-------|------|------|-------|---------|-----------|
| jina-embeddings-v3 | Embed | 570M | Fast | ★★★★ | 89 |
| jina-embeddings-v4 | Embed | 3.8B | Medium | ★★★★★ | 30+ |
| **bge-m3** | Embed | 568M | Fast | ★★★★ | 100+ |
| jina-reranker-v3 | Rerank | ~500M | Fast | ★★★★ | Multi |
| **bge-reranker-v2-m3** | Rerank | 568M | Fast | ★★★★ | 100+ |
| **bge-reranker-v2-gemma** | Rerank | 2.5B | Slow | ★★★★★ | Multi |

---

## Benefits of Adding BGE

1. **Redundancy** - If Jina models have issues, BGE is backup
2. **Cost comparison** - Test which performs better for your use case
3. **Sparse retrieval** - BGE-M3 supports BM25-style sparse vectors
4. **Community** - BGE is widely used, good community support
5. **Same API** - No client changes needed, just change model name

---

## Timeline

| Task | Estimate |
|------|----------|
| BGE-M3 wrapper | 4 hours |
| BGE reranker wrappers | 4 hours |
| Testing & debugging | 4 hours |
| Documentation | 2 hours |
| **Total** | **~2 days** |

---

## References

- [BGE-M3 HuggingFace](https://huggingface.co/BAAI/bge-m3)
- [BGE Reranker v2 M3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [BGE Reranker v2 Gemma](https://huggingface.co/BAAI/bge-reranker-v2-gemma)
- [FlagEmbedding GitHub](https://github.com/FlagOpen/FlagEmbedding)
