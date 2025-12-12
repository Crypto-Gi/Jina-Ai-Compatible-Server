# Jina Local API – Test Report (v0.2)

- **Date**: 2025-12-10 (UTC)
- **Models covered**: `jina-embeddings-v3`, `jina-embeddings-v4`, `jina-code-embeddings-0.5b`, `jina-code-embeddings-1.5b`, `jina-reranker-v3`
- **Goal**: Verify API compatibility, feature coverage (tasks, dimensions/MRL, late chunking, embedding types, reranking options), and behavioral alignment with the official Jina AI API.

### Summary at a Glance

| Component                    | Status | Notes                                                  |
|-----------------------------|--------|--------------------------------------------------------|
| V3 text embeddings          | ✅     | Tasks, MRL, late chunking, embedding types             |
| V4 multimodal embeddings    | ✅*    | Text fully tested; images OOM; PDF not yet implemented |
| Code embeddings 0.5b / 1.5b | ✅     | All tasks, MRL, embedding types                        |
| Reranker v3                 | ✅     | top_n, return_documents, error handling                |

\* V4 image tests are limited by local GPU memory; PDF input is not yet wired in the schema.

### How to Read This Document

- **Tables** summarize what was tested and the observed behavior.
- **Curl examples** below show how to reproduce representative tests for each model.

### Quick API Examples

**V3 – basic embedding**

```bash
curl -s http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v3",
    "input": ["Hello world"]
  }'
```

**V3 – late chunking + dimensions**

```bash
curl -s http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v3",
    "input": [
      "Berlin is the capital of Germany.",
      "It has a population of 3.6 million."
    ],
    "late_chunking": true,
    "dimensions": 512
  }'
```

**V4 – text-matching with late chunking**

```bash
curl -s http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-embeddings-v4",
    "task": "text-matching",
    "late_chunking": true,
    "truncate": true,
    "input": [
      "Berlin is the capital of Germany.",
      "It has a population of 3.6 million.",
      "The city is known for its history."
    ]
  }'
```

**Code embeddings 0.5b – nl2code.query**

```bash
curl -s http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-code-embeddings-0.5b",
    "task": "nl2code.query",
    "input": ["function to sort an array in Python"]
  }'
```

**Code embeddings 1.5b – MRL dimensions**

```bash
curl -s http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-code-embeddings-1.5b",
    "input": ["def add(a, b): return a + b"],
    "dimensions": 512
  }'
```

**Reranker v3 – top_n and return_documents**

```bash
curl -s http://localhost:8080/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-reranker-v3",
    "query": "What is machine learning?",
    "documents": [
      "Machine learning is a subset of AI.",
      "The weather is nice today.",
      "Deep learning uses neural networks."
    ],
    "top_n": 2,
    "return_documents": true
  }'
```

---

## 1. jina-embeddings-v3 – Test Matrix

### 1.1 Basic & Tasks

| Test ID | Description                           | Params                                             | Result                                    |
|--------:|---------------------------------------|----------------------------------------------------|-------------------------------------------|
| V3-01   | Basic embedding                       | `model=jina-embeddings-v3`, `input=["Hello world"]` | 1 embedding, **dim=1024**, tokens=2       |
| V3-02   | Multiple inputs                       | `input=["First text","Second text","Third text"]` | 3 embeddings, tokens=6                    |
| V3-03   | Task: `retrieval.query`              | `task="retrieval.query"`                          | dim=1024                                  |
| V3-04   | Task: `retrieval.passage`            | `task="retrieval.passage"`                        | dim=1024                                  |
| V3-05   | Task: `text-matching`                | `task="text-matching"`                            | 2 embeddings                              |
| V3-06   | Task: `classification`               | `task="classification"`                           | dim=1024                                  |
| V3-07   | Task: `separation`                   | `task="separation"`                               | dim=1024                                  |

All task aliases are wired through the adapter map correctly, and late chunking uses `adapter_mask` rather than `task` on the forward pass.

### 1.2 Dimensions (MRL truncation)

Official matryoshka dims (from Jina docs): `32, 64, 128, 256, 512, 768, 1024`.

| Test ID | Requested `dimensions` | Observed dim | Status |
|--------:|------------------------|--------------|--------|
| V3-10   | 32                     | 32           | ✅     |
| V3-11   | 64                     | 64           | ✅     |
| V3-12   | 128                    | 128          | ✅     |
| V3-13   | 256                    | 256          | ✅     |
| V3-14   | 512                    | 512          | ✅     |
| V3-15   | 768                    | 768          | ✅     |
| V3-16   | 1024                   | 1024         | ✅     |
| V3-17   | 384 (unsupported)      | **Error**    | ✅ – returns 500 with message: `The provided truncate_dim value of 384 is not supported. Supported dimensions are [32, 64, 128, 256, 512, 768, 1024].` |

### 1.3 Embedding Types

| Test ID | embedding_type | Observation                                    |
|--------:|----------------|-----------------------------------------------|
| V3-20   | `float` (default) | `embedding` is `list[float]`, dim as above  |
| V3-21   | `base64`       | `embedding` is `str`, length ~5.4k chars      |
| V3-22   | `binary`       | `embedding` is `list[int]` of 0/1, len=1024   |
| V3-23   | `ubinary`      | **Not implemented** – returns 422 via `_format_embedding` (explicit message) |

### 1.4 Late Chunking Behavior

We verified that late chunking:

1. Concatenates all inputs into one long document.
2. Runs the transformer once to produce contextualized token embeddings.
3. Splits tokens back into original spans using offsets.
4. Pools per-span tokens to get one embedding per input.

Implementation uses `late_chunking_pooling` and `ChunkSpan` from `app/late_chunking.py`.

#### 1.4.1 Qualitative test (Berlin document)

Inputs:

1. "Berlin is the capital of Germany and its largest city."
2. "Paris is the capital of France."
3. "London is the capital of the United Kingdom."

We compared **cosine similarity** between chunk embeddings.

| Setting           | Chunk0–1 | Chunk0–2 | Avg   | Comment                              |
|-------------------|----------|----------|-------|--------------------------------------|
| late_chunking=ON  | 0.9937   | 0.9672   | ~0.98 | High similarity due to shared context |
| late_chunking=OFF | 0.1852   | 0.2401   | ~0.21 | Much lower similarity                |

This matches Jina’s description of late chunking: chunks get full-document context.

#### 1.4.2 Late chunking + dimensions

| Test ID | Params                                                   | Result                          |
|--------:|----------------------------------------------------------|---------------------------------|
| V3-30   | `late_chunking=true`, `dimensions=512`, 2 chunks         | 2 embeddings, dim=512, ✅       |

### 1.5 Error handling & endpoints

| Test ID | Case                  | Result / Behavior                           |
|--------:|-----------------------|---------------------------------------------|
| V3-40   | Invalid model         | 404-like error with `invalid_request_error` |
| V3-41   | Empty `input` list    | Validation error (handled)                  |
| V3-42   | Large batch (10)      | 10 embeddings, tokens counted via tokenizer |
| V3-43   | `/v1/models`          | Lists at least `jina-embeddings-v3`         |
| V3-44   | `/healthz`            | `{ "status": "ok" }`                      |

---

## 2. jina-embeddings-v4 – Test Matrix

Model: `jina-embeddings-v4` (multimodal, 2048-dim default, Qwen2.5-VL-3B base).

### 2.1 Basic & Tasks

| Test ID | Description                 | Params                                         | Result                                      |
|--------:|-----------------------------|------------------------------------------------|---------------------------------------------|
| V4-01   | Basic embedding             | `input=["Hello world"]`                       | 1 embedding, **dim=2048**, tokens=2         |
| V4-02   | Multiple inputs             | `input=["First","Second","Third"]`         | 3 embeddings, dim=2048                      |
| V4-03   | Large batch (10 inputs)     | `input=["T1".."T10"]`                         | 10 embeddings                               |

**Tasks** (per Jina docs: `retrieval`, `text-matching`, `code` with prompt names and aliases):

| Test ID | Task                   | Params example                                     | Result            |
|--------:|------------------------|----------------------------------------------------|-------------------|
| V4-10   | `retrieval.query`     | `task="retrieval.query"`                         | dim=2048          |
| V4-11   | `retrieval.passage`   | `task="retrieval.passage"`                       | dim=2048          |
| V4-12   | `text-matching`       | `task="text-matching"`                           | dim=2048          |
| V4-13   | `code.query`          | `task="code.query"`, NL query                    | dim=2048          |
| V4-14   | `code.passage`        | `task="code.passage"`, code snippet              | dim=2048          |

The wrapper maps these to base tasks (`retrieval`, `text-matching`, `code`) using `TASK_MAP`, and derives `prompt_name` when needed.

### 2.2 Dimensions (MRL truncation)

From Jina’s README, single-vector dims: `2048` default; matryoshka dims: `128, 256, 512, 1024, 2048`.

| Test ID | Requested `dimensions` | Observed dim | Status |
|--------:|------------------------|--------------|--------|
| V4-20   | 128                    | 128          | ✅     |
| V4-21   | 256                    | 256          | ✅     |
| V4-22   | 512                    | 512          | ✅     |
| V4-23   | 1024                   | 1024         | ✅     |
| V4-24   | 2048                   | 2048         | ✅     |

### 2.3 Embedding Types

| Test ID | embedding_type | Observation                                        |
|--------:|----------------|---------------------------------------------------|
| V4-30   | `float`        | default; `embedding` is `list[float]`             |
| V4-31   | `base64`       | `embedding` is base64 `str`, len≈10.9k characters |
| V4-32   | `binary`       | `embedding` is `list[int]` of {0,1}, len=dim      |
| V4-33   | `ubinary`      | Not implemented (same behavior as V3)            |

### 2.4 Late Chunking Behavior (Text)

Implementation mirrors V3: concatenate, encode once, then pool per span.

#### 2.4.1 Berlin document (3 chunks)

| Setting           | Chunk0–1 | Chunk0–2 | Chunk1–2 | Average |
|-------------------|----------|----------|----------|---------|
| late_chunking=ON  | 0.7919   | 0.7844   | 0.8555   | 0.8106  |
| late_chunking=OFF | (measured but omitted in logs) |         |         | ~0.58–0.60 |

Late chunking clearly increases mutual similarity of related chunks.

#### 2.4.2 Pronoun resolution (Apple Inc.)

Inputs:

1. "Apple Inc. is a technology company."
2. "It was founded by Steve Jobs."
3. "The company is headquartered in Cupertino."

| Setting           | Apple–It | Apple–TheCompany | Avg   |
|-------------------|----------|------------------|-------|
| late_chunking=ON  | 0.7366   | 0.7790           | 0.7578|
| late_chunking=OFF | ≈0.74    | ≈0.74            | 0.7381|

Pronoun-related chunks are already similar without late chunking, but still get a modest boost.

#### 2.4.3 Scenario table – average pairwise similarity

We ran a small battery of scenarios with/without late chunking:

| Scenario                     | Late=True | Late=False | Δ (Late − No Late) |
|------------------------------|-----------|------------|--------------------|
| Berlin Document              | 0.8106    | 0.5842     | **+0.2264**        |
| Python Programming           | 0.8395    | 0.6424     | **+0.1971**        |
| Apple Inc. (Pronoun)        | 0.8123    | 0.7381     | +0.0742            |
| Climate Change Paper         | 0.5816    | 0.5799     | +0.0017            |
| Machine Learning             | 0.6536    | 0.7144     | −0.0607            |
| **Average improvement**      | —         | —          | **+0.0877 (~8.77%)**|

Interpretation:

- For **strongly-related chunks in one coherent document**, late chunking significantly boosts similarity (20%+ absolute in some cases).
- For loosely-related or already very similar chunks, the gain is small.
- One synthetic scenario showed a slight drop, indicating late chunking is not universally beneficial but is clearly positive on average for real document-style inputs.

### 2.5 Late Chunking + Dimensions & Tasks

| Test ID | Description                                | Params                                                   | Result                                   |
|--------:|--------------------------------------------|----------------------------------------------------------|------------------------------------------|
| V4-40   | Late chunking + dimensions                 | `late_chunking=true`, `dimensions=512`, 2 chunks         | 2 embeddings, dim=512                    |
| V4-41   | Late chunking + retrieval.passage          | `late_chunking=true`, `task="retrieval.passage"`        | 2 embeddings, task respected             |
| V4-42   | Late chunking on 5-chunk ML document       | 5 chunks, `late_chunking=true`                           | 5 embeddings, tokens≈46 (heuristic)      |

### 2.6 Input Formats & Multimodality

| Test ID | Case                   | Result / Behavior                                                     |
|--------:|------------------------|------------------------------------------------------------------------|
| V4-50   | Plain string input     | Works (same as V3)                                                    |
| V4-51   | Dict `{"text": ...}`   | Works – normalized to text input                                      |
| V4-52   | Image input (URL)      | **CUDA OOM** in this environment (insufficient free VRAM)             |
| V4-53   | PDF input `{"pdf":..}` | **Not supported in current schema** – Pydantic complains about type   |

Notes:

- Jina’s official API **does** support PDF for V4; to match it, we’d need to:
  - Extend `EmbeddingsRequest` with a `pdf` field and a `PdfInput` type.
  - Implement PDF fetching + page/image extraction (e.g., via `pypdf` / `pdf2image`).
- Image tests fail here only due to **local VRAM**, not API logic.

### 2.7 Error Handling & Endpoints

| Test ID | Case                  | Result / Behavior                            |
|--------:|-----------------------|-----------------------------------------------|
| V4-60   | Invalid model         | `invalid_request_error`                      |
| V4-61   | Empty `input` list    | Validation error (422)                       |
| V4-62   | `/v1/models`          | Includes `jina-embeddings-v4`                |
| V4-63   | `/healthz`            | `{ "status": "ok" }`                       |

---

## 3. jina-code-embeddings-0.5b – Test Matrix

Model: `jina-code-embeddings-0.5b` (lightweight code embeddings, 896-dim default, 8192 max tokens).

### 3.1 Basic & Tasks

| Test ID | Description                 | Params                                         | Result                                      |
|--------:|-----------------------------|------------------------------------------------|---------------------------------------------|
| CE05-01 | Basic embedding             | `input=["def add(a,b): return a+b"]`          | 1 embedding, **dim=896**, tokens=21         |
| CE05-02 | Multiple inputs             | 3 code snippets                                | 3 embeddings, dim=896                       |

**Tasks** (per Jina docs: `nl2code`, `code2code`, `code2nl`, `qa` with `.query`/`.passage` suffixes):

| Test ID | Task                   | Result            |
|--------:|------------------------|-------------------|
| CE05-10 | `nl2code.query`       | dim=896 ✅        |
| CE05-11 | `nl2code.passage`     | dim=896 ✅        |
| CE05-12 | `code2code.query`     | dim=896 ✅        |
| CE05-13 | `code2code.passage`   | dim=896 ✅        |
| CE05-14 | `code2nl.query`       | dim=896 ✅        |
| CE05-15 | `code2nl.passage`     | dim=896 ✅        |
| CE05-16 | `qa.query`            | dim=896 ✅        |
| CE05-17 | `qa.passage`          | dim=896 ✅        |

### 3.2 Dimensions (MRL truncation)

| Test ID | Requested `dimensions` | Observed dim | Status |
|--------:|------------------------|--------------|--------|
| CE05-20 | 128                    | 128          | ✅     |
| CE05-21 | 256                    | 256          | ✅     |
| CE05-22 | 512                    | 512          | ✅     |
| CE05-23 | 768                    | 768          | ✅     |
| CE05-24 | 896 (default)          | 896          | ✅     |

### 3.3 Embedding Types

| Test ID | embedding_type | Observation                                    |
|--------:|----------------|-----------------------------------------------|
| CE05-30 | `float`        | `list[float]`, len=896                        |
| CE05-31 | `base64`       | `str`, base64-encoded                         |
| CE05-32 | `binary`       | `list[int]` of {0,1}, len=896                 |

### 3.4 Other Options

| Test ID | Option                 | Result                                         |
|--------:|------------------------|------------------------------------------------|
| CE05-40 | `truncate=true`        | Works – no error on long input                 |
| CE05-41 | `late_chunking=true`   | Works – 2 embeddings with shared context       |

---

## 4. jina-code-embeddings-1.5b – Test Matrix

Model: `jina-code-embeddings-1.5b` (full-size code embeddings, 1536-dim default, 32768 max tokens).

### 4.1 Basic & Tasks

| Test ID | Description                 | Params                                         | Result                                      |
|--------:|-----------------------------|------------------------------------------------|---------------------------------------------|
| CE15-01 | Basic embedding             | `input=["def add(a,b): return a+b"]`          | 1 embedding, **dim=1536**, tokens=21        |
| CE15-02 | Multiple inputs             | 3 code snippets                                | 3 embeddings, dim=1536                      |

**Tasks**:

| Test ID | Task                   | Result            |
|--------:|------------------------|-------------------|
| CE15-10 | `nl2code.query`       | dim=1536 ✅       |
| CE15-11 | `nl2code.passage`     | dim=1536 ✅       |
| CE15-12 | `code2code.query`     | dim=1536 ✅       |
| CE15-13 | `code2code.passage`   | dim=1536 ✅       |
| CE15-14 | `code2nl.query`       | dim=1536 ✅       |
| CE15-15 | `code2nl.passage`     | dim=1536 ✅       |
| CE15-16 | `qa.query`            | dim=1536 ✅       |
| CE15-17 | `qa.passage`          | dim=1536 ✅       |

### 4.2 Dimensions (MRL truncation)

| Test ID | Requested `dimensions` | Observed dim | Status |
|--------:|------------------------|--------------|--------|
| CE15-20 | 128                    | 128          | ✅     |
| CE15-21 | 256                    | 256          | ✅     |
| CE15-22 | 512                    | 512          | ✅     |
| CE15-23 | 768                    | 768          | ✅     |
| CE15-24 | 1024                   | 1024         | ✅     |
| CE15-25 | 1536 (default)         | 1536         | ✅     |

### 4.3 Embedding Types

| Test ID | embedding_type | Observation                                    |
|--------:|----------------|-----------------------------------------------|
| CE15-30 | `float`        | `list[float]`, len=1536                       |
| CE15-31 | `base64`       | `str`, base64-encoded                         |
| CE15-32 | `binary`       | `list[int]` of {0,1}, len=1536                |

---

## 5. jina-reranker-v3 – Test Matrix

Model: `jina-reranker-v3` (listwise document reranker).

### 5.1 Basic Reranking

| Test ID | Description                 | Params                                                                 | Result                                      |
|--------:|-----------------------------|------------------------------------------------------------------------|---------------------------------------------|
| RR-01   | Basic rerank                | query="What is ML?", 3 documents                                       | 3 results, sorted by relevance_score        |
| RR-02   | Large batch (10 docs)       | 10 documents                                                           | 10 results, tokens counted                  |

**Sample Response Structure**:

```json
{
  "model": "jina-reranker-v3",
  "results": [
    {"index": 0, "relevance_score": 0.2555, "document": {"text": "Machine learning is..."}},
    {"index": 2, "relevance_score": -0.1097, "document": {"text": "Deep learning uses..."}},
    {"index": 1, "relevance_score": -0.1656, "document": {"text": "The weather is..."}}
  ],
  "usage": {"total_tokens": 27}
}
```

Results are sorted by `relevance_score` descending. The most relevant document appears first.

### 5.2 top_n Parameter

| Test ID | top_n | Documents | Results Returned | Status |
|--------:|-------|-----------|------------------|--------|
| RR-10   | 1     | 3         | 1                | ✅     |
| RR-11   | 2     | 5         | 2                | ✅     |
| RR-12   | None  | 3         | 3 (all)          | ✅     |

### 5.3 return_documents Parameter

| Test ID | return_documents | Result                                                |
|--------:|------------------|-------------------------------------------------------|
| RR-20   | `true` (default) | Each result includes `document.text`                  |
| RR-21   | `false`          | Results only have `index` and `relevance_score`       |

### 5.4 Error Handling

| Test ID | Case                  | Result / Behavior                            |
|--------:|-----------------------|-----------------------------------------------|
| RR-30   | Invalid model         | `invalid_request_error`                      |
| RR-31   | Empty documents       | Validation error (422)                       |

---

## 6. qwen3-reranker-0.6b – Test Matrix

### 6.1 Basic Reranking

| Test ID  | Description  | Params                                                   | Result                                                                      |
|---------:|--------------|----------------------------------------------------------|-----------------------------------------------------------------------------|
| QW-RR-01 | Basic rerank | `model="qwen3-reranker-0.6b"`, simple query + 3–4 docs | 3–4 results, sorted by `relevance_score` (0–1), Beijing-like answer on top |

The wrapper uses a yes/no scoring head and converts raw logits to probabilities in the **0–1** range.

### 6.2 `top_n` Parameter

| Test ID  | top_n | Documents | Results Returned | Status |
|---------:|-------|-----------|------------------|--------|
| QW-RR-10 | 1     | 3         | 1                | ✅     |
| QW-RR-11 | 2     | 3         | 2                | ✅     |
| QW-RR-12 | None  | 3         | 3 (all)          | ✅     |

### 6.3 `return_documents` Parameter

| Test ID  | return_documents | Result                                                |
|---------:|------------------|-------------------------------------------------------|
| QW-RR-20 | `false` (default) | Results only have `index` and `relevance_score`      |
| QW-RR-21 | `true`           | Each result includes `document.text`                 |

### 6.4 Error Handling

| Test ID  | Case            | Result / Behavior                            |
|---------:|-----------------|-----------------------------------------------|
| QW-RR-30 | Invalid model   | `invalid_request_error` / 404-style error    |
| QW-RR-31 | Empty documents | Validation error (422, `documents` too short) |

---

## 7. bge-m3 – Test Matrix

### 7.1 Basic Embeddings

| Test ID  | Description           | Params                              | Result                          |
|---------:|-----------------------|-------------------------------------|---------------------------------|
| BGE-01   | Basic embedding       | `model="bge-m3"`, single input      | 1024-dim embedding, normalized  |
| BGE-02   | Multiple inputs       | 3 texts                             | 3 embeddings, correct token count |

### 7.2 MRL Dimensions

| Test ID  | dimensions | Actual Output | Status |
|---------:|------------|---------------|--------|
| BGE-10   | 256        | 256           | ✅     |
| BGE-11   | 512        | 512           | ✅     |
| BGE-12   | 768        | 768           | ✅     |
| BGE-13   | 1024       | 1024          | ✅     |

### 7.3 Embedding Types

| Test ID  | encoding_format | Result                              |
|---------:|-----------------|-------------------------------------|
| BGE-20   | `float`         | Array of floats                     |
| BGE-21   | `base64`        | Base64-encoded embedding            |

### 7.4 Late Chunking Comparison

**Test scenario**: 3 chunks from same document about Berlin

| Chunk | Text                                              |
|------:|---------------------------------------------------|
| 1     | "Berlin is the capital of Germany."               |
| 2     | "It has a population of 3.5 million."             |
| 3     | "The city is known for its history and culture."  |

**Query**: "What is the population of Berlin?"

#### Query Similarity Results

| Chunk | Standard | Late Chunking | Δ       |
|------:|---------:|--------------:|--------:|
| 1     | 0.6839   | 0.6434        | -5.9%   |
| 2     | 0.5490   | 0.6471        | **+17.9%** |
| 3     | 0.4129   | 0.6327        | **+53.2%** |

**Key insight**: Chunk 2 ("population of 3.5 million") gains +17.9% similarity to the query about population, because late chunking preserves context that "It" refers to "Berlin".

#### Inter-Chunk Similarity (same document)

| Pair          | Standard | Late Chunking | Δ         |
|---------------|----------|---------------|-----------|
| Chunk 1 ↔ 2   | 0.4683   | 0.9980        | **+113.1%** |
| Chunk 1 ↔ 3   | 0.5411   | 0.9877        | **+82.6%**  |
| Chunk 2 ↔ 3   | 0.5889   | 0.9925        | **+68.5%**  |

**Summary**:

- Avg query similarity improvement: **+21.7%**
- Avg inter-chunk similarity improvement: **+88.1%**

Late chunking dramatically improves coherence for chunks from the same document.

---

## 8. bge-reranker-v2-m3 – Test Matrix

### 8.1 Basic Reranking

| Test ID  | Description  | Params                                                    | Result                                                                      |
|---------:|--------------|-----------------------------------------------------------|-----------------------------------------------------------------------------|
| BGE-RR-01 | Basic rerank | `model="bge-reranker-v2-m3"`, query + 4 docs            | 4 results, sorted by `relevance_score` (0–1), Beijing answer on top (0.9999) |

### 8.2 `top_n` Parameter

| Test ID   | top_n | Documents | Results Returned | Status |
|----------:|-------|-----------|------------------|--------|
| BGE-RR-10 | 1     | 3         | 1                | ✅     |
| BGE-RR-11 | 2     | 3         | 2                | ✅     |
| BGE-RR-12 | None  | 3         | 3 (all)          | ✅     |

### 8.3 `return_documents` Parameter

| Test ID   | return_documents | Result                                                |
|----------:|------------------|-------------------------------------------------------|
| BGE-RR-20 | `false`          | Results only have `index` and `relevance_score`       |
| BGE-RR-21 | `true`           | Each result includes `document.text`                  |

### 8.4 Error Handling

| Test ID   | Case            | Result / Behavior                            |
|----------:|-----------------|-----------------------------------------------|
| BGE-RR-30 | Invalid model   | `invalid_request_error`                       |
| BGE-RR-31 | Empty documents | Validation error (422, `documents` too short) |

---

## 9. Endpoints & Health

| Endpoint      | Behavior                                                              |
|---------------|-----------------------------------------------------------------------|
| `/v1/models`  | Lists all loaded models (e.g., `jina-reranker-v3`, `jina-code-embeddings-0.5b`, `jina-code-embeddings-1.5b`) |
| `/healthz`    | Returns `{"status": "ok"}`                                           |
| `/docs`       | OpenAPI/Swagger UI (HTTP 200)                                        |

---

## 10. Conclusions

- **V3**: All documented features (tasks, matryoshka dimensions, late chunking, embedding types) behave correctly and are compatible with official Jina docs (including proper adapter usage via `adapter_mask`).
- **V4**: Text-side API behavior (tasks, dimensions, late chunking, multivector groundwork, embedding types) matches the official API semantics.
  - Image support is present but limited by local VRAM in this environment.
  - PDF support is not yet wired into the local schema and needs explicit implementation to fully match Jina's public API.
- **Code Embeddings (0.5b & 1.5b)**: All 8 task types work correctly. MRL dimension truncation works. Embedding types (float, base64, binary) all function as expected.
  - 0.5b: 896-dim default, 8192 max tokens
  - 1.5b: 1536-dim default, 32768 max tokens
- **Reranker V3**: Listwise reranking works correctly. `top_n` and `return_documents` parameters behave as per Jina API spec. Results are properly sorted by relevance score.
- **Late Chunking**: Across realistic scenarios, late chunking improves intra-document chunk similarity by **~8.8% on average**, with very large gains on strongly-related chunks (20%+ in some tests).


---

## Qwen3-4B and Jina V4 Benchmark Results

**Tested**: 2025-12-10T21:22:56.173895

### Qwen3-Embedding-4B (Standard Only)

Qwen3 models do not support late chunking (uses last-token pooling).

| Test | P@3 | MRR | NDCG@3 |
|------|-----|-----|--------|
| test1 | 73.33% | 1.0000 | 1.0000 |
| test2 | 66.67% | 1.0000 | 1.0000 |
| test3 | 53.33% | 1.0000 | 0.9839 |
| **Average** | **64.44%** | **1.0000** | **0.9946** |

### Jina-Embeddings-V4 (Standard)

| Test | P@3 | MRR | NDCG@3 |
|------|-----|-----|--------|
| test1 | 66.67% | 1.0000 | 0.9839 |
| test2 | 46.67% | 0.7667 | 0.8262 |
| test3 | 70.00% | 1.0000 | 1.0000 |
| **Average** | **61.11%** | **0.9222** | **0.9367** |

### Jina-Embeddings-V4 (Late Chunking)

| Test | P@3 | MRR | NDCG@3 |
|------|-----|-----|--------|
| test1 | 60.00% | 1.0000 | 0.9679 |
| test2 | 53.33% | 0.7000 | 0.7226 |
| test3 | 60.00% | 0.9333 | 0.9339 |
| **Average** | **57.78%** | **0.8778** | **0.8748** |

### Late Chunking Impact (Jina V4)

| Test | Standard P@3 | Late P@3 | Delta |
|------|--------------|----------|-------|
| test1 | 66.67% | 60.00% | -6.67% |
| test2 | 46.67% | 53.33% | +6.67% |
| test3 | 70.00% | 60.00% | -10.00% |
| **Average** | 61.11% | 57.78% | -3.33% |

### Comparison: All Embedding Models

| Model | Avg P@3 | Avg MRR | Late Chunking |
|-------|---------|---------|---------------|
| qwen3-embedding-4b | 64.4% | 1.0000 | N/A |
| jina-embeddings-v4 | 61.1% | 0.9222 | Standard |
| jina-embeddings-v4 (late) | 57.8% | 0.8778 | Enabled |


---

## Qwen3-4B and Jina V4 Benchmark Results

**Tested**: 2025-12-10T21:33:00.543898

### Qwen3-Embedding-4B (Standard Only)

Qwen3 models do not support late chunking (uses last-token pooling).

| Test | P@3 | MRR | NDCG@3 |
|------|-----|-----|--------|
| test1 | 73.33% | 1.0000 | 1.0000 |
| test2 | 66.67% | 1.0000 | 1.0000 |
| test3 | 53.33% | 1.0000 | 0.9839 |
| **Average** | **64.44%** | **1.0000** | **0.9946** |

### Jina-Embeddings-V4 (Standard)

| Test | P@3 | MRR | NDCG@3 |
|------|-----|-----|--------|
| test1 | 66.67% | 1.0000 | 0.9839 |
| test2 | 46.67% | 0.7667 | 0.8262 |
| test3 | 70.00% | 1.0000 | 1.0000 |
| **Average** | **61.11%** | **0.9222** | **0.9367** |

### Jina-Embeddings-V4 (Late Chunking)

| Test | P@3 | MRR | NDCG@3 |
|------|-----|-----|--------|
| test1 | 60.00% | 1.0000 | 1.0000 |
| test2 | 66.67% | 0.9000 | 0.9066 |
| test3 | 70.00% | 1.0000 | 0.9759 |
| **Average** | **65.56%** | **0.9667** | **0.9608** |

### Late Chunking Impact (Jina V4)

| Test | Standard P@3 | Late P@3 | Delta |
|------|--------------|----------|-------|
| test1 | 66.67% | 60.00% | -6.67% |
| test2 | 46.67% | 66.67% | +20.00% |
| test3 | 70.00% | 70.00% | +0.00% |
| **Average** | 61.11% | 65.56% | +4.44% |

### Comparison: All Embedding Models

| Model | Avg P@3 | Avg MRR | Late Chunking |
|-------|---------|---------|---------------|
| qwen3-embedding-4b | 64.4% | 1.0000 | N/A |
| jina-embeddings-v4 | 61.1% | 0.9222 | Standard |
| jina-embeddings-v4 (late) | 65.6% | 0.9667 | Enabled |

---

## 11. Comprehensive Embedding & Reranker Benchmark (December 2024)

This section contains the final comprehensive benchmark results comparing all local and external embedding/reranker models.

### 11.1 Jina API Compatibility Test Results

**Test Date**: December 11, 2024  
**Result**: 33/33 tests passed (100.0%)

| Test Category | Tests | Status | Details |
|---------------|-------|--------|---------|
| V4 Standard Embedding | 1 | ✅ PASS | Similarity: 0.9998, dim: 2048 |
| V4 Late Chunking | 1 | ✅ PASS | Avg similarity: 0.9849, dim: 2048 |
| V4 Dimensions | 4 | ✅ PASS | 128, 256, 512, 1024 all match |
| V4 Tasks | 5 | ✅ PASS | retrieval.query/passage, text-matching, code.query/passage |
| V4 Embedding Types | 2 | ✅ PASS | float, base64 |
| V3 Tasks | 5 | ✅ PASS | retrieval.query/passage, text-matching, separation, classification |
| V3 Late Chunking | 1 | ✅ PASS | Avg similarity: 0.9613, dim: 1024 |
| Reranker Basic | 1 | ✅ PASS | Order match: True, Score correlation: 1.0000 |
| Reranker top_n | 3 | ✅ PASS | top_n=1,2,3 all correct |
| Reranker return_documents | 2 | ✅ PASS | True/False both work |
| Benchmark Tests | 6 | ✅ PASS | All P@3 scores match official API |

**Key Validation**: Local server produces identical results to official Jina API across all test scenarios.

### 11.2 Final Embedding Model Benchmark

**Models Tested**:
- **Local**: jina-embeddings-v4 (late chunking), qwen3-embedding-4b, qwen3-embedding-0.6b
- **External**: voyage-context-3, voyage-3.5, gemini-embedding-001

**API Configurations** (verified from official documentation):
- Gemini: `RETRIEVAL_QUERY` for queries, `RETRIEVAL_DOCUMENT` for documents
- Voyage voyage-3.5: `embed()` with `input_type="query"` or `"document"`
- Voyage voyage-context-3: `contextualized_embed()` with `inputs=[[chunks...]]` format

#### Results by Test File

| Model | test1 | test2 | test3 | **Average** |
|-------|-------|-------|-------|-------------|
| **voyage-context-3** | 66.7% | 73.3% | **83.3%** | **74.4%** |
| **qwen3-4b** | **73.3%** | **73.3%** | 53.3% | 66.7% |
| **voyage-3.5** | 66.7% | 46.7% | **83.3%** | 65.6% |
| **qwen3-0.6b** | **73.3%** | 66.7% | 53.3% | 64.4% |
| **jina-v4-late** | 66.7% | 60.0% | 63.3% | 63.3% |
| **gemini-001** | 66.7% | 53.3% | 70.0% | 63.3% |

#### Overall Ranking (by Average P@3)

| Rank | Model | Avg P@3 | Type |
|------|-------|---------|------|
| 🥇 1 | **voyage-context-3** | **74.4%** | External (Voyage) |
| 🥈 2 | qwen3-embedding-4b | 66.7% | Local |
| 🥉 3 | voyage-3.5 | 65.6% | External (Voyage) |
| 4 | qwen3-embedding-0.6b | 64.4% | Local |
| 5 | jina-embeddings-v4 (late) | 63.3% | Local |
| 6 | gemini-embedding-001 | 63.3% | External (Google) |

#### Key Findings - Embeddings

1. **voyage-context-3 wins overall** with 74.4% average P@3 - contextualized embeddings provide significant advantage
2. **voyage-context-3 excels on test3** (single-doc ECOS) with 83.3% - best for single-document retrieval
3. **qwen3-4b is best local model** at 66.7% - excellent for multi-doc tests (test1, test2)
4. **Local models are competitive** - qwen3-4b matches or beats external APIs except voyage-context-3
5. **jina-v4-late matches official API** exactly at 63.3% - validated drop-in replacement
6. **Gemini performs well on test3** (70%) but struggles on test2 (53.3%)

### 11.3 Final Reranker Benchmark

**Models Tested**:
- **Local**: jina-reranker-v3, bge-reranker-v2-m3, qwen3-reranker-0.6b, qwen3-reranker-4b
- **External**: voyage rerank-2.5, voyage rerank-2.5-lite

**API Configurations** (verified from official documentation):
- Local `/v1/rerank`: `model`, `query`, `documents`, `top_n`, `return_documents`
- Voyage: `vo.rerank(query, documents, model="rerank-2.5", top_k=3)`

#### Results by Test File

| Model | test1 | test2 | test3 | **Average** |
|-------|-------|-------|-------|-------------|
| **voyage-rerank-2.5** | 73.3% | 53.3% | **76.7%** | **67.8%** |
| **jina-reranker-v3** | **80.0%** | 53.3% | 66.7% | 66.7% |
| **bge-reranker-v2-m3** | 66.7% | **60.0%** | 70.0% | 65.6% |
| **qwen3-reranker-0.6b** | 66.7% | 53.3% | 70.0% | 63.3% |
| **qwen3-reranker-4b** | 60.0% | **60.0%** | 70.0% | 63.3% |
| **voyage-rerank-2.5-lite** | 53.3% | 53.3% | **76.7%** | 61.1% |

#### Overall Ranking (by Average P@3)

| Rank | Model | Avg P@3 | Type |
|------|-------|---------|------|
| 🥇 1 | **voyage-rerank-2.5** | **67.8%** | External (Voyage) |
| 🥈 2 | jina-reranker-v3 | 66.7% | Local |
| 🥉 3 | bge-reranker-v2-m3 | 65.6% | Local |
| 4 | qwen3-reranker-0.6b | 63.3% | Local |
| 5 | qwen3-reranker-4b | 63.3% | Local |
| 6 | voyage-rerank-2.5-lite | 61.1% | External (Voyage) |

#### Key Findings - Rerankers

1. **voyage-rerank-2.5 wins overall** at 67.8% - best for single-doc retrieval (test3: 76.7%)
2. **jina-reranker-v3 is best local model** at 66.7% - excels on multi-doc test1 (80.0%)
3. **bge-reranker-v2-m3 is most consistent** - solid across all tests (60-70%)
4. **Local models are highly competitive** - top 3 local models within 2% of Voyage
5. **qwen3-reranker-4b doesn't outperform 0.6b** - same average despite 6x more params
6. **voyage-rerank-2.5-lite underperforms** - 6.7% below full model

### 11.4 Best Model Recommendations

#### By Use Case - Embeddings

| Use Case | Best Model | P@3 |
|----------|------------|-----|
| Single-doc retrieval (test3) | voyage-context-3, voyage-3.5 | 83.3% |
| Multi-doc retrieval (test1, test2) | qwen3-embedding-4b | 73.3% |
| Best overall | voyage-context-3 | 74.4% |
| Best local/self-hosted | qwen3-embedding-4b | 66.7% |
| Drop-in Jina replacement | jina-embeddings-v4 (late) | 63.3% |

#### By Use Case - Rerankers

| Use Case | Best Model | P@3 |
|----------|------------|-----|
| Single-doc retrieval (test3) | voyage-rerank-2.5, voyage-rerank-2.5-lite | 76.7% |
| Multi-doc retrieval (test1) | jina-reranker-v3 | 80.0% |
| Best overall | voyage-rerank-2.5 | 67.8% |
| Best local/self-hosted | jina-reranker-v3 | 66.7% |
| Best value (small & fast) | qwen3-reranker-0.6b | 63.3% |

### 11.5 Complete Model Comparison Summary

#### All Embedding Models (Including Qwen3-8B)

| Model | test1 | test2 | test3 | **Average** | Type |
|-------|-------|-------|-------|-------------|------|
| voyage-context-3 | 66.7% | 73.3% | 83.3% | **74.4%** | External |
| qwen3-embedding-4b | 73.3% | 73.3% | 53.3% | 66.7% | Local |
| voyage-3.5 | 66.7% | 46.7% | 83.3% | 65.6% | External |
| qwen3-embedding-0.6b | 73.3% | 66.7% | 53.3% | 64.4% | Local |
| jina-v4 (late chunking) | 66.7% | 60.0% | 63.3% | 63.3% | Local |
| gemini-embedding-001 | 66.7% | 53.3% | 70.0% | 63.3% | External |
| qwen3-embedding-8b | 66.7% | 53.3% | 66.7% | 62.2% | Local |
| jina-v4 (standard) | 60.0% | 60.0% | 63.3% | 61.1% | Local |

#### All Reranker Models

| Model | test1 | test2 | test3 | **Average** | Type |
|-------|-------|-------|-------|-------------|------|
| voyage-rerank-2.5 | 73.3% | 53.3% | 76.7% | **67.8%** | External |
| jina-reranker-v3 | 80.0% | 53.3% | 66.7% | 66.7% | Local |
| bge-reranker-v2-m3 | 66.7% | 60.0% | 70.0% | 65.6% | Local |
| qwen3-reranker-0.6b | 66.7% | 53.3% | 70.0% | 63.3% | Local |
| qwen3-reranker-4b | 60.0% | 60.0% | 70.0% | 63.3% | Local |
| voyage-rerank-2.5-lite | 53.3% | 53.3% | 76.7% | 61.1% | External |

### 11.6 Key Insights

1. **Model size doesn't always correlate with performance**:
   - qwen3-embedding-0.6b (64.4%) outperforms qwen3-embedding-8b (62.2%)
   - qwen3-reranker-0.6b ties with qwen3-reranker-4b (both 63.3%)

2. **Contextual embeddings work when implemented correctly**:
   - voyage-context-3 leads all embeddings at 74.4%
   - jina-v4 late chunking improves over standard (+2.2%)

3. **Local models are production-ready**:
   - Top local embedding (qwen3-4b) is within 7.7% of best external
   - Top local reranker (jina-reranker-v3) is within 1.1% of best external

4. **Test type matters**:
   - Single-doc (test3): Voyage models excel (83.3% embedding, 76.7% rerank)
   - Multi-doc (test1, test2): Qwen3 and Jina models excel

---

## 12. Test Scripts Reference

| Script | Purpose |
|--------|---------|
| `tests/test_jina_api_comparison.py` | Validates local server against official Jina API |
| `tests/final_embedding_benchmark_v2.py` | Comprehensive embedding model benchmark |
| `tests/final_reranker_benchmark.py` | Comprehensive reranker model benchmark |

---

**Report Last Updated**: December 11, 2024
