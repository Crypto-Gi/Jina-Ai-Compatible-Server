# Model Benchmark Results

**Generated**: 2025-12-10 18:30:05

## Executive Summary

### Test Dataset

- **3 test files**: test1 (financial analytics), test2 (trade evaluation), test3 (ECOS release notes)
- **Total queries**: 30 (20 retrieval + 10 rerank per test × 3 tests = 60 total evaluations)
- **test3** uses sequential document chunks to evaluate contextual understanding

### Overall Rankings

#### 🏆 Retrieval (Embedding) - Winner: BGE-M3

| Rank | Model | P@3 | MRR | NDCG@3 |
|------|-------|-----|-----|--------|
| 🥇 1 | `bge-m3` | **63.33%** | 0.9667 | 0.7025 |
| 🥈 2 | `qwen3-embedding-0.6b` | 61.67% | **1.0000** | **0.7117** |
| 🥉 3 | `jina-embeddings-v3` | 61.67% | 0.9750 | 0.6908 |

#### 🏆 Reranking - Winner: Qwen3

| Rank | Model | P@3 | MRR | NDCG@3 |
|------|-------|-----|-----|--------|
| 🥇 1 | `qwen3-reranker-0.6b` | **66.67%** | **0.9500** | **0.7235** |
| 🥈 2 | `bge-reranker-v2-m3` | 65.00% | 0.8917 | 0.6913 |
| 🥉 3 | `jina-reranker-v3` | 63.33% | 0.8917 | 0.6765 |

### Key Findings

1. **BGE-M3 leads in retrieval P@3** (63.33%) - best at finding relevant documents
2. **Qwen3 has perfect MRR** (1.0) - always ranks the best answer first
3. **Qwen3 reranker is strongest** - highest scores across all metrics
4. **Jina V3 is competitive** but slightly behind on this benchmark
5. **All models perform well** - differences are within ~5% margin

### Recommendations

| Use Case | Recommended Model |
|----------|-------------------|
| **Best overall retrieval** | `bge-m3` |
| **Best reranking accuracy** | `qwen3-reranker-0.6b` |
| **Best first-result accuracy** | `qwen3-embedding-0.6b` (MRR=1.0) |
| **Jina API compatibility** | `jina-embeddings-v3` + `jina-reranker-v3` |
| **Late chunking support** | `bge-m3` or `jina-embeddings-v3` (Qwen3 not supported) |

### Notes

- Qwen3 does **not** support late chunking (uses last-token pooling)
- Jina V4 was not tested (requires ~6GB VRAM, OOM on test system)
- All tests used standard embedding (no late chunking) for fair comparison

## Overall Comparison

### Retrieval (Embedding) Performance

| Model Family | Embedding Model | P@3 | MRR | NDCG@3 |
|--------------|-----------------|-----|-----|--------|
| BGE (BAAI) | `bge-m3` | 63.33% | 0.9667 | 0.7025 |
| Qwen3 (Alibaba) | `qwen3-embedding-0.6b` | 61.67% | 1.0000 | 0.7117 |
| Jina V3 | `jina-embeddings-v3` | 61.67% | 0.9750 | 0.6908 |

### Reranking Performance

| Model Family | Reranker Model | P@3 | MRR | NDCG@3 |
|--------------|----------------|-----|-----|--------|
| Jina V3 | `jina-reranker-v3` | 63.33% | 0.8917 | 0.6765 |
| Qwen3 (Alibaba) | `qwen3-reranker-0.6b` | 66.67% | 0.9500 | 0.7235 |
| BGE (BAAI) | `bge-reranker-v2-m3` | 65.00% | 0.8917 | 0.6913 |


---

## BGE (BAAI) Detailed Results

- **Embedding Model**: `bge-m3`
- **Reranker Model**: `bge-reranker-v2-m3`
- **Tested**: 2025-12-10T18:29:58.412549

### test1

#### Retrieval Results

| Query | Predicted Top 3 | Gold Top 3 | P@3 | MRR |
|-------|-----------------|------------|-----|-----|
| Q1 | S10, S9, S3 | S10, S1, S9 | 66.67% | 1.00 |
| Q2 | S3, S9, S1 | S3, S2, S1 | 66.67% | 1.00 |
| Q3 | S7, S9, S5 | S7, S5, S6 | 66.67% | 1.00 |
| Q4 | S8, S5, S9 | S8, S5, S7 | 66.67% | 1.00 |
| Q5 | S2, S1, S9 | S2, S1, S3 | 66.67% | 1.00 |
| **Avg** | | | **66.67%** | **1.00** |

#### Rerank Results

| Query | Predicted Top 3 | Gold Top 3 | P@3 | MRR |
|-------|-----------------|------------|-----|-----|
| R1 | S10, S2, S5 | S1, S2, S10 | 66.67% | 1.00 |
| R2 | S6, S4, S5 | S4, S7, S9 | 33.33% | 0.50 |
| R3 | S6, S2, S5 | S3, S5, S6 | 66.67% | 1.00 |
| R4 | S3, S2, S8 | S8, S5, S7 | 33.33% | 0.33 |
| R5 | S9, S7, S10 | S9, S5, S6 | 33.33% | 1.00 |
| **Avg** | | | **46.67%** | **0.77** |

### test2

#### Retrieval Results

| Query | Predicted Top 3 | Gold Top 3 | P@3 | MRR |
|-------|-----------------|------------|-----|-----|
| Q1 | S5, S6, S10 | S5, S10, S6 | 100.00% | 1.00 |
| Q2 | S1, S9, S2 | S1, S2, S3 | 66.67% | 1.00 |
| Q3 | S2, S5, S9 | S2, S1, S3 | 33.33% | 1.00 |
| Q4 | S8, S3, S2 | S8, S4, S7 | 33.33% | 1.00 |
| Q5 | S5, S2, S9 | S9, S8, S1 | 33.33% | 0.33 |
| **Avg** | | | **53.33%** | **0.87** |

#### Rerank Results

| Query | Predicted Top 3 | Gold Top 3 | P@3 | MRR |
|-------|-----------------|------------|-----|-----|
| R1 | S2, S10, S1 | S5, S6, S10 | 33.33% | 0.50 |
| R2 | S5, S6, S2 | S5, S6, S4 | 66.67% | 1.00 |
| R3 | S9, S2, S8 | S8, S9, S1 | 66.67% | 1.00 |
| R4 | S2, S4, S5 | S4, S8, S3 | 33.33% | 0.50 |
| R5 | S7, S10, S3 | S7, S3, S10 | 100.00% | 1.00 |
| **Avg** | | | **60.00%** | **0.80** |

### test3_ecos_release_notes

#### Retrieval Results

| Query | Predicted Top 3 | Gold Top 3 | P@3 | MRR |
|-------|-----------------|------------|-----|-----|
| Q1 | C2, C1, C12 | C1, C2, C10 | 66.67% | 1.00 |
| Q2 | C2, C10, C14 | C2, C1, C14 | 66.67% | 1.00 |
| Q3 | C16, C1, C2 | C16, C15, C1 | 66.67% | 1.00 |
| Q4 | C17, C6, C16 | C17, C6, C7 | 66.67% | 1.00 |
| Q5 | C15, C2, C1 | C15, C2, C1 | 100.00% | 1.00 |
| Q6 | C8, C9, C6 | C8, C7, C6 | 66.67% | 1.00 |
| Q7 | C12, C11, C9 | C12, C11, C10 | 66.67% | 1.00 |
| Q8 | C7, C17, C3 | C7, C17, C6 | 66.67% | 1.00 |
| Q9 | C14, C17, C10 | C14, C2, C5 | 33.33% | 1.00 |
| Q10 | C10, C6, C2 | C10, C6, C5 | 66.67% | 1.00 |
| **Avg** | | | **66.67%** | **1.00** |

#### Rerank Results

| Query | Predicted Top 3 | Gold Top 3 | P@3 | MRR |
|-------|-----------------|------------|-----|-----|
| R1 | C1, C9, C15 | C1, C9, C8 | 66.67% | 1.00 |
| R2 | C11, C12, C8 | C11, C12, C10 | 66.67% | 1.00 |
| R3 | C3, C5, C6 | C3, C4, C5 | 66.67% | 1.00 |
| R4 | C9, C7, C1 | C7, C8, C9 | 66.67% | 1.00 |
| R5 | C13, C4, C1 | C13, C4, C1 | 100.00% | 1.00 |
| R6 | C8, C9, C6 | C8, C6, C9 | 100.00% | 1.00 |
| R7 | C6, C7, C14 | C6, C7, C14 | 100.00% | 1.00 |
| R8 | C9, C7, C8 | C9, C16, C8 | 66.67% | 1.00 |
| R9 | C4, C3, C6 | C4, C3, C5 | 66.67% | 1.00 |
| R10 | C3, C5, C6 | C5, C3, C4 | 66.67% | 1.00 |
| **Avg** | | | **76.67%** | **1.00** |

---

## Qwen3 (Alibaba) Detailed Results

- **Embedding Model**: `qwen3-embedding-0.6b`
- **Reranker Model**: `qwen3-reranker-0.6b`
- **Tested**: 2025-12-10T18:31:00.172933

### test1

#### Retrieval Results

| Query | Predicted Top 3 | Gold Top 3 | P@3 | MRR |
|-------|-----------------|------------|-----|-----|
| Q1 | S10, S9, S1 | S10, S1, S9 | 100.00% | 1.00 |
| Q2 | S3, S9, S1 | S3, S2, S1 | 66.67% | 1.00 |
| Q3 | S6, S7, S9 | S7, S5, S6 | 66.67% | 1.00 |
| Q4 | S8, S5, S9 | S8, S5, S7 | 66.67% | 1.00 |
| Q5 | S1, S2, S9 | S2, S1, S3 | 66.67% | 1.00 |
| **Avg** | | | **73.33%** | **1.00** |

#### Rerank Results

| Query | Predicted Top 3 | Gold Top 3 | P@3 | MRR |
|-------|-----------------|------------|-----|-----|
| R1 | S1, S2, S10 | S1, S2, S10 | 100.00% | 1.00 |
| R2 | S4, S7, S10 | S4, S7, S9 | 66.67% | 1.00 |
| R3 | S6, S2, S5 | S3, S5, S6 | 66.67% | 1.00 |
| R4 | S8, S3, S5 | S8, S5, S7 | 66.67% | 1.00 |
| R5 | S9, S5, S3 | S9, S5, S6 | 66.67% | 1.00 |
| **Avg** | | | **73.33%** | **1.00** |

### test2

#### Retrieval Results

| Query | Predicted Top 3 | Gold Top 3 | P@3 | MRR |
|-------|-----------------|------------|-----|-----|
| Q1 | S10, S5, S6 | S5, S10, S6 | 100.00% | 1.00 |
| Q2 | S1, S2, S10 | S1, S2, S3 | 66.67% | 1.00 |
| Q3 | S2, S5, S8 | S2, S1, S3 | 33.33% | 1.00 |
| Q4 | S8, S4, S2 | S8, S4, S7 | 66.67% | 1.00 |
| Q5 | S9, S8, S2 | S9, S8, S1 | 66.67% | 1.00 |
| **Avg** | | | **66.67%** | **1.00** |

#### Rerank Results

| Query | Predicted Top 3 | Gold Top 3 | P@3 | MRR |
|-------|-----------------|------------|-----|-----|
| R1 | S2, S10, S1 | S5, S6, S10 | 33.33% | 0.50 |
| R2 | S5, S6, S2 | S5, S6, S4 | 66.67% | 1.00 |
| R3 | S9, S5, S8 | S8, S9, S1 | 66.67% | 1.00 |
| R4 | S5, S4, S2 | S4, S8, S3 | 33.33% | 0.50 |
| R5 | S10, S7, S3 | S7, S3, S10 | 100.00% | 1.00 |
| **Avg** | | | **60.00%** | **0.80** |

### test3_ecos_release_notes

#### Retrieval Results

| Query | Predicted Top 3 | Gold Top 3 | P@3 | MRR |
|-------|-----------------|------------|-----|-----|
| Q1 | C1, C2, C12 | C1, C2, C10 | 66.67% | 1.00 |
| Q2 | C2, C1, C12 | C2, C1, C14 | 66.67% | 1.00 |
| Q3 | C16, C10, C4 | C16, C15, C1 | 33.33% | 1.00 |
| Q4 | C17, C12, C4 | C17, C6, C7 | 33.33% | 1.00 |
| Q5 | C15, C2, C12 | C15, C2, C1 | 66.67% | 1.00 |
| Q6 | C8, C17, C9 | C8, C7, C6 | 33.33% | 1.00 |
| Q7 | C12, C10, C14 | C12, C11, C10 | 66.67% | 1.00 |
| Q8 | C7, C17, C3 | C7, C17, C6 | 66.67% | 1.00 |
| Q9 | C14, C17, C16 | C14, C2, C5 | 33.33% | 1.00 |
| Q10 | C10, C6, C2 | C10, C6, C5 | 66.67% | 1.00 |
| **Avg** | | | **53.33%** | **1.00** |

#### Rerank Results

| Query | Predicted Top 3 | Gold Top 3 | P@3 | MRR |
|-------|-----------------|------------|-----|-----|
| R1 | C1, C6, C15 | C1, C9, C8 | 33.33% | 1.00 |
| R2 | C11, C12, C4 | C11, C12, C10 | 66.67% | 1.00 |
| R3 | C3, C5, C6 | C3, C4, C5 | 66.67% | 1.00 |
| R4 | C9, C7, C8 | C7, C8, C9 | 100.00% | 1.00 |
| R5 | C13, C14, C4 | C13, C4, C1 | 66.67% | 1.00 |
| R6 | C8, C9, C7 | C8, C6, C9 | 66.67% | 1.00 |
| R7 | C6, C7, C5 | C6, C7, C14 | 66.67% | 1.00 |
| R8 | C9, C7, C15 | C9, C16, C8 | 33.33% | 1.00 |
| R9 | C4, C5, C3 | C4, C3, C5 | 100.00% | 1.00 |
| R10 | C5, C3, C6 | C5, C3, C4 | 66.67% | 1.00 |
| **Avg** | | | **66.67%** | **1.00** |

---

## Jina V3 Detailed Results

- **Embedding Model**: `jina-embeddings-v3`
- **Reranker Model**: `jina-reranker-v3`
- **Tested**: 2025-12-10T18:31:55.859356

### test1

#### Retrieval Results

| Query | Predicted Top 3 | Gold Top 3 | P@3 | MRR |
|-------|-----------------|------------|-----|-----|
| Q1 | S10, S9, S2 | S10, S1, S9 | 66.67% | 1.00 |
| Q2 | S9, S3, S6 | S3, S2, S1 | 33.33% | 0.50 |
| Q3 | S7, S6, S9 | S7, S5, S6 | 66.67% | 1.00 |
| Q4 | S8, S5, S7 | S8, S5, S7 | 100.00% | 1.00 |
| Q5 | S2, S1, S9 | S2, S1, S3 | 66.67% | 1.00 |
| **Avg** | | | **66.67%** | **0.90** |

#### Rerank Results

| Query | Predicted Top 3 | Gold Top 3 | P@3 | MRR |
|-------|-----------------|------------|-----|-----|
| R1 | S10, S1, S2 | S1, S2, S10 | 100.00% | 1.00 |
| R2 | S6, S4, S5 | S4, S7, S9 | 33.33% | 0.50 |
| R3 | S2, S6, S5 | S3, S5, S6 | 66.67% | 0.50 |
| R4 | S8, S3, S1 | S8, S5, S7 | 33.33% | 1.00 |
| R5 | S9, S3, S10 | S9, S5, S6 | 33.33% | 1.00 |
| **Avg** | | | **53.33%** | **0.80** |

### test2

#### Retrieval Results

| Query | Predicted Top 3 | Gold Top 3 | P@3 | MRR |
|-------|-----------------|------------|-----|-----|
| Q1 | S6, S5, S10 | S5, S10, S6 | 100.00% | 1.00 |
| Q2 | S1, S10, S2 | S1, S2, S3 | 66.67% | 1.00 |
| Q3 | S2, S5, S9 | S2, S1, S3 | 33.33% | 1.00 |
| Q4 | S8, S3, S9 | S8, S4, S7 | 33.33% | 1.00 |
| Q5 | S9, S3, S5 | S9, S8, S1 | 33.33% | 1.00 |
| **Avg** | | | **53.33%** | **1.00** |

#### Rerank Results

| Query | Predicted Top 3 | Gold Top 3 | P@3 | MRR |
|-------|-----------------|------------|-----|-----|
| R1 | S2, S1, S10 | S5, S6, S10 | 33.33% | 0.33 |
| R2 | S5, S6, S2 | S5, S6, S4 | 66.67% | 1.00 |
| R3 | S2, S9, S1 | S8, S9, S1 | 66.67% | 0.50 |
| R4 | S4, S2, S5 | S4, S8, S3 | 33.33% | 1.00 |
| R5 | S10, S7, S6 | S7, S3, S10 | 66.67% | 1.00 |
| **Avg** | | | **53.33%** | **0.77** |

### test3_ecos_release_notes

#### Retrieval Results

| Query | Predicted Top 3 | Gold Top 3 | P@3 | MRR |
|-------|-----------------|------------|-----|-----|
| Q1 | C2, C1, C10 | C1, C2, C10 | 100.00% | 1.00 |
| Q2 | C2, C10, C14 | C2, C1, C14 | 66.67% | 1.00 |
| Q3 | C16, C9, C10 | C16, C15, C1 | 33.33% | 1.00 |
| Q4 | C17, C7, C14 | C17, C6, C7 | 66.67% | 1.00 |
| Q5 | C15, C2, C1 | C15, C2, C1 | 100.00% | 1.00 |
| Q6 | C8, C9, C7 | C8, C7, C6 | 66.67% | 1.00 |
| Q7 | C12, C9, C8 | C12, C11, C10 | 33.33% | 1.00 |
| Q8 | C7, C17, C9 | C7, C17, C6 | 66.67% | 1.00 |
| Q9 | C14, C17, C10 | C14, C2, C5 | 33.33% | 1.00 |
| Q10 | C10, C6, C2 | C10, C6, C5 | 66.67% | 1.00 |
| **Avg** | | | **63.33%** | **1.00** |

#### Rerank Results

| Query | Predicted Top 3 | Gold Top 3 | P@3 | MRR |
|-------|-----------------|------------|-----|-----|
| R1 | C1, C9, C15 | C1, C9, C8 | 66.67% | 1.00 |
| R2 | C11, C12, C9 | C11, C12, C10 | 66.67% | 1.00 |
| R3 | C3, C5, C2 | C3, C4, C5 | 66.67% | 1.00 |
| R4 | C9, C7, C8 | C7, C8, C9 | 100.00% | 1.00 |
| R5 | C13, C1, C10 | C13, C4, C1 | 66.67% | 1.00 |
| R6 | C8, C9, C6 | C8, C6, C9 | 100.00% | 1.00 |
| R7 | C6, C5, C7 | C6, C7, C14 | 66.67% | 1.00 |
| R8 | C9, C7, C1 | C9, C16, C8 | 33.33% | 1.00 |
| R9 | C4, C3, C5 | C4, C3, C5 | 100.00% | 1.00 |
| R10 | C5, C3, C6 | C5, C3, C4 | 66.67% | 1.00 |
| **Avg** | | | **73.33%** | **1.00** |

---

## Late Chunking Comparison

Testing whether `late_chunking=true` improves retrieval accuracy.

**Note**: Late chunking only applies to embeddings, not rerankers.
Qwen3 does not support late chunking (uses last-token pooling).

### Results

| Model | Mode | P@3 | MRR | Δ P@3 | Δ MRR |
|-------|------|-----|-----|-------|-------|
| `bge-m3` | Standard | 63.33% | 0.9667 | - | - |
| `bge-m3` | Late Chunking | 28.33% | 0.5490 | -35.00% | -0.4177 |

### Analysis

- **bge-m3**: Late chunking decreased P@3 by -35.00%

---

## Late Chunking Comparison

Testing whether `late_chunking=true` improves retrieval accuracy.

**Note**: Late chunking only applies to embeddings, not rerankers.
Qwen3 does not support late chunking (uses last-token pooling).

### Results

| Model | Mode | P@3 | MRR | Δ P@3 | Δ MRR |
|-------|------|-----|-----|-------|-------|
| `jina-embeddings-v3` | Standard | 61.67% | 0.9167 | - | - |
| `jina-embeddings-v3` | Late Chunking | 36.67% | 0.6789 | -25.00% | -0.2378 |

### Analysis

- **jina-embeddings-v3**: Late chunking decreased P@3 by -25.00%


---

## Late Chunking Analysis (Correct Usage) - bge-m3

**Key Insight**: Late chunking should only be used when all input texts are chunks from the **SAME document**.

### Test Results

#### test3 (Single Document - ECOS Release Notes)
All 17 chunks are from ONE document → Late chunking is **appropriate**

| Mode | P@3 | MRR | 
|------|-----|-----|
| Standard | 66.67% | 1.0000 |
| Late Chunking | 20.00% | 0.3167 |
| **Delta** | -46.67% | -0.6833 |

#### test1 (Multi-Document - Financial Analytics)
Each chunk is from a DIFFERENT document → Late chunking is **NOT appropriate**

| Mode | P@3 | MRR |
|------|-----|-----|
| Standard (correct) | 66.67% | 1.0000 |
| Late Chunking (wrong) | 26.67% | 0.4667 |
| **Delta** | -40.00% | -0.5333 |

#### test2 (Multi-Document - Trade Evaluation)
Each chunk is from a DIFFERENT document → Late chunking is **NOT appropriate**

| Mode | P@3 | MRR |
|------|-----|-----|
| Standard (correct) | 53.33% | 0.8667 |
| Late Chunking (wrong) | 46.67% | 0.8000 |
| **Delta** | -6.67% | -0.0667 |

### Conclusion

- **Single-document corpus (test3)**: Late chunking delta = -46.67%
- **Multi-document corpus (test1)**: Late chunking delta = -40.00%
- **Multi-document corpus (test2)**: Late chunking delta = -6.67%

**Recommendation**: Only use `late_chunking=true` when embedding chunks from a single long document. 
For corpora with independent documents, use standard embedding.


---

## Late Chunking Analysis (Correct Usage) - jina-embeddings-v3

**Key Insight**: Late chunking should only be used when all input texts are chunks from the **SAME document**.

### Test Results

#### test3 (Single Document - ECOS Release Notes)
All 17 chunks are from ONE document → Late chunking is **appropriate**

| Mode | P@3 | MRR | 
|------|-----|-----|
| Standard | 60.00% | 1.0000 |
| Late Chunking | 23.33% | 0.5333 |
| **Delta** | -36.67% | -0.4667 |

#### test1 (Multi-Document - Financial Analytics)
Each chunk is from a DIFFERENT document → Late chunking is **NOT appropriate**

| Mode | P@3 | MRR |
|------|-----|-----|
| Standard (correct) | 73.33% | 0.8667 |
| Late Chunking (wrong) | 60.00% | 1.0000 |
| **Delta** | -13.33% | +0.1333 |

#### test2 (Multi-Document - Trade Evaluation)
Each chunk is from a DIFFERENT document → Late chunking is **NOT appropriate**

| Mode | P@3 | MRR |
|------|-----|-----|
| Standard (correct) | 53.33% | 0.8000 |
| Late Chunking (wrong) | 40.00% | 0.5333 |
| **Delta** | -13.33% | -0.2667 |

### Conclusion

- **Single-document corpus (test3)**: Late chunking delta = -36.67%
- **Multi-document corpus (test1)**: Late chunking delta = -13.33%
- **Multi-document corpus (test2)**: Late chunking delta = -13.33%

**Recommendation**: Only use `late_chunking=true` when embedding chunks from a single long document. 
For corpora with independent documents, use standard embedding.


---

## Voyage AI Benchmark Results

**Models Tested**:
- `voyage-3.5`: Standard embeddings (no contextual awareness)
- `voyage-context-3`: Contextual chunk embeddings (chunks share document context)

### test3 (Single Document - ECOS Release Notes)
All 17 chunks are from ONE document → Contextual embeddings **appropriate**

| Mode | P@3 | MRR |
|------|-----|-----|
| voyage-3.5 (standard) | 83.33% | 1.0000 |
| voyage-context-3 (contextual) | 26.67% | 0.5500 |
| **Delta** | -56.67% | -0.4500 |

### test1 (Multi-Document - Financial Analytics)
Each chunk is from a DIFFERENT document

| Mode | P@3 | MRR |
|------|-----|-----|
| voyage-3.5 (standard) | 66.67% | 1.0000 |
| voyage-context-3 (per-doc, correct) | 20.00% | 0.2000 |
| voyage-context-3 (all-as-one, WRONG) | 20.00% | 0.3000 |

### test2 (Multi-Document - Trade Evaluation)
Each chunk is from a DIFFERENT document

| Mode | P@3 | MRR |
|------|-----|-----|
| voyage-3.5 (standard) | 46.67% | 1.0000 |
| voyage-context-3 (per-doc, correct) | 46.67% | 0.5667 |
| voyage-context-3 (all-as-one, WRONG) | 46.67% | 0.6000 |

### Key Findings

1. **Single-doc corpus (test3)**: Contextual delta = -56.67%
2. **Multi-doc corpus (test1)**: Wrong usage delta = -46.67%
3. **Multi-doc corpus (test2)**: Wrong usage delta = +0.00%

### Comparison with Local Models

| Model | test3 P@3 | test1 P@3 | test2 P@3 |
|-------|-----------|-----------|-----------|
| voyage-3.5 | 83.3% | 66.7% | 46.7% |
| voyage-context-3 | 26.7% | 20.0% | 46.7% |


---

## Voyage AI Reranker Benchmark Results

**Model**: `rerank-2`

### Reranking Performance

| Test Dataset | P@3 | MRR |
|--------------|-----|-----|
| test1 (Financial Analytics) | 46.67% | 0.8667 |
| test2 (Trade Evaluation) | 53.33% | 0.7333 |
| test3 (ECOS Release Notes) | 60.00% | 1.0000 |
| **Average** | 53.33% | 0.8667 |
