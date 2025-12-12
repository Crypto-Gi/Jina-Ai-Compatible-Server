#!/usr/bin/env python3
"""
Voyage AI Benchmark - Compare voyage-3.5 (standard) vs voyage-context-3 (contextual)

This tests Voyage AI's contextual embeddings against our benchmark datasets
to see if their approach to late/contextual chunking performs better.

Voyage context-3 API:
- inputs: List[List[str]] - each inner list is chunks from ONE document
- The model encodes each chunk with context from other chunks in the same list
"""

import json
import os
import numpy as np
from typing import List, Dict, Optional
import voyageai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
STANDARD_MODEL = "voyage-3.5"  # Standard embeddings (no context)
CONTEXT_MODEL = "voyage-context-3"  # Contextual embeddings
OUTPUT_DIMENSION = 1024

# Initialize client
client = voyageai.Client(api_key=VOYAGE_API_KEY)

def get_standard_embeddings(texts: List[str], input_type: str = "document") -> List[List[float]]:
    """Get standard embeddings using voyage-3.5."""
    result = client.embed(
        texts=texts,
        model=STANDARD_MODEL,
        input_type=input_type,
        output_dimension=OUTPUT_DIMENSION
    )
    return result.embeddings

def get_contextual_embeddings_single_doc(chunks: List[str], input_type: str = "document") -> List[List[float]]:
    """
    Get contextual embeddings for chunks from a SINGLE document.
    All chunks are passed together so they share context.
    """
    # For contextualized embeddings, inputs is List[List[str]]
    # Each inner list is chunks from one document
    result = client.contextualized_embed(
        inputs=[chunks],  # Single document with multiple chunks
        model=CONTEXT_MODEL,
        input_type=input_type,
        output_dimension=OUTPUT_DIMENSION
    )
    # Result.results[0].embeddings contains embeddings for all chunks in the first (only) document
    return result.results[0].embeddings

def get_contextual_embeddings_multi_doc(corpus_texts: List[str], input_type: str = "document") -> List[List[float]]:
    """
    Get contextual embeddings treating each text as a separate document.
    This is equivalent to standard embeddings (no cross-document context).
    """
    # Each text is its own document (no shared context)
    inputs = [[text] for text in corpus_texts]
    result = client.contextualized_embed(
        inputs=inputs,
        model=CONTEXT_MODEL,
        input_type=input_type,
        output_dimension=OUTPUT_DIMENSION
    )
    # Each result.results[i].embeddings[0] is the single chunk embedding for document i
    return [r.embeddings[0] for r in result.results]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity."""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def run_retrieval(query_emb: List[float], corpus_embs: List[List[float]], corpus_ids: List[str], k: int = 3) -> List[str]:
    """Retrieve top-k most similar corpus items."""
    similarities = [(corpus_ids[i], cosine_similarity(query_emb, emb)) for i, emb in enumerate(corpus_embs)]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in similarities[:k]]

def precision_at_k(predicted: List[str], gold: List[str], k: int = 3) -> float:
    """Calculate precision@k."""
    predicted_set = set(predicted[:k])
    gold_set = set(gold[:k])
    return len(predicted_set & gold_set) / k

def mrr(predicted: List[str], gold: List[str]) -> float:
    """Calculate Mean Reciprocal Rank."""
    for i, pred in enumerate(predicted):
        if pred in gold:
            return 1.0 / (i + 1)
    return 0.0

def test_single_document_corpus(test_file: str) -> Dict:
    """
    Test on single-document corpus (test3 - ECOS release notes).
    All chunks are from ONE document, so contextual embeddings should help.
    """
    print(f"\n  Loading {test_file}...")
    with open(test_file) as f:
        data = json.load(f)
    
    corpus = data["corpus"]
    queries = [q for q in data["queries"] if q["task"] == "retrieval"]
    
    corpus_texts = [item["text"] for item in corpus]
    corpus_ids = [item["id"] for item in corpus]
    
    # Standard embeddings (voyage-3.5)
    print("    Getting standard embeddings (voyage-3.5)...")
    standard_corpus_embs = get_standard_embeddings(corpus_texts, input_type="document")
    
    # Contextual embeddings - all chunks as ONE document (correct usage)
    print("    Getting contextual embeddings (voyage-context-3, single doc)...")
    contextual_corpus_embs = get_contextual_embeddings_single_doc(corpus_texts, input_type="document")
    
    results = {
        "standard": {"p3": [], "mrr": []},
        "contextual": {"p3": [], "mrr": []}
    }
    
    for query in queries:
        # Query embedding (always standard, no context needed for query)
        query_emb = get_standard_embeddings([query["query"]], input_type="query")[0]
        gold = query["gold_top3"]
        
        # Standard retrieval
        pred_std = run_retrieval(query_emb, standard_corpus_embs, corpus_ids)
        results["standard"]["p3"].append(precision_at_k(pred_std, gold))
        results["standard"]["mrr"].append(mrr(pred_std, gold))
        
        # Contextual retrieval
        pred_ctx = run_retrieval(query_emb, contextual_corpus_embs, corpus_ids)
        results["contextual"]["p3"].append(precision_at_k(pred_ctx, gold))
        results["contextual"]["mrr"].append(mrr(pred_ctx, gold))
    
    return {
        "standard_p3": np.mean(results["standard"]["p3"]),
        "standard_mrr": np.mean(results["standard"]["mrr"]),
        "contextual_p3": np.mean(results["contextual"]["p3"]),
        "contextual_mrr": np.mean(results["contextual"]["mrr"]),
    }

def test_multi_document_corpus(test_file: str) -> Dict:
    """
    Test on multi-document corpus (test1, test2).
    Each chunk is from a DIFFERENT document, so contextual embeddings should NOT be used
    across documents (would be incorrect).
    """
    print(f"\n  Loading {test_file}...")
    with open(test_file) as f:
        data = json.load(f)
    
    corpus = data["corpus"]
    queries = [q for q in data["queries"] if q["task"] == "retrieval"]
    
    corpus_texts = [item["text"] for item in corpus]
    corpus_ids = [item["id"] for item in corpus]
    
    # Standard embeddings (voyage-3.5)
    print("    Getting standard embeddings (voyage-3.5)...")
    standard_corpus_embs = get_standard_embeddings(corpus_texts, input_type="document")
    
    # Contextual embeddings - each text as separate document (correct for multi-doc)
    print("    Getting contextual embeddings (voyage-context-3, multi doc)...")
    contextual_corpus_embs = get_contextual_embeddings_multi_doc(corpus_texts, input_type="document")
    
    # Also test INCORRECT usage: all texts as one document (to show the problem)
    print("    Getting contextual embeddings (voyage-context-3, WRONG: all as one doc)...")
    contextual_wrong_embs = get_contextual_embeddings_single_doc(corpus_texts, input_type="document")
    
    results = {
        "standard": {"p3": [], "mrr": []},
        "contextual_correct": {"p3": [], "mrr": []},
        "contextual_wrong": {"p3": [], "mrr": []}
    }
    
    for query in queries:
        query_emb = get_standard_embeddings([query["query"]], input_type="query")[0]
        gold = query["gold_top3"]
        
        # Standard retrieval
        pred_std = run_retrieval(query_emb, standard_corpus_embs, corpus_ids)
        results["standard"]["p3"].append(precision_at_k(pred_std, gold))
        results["standard"]["mrr"].append(mrr(pred_std, gold))
        
        # Contextual correct (each doc separate)
        pred_ctx = run_retrieval(query_emb, contextual_corpus_embs, corpus_ids)
        results["contextual_correct"]["p3"].append(precision_at_k(pred_ctx, gold))
        results["contextual_correct"]["mrr"].append(mrr(pred_ctx, gold))
        
        # Contextual wrong (all as one doc)
        pred_wrong = run_retrieval(query_emb, contextual_wrong_embs, corpus_ids)
        results["contextual_wrong"]["p3"].append(precision_at_k(pred_wrong, gold))
        results["contextual_wrong"]["mrr"].append(mrr(pred_wrong, gold))
    
    return {
        "standard_p3": np.mean(results["standard"]["p3"]),
        "standard_mrr": np.mean(results["standard"]["mrr"]),
        "contextual_correct_p3": np.mean(results["contextual_correct"]["p3"]),
        "contextual_correct_mrr": np.mean(results["contextual_correct"]["mrr"]),
        "contextual_wrong_p3": np.mean(results["contextual_wrong"]["p3"]),
        "contextual_wrong_mrr": np.mean(results["contextual_wrong"]["mrr"]),
    }

def append_results_to_report(results: Dict, output_file: str):
    """Append Voyage AI results to report."""
    
    content = f"""

---

## Voyage AI Benchmark Results

**Models Tested**:
- `voyage-3.5`: Standard embeddings (no contextual awareness)
- `voyage-context-3`: Contextual chunk embeddings (chunks share document context)

### test3 (Single Document - ECOS Release Notes)
All 17 chunks are from ONE document → Contextual embeddings **appropriate**

| Mode | P@3 | MRR |
|------|-----|-----|
| voyage-3.5 (standard) | {results['test3']['standard_p3']*100:.2f}% | {results['test3']['standard_mrr']:.4f} |
| voyage-context-3 (contextual) | {results['test3']['contextual_p3']*100:.2f}% | {results['test3']['contextual_mrr']:.4f} |
| **Delta** | {(results['test3']['contextual_p3'] - results['test3']['standard_p3'])*100:+.2f}% | {results['test3']['contextual_mrr'] - results['test3']['standard_mrr']:+.4f} |

### test1 (Multi-Document - Financial Analytics)
Each chunk is from a DIFFERENT document

| Mode | P@3 | MRR |
|------|-----|-----|
| voyage-3.5 (standard) | {results['test1']['standard_p3']*100:.2f}% | {results['test1']['standard_mrr']:.4f} |
| voyage-context-3 (per-doc, correct) | {results['test1']['contextual_correct_p3']*100:.2f}% | {results['test1']['contextual_correct_mrr']:.4f} |
| voyage-context-3 (all-as-one, WRONG) | {results['test1']['contextual_wrong_p3']*100:.2f}% | {results['test1']['contextual_wrong_mrr']:.4f} |

### test2 (Multi-Document - Trade Evaluation)
Each chunk is from a DIFFERENT document

| Mode | P@3 | MRR |
|------|-----|-----|
| voyage-3.5 (standard) | {results['test2']['standard_p3']*100:.2f}% | {results['test2']['standard_mrr']:.4f} |
| voyage-context-3 (per-doc, correct) | {results['test2']['contextual_correct_p3']*100:.2f}% | {results['test2']['contextual_correct_mrr']:.4f} |
| voyage-context-3 (all-as-one, WRONG) | {results['test2']['contextual_wrong_p3']*100:.2f}% | {results['test2']['contextual_wrong_mrr']:.4f} |

### Key Findings

1. **Single-doc corpus (test3)**: Contextual delta = {(results['test3']['contextual_p3'] - results['test3']['standard_p3'])*100:+.2f}%
2. **Multi-doc corpus (test1)**: Wrong usage delta = {(results['test1']['contextual_wrong_p3'] - results['test1']['standard_p3'])*100:+.2f}%
3. **Multi-doc corpus (test2)**: Wrong usage delta = {(results['test2']['contextual_wrong_p3'] - results['test2']['standard_p3'])*100:+.2f}%

### Comparison with Local Models

| Model | test3 P@3 | test1 P@3 | test2 P@3 |
|-------|-----------|-----------|-----------|
| voyage-3.5 | {results['test3']['standard_p3']*100:.1f}% | {results['test1']['standard_p3']*100:.1f}% | {results['test2']['standard_p3']*100:.1f}% |
| voyage-context-3 | {results['test3']['contextual_p3']*100:.1f}% | {results['test1']['contextual_correct_p3']*100:.1f}% | {results['test2']['contextual_correct_p3']*100:.1f}% |
"""
    
    with open(output_file, "a") as f:
        f.write(content)
    
    print(f"\nResults appended to: {output_file}")

if __name__ == "__main__":
    test_files = {
        "test1": "/home/mir/projects/Jina-AI/tests/test1.json",
        "test2": "/home/mir/projects/Jina-AI/tests/test2.json",
        "test3": "/home/mir/projects/Jina-AI/tests/test3_ecos_release_notes.json"
    }
    output_file = "/home/mir/projects/Jina-AI/tests/BENCHMARK_RESULTS.md"
    
    print("=" * 60)
    print("Voyage AI Benchmark")
    print("=" * 60)
    print(f"Standard Model: {STANDARD_MODEL}")
    print(f"Contextual Model: {CONTEXT_MODEL}")
    print(f"Output Dimension: {OUTPUT_DIMENSION}")
    
    results = {}
    
    # Test3 - single document corpus
    print("\n" + "=" * 60)
    print("Testing test3 (single document - contextual appropriate)")
    print("=" * 60)
    results["test3"] = test_single_document_corpus(test_files["test3"])
    print(f"  Standard P@3: {results['test3']['standard_p3']*100:.1f}%")
    print(f"  Contextual P@3: {results['test3']['contextual_p3']*100:.1f}%")
    print(f"  Delta: {(results['test3']['contextual_p3'] - results['test3']['standard_p3'])*100:+.1f}%")
    
    # Test1 - multi document corpus
    print("\n" + "=" * 60)
    print("Testing test1 (multi-document)")
    print("=" * 60)
    results["test1"] = test_multi_document_corpus(test_files["test1"])
    print(f"  Standard P@3: {results['test1']['standard_p3']*100:.1f}%")
    print(f"  Contextual (correct) P@3: {results['test1']['contextual_correct_p3']*100:.1f}%")
    print(f"  Contextual (wrong) P@3: {results['test1']['contextual_wrong_p3']*100:.1f}%")
    
    # Test2 - multi document corpus
    print("\n" + "=" * 60)
    print("Testing test2 (multi-document)")
    print("=" * 60)
    results["test2"] = test_multi_document_corpus(test_files["test2"])
    print(f"  Standard P@3: {results['test2']['standard_p3']*100:.1f}%")
    print(f"  Contextual (correct) P@3: {results['test2']['contextual_correct_p3']*100:.1f}%")
    print(f"  Contextual (wrong) P@3: {results['test2']['contextual_wrong_p3']*100:.1f}%")
    
    # Append to report
    append_results_to_report(results, output_file)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
