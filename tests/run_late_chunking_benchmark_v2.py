#!/usr/bin/env python3
"""
Late Chunking Benchmark v2 - Correct Usage

Late chunking should be applied PER-DOCUMENT, not across the entire corpus.
This script tests late chunking correctly by:
1. For test3 (ECOS release notes): All chunks are from ONE document, so late chunking applies
2. For test1/test2: Chunks are from DIFFERENT documents, so late chunking should NOT be used

This demonstrates when late chunking helps vs hurts.
"""

import json
import requests
import numpy as np
from typing import List, Dict

BASE_URL = "http://localhost:8080"

def get_embeddings(texts: List[str], model: str, late_chunking: bool = False) -> List[List[float]]:
    """Get embeddings for texts."""
    resp = requests.post(
        f"{BASE_URL}/v1/embeddings",
        json={
            "model": model,
            "input": texts,
            "late_chunking": late_chunking
        }
    )
    data = resp.json()
    return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]

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

def test_single_document_late_chunking(test_file: str, model: str) -> Dict:
    """
    Test late chunking on a single-document corpus (test3).
    All chunks belong to the same document, so late chunking should help.
    """
    with open(test_file) as f:
        data = json.load(f)
    
    corpus = data["corpus"]
    queries = [q for q in data["queries"] if q["task"] == "retrieval"]
    
    corpus_texts = [item["text"] for item in corpus]
    corpus_ids = [item["id"] for item in corpus]
    
    # Standard encoding - each chunk independently
    print("    Standard encoding...")
    standard_embs = get_embeddings(corpus_texts, model, late_chunking=False)
    
    # Late chunking - all chunks as one document (CORRECT usage for single-doc corpus)
    print("    Late chunking (single document)...")
    late_embs = get_embeddings(corpus_texts, model, late_chunking=True)
    
    results = {"standard": {"p3": [], "mrr": []}, "late": {"p3": [], "mrr": []}}
    
    for query in queries:
        query_emb_std = get_embeddings([query["query"]], model, late_chunking=False)[0]
        query_emb_late = get_embeddings([query["query"]], model, late_chunking=False)[0]  # Query never uses late chunking
        
        gold = query["gold_top3"]
        
        # Standard retrieval
        pred_std = run_retrieval(query_emb_std, standard_embs, corpus_ids)
        results["standard"]["p3"].append(precision_at_k(pred_std, gold))
        results["standard"]["mrr"].append(mrr(pred_std, gold))
        
        # Late chunking retrieval
        pred_late = run_retrieval(query_emb_late, late_embs, corpus_ids)
        results["late"]["p3"].append(precision_at_k(pred_late, gold))
        results["late"]["mrr"].append(mrr(pred_late, gold))
    
    return {
        "standard_p3": np.mean(results["standard"]["p3"]),
        "standard_mrr": np.mean(results["standard"]["mrr"]),
        "late_p3": np.mean(results["late"]["p3"]),
        "late_mrr": np.mean(results["late"]["mrr"]),
    }

def test_multi_document_no_late_chunking(test_file: str, model: str) -> Dict:
    """
    Test on multi-document corpus (test1, test2).
    Each chunk is from a DIFFERENT document, so late chunking should NOT be used.
    This shows what happens when late chunking is misused.
    """
    with open(test_file) as f:
        data = json.load(f)
    
    corpus = data["corpus"]
    queries = [q for q in data["queries"] if q["task"] == "retrieval"]
    
    corpus_texts = [item["text"] for item in corpus]
    corpus_ids = [item["id"] for item in corpus]
    
    # Standard encoding - correct for multi-doc
    print("    Standard encoding (correct for multi-doc)...")
    standard_embs = get_embeddings(corpus_texts, model, late_chunking=False)
    
    # Late chunking - INCORRECT for multi-doc (demonstrates the problem)
    print("    Late chunking (INCORRECT for multi-doc)...")
    late_embs = get_embeddings(corpus_texts, model, late_chunking=True)
    
    results = {"standard": {"p3": [], "mrr": []}, "late": {"p3": [], "mrr": []}}
    
    for query in queries:
        query_emb = get_embeddings([query["query"]], model, late_chunking=False)[0]
        gold = query["gold_top3"]
        
        pred_std = run_retrieval(query_emb, standard_embs, corpus_ids)
        results["standard"]["p3"].append(precision_at_k(pred_std, gold))
        results["standard"]["mrr"].append(mrr(pred_std, gold))
        
        pred_late = run_retrieval(query_emb, late_embs, corpus_ids)
        results["late"]["p3"].append(precision_at_k(pred_late, gold))
        results["late"]["mrr"].append(mrr(pred_late, gold))
    
    return {
        "standard_p3": np.mean(results["standard"]["p3"]),
        "standard_mrr": np.mean(results["standard"]["mrr"]),
        "late_p3": np.mean(results["late"]["p3"]),
        "late_mrr": np.mean(results["late"]["mrr"]),
    }

def append_results_to_report(results: Dict, model: str, output_file: str):
    """Append late chunking v2 results to report."""
    
    content = f"""

---

## Late Chunking Analysis (Correct Usage) - {model}

**Key Insight**: Late chunking should only be used when all input texts are chunks from the **SAME document**.

### Test Results

#### test3 (Single Document - ECOS Release Notes)
All 17 chunks are from ONE document → Late chunking is **appropriate**

| Mode | P@3 | MRR | 
|------|-----|-----|
| Standard | {results['test3']['standard_p3']*100:.2f}% | {results['test3']['standard_mrr']:.4f} |
| Late Chunking | {results['test3']['late_p3']*100:.2f}% | {results['test3']['late_mrr']:.4f} |
| **Delta** | {(results['test3']['late_p3'] - results['test3']['standard_p3'])*100:+.2f}% | {results['test3']['late_mrr'] - results['test3']['standard_mrr']:+.4f} |

#### test1 (Multi-Document - Financial Analytics)
Each chunk is from a DIFFERENT document → Late chunking is **NOT appropriate**

| Mode | P@3 | MRR |
|------|-----|-----|
| Standard (correct) | {results['test1']['standard_p3']*100:.2f}% | {results['test1']['standard_mrr']:.4f} |
| Late Chunking (wrong) | {results['test1']['late_p3']*100:.2f}% | {results['test1']['late_mrr']:.4f} |
| **Delta** | {(results['test1']['late_p3'] - results['test1']['standard_p3'])*100:+.2f}% | {results['test1']['late_mrr'] - results['test1']['standard_mrr']:+.4f} |

#### test2 (Multi-Document - Trade Evaluation)
Each chunk is from a DIFFERENT document → Late chunking is **NOT appropriate**

| Mode | P@3 | MRR |
|------|-----|-----|
| Standard (correct) | {results['test2']['standard_p3']*100:.2f}% | {results['test2']['standard_mrr']:.4f} |
| Late Chunking (wrong) | {results['test2']['late_p3']*100:.2f}% | {results['test2']['late_mrr']:.4f} |
| **Delta** | {(results['test2']['late_p3'] - results['test2']['standard_p3'])*100:+.2f}% | {results['test2']['late_mrr'] - results['test2']['standard_mrr']:+.4f} |

### Conclusion

- **Single-document corpus (test3)**: Late chunking delta = {(results['test3']['late_p3'] - results['test3']['standard_p3'])*100:+.2f}%
- **Multi-document corpus (test1)**: Late chunking delta = {(results['test1']['late_p3'] - results['test1']['standard_p3'])*100:+.2f}%
- **Multi-document corpus (test2)**: Late chunking delta = {(results['test2']['late_p3'] - results['test2']['standard_p3'])*100:+.2f}%

**Recommendation**: Only use `late_chunking=true` when embedding chunks from a single long document. 
For corpora with independent documents, use standard embedding.
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
    
    # Check which model is loaded
    resp = requests.get(f"{BASE_URL}/v1/models")
    loaded_models = [m["id"] for m in resp.json()["data"]]
    print(f"Loaded models: {loaded_models}")
    
    # Find embedding model
    embedding_model = None
    for m in loaded_models:
        if "embedding" in m or m == "bge-m3":
            embedding_model = m
            break
    
    if not embedding_model:
        print("No embedding model loaded!")
        exit(1)
    
    print(f"\n{'='*60}")
    print(f"Late Chunking Analysis (Correct Usage): {embedding_model}")
    print(f"{'='*60}")
    
    results = {}
    
    # Test3 - single document, late chunking appropriate
    print(f"\n  test3 (single document - late chunking appropriate):")
    results["test3"] = test_single_document_late_chunking(test_files["test3"], embedding_model)
    
    # Test1 - multi document, late chunking NOT appropriate
    print(f"\n  test1 (multi-document - late chunking NOT appropriate):")
    results["test1"] = test_multi_document_no_late_chunking(test_files["test1"], embedding_model)
    
    # Test2 - multi document, late chunking NOT appropriate
    print(f"\n  test2 (multi-document - late chunking NOT appropriate):")
    results["test2"] = test_multi_document_no_late_chunking(test_files["test2"], embedding_model)
    
    # Print summary
    print(f"\n  Summary for {embedding_model}:")
    print(f"    test3 (single-doc): Standard P@3={results['test3']['standard_p3']*100:.1f}%, Late P@3={results['test3']['late_p3']*100:.1f}% (delta: {(results['test3']['late_p3']-results['test3']['standard_p3'])*100:+.1f}%)")
    print(f"    test1 (multi-doc):  Standard P@3={results['test1']['standard_p3']*100:.1f}%, Late P@3={results['test1']['late_p3']*100:.1f}% (delta: {(results['test1']['late_p3']-results['test1']['standard_p3'])*100:+.1f}%)")
    print(f"    test2 (multi-doc):  Standard P@3={results['test2']['standard_p3']*100:.1f}%, Late P@3={results['test2']['late_p3']*100:.1f}% (delta: {(results['test2']['late_p3']-results['test2']['standard_p3'])*100:+.1f}%)")
    
    append_results_to_report(results, embedding_model, output_file)
    
    print("\nDone!")
