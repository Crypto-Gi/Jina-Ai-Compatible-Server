#!/usr/bin/env python3
"""
Benchmark Qwen3-4B and Jina V4 embedding models.
- Qwen3-4B: Standard embeddings only (no late chunking support)
- Jina V4: Both standard and late chunking
"""

import json
import requests
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple

API_BASE = "http://localhost:8080"

def get_embeddings(texts: List[str], model: str, late_chunking: bool = False) -> List[List[float]]:
    """Get embeddings from the local API."""
    payload = {
        "input": texts,
        "model": model
    }
    if late_chunking:
        payload["late_chunking"] = True
    
    response = requests.post(f"{API_BASE}/v1/embeddings", json=payload)
    response.raise_for_status()
    data = response.json()
    return [item["embedding"] for item in data["data"]]

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

def ndcg_at_k(predicted: List[str], gold: List[str], k: int = 3) -> float:
    """Calculate NDCG@k."""
    def dcg(relevances):
        return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
    
    relevances = [1 if p in gold else 0 for p in predicted[:k]]
    ideal_relevances = sorted(relevances, reverse=True)
    
    dcg_val = dcg(relevances)
    idcg_val = dcg(ideal_relevances)
    
    return dcg_val / idcg_val if idcg_val > 0 else 0.0

def test_model(model: str, test_files: Dict[str, str], late_chunking: bool = False) -> Dict:
    """Test a model on all test files."""
    mode = "late_chunking" if late_chunking else "standard"
    print(f"\n{'='*60}")
    print(f"Testing {model} ({mode})")
    print(f"{'='*60}")
    
    all_results = {}
    
    for test_name, test_path in test_files.items():
        print(f"\n  {test_name}:")
        with open(test_path) as f:
            data = json.load(f)
        
        corpus = data["corpus"]
        retrieval_queries = [q for q in data["queries"] if q["task"] == "retrieval"]
        
        corpus_texts = [item["text"] for item in corpus]
        corpus_ids = [item["id"] for item in corpus]
        
        # Get corpus embeddings
        print(f"    Encoding corpus ({len(corpus_texts)} chunks)...")
        corpus_embs = get_embeddings(corpus_texts, model, late_chunking=late_chunking)
        
        # Test retrieval
        results = {"p3": [], "mrr": [], "ndcg3": []}
        
        for query in retrieval_queries:
            # Use same encoding mode for query as corpus
            query_emb = get_embeddings([query["query"]], model, late_chunking=late_chunking)[0]
            gold = query["gold_top3"]
            
            predicted = run_retrieval(query_emb, corpus_embs, corpus_ids)
            
            results["p3"].append(precision_at_k(predicted, gold))
            results["mrr"].append(mrr(predicted, gold))
            results["ndcg3"].append(ndcg_at_k(predicted, gold))
        
        avg_p3 = np.mean(results["p3"])
        avg_mrr = np.mean(results["mrr"])
        avg_ndcg3 = np.mean(results["ndcg3"])
        
        print(f"    P@3: {avg_p3*100:.1f}%, MRR: {avg_mrr:.4f}, NDCG@3: {avg_ndcg3:.4f}")
        
        all_results[test_name] = {
            "p3": avg_p3,
            "mrr": avg_mrr,
            "ndcg3": avg_ndcg3
        }
    
    # Calculate averages
    avg_p3 = np.mean([r["p3"] for r in all_results.values()])
    avg_mrr = np.mean([r["mrr"] for r in all_results.values()])
    avg_ndcg3 = np.mean([r["ndcg3"] for r in all_results.values()])
    
    all_results["average"] = {"p3": avg_p3, "mrr": avg_mrr, "ndcg3": avg_ndcg3}
    
    print(f"\n  AVERAGE: P@3={avg_p3*100:.1f}%, MRR={avg_mrr:.4f}, NDCG@3={avg_ndcg3:.4f}")
    
    return all_results

def generate_report(results: Dict) -> str:
    """Generate markdown report."""
    report = f"""

---

## Qwen3-4B and Jina V4 Benchmark Results

**Tested**: {datetime.now().isoformat()}

### Qwen3-Embedding-4B (Standard Only)

Qwen3 models do not support late chunking (uses last-token pooling).

| Test | P@3 | MRR | NDCG@3 |
|------|-----|-----|--------|
| test1 | {results['qwen3-embedding-4b']['test1']['p3']*100:.2f}% | {results['qwen3-embedding-4b']['test1']['mrr']:.4f} | {results['qwen3-embedding-4b']['test1']['ndcg3']:.4f} |
| test2 | {results['qwen3-embedding-4b']['test2']['p3']*100:.2f}% | {results['qwen3-embedding-4b']['test2']['mrr']:.4f} | {results['qwen3-embedding-4b']['test2']['ndcg3']:.4f} |
| test3 | {results['qwen3-embedding-4b']['test3']['p3']*100:.2f}% | {results['qwen3-embedding-4b']['test3']['mrr']:.4f} | {results['qwen3-embedding-4b']['test3']['ndcg3']:.4f} |
| **Average** | **{results['qwen3-embedding-4b']['average']['p3']*100:.2f}%** | **{results['qwen3-embedding-4b']['average']['mrr']:.4f}** | **{results['qwen3-embedding-4b']['average']['ndcg3']:.4f}** |

### Jina-Embeddings-V4 (Standard)

| Test | P@3 | MRR | NDCG@3 |
|------|-----|-----|--------|
| test1 | {results['jina-embeddings-v4']['test1']['p3']*100:.2f}% | {results['jina-embeddings-v4']['test1']['mrr']:.4f} | {results['jina-embeddings-v4']['test1']['ndcg3']:.4f} |
| test2 | {results['jina-embeddings-v4']['test2']['p3']*100:.2f}% | {results['jina-embeddings-v4']['test2']['mrr']:.4f} | {results['jina-embeddings-v4']['test2']['ndcg3']:.4f} |
| test3 | {results['jina-embeddings-v4']['test3']['p3']*100:.2f}% | {results['jina-embeddings-v4']['test3']['mrr']:.4f} | {results['jina-embeddings-v4']['test3']['ndcg3']:.4f} |
| **Average** | **{results['jina-embeddings-v4']['average']['p3']*100:.2f}%** | **{results['jina-embeddings-v4']['average']['mrr']:.4f}** | **{results['jina-embeddings-v4']['average']['ndcg3']:.4f}** |

### Jina-Embeddings-V4 (Late Chunking)

| Test | P@3 | MRR | NDCG@3 |
|------|-----|-----|--------|
| test1 | {results['jina-embeddings-v4-late']['test1']['p3']*100:.2f}% | {results['jina-embeddings-v4-late']['test1']['mrr']:.4f} | {results['jina-embeddings-v4-late']['test1']['ndcg3']:.4f} |
| test2 | {results['jina-embeddings-v4-late']['test2']['p3']*100:.2f}% | {results['jina-embeddings-v4-late']['test2']['mrr']:.4f} | {results['jina-embeddings-v4-late']['test2']['ndcg3']:.4f} |
| test3 | {results['jina-embeddings-v4-late']['test3']['p3']*100:.2f}% | {results['jina-embeddings-v4-late']['test3']['mrr']:.4f} | {results['jina-embeddings-v4-late']['test3']['ndcg3']:.4f} |
| **Average** | **{results['jina-embeddings-v4-late']['average']['p3']*100:.2f}%** | **{results['jina-embeddings-v4-late']['average']['mrr']:.4f}** | **{results['jina-embeddings-v4-late']['average']['ndcg3']:.4f}** |

### Late Chunking Impact (Jina V4)

| Test | Standard P@3 | Late P@3 | Delta |
|------|--------------|----------|-------|
| test1 | {results['jina-embeddings-v4']['test1']['p3']*100:.2f}% | {results['jina-embeddings-v4-late']['test1']['p3']*100:.2f}% | {(results['jina-embeddings-v4-late']['test1']['p3'] - results['jina-embeddings-v4']['test1']['p3'])*100:+.2f}% |
| test2 | {results['jina-embeddings-v4']['test2']['p3']*100:.2f}% | {results['jina-embeddings-v4-late']['test2']['p3']*100:.2f}% | {(results['jina-embeddings-v4-late']['test2']['p3'] - results['jina-embeddings-v4']['test2']['p3'])*100:+.2f}% |
| test3 | {results['jina-embeddings-v4']['test3']['p3']*100:.2f}% | {results['jina-embeddings-v4-late']['test3']['p3']*100:.2f}% | {(results['jina-embeddings-v4-late']['test3']['p3'] - results['jina-embeddings-v4']['test3']['p3'])*100:+.2f}% |
| **Average** | {results['jina-embeddings-v4']['average']['p3']*100:.2f}% | {results['jina-embeddings-v4-late']['average']['p3']*100:.2f}% | {(results['jina-embeddings-v4-late']['average']['p3'] - results['jina-embeddings-v4']['average']['p3'])*100:+.2f}% |

### Comparison: All Embedding Models

| Model | Avg P@3 | Avg MRR | Late Chunking |
|-------|---------|---------|---------------|
| qwen3-embedding-4b | {results['qwen3-embedding-4b']['average']['p3']*100:.1f}% | {results['qwen3-embedding-4b']['average']['mrr']:.4f} | N/A |
| jina-embeddings-v4 | {results['jina-embeddings-v4']['average']['p3']*100:.1f}% | {results['jina-embeddings-v4']['average']['mrr']:.4f} | Standard |
| jina-embeddings-v4 (late) | {results['jina-embeddings-v4-late']['average']['p3']*100:.1f}% | {results['jina-embeddings-v4-late']['average']['mrr']:.4f} | Enabled |
"""
    return report

if __name__ == "__main__":
    test_files = {
        "test1": "/home/mir/projects/Jina-AI/tests/test1.json",
        "test2": "/home/mir/projects/Jina-AI/tests/test2.json",
        "test3": "/home/mir/projects/Jina-AI/tests/test3_ecos_release_notes.json"
    }
    
    results = {}
    
    # Test Qwen3-4B (standard only)
    results["qwen3-embedding-4b"] = test_model("qwen3-embedding-4b", test_files, late_chunking=False)
    
    # Test Jina V4 (standard)
    results["jina-embeddings-v4"] = test_model("jina-embeddings-v4", test_files, late_chunking=False)
    
    # Test Jina V4 (late chunking)
    results["jina-embeddings-v4-late"] = test_model("jina-embeddings-v4", test_files, late_chunking=True)
    
    # Generate and save report
    report = generate_report(results)
    
    # Append to EMBEDDINGS_TEST_RESULTS.md
    with open("/home/mir/projects/Jina-AI/docs/EMBEDDINGS_TEST_RESULTS.md", "a") as f:
        f.write(report)
    print(f"\nResults appended to: /home/mir/projects/Jina-AI/docs/EMBEDDINGS_TEST_RESULTS.md")
    
    # Save raw results as JSON
    with open("/home/mir/projects/Jina-AI/tests/qwen4b_jinav4_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Raw results saved to: /home/mir/projects/Jina-AI/tests/qwen4b_jinav4_results.json")
    
    print("\n" + "="*60)
    print("Benchmark complete!")
    print("="*60)
