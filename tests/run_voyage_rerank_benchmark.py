#!/usr/bin/env python3
"""
Voyage AI Reranker Benchmark - Test rerank-2 against our datasets
"""

import json
import os
import numpy as np
from typing import List, Dict
import voyageai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
RERANK_MODEL = "rerank-2"

# Initialize client
client = voyageai.Client(api_key=VOYAGE_API_KEY)

def rerank_documents(query: str, documents: List[str], top_k: int = 3) -> List[int]:
    """Rerank documents and return indices of top-k."""
    result = client.rerank(
        query=query,
        documents=documents,
        model=RERANK_MODEL,
        top_k=top_k
    )
    # Return indices of top results
    return [r.index for r in result.results]

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

def test_reranking(test_file: str) -> Dict:
    """Test reranking on a dataset."""
    print(f"\n  Loading {test_file}...")
    with open(test_file) as f:
        data = json.load(f)
    
    corpus = data["corpus"]
    queries = [q for q in data["queries"] if q["task"] == "rerank"]
    
    corpus_texts = [item["text"] for item in corpus]
    corpus_ids = [item["id"] for item in corpus]
    
    results = {"p3": [], "mrr": []}
    
    for query in queries:
        print(f"    Reranking for query: {query['id']}...")
        
        # Get reranked indices
        reranked_indices = rerank_documents(query["query"], corpus_texts, top_k=3)
        
        # Convert indices to IDs
        predicted_ids = [corpus_ids[i] for i in reranked_indices]
        gold = query["gold_top3"]
        
        results["p3"].append(precision_at_k(predicted_ids, gold))
        results["mrr"].append(mrr(predicted_ids, gold))
    
    return {
        "p3": np.mean(results["p3"]),
        "mrr": np.mean(results["mrr"]),
    }

def append_results_to_report(results: Dict, output_file: str):
    """Append Voyage reranker results to report."""
    
    content = f"""

---

## Voyage AI Reranker Benchmark Results

**Model**: `rerank-2`

### Reranking Performance

| Test Dataset | P@3 | MRR |
|--------------|-----|-----|
| test1 (Financial Analytics) | {results['test1']['p3']*100:.2f}% | {results['test1']['mrr']:.4f} |
| test2 (Trade Evaluation) | {results['test2']['p3']*100:.2f}% | {results['test2']['mrr']:.4f} |
| test3 (ECOS Release Notes) | {results['test3']['p3']*100:.2f}% | {results['test3']['mrr']:.4f} |
| **Average** | {np.mean([results['test1']['p3'], results['test2']['p3'], results['test3']['p3']])*100:.2f}% | {np.mean([results['test1']['mrr'], results['test2']['mrr'], results['test3']['mrr']]):.4f} |
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
    print("Voyage AI Reranker Benchmark")
    print("=" * 60)
    print(f"Model: {RERANK_MODEL}")
    
    results = {}
    
    for name, path in test_files.items():
        print(f"\n{'=' * 60}")
        print(f"Testing {name}")
        print("=" * 60)
        results[name] = test_reranking(path)
        print(f"  P@3: {results[name]['p3']*100:.1f}%")
        print(f"  MRR: {results[name]['mrr']:.4f}")
    
    # Append to report
    append_results_to_report(results, output_file)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
