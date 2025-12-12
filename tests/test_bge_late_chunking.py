#!/usr/bin/env python3
"""
Test BGE-M3 Late Chunking vs Standard Embeddings
=================================================
Compare the current late chunking implementation against standard embeddings
to see if the offset mapping fix improves results.
"""

import json
import sys
from pathlib import Path

import numpy as np
import requests

# Test data - using test3_ecos_release_notes.json (single-doc ECOS) since that's where late chunking should help most
TEST_FILE = Path(__file__).parent / "test3_ecos_release_notes.json"
API_URL = "http://localhost:8080/v1/embeddings"


def load_test_data():
    """Load test data from test3_ecos_release_notes.json."""
    with open(TEST_FILE) as f:
        data = json.load(f)
    
    # Extract corpus texts from the structured format
    corpus = [item["text"] for item in data["corpus"]]
    
    # Extract retrieval queries and convert gold_top3 IDs to indices
    id_to_idx = {item["id"]: i for i, item in enumerate(data["corpus"])}
    queries = []
    for q in data["queries"]:
        if q["task"] == "retrieval":
            gold_indices = [id_to_idx[cid] for cid in q["gold_top3"] if cid in id_to_idx]
            queries.append({
                "query": q["query"],
                "gold": gold_indices,
            })
    
    return {"corpus": corpus, "queries": queries}


def get_embeddings(texts: list[str], late_chunking: bool = False) -> np.ndarray:
    """Get embeddings from the local API."""
    response = requests.post(
        API_URL,
        json={
            "model": "bge-m3",
            "input": texts,
            "late_chunking": late_chunking,
        },
        timeout=120,
    )
    response.raise_for_status()
    result = response.json()
    embeddings = [d["embedding"] for d in result["data"]]
    return np.array(embeddings)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def precision_at_k(retrieved: list[int], relevant: list[int], k: int = 3) -> float:
    """Calculate Precision@K."""
    top_k = retrieved[:k]
    hits = len(set(top_k) & set(relevant))
    return hits / k


def run_benchmark(corpus: list[str], queries: list[dict], mode: str) -> dict:
    """Run benchmark for a given mode (standard or late_chunking)."""
    late_chunking = mode == "late_chunking"
    
    print(f"\n{'='*60}")
    print(f"Mode: {mode.upper()}")
    print(f"{'='*60}")
    
    # Get corpus embeddings
    print(f"Encoding {len(corpus)} corpus chunks...")
    corpus_embeddings = get_embeddings(corpus, late_chunking=late_chunking)
    print(f"Corpus embeddings shape: {corpus_embeddings.shape}")
    
    # Run queries
    results = []
    for i, query_data in enumerate(queries):
        query = query_data["query"]
        gold = query_data["gold"]  # Already top 3 relevant indices
        
        # Get query embedding (always standard for queries)
        query_embedding = get_embeddings([query], late_chunking=False)[0]
        
        # Compute similarities
        similarities = [
            cosine_similarity(query_embedding, corpus_embeddings[j])
            for j in range(len(corpus))
        ]
        
        # Rank by similarity
        ranked = sorted(range(len(similarities)), key=lambda x: similarities[x], reverse=True)
        
        # Calculate P@3
        p_at_3 = precision_at_k(ranked, gold, k=3)
        results.append(p_at_3)
        
        print(f"  Query {i+1}: P@3 = {p_at_3:.2%} | Top 3: {ranked[:3]} | Gold: {gold}")
    
    avg_p3 = np.mean(results)
    print(f"\n  Average P@3: {avg_p3:.2%}")
    
    return {
        "mode": mode,
        "p_at_3_scores": results,
        "avg_p_at_3": avg_p3,
    }


def main():
    print("="*60)
    print("BGE-M3 Late Chunking Benchmark")
    print("="*60)
    
    # Load test data
    data = load_test_data()
    corpus = data["corpus"]
    queries = data["queries"]
    
    print(f"\nTest file: {TEST_FILE}")
    print(f"Corpus size: {len(corpus)} chunks")
    print(f"Queries: {len(queries)}")
    
    # Run standard benchmark
    standard_results = run_benchmark(corpus, queries, "standard")
    
    # Run late chunking benchmark
    late_results = run_benchmark(corpus, queries, "late_chunking")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Standard P@3:      {standard_results['avg_p_at_3']:.2%}")
    print(f"Late Chunking P@3: {late_results['avg_p_at_3']:.2%}")
    
    delta = late_results['avg_p_at_3'] - standard_results['avg_p_at_3']
    print(f"Delta:             {delta:+.2%}")
    
    if delta > 0:
        print("\n✅ Late chunking IMPROVED results!")
    elif delta < 0:
        print("\n❌ Late chunking HURT results.")
    else:
        print("\n➖ No difference.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
