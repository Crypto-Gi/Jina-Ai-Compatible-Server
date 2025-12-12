#!/usr/bin/env python3
"""
Final Reranker Benchmark
========================

Compares local reranker models against Voyage AI rerankers on the three
retrieval test JSONs using P@3.

Local models (via Jina-compatible /v1/rerank):
- jina-reranker-v3
- bge-reranker-v2-m3
- qwen3-reranker-0.6b
- qwen3-reranker-4b

Remote Voyage models (via official voyageai Python client):
- rerank-2.5
- rerank-2.5-lite

Metric:
- For each query, call reranker over the full corpus texts and collect top-3
  ranked documents.
- Compute P@3 = |top3 ∩ gold_top3| / 3, using the `gold_top3` IDs from the
  JSON files.
- Report average P@3 per test per model, plus overall ranking.

This script has been configured using the official Voyage docs:
- Rerank API (Python client): vo.rerank(query, documents, model="rerank-2.5", top_k=3)
  https://docs.voyageai.com/docs/reranker
- Rerank REST API schema: POST https://api.voyageai.com/v1/rerank

And local rerank schema from app/schemas/rerank.py:
- POST /v1/rerank with fields: model, query, documents, top_n, return_documents.
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import voyageai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

LOCAL_API_BASE = os.environ.get("LOCAL_API_BASE", "http://localhost:8080/v1")
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")

TEST_FILES: Dict[str, str] = {
    "test1": "/home/mir/projects/Jina-AI/tests/test1.json",
    "test2": "/home/mir/projects/Jina-AI/tests/test2.json",
    "test3": "/home/mir/projects/Jina-AI/tests/test3_ecos_release_notes.json",
}

# Model configurations
MODELS: Dict[str, Dict[str, Any]] = {
    # Local rerankers
    "jina-reranker-v3": {"type": "local", "model_id": "jina-reranker-v3"},
    "bge-reranker-v2-m3": {"type": "local", "model_id": "bge-reranker-v2-m3"},
    "qwen3-reranker-0.6b": {"type": "local", "model_id": "qwen3-reranker-0.6b"},
    "qwen3-reranker-4b": {"type": "local", "model_id": "qwen3-reranker-4b"},
    # Voyage rerankers
    "voyage-rerank-2.5": {"type": "voyage", "model_id": "rerank-2.5"},
    "voyage-rerank-2.5-lite": {"type": "voyage", "model_id": "rerank-2.5-lite"},
}


def cosine_similarity(a, b) -> float:
    """Cosine similarity helper (not strictly needed for rerank, but kept for debug)."""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def call_local_rerank(
    model_id: str,
    query: str,
    documents: List[str],
    top_n: int = 3,
) -> Optional[List[int]]:
    """Call local /v1/rerank and return list of top indices.

    Uses schema from app/schemas/rerank.py (Jina-compatible):
    - model, query, documents, top_n, return_documents.
    """

    payload = {
        "model": model_id,
        "query": query,
        "documents": documents,
        "top_n": top_n,
        "return_documents": False,
    }

    try:
        resp = requests.post(f"{LOCAL_API_BASE}/rerank", json=payload, timeout=120)
    except Exception as e:
        print(f"      Local rerank exception ({model_id}): {e}")
        return None

    if resp.status_code != 200:
        print(f"      Local rerank error ({model_id}) {resp.status_code}: {resp.text[:200]}")
        return None

    data = resp.json()
    results = data.get("results", [])
    if not isinstance(results, list) or not results:
        print(f"      Local rerank ({model_id}) returned empty results")
        return None

    # Results are already sorted by relevance_score desc per schema
    indices = [r.get("index") for r in results[:top_n] if isinstance(r.get("index"), int)]
    if len(indices) == 0:
        return None
    return indices


def call_voyage_rerank(
    client: voyageai.Client,
    model_id: str,
    query: str,
    documents: List[str],
    top_k: int = 3,
) -> Optional[List[int]]:
    """Call Voyage rerank API via official Python client and return top indices.

    Official usage (from docs.voyageai.com/docs/reranker):

    ```python
    reranking = vo.rerank(query, documents, model="rerank-2.5", top_k=3)
    for r in reranking.results:
        print(r.document, r.relevance_score, r.index)
    ```
    """

    try:
        reranking = client.rerank(query, documents, model=model_id, top_k=top_k)
    except Exception as e:
        print(f"      Voyage rerank exception ({model_id}): {e}")
        return None

    results = getattr(reranking, "results", None)
    if not results:
        print(f"      Voyage rerank ({model_id}) returned empty results")
        return None

    indices: List[int] = []
    for r in results[:top_k]:
        idx = getattr(r, "index", None)
        if isinstance(idx, int):
            indices.append(idx)
    if not indices:
        return None
    return indices


def run_benchmark_for_model(
    model_name: str,
    config: Dict[str, Any],
    test_name: str,
    test_file: str,
    voyage_client: Optional[voyageai.Client] = None,
) -> Optional[float]:
    """Run P@3 benchmark for a single model on a single test file."""

    with open(test_file, "r") as f:
        data = json.load(f)

    corpus = data["corpus"]
    queries = [q for q in data["queries"] if q.get("task") == "retrieval"]

    corpus_texts = [item["text"] for item in corpus]
    corpus_ids = [item["id"] for item in corpus]

    model_type = config["type"]

    p3_scores: List[float] = []

    for q in queries:
        q_text = q["query"]
        gold_ids = q["gold_top3"]

        if model_type == "local":
            indices = call_local_rerank(config["model_id"], q_text, corpus_texts, top_n=3)
        elif model_type == "voyage":
            if voyage_client is None:
                print("      Voyage client not initialized")
                return None
            indices = call_voyage_rerank(voyage_client, config["model_id"], q_text, corpus_texts, top_k=3)
        else:
            print(f"      Unknown model type: {model_type}")
            return None

        if indices is None:
            continue

        top3_ids = [corpus_ids[i] for i in indices if 0 <= i < len(corpus_ids)]
        p3 = len(set(top3_ids) & set(gold_ids)) / 3.0
        p3_scores.append(p3)

    if not p3_scores:
        return 0.0

    return float(np.mean(p3_scores) * 100.0)


def main() -> None:
    print("=" * 80)
    print("FINAL RERANKER BENCHMARK")
    print("=" * 80)
    print("\nLocal models: jina-reranker-v3, bge-reranker-v2-m3, qwen3-reranker-0.6b, qwen3-reranker-4b")
    print("Voyage models: rerank-2.5, rerank-2.5-lite (via voyageai.Client.rerank)")
    print(f"Tests: {', '.join(TEST_FILES.keys())}")
    print()

    if VOYAGE_API_KEY is None:
        print("[WARN] VOYAGE_API_KEY not set in environment. Voyage rerankers will fail.")

    voyage_client: Optional[voyageai.Client] = None
    if VOYAGE_API_KEY:
        voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)

    results: Dict[str, Dict[str, Optional[float]]] = {}

    for model_name, config in MODELS.items():
        print("\n" + "=" * 80)
        print(f"MODEL: {model_name}")
        print("=" * 80)

        results[model_name] = {}

        for test_name, test_file in TEST_FILES.items():
            print(f"  [{test_name}] ", end="", flush=True)
            p3 = run_benchmark_for_model(model_name, config, test_name, test_file, voyage_client)
            if p3 is None:
                results[model_name][test_name] = None
                print("FAILED")
            else:
                results[model_name][test_name] = p3
                print(f"P@3 = {p3:.1f}%")

    # Summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    print("\n## Results by Test File")
    print(f"\n{'Model':<22} {'test1':<10} {'test2':<10} {'test3':<10} {'Average':<10}")
    print("-" * 70)

    for model_name in MODELS.keys():
        r = results.get(model_name, {})
        t1 = r.get("test1")
        t2 = r.get("test2")
        t3 = r.get("test3")

        scores = [s for s in [t1, t2, t3] if s is not None]
        avg = float(np.mean(scores)) if scores else None

        t1_str = f"{t1:.1f}%" if t1 is not None else "FAIL"
        t2_str = f"{t2:.1f}%" if t2 is not None else "FAIL"
        t3_str = f"{t3:.1f}%" if t3 is not None else "FAIL"
        avg_str = f"{avg:.1f}%" if avg is not None else "N/A"

        print(f"{model_name:<22} {t1_str:<10} {t2_str:<10} {t3_str:<10} {avg_str:<10}")

    # Ranking
    print("\n## Overall Ranking (by Average P@3)")
    print(f"{'Rank':<6} {'Model':<22} {'Avg P@3':<10} {'Type':<15}")
    print("-" * 60)

    ranking_rows = []
    for model_name, config in MODELS.items():
        r = results.get(model_name, {})
        scores = [s for s in r.values() if s is not None]
        avg = float(np.mean(scores)) if scores else 0.0
        ranking_rows.append((model_name, avg, config["type"]))

    ranking_rows.sort(key=lambda x: x[1], reverse=True)

    for i, (model_name, avg, mtype) in enumerate(ranking_rows, start=1):
        medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else "  "))
        print(f"{medal} {i:<4} {model_name:<22} {avg:>7.1f}%   {mtype:<15}")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
