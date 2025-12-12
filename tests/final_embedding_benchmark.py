#!/usr/bin/env python3
"""
Final Embedding Model Benchmark
==============================
Compares all embedding models across test files.

Models tested:
- Local: jina-embeddings-v4 (late chunking), qwen3-embedding-4b, qwen3-embedding-0.6b
- External: voyage-context-3, voyage-embedding-3.5, gemini-embedding-001
"""

import json
import os
import time
from typing import Optional

import numpy as np
import requests
import voyageai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys (loaded from .env file)
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

LOCAL_API_BASE = "http://localhost:8080/v1"

TEST_FILES = {
    "test1": "/home/mir/projects/Jina-AI/tests/test1.json",
    "test2": "/home/mir/projects/Jina-AI/tests/test2.json",
    "test3": "/home/mir/projects/Jina-AI/tests/test3_ecos_release_notes.json",
}

# Model configurations
MODELS = {
    # Local models
    "jina-v4-late": {
        "type": "local",
        "model_id": "jina-embeddings-v4",
        "late_chunking": True,
        "task_corpus": "retrieval.passage",
        "task_query": "retrieval.query",
    },
    "qwen3-4b": {
        "type": "local",
        "model_id": "qwen3-embedding-4b",
        "late_chunking": False,
        "task_corpus": "retrieval.passage",
        "task_query": "retrieval.query",
    },
    "qwen3-0.6b": {
        "type": "local",
        "model_id": "qwen3-embedding-0.6b",
        "late_chunking": False,
        "task_corpus": "retrieval.passage",
        "task_query": "retrieval.query",
    },
    # External APIs
    "voyage-large-2": {
        "type": "voyage",
        "model_id": "voyage-large-2",
        "contextualized": False,
    },
    "voyage-multilingual-2": {
        "type": "voyage",
        "model_id": "voyage-multilingual-2",
        "contextualized": False,
    },
    "gemini-001": {
        "type": "gemini",
        "model_id": "gemini-embedding-001",
    },
}


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_local_embeddings(texts: list[str], model_id: str, task: str, late_chunking: bool = False) -> Optional[list]:
    """Get embeddings from local server."""
    payload = {"model": model_id, "input": texts, "task": task}
    if late_chunking:
        payload["late_chunking"] = True
    
    try:
        resp = requests.post(f"{LOCAL_API_BASE}/embeddings", json=payload, timeout=120)
        if resp.status_code != 200:
            print(f"      Error {resp.status_code}: {resp.text[:100]}")
            return None
        return [d["embedding"] for d in resp.json()["data"]]
    except Exception as e:
        print(f"      Exception: {e}")
        return None


def get_voyage_embeddings(texts: list[str], model_id: str, input_type: str = "document") -> Optional[list]:
    """Get embeddings from Voyage AI API."""
    try:
        client = voyageai.Client(api_key=VOYAGE_API_KEY)
        result = client.embed(texts, model=model_id, input_type=input_type)
        return result.embeddings
    except Exception as e:
        print(f"      Voyage error: {e}")
        return None


def get_voyage_contextualized_embeddings(texts: list[str], model_id: str) -> Optional[list]:
    """Get contextualized embeddings from Voyage AI API (all texts as one document)."""
    try:
        client = voyageai.Client(api_key=VOYAGE_API_KEY)
        # For contextualized embeddings, wrap all texts as chunks of one document
        result = client.contextualized_embed(inputs=[texts], model=model_id, input_type="document")
        return result.results[0].embeddings
    except Exception as e:
        print(f"      Voyage contextualized error: {e}")
        return None


def get_gemini_embeddings(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> Optional[list]:
    """Get embeddings from Gemini API.
    
    Task types:
    - RETRIEVAL_QUERY: For search queries
    - RETRIEVAL_DOCUMENT: For documents to be searched
    - SEMANTIC_SIMILARITY: For comparing text similarity
    - CLASSIFICATION: For text classification
    - CLUSTERING: For clustering texts
    """
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={GEMINI_API_KEY}"
        
        embeddings = []
        for text in texts:
            payload = {
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": text}]},
                "taskType": task_type,
            }
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code != 200:
                print(f"      Gemini error {resp.status_code}: {resp.text[:100]}")
                return None
            data = resp.json()
            embeddings.append(data["embedding"]["values"])
            time.sleep(0.1)  # Rate limiting
        
        return embeddings
    except Exception as e:
        print(f"      Gemini exception: {e}")
        return None


def run_benchmark(test_name: str, test_file: str, model_name: str, config: dict) -> Optional[float]:
    """Run benchmark for a specific model and test file."""
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    corpus = data["corpus"]
    queries = [q for q in data["queries"] if q["task"] == "retrieval"]
    
    corpus_texts = [item["text"] for item in corpus]
    corpus_ids = [item["id"] for item in corpus]
    
    model_type = config["type"]
    
    # Get corpus embeddings
    if model_type == "local":
        corpus_embs = get_local_embeddings(
            corpus_texts, 
            config["model_id"], 
            config["task_corpus"],
            config.get("late_chunking", False)
        )
    elif model_type == "voyage":
        if config.get("contextualized"):
            corpus_embs = get_voyage_contextualized_embeddings(corpus_texts, config["model_id"])
        else:
            corpus_embs = get_voyage_embeddings(corpus_texts, config["model_id"], "document")
    elif model_type == "gemini":
        corpus_embs = get_gemini_embeddings(corpus_texts, "RETRIEVAL_DOCUMENT")
    else:
        print(f"      Unknown model type: {model_type}")
        return None
    
    if corpus_embs is None:
        return None
    
    # Calculate P@3 for each query
    p3_scores = []
    
    for query in queries:
        # Get query embedding
        if model_type == "local":
            query_embs = get_local_embeddings(
                [query["query"]], 
                config["model_id"], 
                config["task_query"],
                False  # No late chunking for queries
            )
        elif model_type == "voyage":
            query_embs = get_voyage_embeddings([query["query"]], config["model_id"], "query")
        elif model_type == "gemini":
            query_embs = get_gemini_embeddings([query["query"]], "RETRIEVAL_QUERY")
        
        if query_embs is None:
            continue
        
        query_emb = query_embs[0]
        
        # Get top 3
        sims = [(corpus_ids[i], cosine_similarity(query_emb, emb)) 
                for i, emb in enumerate(corpus_embs)]
        sims.sort(key=lambda x: x[1], reverse=True)
        top3 = [s[0] for s in sims[:3]]
        
        gold = query["gold_top3"]
        p3 = len(set(top3) & set(gold)) / 3
        p3_scores.append(p3)
    
    return np.mean(p3_scores) * 100 if p3_scores else 0


def main():
    print("=" * 80)
    print("FINAL EMBEDDING MODEL BENCHMARK")
    print("=" * 80)
    print(f"\nModels: {', '.join(MODELS.keys())}")
    print(f"Tests: {', '.join(TEST_FILES.keys())}")
    print()
    
    results = {}
    
    for model_name, config in MODELS.items():
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print(f"{'='*80}")
        
        results[model_name] = {}
        
        for test_name, test_file in TEST_FILES.items():
            print(f"  [{test_name}] ", end="", flush=True)
            p3 = run_benchmark(test_name, test_file, model_name, config)
            
            if p3 is not None:
                results[model_name][test_name] = p3
                print(f"P@3 = {p3:.1f}%")
            else:
                results[model_name][test_name] = None
                print("FAILED")
    
    # Summary
    print("\n")
    print("=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    print("\n## Results by Test File")
    print(f"\n{'Model':<20} {'test1':<12} {'test2':<12} {'test3':<12} {'Average':<12}")
    print("-" * 70)
    
    for model_name in MODELS.keys():
        r = results.get(model_name, {})
        t1 = r.get("test1")
        t2 = r.get("test2")
        t3 = r.get("test3")
        
        scores = [s for s in [t1, t2, t3] if s is not None]
        avg = np.mean(scores) if scores else None
        
        t1_str = f"{t1:.1f}%" if t1 is not None else "FAIL"
        t2_str = f"{t2:.1f}%" if t2 is not None else "FAIL"
        t3_str = f"{t3:.1f}%" if t3 is not None else "FAIL"
        avg_str = f"{avg:.1f}%" if avg is not None else "N/A"
        
        print(f"{model_name:<20} {t1_str:<12} {t2_str:<12} {t3_str:<12} {avg_str:<12}")
    
    # Ranking
    print("\n## Overall Ranking (by Average P@3)")
    print(f"{'Rank':<6} {'Model':<20} {'Avg P@3':<12} {'Type':<15}")
    print("-" * 55)
    
    rankings = []
    for model_name, config in MODELS.items():
        r = results.get(model_name, {})
        scores = [s for s in r.values() if s is not None]
        avg = np.mean(scores) if scores else 0
        model_type = config["type"]
        rankings.append((model_name, avg, model_type))
    
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model_name, avg, model_type) in enumerate(rankings, 1):
        medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else "  "))
        print(f"{medal} {i:<4} {model_name:<20} {avg:.1f}%        {model_type:<15}")
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    main()
