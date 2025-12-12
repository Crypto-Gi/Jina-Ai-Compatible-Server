#!/usr/bin/env python3
"""
Late Chunking Benchmark Script
==============================
Tests retrieval performance with late_chunking=true vs late_chunking=false
for BGE-M3 and Jina V3 embeddings only (not rerankers).
"""

import json
import requests
import numpy as np
from datetime import datetime
from pathlib import Path

BASE_URL = "http://localhost:8080"

def load_test_data(test_file: str) -> dict:
    with open(test_file) as f:
        return json.load(f)

def get_embeddings(model: str, texts: list[str], late_chunking: bool = False) -> list[list[float]]:
    """Get embeddings with optional late chunking."""
    if late_chunking:
        # Late chunking requires dict format with text key
        payload = {
            "model": model,
            "input": [{"text": t} for t in texts],
            "late_chunking": True
        }
    else:
        payload = {
            "model": model,
            "input": texts,
            "late_chunking": False
        }
    
    resp = requests.post(f"{BASE_URL}/v1/embeddings", json=payload)
    resp.raise_for_status()
    data = resp.json()
    return [item["embedding"] for item in data["data"]]

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def precision_at_k(predicted: list[str], gold: list[str], k: int = 3) -> float:
    predicted_k = predicted[:k]
    gold_k = set(gold[:k])
    hits = sum(1 for p in predicted_k if p in gold_k)
    return hits / k

def mrr(predicted: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    for i, p in enumerate(predicted):
        if p in gold_set:
            return 1.0 / (i + 1)
    return 0.0

def run_retrieval_test(model: str, test_data: dict, late_chunking: bool) -> list[dict]:
    """Run retrieval tests with or without late chunking."""
    corpus = test_data["corpus"]
    corpus_texts = [s["text"] for s in corpus]
    corpus_ids = [s["id"] for s in corpus]
    
    mode = "late_chunking" if late_chunking else "standard"
    print(f"      Embedding corpus ({mode})...")
    corpus_embs = get_embeddings(model, corpus_texts, late_chunking=late_chunking)
    
    results = []
    retrieval_queries = [q for q in test_data["queries"] if q["task"] == "retrieval"]
    
    for query in retrieval_queries:
        # Query embedding (never use late chunking for single query)
        query_emb = get_embeddings(model, [query["query"]], late_chunking=False)[0]
        
        # Compute similarities
        sims = [(corpus_ids[i], cosine_similarity(query_emb, corpus_embs[i])) 
                for i in range(len(corpus_embs))]
        sims.sort(key=lambda x: x[1], reverse=True)
        
        predicted = [s[0] for s in sims]
        gold = query["gold_top3"]
        
        results.append({
            "query_id": query["id"],
            "predicted_top3": predicted[:3],
            "gold_top3": gold,
            "precision@3": precision_at_k(predicted, gold, 3),
            "mrr": mrr(predicted, gold),
        })
    
    return results

def compute_summary(results: list[dict]) -> dict:
    if not results:
        return {"precision@3": 0, "mrr": 0}
    return {
        "precision@3": round(np.mean([r["precision@3"] for r in results]), 4),
        "mrr": round(np.mean([r["mrr"] for r in results]), 4),
    }

def run_late_chunking_comparison(model: str, test_files: list[str]) -> dict:
    """Compare standard vs late chunking for a model."""
    print(f"\n{'='*60}")
    print(f"Late Chunking Test: {model}")
    print(f"{'='*60}")
    
    all_standard = []
    all_late = []
    
    for test_file in test_files:
        test_name = Path(test_file).stem
        print(f"\n  {test_name}:")
        test_data = load_test_data(test_file)
        
        # Standard encoding
        print(f"    Running standard encoding...")
        standard_results = run_retrieval_test(model, test_data, late_chunking=False)
        all_standard.extend(standard_results)
        
        # Late chunking
        print(f"    Running late chunking...")
        late_results = run_retrieval_test(model, test_data, late_chunking=True)
        all_late.extend(late_results)
    
    standard_summary = compute_summary(all_standard)
    late_summary = compute_summary(all_late)
    
    # Calculate improvement
    p3_delta = late_summary["precision@3"] - standard_summary["precision@3"]
    mrr_delta = late_summary["mrr"] - standard_summary["mrr"]
    
    print(f"\n  Results for {model}:")
    print(f"    Standard   - P@3: {standard_summary['precision@3']:.2%}, MRR: {standard_summary['mrr']:.4f}")
    print(f"    Late Chunk - P@3: {late_summary['precision@3']:.2%}, MRR: {late_summary['mrr']:.4f}")
    print(f"    Delta      - P@3: {p3_delta:+.2%}, MRR: {mrr_delta:+.4f}")
    
    return {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "standard": standard_summary,
        "late_chunking": late_summary,
        "delta": {
            "precision@3": round(p3_delta, 4),
            "mrr": round(mrr_delta, 4),
        }
    }

def append_to_report(results: list[dict], output_file: str):
    """Append late chunking results to the benchmark report."""
    
    # Read existing report
    with open(output_file) as f:
        content = f.read()
    
    # Build late chunking section
    lines = []
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Late Chunking Comparison")
    lines.append("")
    lines.append("Testing whether `late_chunking=true` improves retrieval accuracy.")
    lines.append("")
    lines.append("**Note**: Late chunking only applies to embeddings, not rerankers.")
    lines.append("Qwen3 does not support late chunking (uses last-token pooling).")
    lines.append("")
    lines.append("### Results")
    lines.append("")
    lines.append("| Model | Mode | P@3 | MRR | Δ P@3 | Δ MRR |")
    lines.append("|-------|------|-----|-----|-------|-------|")
    
    for r in results:
        # Standard row
        lines.append(f"| `{r['model']}` | Standard | {r['standard']['precision@3']:.2%} | {r['standard']['mrr']:.4f} | - | - |")
        # Late chunking row
        p3_delta = f"{r['delta']['precision@3']:+.2%}" if r['delta']['precision@3'] != 0 else "0%"
        mrr_delta = f"{r['delta']['mrr']:+.4f}" if r['delta']['mrr'] != 0 else "0"
        lines.append(f"| `{r['model']}` | Late Chunking | {r['late_chunking']['precision@3']:.2%} | {r['late_chunking']['mrr']:.4f} | {p3_delta} | {mrr_delta} |")
    
    lines.append("")
    
    # Summary
    lines.append("### Analysis")
    lines.append("")
    for r in results:
        if r['delta']['precision@3'] > 0:
            lines.append(f"- **{r['model']}**: Late chunking improved P@3 by {r['delta']['precision@3']:+.2%}")
        elif r['delta']['precision@3'] < 0:
            lines.append(f"- **{r['model']}**: Late chunking decreased P@3 by {r['delta']['precision@3']:.2%}")
        else:
            lines.append(f"- **{r['model']}**: No change with late chunking")
    
    lines.append("")
    
    content += "\n".join(lines)
    
    with open(output_file, "w") as f:
        f.write(content)
    
    print(f"\nLate chunking results appended to: {output_file}")

if __name__ == "__main__":
    test_files = [
        "/home/mir/projects/Jina-AI/tests/test1.json",
        "/home/mir/projects/Jina-AI/tests/test2.json",
        "/home/mir/projects/Jina-AI/tests/test3_ecos_release_notes.json"
    ]
    output_file = "/home/mir/projects/Jina-AI/tests/BENCHMARK_RESULTS.md"
    
    # Check which model is loaded
    resp = requests.get(f"{BASE_URL}/v1/models")
    loaded_models = [m["id"] for m in resp.json()["data"]]
    print(f"Loaded models: {loaded_models}")
    
    results = []
    
    if "bge-m3" in loaded_models:
        result = run_late_chunking_comparison("bge-m3", test_files)
        results.append(result)
    
    if "jina-embeddings-v3" in loaded_models:
        result = run_late_chunking_comparison("jina-embeddings-v3", test_files)
        results.append(result)
    
    if results:
        append_to_report(results, output_file)
    else:
        print("No supported models loaded for late chunking test (need bge-m3 or jina-embeddings-v3)")
