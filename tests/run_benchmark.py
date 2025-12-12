#!/usr/bin/env python3
"""
Model Benchmark Script
======================
Runs retrieval and rerank tests against test1.json and test2.json
Compares results across BGE, Qwen3, and Jina model families.
"""

import json
import sys
import requests
import numpy as np
from datetime import datetime
from pathlib import Path

BASE_URL = "http://localhost:8080"

def load_test_data(test_file: str) -> dict:
    """Load test JSON file."""
    with open(test_file) as f:
        return json.load(f)

def get_embeddings(model: str, texts: list[str], task: str = None) -> list[list[float]]:
    """Get embeddings from the API."""
    payload = {"model": model, "input": texts}
    if task:
        payload["task"] = task
    
    resp = requests.post(f"{BASE_URL}/v1/embeddings", json=payload)
    resp.raise_for_status()
    data = resp.json()
    return [item["embedding"] for item in data["data"]]

def rerank(model: str, query: str, documents: list[str], top_n: int = None) -> list[dict]:
    """Rerank documents using the API."""
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "return_documents": False
    }
    if top_n:
        payload["top_n"] = top_n
    
    resp = requests.post(f"{BASE_URL}/v1/rerank", json=payload)
    resp.raise_for_status()
    return resp.json()["results"]

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def precision_at_k(predicted: list[str], gold: list[str], k: int = 3) -> float:
    """Precision@K: fraction of predicted top-K that are in gold top-K."""
    predicted_k = predicted[:k]
    gold_k = set(gold[:k])
    hits = sum(1 for p in predicted_k if p in gold_k)
    return hits / k

def mrr(predicted: list[str], gold: list[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first correct answer."""
    gold_set = set(gold)
    for i, p in enumerate(predicted):
        if p in gold_set:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(predicted: list[str], gold: list[str], k: int = 3) -> float:
    """NDCG@K: Normalized Discounted Cumulative Gain."""
    def dcg(ranking: list[str], gold_set: set, k: int) -> float:
        score = 0.0
        for i, doc in enumerate(ranking[:k]):
            if doc in gold_set:
                # Relevance = 1 if in gold, 0 otherwise
                score += 1.0 / np.log2(i + 2)  # i+2 because log2(1)=0
        return score
    
    gold_set = set(gold[:k])
    actual_dcg = dcg(predicted, gold_set, k)
    ideal_dcg = dcg(gold[:k], gold_set, k)  # Perfect ranking
    
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg

def run_retrieval_test(embedding_model: str, test_data: dict, task: str = "retrieval.query") -> list[dict]:
    """Run retrieval tests using embeddings."""
    corpus = test_data["corpus"]
    corpus_texts = [s["text"] for s in corpus]
    corpus_ids = [s["id"] for s in corpus]
    
    # Get corpus embeddings (no task for passages)
    print(f"    Embedding {len(corpus_texts)} corpus texts...")
    corpus_embs = get_embeddings(embedding_model, corpus_texts)
    
    results = []
    retrieval_queries = [q for q in test_data["queries"] if q["task"] == "retrieval"]
    
    for query in retrieval_queries:
        # Get query embedding with task
        query_emb = get_embeddings(embedding_model, [query["query"]], task=task)[0]
        
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
            "ndcg@3": ndcg_at_k(predicted, gold, 3),
            "scores": {s[0]: round(s[1], 4) for s in sims[:5]}
        })
    
    return results

def run_rerank_test(reranker_model: str, test_data: dict) -> list[dict]:
    """Run rerank tests using reranker."""
    corpus = test_data["corpus"]
    corpus_texts = [s["text"] for s in corpus]
    corpus_ids = [s["id"] for s in corpus]
    
    results = []
    rerank_queries = [q for q in test_data["queries"] if q["task"] == "rerank"]
    
    for query in rerank_queries:
        # Rerank all documents
        rerank_results = rerank(reranker_model, query["query"], corpus_texts)
        
        # Map back to IDs
        predicted = [corpus_ids[r["index"]] for r in rerank_results]
        gold = query["gold_top3"]
        
        results.append({
            "query_id": query["id"],
            "predicted_top3": predicted[:3],
            "gold_top3": gold,
            "precision@3": precision_at_k(predicted, gold, 3),
            "mrr": mrr(predicted, gold),
            "ndcg@3": ndcg_at_k(predicted, gold, 3),
            "scores": {corpus_ids[r["index"]]: round(r["relevance_score"], 4) for r in rerank_results[:5]}
        })
    
    return results

def compute_summary(results: list[dict]) -> dict:
    """Compute summary statistics."""
    if not results:
        return {"precision@3": 0, "mrr": 0, "ndcg@3": 0}
    
    return {
        "precision@3": round(np.mean([r["precision@3"] for r in results]), 4),
        "mrr": round(np.mean([r["mrr"] for r in results]), 4),
        "ndcg@3": round(np.mean([r["ndcg@3"] for r in results]), 4)
    }

def run_model_family_tests(
    family_name: str,
    embedding_model: str,
    reranker_model: str,
    test_files: list[str],
    embedding_task: str = "retrieval.query"
) -> dict:
    """Run all tests for a model family."""
    print(f"\n{'='*60}")
    print(f"Testing {family_name}")
    print(f"  Embedding: {embedding_model}")
    print(f"  Reranker: {reranker_model}")
    print(f"{'='*60}")
    
    family_results = {
        "family": family_name,
        "embedding_model": embedding_model,
        "reranker_model": reranker_model,
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }
    
    all_retrieval = []
    all_rerank = []
    
    for test_file in test_files:
        test_name = Path(test_file).stem
        print(f"\n  Loading {test_name}...")
        test_data = load_test_data(test_file)
        
        # Retrieval tests
        print(f"  Running retrieval tests with {embedding_model}...")
        retrieval_results = run_retrieval_test(embedding_model, test_data, task=embedding_task)
        all_retrieval.extend(retrieval_results)
        
        # Rerank tests
        print(f"  Running rerank tests with {reranker_model}...")
        rerank_results = run_rerank_test(reranker_model, test_data)
        all_rerank.extend(rerank_results)
        
        family_results["tests"][test_name] = {
            "retrieval": {
                "results": retrieval_results,
                "summary": compute_summary(retrieval_results)
            },
            "rerank": {
                "results": rerank_results,
                "summary": compute_summary(rerank_results)
            }
        }
    
    # Overall summary
    family_results["overall"] = {
        "retrieval": compute_summary(all_retrieval),
        "rerank": compute_summary(all_rerank)
    }
    
    print(f"\n  {family_name} Overall Results:")
    print(f"    Retrieval - P@3: {family_results['overall']['retrieval']['precision@3']:.2%}, "
          f"MRR: {family_results['overall']['retrieval']['mrr']:.4f}, "
          f"NDCG@3: {family_results['overall']['retrieval']['ndcg@3']:.4f}")
    print(f"    Rerank    - P@3: {family_results['overall']['rerank']['precision@3']:.2%}, "
          f"MRR: {family_results['overall']['rerank']['mrr']:.4f}, "
          f"NDCG@3: {family_results['overall']['rerank']['ndcg@3']:.4f}")
    
    return family_results

def generate_markdown_report(all_results: list[dict], output_file: str):
    """Generate markdown report from all results."""
    
    lines = []
    lines.append("# Model Benchmark Results")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Executive Summary (placeholder - filled after all tests)
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("<!-- EXECUTIVE_SUMMARY_PLACEHOLDER -->")
    lines.append("")
    
    # Overall Comparison Table
    lines.append("## Overall Comparison")
    lines.append("")
    lines.append("### Retrieval (Embedding) Performance")
    lines.append("")
    lines.append("| Model Family | Embedding Model | P@3 | MRR | NDCG@3 |")
    lines.append("|--------------|-----------------|-----|-----|--------|")
    
    for result in all_results:
        r = result["overall"]["retrieval"]
        lines.append(f"| {result['family']} | `{result['embedding_model']}` | "
                    f"{r['precision@3']:.2%} | {r['mrr']:.4f} | {r['ndcg@3']:.4f} |")
    
    lines.append("")
    lines.append("### Reranking Performance")
    lines.append("")
    lines.append("| Model Family | Reranker Model | P@3 | MRR | NDCG@3 |")
    lines.append("|--------------|----------------|-----|-----|--------|")
    
    for result in all_results:
        r = result["overall"]["rerank"]
        lines.append(f"| {result['family']} | `{result['reranker_model']}` | "
                    f"{r['precision@3']:.2%} | {r['mrr']:.4f} | {r['ndcg@3']:.4f} |")
    
    lines.append("")
    
    # Detailed results per family
    for result in all_results:
        lines.append(f"---")
        lines.append("")
        lines.append(f"## {result['family']} Detailed Results")
        lines.append("")
        lines.append(f"- **Embedding Model**: `{result['embedding_model']}`")
        lines.append(f"- **Reranker Model**: `{result['reranker_model']}`")
        lines.append(f"- **Tested**: {result['timestamp']}")
        lines.append("")
        
        for test_name, test_results in result["tests"].items():
            lines.append(f"### {test_name}")
            lines.append("")
            
            # Retrieval
            lines.append("#### Retrieval Results")
            lines.append("")
            lines.append("| Query | Predicted Top 3 | Gold Top 3 | P@3 | MRR |")
            lines.append("|-------|-----------------|------------|-----|-----|")
            
            for r in test_results["retrieval"]["results"]:
                pred = ", ".join(r["predicted_top3"])
                gold = ", ".join(r["gold_top3"])
                lines.append(f"| {r['query_id']} | {pred} | {gold} | {r['precision@3']:.2%} | {r['mrr']:.2f} |")
            
            summary = test_results["retrieval"]["summary"]
            lines.append(f"| **Avg** | | | **{summary['precision@3']:.2%}** | **{summary['mrr']:.2f}** |")
            lines.append("")
            
            # Rerank
            lines.append("#### Rerank Results")
            lines.append("")
            lines.append("| Query | Predicted Top 3 | Gold Top 3 | P@3 | MRR |")
            lines.append("|-------|-----------------|------------|-----|-----|")
            
            for r in test_results["rerank"]["results"]:
                pred = ", ".join(r["predicted_top3"])
                gold = ", ".join(r["gold_top3"])
                lines.append(f"| {r['query_id']} | {pred} | {gold} | {r['precision@3']:.2%} | {r['mrr']:.2f} |")
            
            summary = test_results["rerank"]["summary"]
            lines.append(f"| **Avg** | | | **{summary['precision@3']:.2%}** | **{summary['mrr']:.2f}** |")
            lines.append("")
    
    # Write report
    with open(output_file, "w") as f:
        f.write("\n".join(lines))
    
    print(f"\nReport saved to: {output_file}")

def append_family_to_report(family_result: dict, output_file: str):
    """Append a single family's results to the report."""
    
    # Check if file exists
    if Path(output_file).exists():
        with open(output_file) as f:
            content = f.read()
    else:
        content = f"""# Model Benchmark Results

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

<!-- EXECUTIVE_SUMMARY_PLACEHOLDER -->

## Overall Comparison

### Retrieval (Embedding) Performance

| Model Family | Embedding Model | P@3 | MRR | NDCG@3 |
|--------------|-----------------|-----|-----|--------|

### Reranking Performance

| Model Family | Reranker Model | P@3 | MRR | NDCG@3 |
|--------------|----------------|-----|-----|--------|

"""
    
    # Add to comparison tables
    r_ret = family_result["overall"]["retrieval"]
    r_rer = family_result["overall"]["rerank"]
    
    retrieval_row = (f"| {family_result['family']} | `{family_result['embedding_model']}` | "
                   f"{r_ret['precision@3']:.2%} | {r_ret['mrr']:.4f} | {r_ret['ndcg@3']:.4f} |")
    
    rerank_row = (f"| {family_result['family']} | `{family_result['reranker_model']}` | "
                 f"{r_rer['precision@3']:.2%} | {r_rer['mrr']:.4f} | {r_rer['ndcg@3']:.4f} |")
    
    # Insert retrieval row
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("### Reranking Performance"):
            # Insert before this line
            lines.insert(i - 1, retrieval_row)
            break
    
    content = "\n".join(lines)
    
    # Insert rerank row (find the empty line after rerank table header)
    lines = content.split("\n")
    in_rerank_table = False
    for i, line in enumerate(lines):
        if "### Reranking Performance" in line:
            in_rerank_table = True
        if in_rerank_table and line.startswith("|---"):
            # Insert after the header separator
            lines.insert(i + 1, rerank_row)
            break
    
    content = "\n".join(lines)
    
    # Add detailed results section
    detail_lines = []
    detail_lines.append("")
    detail_lines.append("---")
    detail_lines.append("")
    detail_lines.append(f"## {family_result['family']} Detailed Results")
    detail_lines.append("")
    detail_lines.append(f"- **Embedding Model**: `{family_result['embedding_model']}`")
    detail_lines.append(f"- **Reranker Model**: `{family_result['reranker_model']}`")
    detail_lines.append(f"- **Tested**: {family_result['timestamp']}")
    detail_lines.append("")
    
    for test_name, test_results in family_result["tests"].items():
        detail_lines.append(f"### {test_name}")
        detail_lines.append("")
        
        # Retrieval
        detail_lines.append("#### Retrieval Results")
        detail_lines.append("")
        detail_lines.append("| Query | Predicted Top 3 | Gold Top 3 | P@3 | MRR |")
        detail_lines.append("|-------|-----------------|------------|-----|-----|")
        
        for r in test_results["retrieval"]["results"]:
            pred = ", ".join(r["predicted_top3"])
            gold = ", ".join(r["gold_top3"])
            detail_lines.append(f"| {r['query_id']} | {pred} | {gold} | {r['precision@3']:.2%} | {r['mrr']:.2f} |")
        
        summary = test_results["retrieval"]["summary"]
        detail_lines.append(f"| **Avg** | | | **{summary['precision@3']:.2%}** | **{summary['mrr']:.2f}** |")
        detail_lines.append("")
        
        # Rerank
        detail_lines.append("#### Rerank Results")
        detail_lines.append("")
        detail_lines.append("| Query | Predicted Top 3 | Gold Top 3 | P@3 | MRR |")
        detail_lines.append("|-------|-----------------|------------|-----|-----|")
        
        for r in test_results["rerank"]["results"]:
            pred = ", ".join(r["predicted_top3"])
            gold = ", ".join(r["gold_top3"])
            detail_lines.append(f"| {r['query_id']} | {pred} | {gold} | {r['precision@3']:.2%} | {r['mrr']:.2f} |")
        
        summary = test_results["rerank"]["summary"]
        detail_lines.append(f"| **Avg** | | | **{summary['precision@3']:.2%}** | **{summary['mrr']:.2f}** |")
        detail_lines.append("")
    
    content += "\n".join(detail_lines)
    
    with open(output_file, "w") as f:
        f.write(content)
    
    print(f"  Results appended to: {output_file}")

if __name__ == "__main__":
    test_files = [
        "/home/mir/projects/Jina-AI/tests/test1.json",
        "/home/mir/projects/Jina-AI/tests/test2.json",
        "/home/mir/projects/Jina-AI/tests/test3_ecos_release_notes.json"
    ]
    output_file = "/home/mir/projects/Jina-AI/tests/BENCHMARK_RESULTS.md"
    
    # Check which models are loaded
    resp = requests.get(f"{BASE_URL}/v1/models")
    loaded_models = [m["id"] for m in resp.json()["data"]]
    print(f"Loaded models: {loaded_models}")
    
    # Determine which family to test based on loaded models
    if "bge-m3" in loaded_models:
        result = run_model_family_tests(
            family_name="BGE (BAAI)",
            embedding_model="bge-m3",
            reranker_model="bge-reranker-v2-m3",
            test_files=test_files,
            embedding_task="retrieval.query"  # BGE ignores this but we pass for consistency
        )
        append_family_to_report(result, output_file)
        
    elif "qwen3-embedding-0.6b" in loaded_models:
        result = run_model_family_tests(
            family_name="Qwen3 (Alibaba)",
            embedding_model="qwen3-embedding-0.6b",
            reranker_model="qwen3-reranker-0.6b",
            test_files=test_files,
            embedding_task="retrieval.query"
        )
        append_family_to_report(result, output_file)
        
    elif "jina-embeddings-v3" in loaded_models:
        result = run_model_family_tests(
            family_name="Jina V3",
            embedding_model="jina-embeddings-v3",
            reranker_model="jina-reranker-v3",
            test_files=test_files,
            embedding_task="retrieval.query"
        )
        append_family_to_report(result, output_file)
        
    elif "jina-embeddings-v4" in loaded_models:
        result = run_model_family_tests(
            family_name="Jina V4",
            embedding_model="jina-embeddings-v4",
            reranker_model="jina-reranker-v3",  # V4 uses same reranker
            test_files=test_files,
            embedding_task="retrieval"  # V4 uses different task names
        )
        append_family_to_report(result, output_file)
    
    else:
        print(f"No recognized model family loaded. Loaded: {loaded_models}")
        sys.exit(1)
    
    print("\nDone!")
