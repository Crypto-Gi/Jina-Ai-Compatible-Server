#!/usr/bin/env python3
"""
Voyage AI Benchmark V2 - Corrected Implementation

Key fixes from V1:
1. For contextual embeddings, queries should ALSO use contextualized_embed with input_type="query"
2. Per Voyage docs: queries should be [[query]] (single element inner list)

Reference: https://docs.voyageai.com/docs/contextualized-chunk-embeddings
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
STANDARD_MODEL = "voyage-3.5"
CONTEXT_MODEL = "voyage-context-3"
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


def get_contextual_query_embedding(query: str) -> List[float]:
    """
    Get contextual embedding for a QUERY.
    Per Voyage docs: queries should use contextualized_embed with [[query]]
    """
    result = client.contextualized_embed(
        inputs=[[query]],  # Single query in its own list
        model=CONTEXT_MODEL,
        input_type="query",
        output_dimension=OUTPUT_DIMENSION
    )
    return result.results[0].embeddings[0]


def get_contextual_embeddings_single_doc(chunks: List[str]) -> List[List[float]]:
    """
    Get contextual embeddings for chunks from a SINGLE document.
    All chunks are passed together so they share context.
    """
    result = client.contextualized_embed(
        inputs=[chunks],  # All chunks from one document
        model=CONTEXT_MODEL,
        input_type="document",
        output_dimension=OUTPUT_DIMENSION
    )
    return result.results[0].embeddings


def get_contextual_embeddings_multi_doc(corpus_texts: List[str]) -> List[List[float]]:
    """
    Get contextual embeddings treating each text as a separate document.
    Each text is its own document (no shared context).
    """
    inputs = [[text] for text in corpus_texts]
    result = client.contextualized_embed(
        inputs=inputs,
        model=CONTEXT_MODEL,
        input_type="document",
        output_dimension=OUTPUT_DIMENSION
    )
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
    print("    Getting standard corpus embeddings (voyage-3.5)...")
    standard_corpus_embs = get_standard_embeddings(corpus_texts, input_type="document")
    
    # Contextual embeddings - all chunks as ONE document (correct usage)
    print("    Getting contextual corpus embeddings (voyage-context-3, single doc)...")
    contextual_corpus_embs = get_contextual_embeddings_single_doc(corpus_texts)
    
    results = {
        "standard": {"p3": [], "mrr": []},
        "contextual": {"p3": [], "mrr": []}
    }
    
    for query in queries:
        query_text = query["query"]
        gold = query["gold_top3"]
        
        # Standard: query with standard embedding
        query_emb_std = get_standard_embeddings([query_text], input_type="query")[0]
        pred_std = run_retrieval(query_emb_std, standard_corpus_embs, corpus_ids)
        results["standard"]["p3"].append(precision_at_k(pred_std, gold))
        results["standard"]["mrr"].append(mrr(pred_std, gold))
        
        # Contextual: query with CONTEXTUAL embedding (this is the fix!)
        query_emb_ctx = get_contextual_query_embedding(query_text)
        pred_ctx = run_retrieval(query_emb_ctx, contextual_corpus_embs, corpus_ids)
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
    Each chunk is from a DIFFERENT document.
    """
    print(f"\n  Loading {test_file}...")
    with open(test_file) as f:
        data = json.load(f)
    
    corpus = data["corpus"]
    queries = [q for q in data["queries"] if q["task"] == "retrieval"]
    
    corpus_texts = [item["text"] for item in corpus]
    corpus_ids = [item["id"] for item in corpus]
    
    # Standard embeddings (voyage-3.5)
    print("    Getting standard corpus embeddings (voyage-3.5)...")
    standard_corpus_embs = get_standard_embeddings(corpus_texts, input_type="document")
    
    # Contextual embeddings - each text as separate document (correct for multi-doc)
    print("    Getting contextual corpus embeddings (voyage-context-3, per-doc)...")
    contextual_corpus_embs = get_contextual_embeddings_multi_doc(corpus_texts)
    
    # Also test INCORRECT usage: all texts as one document
    print("    Getting contextual corpus embeddings (voyage-context-3, WRONG: all as one)...")
    contextual_wrong_embs = get_contextual_embeddings_single_doc(corpus_texts)
    
    results = {
        "standard": {"p3": [], "mrr": []},
        "contextual_correct": {"p3": [], "mrr": []},
        "contextual_wrong": {"p3": [], "mrr": []}
    }
    
    for query in queries:
        query_text = query["query"]
        gold = query["gold_top3"]
        
        # Standard: query with standard embedding
        query_emb_std = get_standard_embeddings([query_text], input_type="query")[0]
        pred_std = run_retrieval(query_emb_std, standard_corpus_embs, corpus_ids)
        results["standard"]["p3"].append(precision_at_k(pred_std, gold))
        results["standard"]["mrr"].append(mrr(pred_std, gold))
        
        # Contextual correct: query with CONTEXTUAL embedding
        query_emb_ctx = get_contextual_query_embedding(query_text)
        pred_ctx = run_retrieval(query_emb_ctx, contextual_corpus_embs, corpus_ids)
        results["contextual_correct"]["p3"].append(precision_at_k(pred_ctx, gold))
        results["contextual_correct"]["mrr"].append(mrr(pred_ctx, gold))
        
        # Contextual wrong: query with CONTEXTUAL embedding, corpus all-as-one
        pred_wrong = run_retrieval(query_emb_ctx, contextual_wrong_embs, corpus_ids)
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


if __name__ == "__main__":
    test_files = {
        "test1": "/home/mir/projects/Jina-AI/tests/test1.json",
        "test2": "/home/mir/projects/Jina-AI/tests/test2.json",
        "test3": "/home/mir/projects/Jina-AI/tests/test3_ecos_release_notes.json"
    }
    
    print("=" * 60)
    print("Voyage AI Benchmark V2 (Corrected)")
    print("=" * 60)
    print(f"Standard Model: {STANDARD_MODEL}")
    print(f"Contextual Model: {CONTEXT_MODEL}")
    print(f"Output Dimension: {OUTPUT_DIMENSION}")
    print("\nKey fix: Using contextualized_embed for QUERIES too!")
    
    results = {}
    
    # Test3 - single document corpus
    print("\n" + "=" * 60)
    print("Testing test3 (single document - contextual appropriate)")
    print("=" * 60)
    results["test3"] = test_single_document_corpus(test_files["test3"])
    print(f"\n  Results:")
    print(f"    Standard P@3: {results['test3']['standard_p3']*100:.1f}%")
    print(f"    Contextual P@3: {results['test3']['contextual_p3']*100:.1f}%")
    print(f"    Delta: {(results['test3']['contextual_p3'] - results['test3']['standard_p3'])*100:+.1f}%")
    
    # Test1 - multi document corpus
    print("\n" + "=" * 60)
    print("Testing test1 (multi-document)")
    print("=" * 60)
    results["test1"] = test_multi_document_corpus(test_files["test1"])
    print(f"\n  Results:")
    print(f"    Standard P@3: {results['test1']['standard_p3']*100:.1f}%")
    print(f"    Contextual (per-doc) P@3: {results['test1']['contextual_correct_p3']*100:.1f}%")
    print(f"    Contextual (all-as-one, WRONG) P@3: {results['test1']['contextual_wrong_p3']*100:.1f}%")
    
    # Test2 - multi document corpus
    print("\n" + "=" * 60)
    print("Testing test2 (multi-document)")
    print("=" * 60)
    results["test2"] = test_multi_document_corpus(test_files["test2"])
    print(f"\n  Results:")
    print(f"    Standard P@3: {results['test2']['standard_p3']*100:.1f}%")
    print(f"    Contextual (per-doc) P@3: {results['test2']['contextual_correct_p3']*100:.1f}%")
    print(f"    Contextual (all-as-one, WRONG) P@3: {results['test2']['contextual_wrong_p3']*100:.1f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\n| Test | Standard P@3 | Contextual P@3 | Delta |")
    print("|------|--------------|----------------|-------|")
    print(f"| test3 (single-doc) | {results['test3']['standard_p3']*100:.1f}% | {results['test3']['contextual_p3']*100:.1f}% | {(results['test3']['contextual_p3'] - results['test3']['standard_p3'])*100:+.1f}% |")
    print(f"| test1 (multi-doc) | {results['test1']['standard_p3']*100:.1f}% | {results['test1']['contextual_correct_p3']*100:.1f}% | {(results['test1']['contextual_correct_p3'] - results['test1']['standard_p3'])*100:+.1f}% |")
    print(f"| test2 (multi-doc) | {results['test2']['standard_p3']*100:.1f}% | {results['test2']['contextual_correct_p3']*100:.1f}% | {(results['test2']['contextual_correct_p3'] - results['test2']['standard_p3'])*100:+.1f}% |")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
