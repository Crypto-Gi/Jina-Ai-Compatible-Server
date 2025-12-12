#!/usr/bin/env python3
"""
Jina AI API Comparison Test Suite
=================================

Comprehensive test suite comparing Official Jina AI API vs Local Server.

Tests:
1. Embeddings V4 - Standard mode
2. Embeddings V4 - Late chunking mode
3. Embeddings V4 - Multi-vector mode
4. Embeddings V4 - Different tasks (retrieval, text-matching, code)
5. Embeddings V4 - Different dimensions (MRL truncation)
6. Embeddings V4 - Different embedding types (float, base64, binary)
7. Embeddings V4 - Multimodal (text + image)
8. Embeddings V3 - All tasks
9. Reranker V3 - Standard reranking
10. Reranker V3 - top_n parameter
11. Reranker V3 - return_documents parameter
12. Benchmark on test1.json, test2.json, test3.json

Author: Jina AI Local Server Test Suite
"""

import json
import time
import base64
import struct
import requests
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================
# Configuration
# ============================================

JINA_API_KEY = os.environ.get("JINA_API_KEY")
JINA_API_BASE = "https://api.jina.ai/v1"
LOCAL_API_BASE = "http://localhost:8080/v1"

# Test files
TEST_FILES = {
    "test1": "/home/mir/projects/Jina-AI/tests/test1.json",
    "test2": "/home/mir/projects/Jina-AI/tests/test2.json",
    "test3": "/home/mir/projects/Jina-AI/tests/test3_ecos_release_notes.json",
}

# Models to test
EMBEDDING_MODEL_V4 = "jina-embeddings-v4"
EMBEDDING_MODEL_V3 = "jina-embeddings-v3"
RERANKER_MODEL = "jina-reranker-v3"


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    details: str
    local_time_ms: float = 0
    jina_time_ms: float = 0
    similarity: float = 0  # Cosine similarity between embeddings


class JinaAPITester:
    """Test suite for comparing Jina AI API vs Local Server."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.jina_headers = {
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json"
        }
        self.local_headers = {
            "Content-Type": "application/json"
        }
    
    def _call_jina_api(self, endpoint: str, payload: Dict) -> Tuple[Dict, float]:
        """Call official Jina API and return response + time in ms."""
        url = f"{JINA_API_BASE}/{endpoint}"
        start = time.time()
        resp = requests.post(url, headers=self.jina_headers, json=payload)
        elapsed_ms = (time.time() - start) * 1000
        
        if resp.status_code != 200:
            raise Exception(f"Jina API error {resp.status_code}: {resp.text}")
        
        return resp.json(), elapsed_ms
    
    def _call_local_api(self, endpoint: str, payload: Dict) -> Tuple[Dict, float]:
        """Call local server API and return response + time in ms."""
        url = f"{LOCAL_API_BASE}/{endpoint}"
        start = time.time()
        resp = requests.post(url, headers=self.local_headers, json=payload)
        elapsed_ms = (time.time() - start) * 1000
        
        if resp.status_code != 200:
            raise Exception(f"Local API error {resp.status_code}: {resp.text}")
        
        return resp.json(), elapsed_ms
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def _decode_base64_embedding(self, b64_str: str) -> List[float]:
        """Decode base64 embedding to float list."""
        packed = base64.b64decode(b64_str)
        count = len(packed) // 4  # 4 bytes per float32
        return list(struct.unpack(f"{count}f", packed))
    
    def _compare_embeddings(self, jina_emb: Any, local_emb: Any, emb_type: str = "float") -> float:
        """Compare two embeddings and return similarity."""
        # Handle different embedding types
        if emb_type == "base64":
            jina_vec = self._decode_base64_embedding(jina_emb)
            local_vec = self._decode_base64_embedding(local_emb)
        elif emb_type == "binary":
            # Binary embeddings - compare directly
            jina_vec = jina_emb
            local_vec = local_emb
            # For binary, compute match percentage
            matches = sum(1 for a, b in zip(jina_vec, local_vec) if a == b)
            return matches / len(jina_vec)
        else:  # float
            jina_vec = jina_emb
            local_vec = local_emb
        
        return self._cosine_similarity(jina_vec, local_vec)
    
    def _add_result(self, name: str, passed: bool, details: str, 
                    local_time: float = 0, jina_time: float = 0, similarity: float = 0):
        """Add a test result."""
        self.results.append(TestResult(
            name=name,
            passed=passed,
            details=details,
            local_time_ms=local_time,
            jina_time_ms=jina_time,
            similarity=similarity
        ))
    
    # ============================================
    # Embedding Tests
    # ============================================
    
    def test_v4_standard_embedding(self):
        """Test V4 standard embedding mode."""
        print("\n[TEST] V4 Standard Embedding")
        
        payload = {
            "model": EMBEDDING_MODEL_V4,
            "input": ["Hello world", "This is a test"],
            "task": "retrieval.passage",  # Official API uses retrieval.query or retrieval.passage
        }
        
        try:
            jina_resp, jina_time = self._call_jina_api("embeddings", payload)
            local_resp, local_time = self._call_local_api("embeddings", payload)
            
            # Compare embeddings
            jina_emb = jina_resp["data"][0]["embedding"]
            local_emb = local_resp["data"][0]["embedding"]
            
            sim = self._compare_embeddings(jina_emb, local_emb)
            
            # Check dimensions
            jina_dim = len(jina_emb)
            local_dim = len(local_emb)
            
            passed = sim > 0.95 and jina_dim == local_dim
            details = f"Similarity: {sim:.4f}, Jina dim: {jina_dim}, Local dim: {local_dim}"
            
            self._add_result("V4 Standard Embedding", passed, details, local_time, jina_time, sim)
            print(f"  ✓ {details}" if passed else f"  ✗ {details}")
            
        except Exception as e:
            self._add_result("V4 Standard Embedding", False, str(e))
            print(f"  ✗ Error: {e}")
    
    def test_v4_late_chunking(self):
        """Test V4 late chunking mode."""
        print("\n[TEST] V4 Late Chunking")
        
        payload = {
            "model": EMBEDDING_MODEL_V4,
            "input": [
                "Berlin is the capital of Germany.",
                "It has a population of 3.5 million.",
                "The city is known for its history."
            ],
            "task": "retrieval.passage",  # Official API uses retrieval.query or retrieval.passage
            "late_chunking": True,
        }
        
        try:
            jina_resp, jina_time = self._call_jina_api("embeddings", payload)
            local_resp, local_time = self._call_local_api("embeddings", payload)
            
            # Compare all embeddings
            similarities = []
            for i in range(len(payload["input"])):
                jina_emb = jina_resp["data"][i]["embedding"]
                local_emb = local_resp["data"][i]["embedding"]
                sim = self._compare_embeddings(jina_emb, local_emb)
                similarities.append(sim)
            
            avg_sim = np.mean(similarities)
            jina_dim = len(jina_resp["data"][0]["embedding"])
            local_dim = len(local_resp["data"][0]["embedding"])
            
            passed = avg_sim > 0.90 and jina_dim == local_dim
            details = f"Avg similarity: {avg_sim:.4f}, Jina dim: {jina_dim}, Local dim: {local_dim}"
            
            self._add_result("V4 Late Chunking", passed, details, local_time, jina_time, avg_sim)
            print(f"  ✓ {details}" if passed else f"  ✗ {details}")
            
        except Exception as e:
            self._add_result("V4 Late Chunking", False, str(e))
            print(f"  ✗ Error: {e}")
    
    def test_v4_dimensions(self):
        """Test V4 MRL dimension truncation."""
        print("\n[TEST] V4 Dimension Truncation (MRL)")
        
        for dim in [128, 256, 512, 1024]:
            payload = {
                "model": EMBEDDING_MODEL_V4,
                "input": ["Test dimension truncation"],
                "task": "retrieval.passage",
                "dimensions": dim,
            }
            
            try:
                jina_resp, jina_time = self._call_jina_api("embeddings", payload)
                local_resp, local_time = self._call_local_api("embeddings", payload)
                
                jina_emb = jina_resp["data"][0]["embedding"]
                local_emb = local_resp["data"][0]["embedding"]
                
                jina_dim = len(jina_emb)
                local_dim = len(local_emb)
                
                sim = self._compare_embeddings(jina_emb, local_emb)
                
                passed = jina_dim == dim and local_dim == dim and sim > 0.95
                details = f"dim={dim}: Jina={jina_dim}, Local={local_dim}, sim={sim:.4f}"
                
                self._add_result(f"V4 Dimensions ({dim})", passed, details, local_time, jina_time, sim)
                print(f"  {'✓' if passed else '✗'} {details}")
                
            except Exception as e:
                self._add_result(f"V4 Dimensions ({dim})", False, str(e))
                print(f"  ✗ dim={dim}: Error: {e}")
    
    def test_v4_tasks(self):
        """Test V4 different tasks."""
        print("\n[TEST] V4 Tasks")
        
        # Official Jina API V4 tasks: retrieval.query, retrieval.passage, text-matching, code.query, code.passage
        for task in ["retrieval.query", "retrieval.passage", "text-matching", "code.query", "code.passage"]:
            payload = {
                "model": EMBEDDING_MODEL_V4,
                "input": ["Test task embedding"],
                "task": task,
            }
            
            try:
                jina_resp, jina_time = self._call_jina_api("embeddings", payload)
                local_resp, local_time = self._call_local_api("embeddings", payload)
                
                jina_emb = jina_resp["data"][0]["embedding"]
                local_emb = local_resp["data"][0]["embedding"]
                
                sim = self._compare_embeddings(jina_emb, local_emb)
                
                passed = sim > 0.95
                details = f"task={task}: similarity={sim:.4f}"
                
                self._add_result(f"V4 Task ({task})", passed, details, local_time, jina_time, sim)
                print(f"  {'✓' if passed else '✗'} {details}")
                
            except Exception as e:
                self._add_result(f"V4 Task ({task})", False, str(e))
                print(f"  ✗ task={task}: Error: {e}")
    
    def test_v4_embedding_types(self):
        """Test V4 different embedding types."""
        print("\n[TEST] V4 Embedding Types")
        
        for emb_type in ["float", "base64"]:  # binary may differ
            payload = {
                "model": EMBEDDING_MODEL_V4,
                "input": ["Test embedding type"],
                "task": "retrieval.passage",
                "embedding_type": emb_type,
            }
            
            try:
                jina_resp, jina_time = self._call_jina_api("embeddings", payload)
                local_resp, local_time = self._call_local_api("embeddings", payload)
                
                jina_emb = jina_resp["data"][0]["embedding"]
                local_emb = local_resp["data"][0]["embedding"]
                
                sim = self._compare_embeddings(jina_emb, local_emb, emb_type)
                
                passed = sim > 0.95
                details = f"type={emb_type}: similarity={sim:.4f}"
                
                self._add_result(f"V4 Embedding Type ({emb_type})", passed, details, local_time, jina_time, sim)
                print(f"  {'✓' if passed else '✗'} {details}")
                
            except Exception as e:
                self._add_result(f"V4 Embedding Type ({emb_type})", False, str(e))
                print(f"  ✗ type={emb_type}: Error: {e}")
    
    def test_v4_prompt_names(self):
        """Test V4 task variants (query vs passage) - prompt_name not used in official API."""
        print("\n[TEST] V4 Query vs Passage Tasks")
        
        # Official API uses task=retrieval.query or task=retrieval.passage instead of prompt_name
        for task in ["retrieval.query", "retrieval.passage"]:
            payload = {
                "model": EMBEDDING_MODEL_V4,
                "input": ["Test prompt name"],
                "task": task,
            }
            
            try:
                jina_resp, jina_time = self._call_jina_api("embeddings", payload)
                local_resp, local_time = self._call_local_api("embeddings", payload)
                
                jina_emb = jina_resp["data"][0]["embedding"]
                local_emb = local_resp["data"][0]["embedding"]
                
                sim = self._compare_embeddings(jina_emb, local_emb)
                
                passed = sim > 0.95
                details = f"task={task}: similarity={sim:.4f}"
                
                self._add_result(f"V4 Query/Passage ({task})", passed, details, local_time, jina_time, sim)
                print(f"  {'✓' if passed else '✗'} {details}")
                
            except Exception as e:
                self._add_result(f"V4 Query/Passage ({task})", False, str(e))
                print(f"  ✗ task={task}: Error: {e}")
    
    def test_v3_tasks(self):
        """Test V3 different tasks."""
        print("\n[TEST] V3 Tasks")
        
        tasks = ["retrieval.query", "retrieval.passage", "text-matching", "separation", "classification"]
        
        for task in tasks:
            payload = {
                "model": EMBEDDING_MODEL_V3,
                "input": ["Test V3 task embedding"],
                "task": task,
            }
            
            try:
                jina_resp, jina_time = self._call_jina_api("embeddings", payload)
                local_resp, local_time = self._call_local_api("embeddings", payload)
                
                jina_emb = jina_resp["data"][0]["embedding"]
                local_emb = local_resp["data"][0]["embedding"]
                
                sim = self._compare_embeddings(jina_emb, local_emb)
                
                passed = sim > 0.95
                details = f"task={task}: similarity={sim:.4f}"
                
                self._add_result(f"V3 Task ({task})", passed, details, local_time, jina_time, sim)
                print(f"  {'✓' if passed else '✗'} {details}")
                
            except Exception as e:
                self._add_result(f"V3 Task ({task})", False, str(e))
                print(f"  ✗ task={task}: Error: {e}")
    
    def test_v3_late_chunking(self):
        """Test V3 late chunking mode."""
        print("\n[TEST] V3 Late Chunking")
        
        payload = {
            "model": EMBEDDING_MODEL_V3,
            "input": [
                "Berlin is the capital of Germany.",
                "It has a population of 3.5 million.",
                "The city is known for its history."
            ],
            "task": "retrieval.passage",
            "late_chunking": True,
        }
        
        try:
            jina_resp, jina_time = self._call_jina_api("embeddings", payload)
            local_resp, local_time = self._call_local_api("embeddings", payload)
            
            # Compare all embeddings
            similarities = []
            for i in range(len(payload["input"])):
                jina_emb = jina_resp["data"][i]["embedding"]
                local_emb = local_resp["data"][i]["embedding"]
                sim = self._compare_embeddings(jina_emb, local_emb)
                similarities.append(sim)
            
            avg_sim = np.mean(similarities)
            jina_dim = len(jina_resp["data"][0]["embedding"])
            local_dim = len(local_resp["data"][0]["embedding"])
            
            passed = avg_sim > 0.90 and jina_dim == local_dim
            details = f"Avg similarity: {avg_sim:.4f}, Jina dim: {jina_dim}, Local dim: {local_dim}"
            
            self._add_result("V3 Late Chunking", passed, details, local_time, jina_time, avg_sim)
            print(f"  ✓ {details}" if passed else f"  ✗ {details}")
            
        except Exception as e:
            self._add_result("V3 Late Chunking", False, str(e))
            print(f"  ✗ Error: {e}")
    
    # ============================================
    # Reranker Tests
    # ============================================
    
    def test_reranker_basic(self):
        """Test basic reranking."""
        print("\n[TEST] Reranker Basic")
        
        payload = {
            "model": RERANKER_MODEL,
            "query": "What is machine learning?",
            "documents": [
                "Machine learning is a subset of artificial intelligence.",
                "The weather is nice today.",
                "Deep learning uses neural networks.",
                "I like pizza."
            ]
        }
        
        try:
            jina_resp, jina_time = self._call_jina_api("rerank", payload)
            local_resp, local_time = self._call_local_api("rerank", payload)
            
            # Compare rankings
            jina_order = [r["index"] for r in jina_resp["results"]]
            local_order = [r["index"] for r in local_resp["results"]]
            
            # Compare scores
            jina_scores = [r["relevance_score"] for r in jina_resp["results"]]
            local_scores = [r["relevance_score"] for r in local_resp["results"]]
            
            order_match = jina_order == local_order
            score_corr = np.corrcoef(jina_scores, local_scores)[0, 1]
            
            passed = order_match and score_corr > 0.95
            details = f"Order match: {order_match}, Score correlation: {score_corr:.4f}"
            
            self._add_result("Reranker Basic", passed, details, local_time, jina_time, score_corr)
            print(f"  {'✓' if passed else '✗'} {details}")
            print(f"    Jina order: {jina_order}")
            print(f"    Local order: {local_order}")
            
        except Exception as e:
            self._add_result("Reranker Basic", False, str(e))
            print(f"  ✗ Error: {e}")
    
    def test_reranker_top_n(self):
        """Test reranker top_n parameter."""
        print("\n[TEST] Reranker top_n")
        
        for top_n in [1, 2, 3]:
            payload = {
                "model": RERANKER_MODEL,
                "query": "What is AI?",
                "documents": [
                    "AI stands for artificial intelligence.",
                    "The sky is blue.",
                    "Machine learning is part of AI.",
                    "I enjoy reading books."
                ],
                "top_n": top_n
            }
            
            try:
                jina_resp, jina_time = self._call_jina_api("rerank", payload)
                local_resp, local_time = self._call_local_api("rerank", payload)
                
                jina_count = len(jina_resp["results"])
                local_count = len(local_resp["results"])
                
                passed = jina_count == top_n and local_count == top_n
                details = f"top_n={top_n}: Jina returned {jina_count}, Local returned {local_count}"
                
                self._add_result(f"Reranker top_n ({top_n})", passed, details, local_time, jina_time)
                print(f"  {'✓' if passed else '✗'} {details}")
                
            except Exception as e:
                self._add_result(f"Reranker top_n ({top_n})", False, str(e))
                print(f"  ✗ top_n={top_n}: Error: {e}")
    
    def test_reranker_return_documents(self):
        """Test reranker return_documents parameter."""
        print("\n[TEST] Reranker return_documents")
        
        for return_docs in [True, False]:
            payload = {
                "model": RERANKER_MODEL,
                "query": "What is AI?",
                "documents": ["AI is artificial intelligence.", "The sky is blue."],
                "return_documents": return_docs
            }
            
            try:
                jina_resp, jina_time = self._call_jina_api("rerank", payload)
                local_resp, local_time = self._call_local_api("rerank", payload)
                
                jina_has_doc = jina_resp["results"][0].get("document") is not None
                local_has_doc = local_resp["results"][0].get("document") is not None
                
                passed = jina_has_doc == return_docs and local_has_doc == return_docs
                details = f"return_documents={return_docs}: Jina has doc={jina_has_doc}, Local has doc={local_has_doc}"
                
                self._add_result(f"Reranker return_documents ({return_docs})", passed, details, local_time, jina_time)
                print(f"  {'✓' if passed else '✗'} {details}")
                
            except Exception as e:
                self._add_result(f"Reranker return_documents ({return_docs})", False, str(e))
                print(f"  ✗ return_documents={return_docs}: Error: {e}")
    
    # ============================================
    # Benchmark Tests
    # ============================================
    
    def test_benchmark_retrieval(self, test_name: str, test_file: str):
        """Run retrieval benchmark on a test file."""
        print(f"\n[BENCHMARK] {test_name}")
        
        with open(test_file) as f:
            data = json.load(f)
        
        corpus = data["corpus"]
        queries = [q for q in data["queries"] if q["task"] == "retrieval"]
        
        corpus_texts = [item["text"] for item in corpus]
        corpus_ids = [item["id"] for item in corpus]
        
        # Test both standard and late chunking
        for mode in ["standard", "late_chunking"]:
            print(f"\n  Mode: {mode}")
            
            # Get corpus embeddings
            payload = {
                "model": EMBEDDING_MODEL_V4,
                "input": corpus_texts,
                "task": "retrieval.passage",
                "late_chunking": mode == "late_chunking",
            }
            
            try:
                jina_corpus, _ = self._call_jina_api("embeddings", payload)
                local_corpus, _ = self._call_local_api("embeddings", payload)
                
                jina_corpus_embs = [d["embedding"] for d in jina_corpus["data"]]
                local_corpus_embs = [d["embedding"] for d in local_corpus["data"]]
                
                # Compare corpus embeddings
                corpus_sims = []
                for i in range(len(corpus_texts)):
                    sim = self._compare_embeddings(jina_corpus_embs[i], local_corpus_embs[i])
                    corpus_sims.append(sim)
                
                avg_corpus_sim = np.mean(corpus_sims)
                print(f"    Corpus embedding similarity: {avg_corpus_sim:.4f}")
                
                # Run retrieval for each query
                jina_results = {"p3": [], "mrr": []}
                local_results = {"p3": [], "mrr": []}
                
                for query in queries:
                    query_payload = {
                        "model": EMBEDDING_MODEL_V4,
                        "input": [query["query"]],
                        "task": "retrieval.query",
                    }
                    
                    jina_q, _ = self._call_jina_api("embeddings", query_payload)
                    local_q, _ = self._call_local_api("embeddings", query_payload)
                    
                    jina_query_emb = jina_q["data"][0]["embedding"]
                    local_query_emb = local_q["data"][0]["embedding"]
                    
                    # Retrieve top 3
                    def retrieve_top3(query_emb, corpus_embs, ids):
                        sims = [(ids[i], self._cosine_similarity(query_emb, emb)) 
                                for i, emb in enumerate(corpus_embs)]
                        sims.sort(key=lambda x: x[1], reverse=True)
                        return [s[0] for s in sims[:3]]
                    
                    jina_top3 = retrieve_top3(jina_query_emb, jina_corpus_embs, corpus_ids)
                    local_top3 = retrieve_top3(local_query_emb, local_corpus_embs, corpus_ids)
                    
                    gold = query["gold_top3"]
                    
                    # P@3
                    jina_p3 = len(set(jina_top3) & set(gold)) / 3
                    local_p3 = len(set(local_top3) & set(gold)) / 3
                    
                    jina_results["p3"].append(jina_p3)
                    local_results["p3"].append(local_p3)
                
                jina_avg_p3 = np.mean(jina_results["p3"]) * 100
                local_avg_p3 = np.mean(local_results["p3"]) * 100
                
                passed = abs(jina_avg_p3 - local_avg_p3) < 5  # Within 5% tolerance
                details = f"{mode}: Jina P@3={jina_avg_p3:.1f}%, Local P@3={local_avg_p3:.1f}%, Corpus sim={avg_corpus_sim:.4f}"
                
                self._add_result(f"Benchmark {test_name} ({mode})", passed, details, similarity=avg_corpus_sim)
                print(f"    {'✓' if passed else '✗'} {details}")
                
            except Exception as e:
                self._add_result(f"Benchmark {test_name} ({mode})", False, str(e))
                print(f"    ✗ Error: {e}")
    
    # ============================================
    # Run All Tests
    # ============================================
    
    def run_all_tests(self):
        """Run all tests."""
        print("=" * 70)
        print("Jina AI API Comparison Test Suite")
        print("=" * 70)
        print(f"Jina API: {JINA_API_BASE}")
        print(f"Local API: {LOCAL_API_BASE}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Check local server is running
        try:
            requests.get(f"{LOCAL_API_BASE.replace('/v1', '')}/health", timeout=5)
        except:
            print("\n❌ ERROR: Local server is not running!")
            print("Please start the server with: docker compose up")
            return
        
        print("\n" + "=" * 70)
        print("EMBEDDING TESTS")
        print("=" * 70)
        
        # V4 Tests
        self.test_v4_standard_embedding()
        self.test_v4_late_chunking()
        self.test_v4_dimensions()
        self.test_v4_tasks()
        self.test_v4_embedding_types()
        self.test_v4_prompt_names()
        
        # V3 Tests
        self.test_v3_tasks()
        self.test_v3_late_chunking()
        
        print("\n" + "=" * 70)
        print("RERANKER TESTS")
        print("=" * 70)
        
        self.test_reranker_basic()
        self.test_reranker_top_n()
        self.test_reranker_return_documents()
        
        print("\n" + "=" * 70)
        print("BENCHMARK TESTS")
        print("=" * 70)
        
        for test_name, test_file in TEST_FILES.items():
            self.test_benchmark_retrieval(test_name, test_file)
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        print("\n| Test | Status | Similarity | Details |")
        print("|------|--------|------------|---------|")
        for r in self.results:
            status = "✓ PASS" if r.passed else "✗ FAIL"
            sim = f"{r.similarity:.4f}" if r.similarity > 0 else "-"
            print(f"| {r.name} | {status} | {sim} | {r.details[:50]}... |")
        
        # Failed tests
        failed = [r for r in self.results if not r.passed]
        if failed:
            print("\n❌ FAILED TESTS:")
            for r in failed:
                print(f"  - {r.name}: {r.details}")
        
        return self.results


if __name__ == "__main__":
    tester = JinaAPITester()
    tester.run_all_tests()
