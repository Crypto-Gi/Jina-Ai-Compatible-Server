"""
Jina Local API Server - BGE Reranker Wrapper
=============================================
Wrapper for BAAI/bge-reranker-v2-m3 model.

Specifications (verified from HuggingFace, FlagEmbedding docs):
- Parameters: 568M
- Max Length: 512 (recommended)
- Architecture: XLM-RoBERTa-Large based
- Output: Normalized scores (0-1 range with sigmoid)
- Languages: 100+
"""

from typing import Any

import torch

from app.logging_config import get_logger
from app.models.base import RerankerModelWrapper

logger = get_logger(__name__)


class BGERerankerWrapper(RerankerModelWrapper):
    """
    Wrapper for BAAI/bge-reranker-v2-m3 model.
    
    Uses FlagEmbedding's FlagReranker for optimal performance.
    Returns normalized relevance scores (0-1 range) for query-document pairs.
    """

    model_id = "bge-reranker-v2-m3"
    hf_model_id = "BAAI/bge-reranker-v2-m3"
    query_max_length = 256
    passage_max_length = 512

    def load(self, device: str, torch_dtype: torch.dtype = torch.float16) -> None:
        """Load the BGE reranker model using FlagReranker."""
        logger.info(
            "Loading model",
            model_id=self.model_id,
            hf_model_id=self.hf_model_id,
            device=device,
        )

        self.device = device
        use_fp16 = device == "cuda" and torch_dtype == torch.float16

        try:
            from FlagEmbedding import FlagReranker

            # FlagReranker handles device placement via devices parameter
            devices = [device] if device == "cuda" else None
            self.model = FlagReranker(
                self.hf_model_id,
                use_fp16=use_fp16,
                query_max_length=self.query_max_length,
                passage_max_length=self.passage_max_length,
                devices=devices,
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower() and device == "cuda":
                logger.warning(
                    "GPU out of memory, falling back to CPU",
                    model_id=self.model_id,
                    error=str(e),
                )
                self.device = "cpu"
                from FlagEmbedding import FlagReranker

                self.model = FlagReranker(
                    self.hf_model_id,
                    use_fp16=False,
                    query_max_length=self.query_max_length,
                    passage_max_length=self.passage_max_length,
                    devices=None,  # CPU
                )
            else:
                raise e

        self._loaded = True
        logger.info("Model loaded successfully", model_id=self.model_id)

    def rerank(
        self,
        query: str,
        documents: list[str | dict[str, Any]],
        top_n: int | None = None,
        return_documents: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Rerank documents by relevance to query.
        
        Uses FlagReranker.compute_score() with normalize=True for 0-1 scores.
        Returns results sorted by score (highest first).
        """
        if not self._loaded:
            raise RuntimeError(f"Model {self.model_id} is not loaded")

        if not documents:
            return [], 0

        # Normalize documents to list of strings
        texts = []
        for doc in documents:
            if isinstance(doc, str):
                texts.append(doc)
            elif isinstance(doc, dict) and "text" in doc:
                texts.append(doc["text"])
            else:
                raise ValueError(f"Document must be string or dict with 'text' key: {doc}")

        # Create query-document pairs for FlagReranker
        # Format: [[query, doc1], [query, doc2], ...]
        pairs = [[query, doc] for doc in texts]

        # Compute normalized scores (0-1 range via sigmoid)
        scores = self.model.compute_score(pairs, normalize=True)

        # Handle single result (returns float instead of list)
        if isinstance(scores, (int, float)):
            scores = [scores]

        # Build results with original indices
        results: list[dict[str, Any]] = []
        for idx, score in enumerate(scores):
            result: dict[str, Any] = {
                "index": idx,
                "relevance_score": float(score),
            }
            if return_documents:
                original_doc = documents[idx]
                if isinstance(original_doc, dict):
                    result["document"] = original_doc
                else:
                    result["document"] = {"text": original_doc}
            results.append(result)

        # Sort by relevance score (descending)
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Apply top_n filter
        if top_n is not None and top_n > 0:
            results = results[:top_n]

        # Estimate token count
        query_tokens = len(query.split())
        doc_tokens = sum(len(t.split()) for t in texts)
        token_count = int((query_tokens + doc_tokens) * 1.3)

        return results, token_count
