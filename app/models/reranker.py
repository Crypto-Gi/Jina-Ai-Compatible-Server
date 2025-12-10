"""
Jina Local API Server - Jina Reranker V3 Wrapper
================================================
Wrapper for jinaai/jina-reranker-v3 model.
Listwise document reranker for multilingual retrieval.
"""

from typing import Any

import torch
from transformers import AutoModel

from app.logging_config import get_logger
from app.models.base import RerankerModelWrapper

logger = get_logger(__name__)


class RerankerWrapper(RerankerModelWrapper):
    """Wrapper for jina-reranker-v3 model."""

    model_id = "jina-reranker-v3"
    hf_model_id = "jinaai/jina-reranker-v3"

    def load(self, device: str, torch_dtype: torch.dtype = torch.float16) -> None:
        """Load the jina-reranker-v3 model."""
        logger.info(
            "Loading model",
            model_id=self.model_id,
            hf_model_id=self.hf_model_id,
            device=device,
        )

        self.device = device

        # Reranker uses dtype="auto" as recommended in HuggingFace model card
        try:
            self.model = AutoModel.from_pretrained(
                self.hf_model_id,
                trust_remote_code=True,
                torch_dtype="auto",  # Per HF model card recommendation
            )
            self.model.to(device)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower() and device == "cuda":
                logger.warning(
                    "GPU out of memory, falling back to CPU",
                    model_id=self.model_id,
                    error=str(e)
                )
                self.device = "cpu"
                self.model = AutoModel.from_pretrained(
                    self.hf_model_id,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                )
                self.model.to("cpu")
            else:
                raise e
        self.model.eval()
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
        
        Uses the model's built-in rerank() method which returns results
        sorted by relevance score (highest first).
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
                # Fallback: try to convert to string or raise error
                # Jina API might support other keys, but 'text' is standard.
                # If dict doesn't have text, maybe use str(doc)?
                # Let's be strict for now or check if there's a specific key.
                raise ValueError(f"Document must be string or dict with 'text' key: {doc}")

        # Use model's rerank method
        with torch.no_grad():
            results = self.model.rerank(
                query=query,
                documents=texts,
                top_n=top_n,
            )

        # Format results to match Jina API
        formatted_results: list[dict[str, Any]] = []
        for r in results:
            result: dict[str, Any] = {
                "index": r["index"],
                "relevance_score": float(r["relevance_score"]),
            }
            if return_documents:
                # Return the original document object/string
                original_doc = documents[r["index"]]
                if isinstance(original_doc, dict):
                    result["document"] = original_doc
                else:
                    result["document"] = {"text": original_doc}
            formatted_results.append(result)

        # Estimate token count
        query_tokens = len(query.split())
        doc_tokens = sum(len(t.split()) for t in texts)
        token_count = int((query_tokens + doc_tokens) * 1.3)

        return formatted_results, token_count
