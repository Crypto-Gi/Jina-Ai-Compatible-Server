"""
Jina Local API Server - Base Model Wrapper Classes
==================================================
Abstract base classes for embedding and reranker model wrappers.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseModelWrapper(ABC):
    """Abstract base class for all model wrappers."""

    model_id: str
    model_type: str  # "embedding" or "reranker"
    hf_model_id: str

    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
        self.device: str = "cpu"
        self._loaded: bool = False

    @abstractmethod
    def load(self, device: str, torch_dtype: torch.dtype = torch.float16) -> None:
        """
        Load the model onto the specified device.
        
        Args:
            device: Target device ('cuda' or 'cpu')
            torch_dtype: Data type for model weights
        """
        pass

    def is_loaded(self) -> bool:
        """Check if the model has been loaded."""
        return self._loaded

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class EmbeddingModelWrapper(BaseModelWrapper):
    """Base class for embedding model wrappers."""

    model_type: str = "embedding"
    default_dimensions: int = 1024
    max_tokens: int = 8192
    supports_multimodal: bool = False

    @abstractmethod
    def encode(
        self,
        inputs: list[str | dict[str, str]],
        task: str | None = None,
        dimensions: int | None = None,
        normalized: bool = True,
        prompt_name: str | None = None,
        late_chunking: bool = False,
        truncate: bool = True,
        **kwargs: Any,
    ) -> tuple[Any, int]:
        """
        Encode inputs to embeddings.
        
        Args:
            inputs: List of input strings or dicts (for multimodal)
            task: Task name for task-specific embeddings
            dimensions: Output dimensions (truncation)
            normalized: Whether to normalize embeddings
            prompt_name: Prompt name (for v4)
            late_chunking: Whether to return token-level embeddings
            truncate: Whether to truncate inputs to max length
            **kwargs: Additional model-specific arguments
            
        Returns:
            Tuple of (embeddings, token_count)
        """
        raise NotImplementedError


class RerankerModelWrapper(BaseModelWrapper):
    """Base class for reranker model wrappers."""

    model_type: str = "reranker"

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        return_documents: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: Search query
            documents: List of document texts
            top_n: Return only top N documents (None = all)
            return_documents: Include document text in results
            
        Returns:
            Tuple of (results list, token count)
            Each result: {index, relevance_score, document?}
        """
        pass
