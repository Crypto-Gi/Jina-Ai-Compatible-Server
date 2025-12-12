"""
Jina Local API Server - BGE-M3 Embedding Wrapper
=================================================
Wrapper for BAAI/bge-m3 multilingual embedding model.

Specifications (verified from HuggingFace, FlagEmbedding docs):
- Parameters: 568M
- Max Tokens: 8,192
- Dimensions: 1024 (default)
- Languages: 100+
- Features: Dense, Sparse (lexical), Multi-vector (ColBERT)
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from app.logging_config import get_logger
from app.models.base import EmbeddingModelWrapper

logger = get_logger(__name__)


class BGEEmbeddingsWrapper(EmbeddingModelWrapper):
    """
    Wrapper for BAAI/bge-m3 embedding model.
    
    Uses FlagEmbedding library for optimal performance.
    Supports dense embeddings, MRL dimension truncation, and late chunking.
    """

    model_id = "bge-m3"
    hf_model_id = "BAAI/bge-m3"
    default_dimensions = 1024
    max_tokens = 8192  # BGE-M3 specific
    supports_multimodal = False

    def load(self, device: str, torch_dtype: torch.dtype = torch.float16) -> None:
        """Load the BGE-M3 model using FlagEmbedding library."""
        logger.info(
            "Loading model",
            model_id=self.model_id,
            hf_model_id=self.hf_model_id,
            device=device,
        )

        self.device = device
        use_fp16 = device == "cuda" and torch_dtype == torch.float16

        try:
            from FlagEmbedding import BGEM3FlagModel

            # FlagEmbedding requires devices as a list
            devices = [device] if device == "cuda" else None
            self.model = BGEM3FlagModel(
                self.hf_model_id,
                use_fp16=use_fp16,
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
                from FlagEmbedding import BGEM3FlagModel

                self.model = BGEM3FlagModel(
                    self.hf_model_id,
                    use_fp16=False,
                    devices=None,  # CPU
                )
            else:
                raise e

        # Get tokenizer for late chunking
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)

        self._loaded = True
        logger.info("Model loaded successfully", model_id=self.model_id)

    def encode(
        self,
        inputs: list[str | dict[str, str]],
        task: str | None = None,  # Ignored for BGE - no task adapters
        dimensions: int | None = None,
        normalized: bool = True,
        prompt_name: str | None = None,
        late_chunking: bool = False,
        truncate: bool = True,
        **kwargs: Any,
    ) -> tuple[Any, int]:
        """
        Encode texts to embeddings using BGE-M3.
        
        Note: BGE-M3 does not support task-specific adapters like Jina.
        The `task` parameter is accepted for API compatibility but ignored.
        """
        if not self._loaded:
            raise RuntimeError(f"Model {self.model_id} is not loaded")

        # Normalize inputs to list of strings
        texts = []
        for inp in inputs:
            if isinstance(inp, str):
                texts.append(inp)
            elif isinstance(inp, dict) and "text" in inp:
                texts.append(inp["text"])
            else:
                raise ValueError(f"Invalid input type for {self.model_id}: {type(inp)}")

        if late_chunking:
            # BGE-M3 does NOT support late chunking - the model architecture causes
            # embedding space collapse (tested: -46.67% P@3 degradation).
            # Unlike Jina V4 or Voyage context-3, BGE-M3 wasn't trained for this use case.
            raise ValueError(
                f"Model {self.model_id} does not support late_chunking. "
                "BGE-M3's architecture causes embedding space collapse when late chunking is applied. "
                "Use jina-embeddings-v4 or voyage-context-3 for contextual embeddings."
            )
        
        return self._encode_standard(texts, dimensions, normalized, truncate)

    def _encode_standard(
        self,
        texts: list[str],
        dimensions: int | None,
        normalized: bool,
        truncate: bool,
    ) -> tuple[torch.Tensor, int]:
        """Standard encoding using BGEM3FlagModel.encode()."""
        # Use FlagEmbedding's encode method
        output = self.model.encode(
            texts,
            max_length=self.max_tokens if truncate else None,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        embeddings = output["dense_vecs"]  # numpy array (N, 1024) - already normalized by FlagEmbedding

        # Convert to tensor
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings, device=self.device)

        # Dimension truncation (MRL)
        if dimensions and dimensions < self.default_dimensions:
            embeddings = embeddings[:, :dimensions]
            # Re-normalize after truncation since we changed dimensions
            if normalized:
                embeddings = F.normalize(embeddings, p=2, dim=-1)
        # Note: FlagEmbedding already returns normalized embeddings, so we only
        # re-normalize if dimensions were truncated

        # Calculate token count
        try:
            token_count = sum(
                len(self.tokenizer.encode(text, add_special_tokens=False))
                for text in texts
            )
        except Exception:
            token_count = int(sum(len(text.split()) for text in texts) * 1.3)

        return embeddings, token_count
