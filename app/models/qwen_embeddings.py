"""
Jina Local API Server - Qwen3 Embedding Wrapper
================================================
Wrapper for Qwen/Qwen3-Embedding models (0.6B, 4B, 8B).

Specifications (verified from HuggingFace):
- Qwen3-Embedding-0.6B: 0.6B params, 32k context, 1024 dim
- Qwen3-Embedding-4B: 4B params, 32k context, 1024 dim
- Qwen3-Embedding-8B: 8B params, 32k context, 1024 dim
- MRL Support: Yes (32-1024 dimensions)
- Instruction-Aware: Yes (prompt_name for queries)
- Languages: 100+
- Requires: transformers>=4.51.0, sentence-transformers>=2.7.0

NOTE: Qwen3 uses LAST TOKEN pooling (not mean pooling like BERT-based models).
Late chunking is NOT supported because it requires mean pooling over token spans.
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from app.logging_config import get_logger
from app.models.base import EmbeddingModelWrapper

logger = get_logger(__name__)


class QwenEmbeddingsWrapper(EmbeddingModelWrapper):
    """
    Wrapper for Qwen3-Embedding models.
    
    Uses SentenceTransformers for optimal performance.
    Supports task-based instruction prompts, MRL dimension truncation.
    
    Qwen3 is instruction-aware: different task instructions yield 1-5% improvement.
    We provide rich, task-specific instructions matching Jina's LoRA adapter tasks.
    """

    default_dimensions = 1024
    max_tokens = 32768  # Qwen3's 32k context - 4x longer than BGE/Jina!
    supports_multimodal = False

    # ==========================================================================
    # TASK INSTRUCTIONS - Enhanced prompts for Qwen3's instruction-aware model
    # ==========================================================================
    # Format: "Instruct: {instruction}\nQuery:{text}"
    # 
    # These instructions are designed to match Jina V3's LoRA adapter tasks
    # while leveraging Qwen3's superior instruction-following and reasoning.
    # 
    # Per Qwen3 docs: "We recommend that developers create tailored instructions
    # specific to their tasks and scenarios."
    # ==========================================================================

    TASK_INSTRUCTIONS = {
        # ======================================================================
        # RETRIEVAL TASKS - Asymmetric (query vs passage)
        # ======================================================================
        "retrieval.query": (
            "Given a search query, retrieve relevant documents or passages that "
            "directly answer or address the query"
        ),
        "retrieval.passage": None,  # Passages don't need instruction prefix
        
        # Aliases for retrieval
        "query": (
            "Given a search query, retrieve relevant documents or passages that "
            "directly answer or address the query"
        ),
        "passage": None,
        
        # ======================================================================
        # TEXT MATCHING - Symmetric similarity (both texts get same treatment)
        # ======================================================================
        "text-matching": (
            "Represent this text for finding semantically similar texts that "
            "share the same meaning, topic, or intent"
        ),
        
        # ======================================================================
        # CLASSIFICATION - Represent text for categorization
        # ======================================================================
        "classification": (
            "Represent this text for classification, capturing the key features "
            "and characteristics that determine its category or label"
        ),
        
        # ======================================================================
        # SEPARATION / CLUSTERING - Represent text for grouping similar items
        # ======================================================================
        "separation": (
            "Represent this text for clustering, capturing the distinctive "
            "semantic features that differentiate it from other topics or themes"
        ),
        "clustering": (
            "Represent this text for clustering, capturing the distinctive "
            "semantic features that differentiate it from other topics or themes"
        ),
        
        # ======================================================================
        # CODE RETRIEVAL - Specialized for code search
        # ======================================================================
        "code.query": (
            "Given a natural language description, retrieve relevant code "
            "snippets, functions, or implementations that match the description"
        ),
        "code.passage": None,  # Code passages don't need instruction
        
        # ======================================================================
        # SCIENTIFIC / ACADEMIC - For research and papers
        # ======================================================================
        "scientific.query": (
            "Given a scientific question or claim, retrieve relevant research "
            "papers, abstracts, or passages that support, refute, or address it"
        ),
        "scientific.passage": None,
        
        # ======================================================================
        # QA / QUESTION ANSWERING - For FAQ and knowledge bases
        # ======================================================================
        "qa.query": (
            "Given a question, retrieve passages or documents that contain "
            "the answer or relevant information to answer the question"
        ),
        "qa.passage": None,
        
        # ======================================================================
        # BITEXT / TRANSLATION - Cross-lingual matching
        # ======================================================================
        "bitext": (
            "Represent this text for finding its translation or semantically "
            "equivalent text in another language"
        ),
        
        # ======================================================================
        # SUMMARIZATION - Match summaries to full documents
        # ======================================================================
        "summarization.query": (
            "Given a summary, retrieve the original document or passage "
            "that this summary was derived from"
        ),
        "summarization.passage": None,
    }

    # Mapping from Jina-style task names to our instruction keys
    # This ensures API compatibility with existing Jina task parameters
    TASK_ALIASES = {
        "retrieval": "retrieval.query",
        "search": "retrieval.query",
        "code": "code.query",
        "scientific": "scientific.query",
        "qa": "qa.query",
        "question-answering": "qa.query",
        "summarization": "summarization.query",
    }

    # Model size configurations
    MODEL_CONFIGS = {
        "0.6b": {
            "model_id": "qwen3-embedding-0.6b",
            "hf_model_id": "Qwen/Qwen3-Embedding-0.6B",
        },
        "4b": {
            "model_id": "qwen3-embedding-4b",
            "hf_model_id": "Qwen/Qwen3-Embedding-4B",
        },
        "8b": {
            "model_id": "qwen3-embedding-8b",
            "hf_model_id": "Qwen/Qwen3-Embedding-8B",
        },
    }

    def __init__(self, model_size: str = "0.6b"):
        """
        Initialize Qwen3 embedding wrapper.
        
        Args:
            model_size: One of "0.6b", "4b", "8b"
        """
        super().__init__()
        self.model_size = model_size.lower()

        if self.model_size not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unknown Qwen3 model size: {model_size}. "
                f"Valid options: {list(self.MODEL_CONFIGS.keys())}"
            )

        config = self.MODEL_CONFIGS[self.model_size]
        self.model_id = config["model_id"]
        self.hf_model_id = config["hf_model_id"]

    def load(self, device: str, torch_dtype: torch.dtype = torch.float16) -> None:
        """Load the Qwen3 embedding model using SentenceTransformers."""
        logger.info(
            "Loading model",
            model_id=self.model_id,
            hf_model_id=self.hf_model_id,
            device=device,
        )

        self.device = device

        # Configure model kwargs for SentenceTransformer
        model_kwargs = {}
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
            if torch_dtype == torch.float16:
                model_kwargs["torch_dtype"] = torch.float16

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                self.hf_model_id,
                model_kwargs=model_kwargs,
                tokenizer_kwargs={"padding_side": "left"},
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower() and device == "cuda":
                logger.warning(
                    "GPU out of memory, falling back to CPU",
                    model_id=self.model_id,
                    error=str(e),
                )
                self.device = "cpu"
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(self.hf_model_id)
            else:
                raise e

        # Get tokenizer for late chunking and token counting
        self.tokenizer = self.model.tokenizer

        self._loaded = True
        logger.info("Model loaded successfully", model_id=self.model_id)

    def _get_instruction(self, task: str | None) -> str | None:
        """
        Get the instruction string for a given task.
        
        Resolves task aliases and returns the appropriate instruction.
        Returns None if no instruction should be applied (e.g., for passages).
        """
        if task is None:
            return None
        
        # Resolve aliases first
        resolved_task = self.TASK_ALIASES.get(task, task)
        
        # Get instruction for the resolved task
        return self.TASK_INSTRUCTIONS.get(resolved_task)

    def _format_with_instruction(self, texts: list[str], instruction: str) -> list[str]:
        """
        Format texts with Qwen3's instruction format.
        
        Format: "Instruct: {instruction}\nQuery:{text}"
        
        This is the official Qwen3 format from their documentation.
        """
        return [f"Instruct: {instruction}\nQuery:{text}" for text in texts]

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
        Encode texts to embeddings using Qwen3-Embedding.
        
        Supports:
        - Task-based instruction prompts (rich, task-specific instructions)
        - MRL dimension truncation (32-1024)
        
        Task Instructions:
        - retrieval.query / query: Search query embedding
        - retrieval.passage / passage: Document/passage embedding (no instruction)
        - text-matching: Symmetric similarity
        - classification: Text categorization
        - separation / clustering: Topic clustering
        - code.query: Code search from natural language
        - scientific.query: Academic/research retrieval
        - qa.query: Question answering
        - bitext: Cross-lingual matching
        - summarization.query: Summary-to-document matching
        
        Note: Late chunking is NOT supported (Qwen3 uses last-token pooling).
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

        # Get instruction for the task
        # Priority: explicit prompt_name (used as instruction) > task instruction > None
        instruction = None
        if prompt_name:
            # If prompt_name is provided, check if it's a known task or use as-is
            instruction = self._get_instruction(prompt_name)
            if instruction is None and prompt_name not in self.TASK_INSTRUCTIONS:
                # Treat prompt_name as a custom instruction string
                instruction = prompt_name
        elif task:
            instruction = self._get_instruction(task)

        if late_chunking:
            # Qwen3 uses last-token pooling, not mean pooling.
            # Late chunking requires mean pooling over token spans, which is incompatible.
            raise ValueError(
                f"late_chunking is not supported for {self.model_id}. "
                "Qwen3 models use last-token pooling, which is incompatible with late chunking. "
                "Use jina-embeddings-v3 or jina-embeddings-v4 for late chunking support."
            )
        
        return self._encode_standard(
            texts, dimensions, normalized, truncate, instruction
        )

    def _encode_standard(
        self,
        texts: list[str],
        dimensions: int | None,
        normalized: bool,
        truncate: bool,
        instruction: str | None,
    ) -> tuple[torch.Tensor, int]:
        """
        Standard encoding using SentenceTransformer.encode().
        
        If an instruction is provided, we format texts using Qwen3's official format:
        "Instruct: {instruction}\nQuery:{text}"
        
        This leverages Qwen3's instruction-aware training for 1-5% improvement.
        """
        # Apply instruction formatting if provided
        if instruction:
            # Format texts with Qwen3's instruction format
            formatted_texts = self._format_with_instruction(texts, instruction)
            # Use prompt=None since we've already formatted the texts
            embeddings = self.model.encode(
                formatted_texts,
                normalize_embeddings=normalized,
                convert_to_tensor=True,
            )
        else:
            # No instruction - encode texts directly (for passages, documents, etc.)
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=normalized,
                convert_to_tensor=True,
            )

        # Convert to tensor if numpy
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings, device=self.device)
        elif isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.to(self.device)

        # Dimension truncation (MRL)
        if dimensions and dimensions < self.default_dimensions:
            embeddings = embeddings[:, :dimensions]
            # Re-normalize after truncation if needed
            if normalized:
                embeddings = F.normalize(embeddings, p=2, dim=-1)

        # Calculate token count
        try:
            token_count = sum(
                len(self.tokenizer.encode(text, add_special_tokens=False))
                for text in texts
            )
        except Exception:
            token_count = int(sum(len(text.split()) for text in texts) * 1.3)

        return embeddings, token_count
