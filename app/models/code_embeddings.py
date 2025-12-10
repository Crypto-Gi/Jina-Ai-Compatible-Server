"""
Jina Local API Server - Jina Code Embeddings Wrapper
====================================================
Wrapper for jinaai/jina-code-embeddings-0.5b and jina-code-embeddings-1.5b models.
Code embeddings supporting various code retrieval tasks.
"""

from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from app.logging_config import get_logger
from app.models.base import EmbeddingModelWrapper

logger = get_logger(__name__)


# Instruction prefixes for different code tasks
INSTRUCTION_CONFIG = {
    "nl2code": {
        "query": "Find the most relevant code snippet given the following query:\n",
        "passage": "Candidate code snippet:\n",
    },
    "qa": {
        "query": "Find the most relevant answer given the following question:\n",
        "passage": "Candidate answer:\n",
    },
    "code2code": {
        "query": "Find an equivalent code snippet given the following code snippet:\n",
        "passage": "Candidate code snippet:\n",
    },
    "code2nl": {
        "query": "Find the most relevant comment given the following code snippet:\n",
        "passage": "Candidate comment:\n",
    },
    "code2completion": {
        "query": "Find the most relevant completion given the following start of code snippet:\n",
        "passage": "Candidate completion:\n",
    },
}


def last_token_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Extract embeddings using last token pooling (decoder-style models).
    """
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]


class CodeEmbeddingsWrapper(EmbeddingModelWrapper):
    """Wrapper for jina-code-embeddings models (0.5b and 1.5b variants)."""

    # These are set per-variant in __init__
    default_dimensions = 1536  # Will be overridden based on variant
    max_tokens = 8192  # Will be overridden based on variant
    supports_multimodal = False
    
    # Variant-specific configurations
    VARIANT_CONFIG = {
        "0.5b": {
            "dimensions": 896,
            "max_tokens": 8192,
            "mrl_sizes": [64, 128, 256, 512, 896],
        },
        "1.5b": {
            "dimensions": 1536,
            "max_tokens": 32768,
            "mrl_sizes": [128, 256, 512, 1024, 1536],
        },
    }

    # Map public task names to internal instruction types
    # Supports both simple names ("nl2code") and dot notation ("nl2code.query")
    TASK_MAP = {
        # Standard tasks (use prompt_name to determine query/passage)
        "nl2code": "nl2code",
        "code2code": "code2code",
        "code2nl": "code2nl",
        "code2completion": "code2completion",
        "qa": "qa",
        # Dot notation tasks (explicitly specify query/passage)
        "nl2code.query": ("nl2code", True),      # query
        "nl2code.passage": ("nl2code", False),   # passage
        "code2code.query": ("code2code", True),
        "code2code.passage": ("code2code", False),
        "code2nl.query": ("code2nl", True),
        "code2nl.passage": ("code2nl", False),
        "code2completion.query": ("code2completion", True),
        "code2completion.passage": ("code2completion", False),
        "qa.query": ("qa", True),
        "qa.passage": ("qa", False),
        # Generic fallbacks
        "text-matching": "nl2code",
        "retrieval": "nl2code",
        "retrieval.query": ("nl2code", True),
        "retrieval.passage": ("nl2code", False),
    }

    def __init__(self, variant: str = "1.5b") -> None:
        """
        Initialize code embeddings wrapper.
        
        Args:
            variant: Model size variant ("0.5b" or "1.5b")
        """
        super().__init__()
        if variant not in ("0.5b", "1.5b"):
            raise ValueError(f"Invalid variant: {variant}. Must be '0.5b' or '1.5b'")
        
        self.variant = variant
        self.model_id = f"jina-code-embeddings-{variant}"
        self.hf_model_id = f"jinaai/jina-code-embeddings-{variant}"
        
        # Set variant-specific dimensions and max tokens
        config = self.VARIANT_CONFIG[variant]
        self.default_dimensions = config["dimensions"]
        self.max_tokens = config["max_tokens"]
        self.mrl_sizes = config["mrl_sizes"]

    def load(self, device: str, torch_dtype: torch.dtype = torch.float16) -> None:
        """Load the jina-code-embeddings model."""
        logger.info(
            "Loading model",
            model_id=self.model_id,
            hf_model_id=self.hf_model_id,
            device=device,
        )

        self.device = device
        dtype = torch_dtype if device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
        try:
            self.model = AutoModel.from_pretrained(
                self.hf_model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
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

    def _add_instruction(self, text: str, task: str, is_query: bool = True) -> str:
        """Add instruction prefix based on task and input type."""
        if task not in INSTRUCTION_CONFIG:
            task = "nl2code"  # Default
        
        role = "query" if is_query else "passage"
        prefix = INSTRUCTION_CONFIG[task][role]
        return f"{prefix}{text}"

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
        Encode texts to embeddings using jina-code-embeddings.
        
        Uses last-token pooling as recommended for decoder-style models.
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

        # Map task and determine if query or passage
        task_mapping = self.TASK_MAP.get(task, "nl2code") if task else "nl2code"
        
        # Handle tuple mappings (task, is_query) for dot notation
        if isinstance(task_mapping, tuple):
            internal_task, is_query = task_mapping
        else:
            internal_task = task_mapping
            # Use prompt_name to determine query/passage for simple task names
            is_query = prompt_name != "passage" if prompt_name else True

        # Add instruction prefixes
        prefixed_texts = [
            self._add_instruction(text, internal_task, is_query) for text in texts
        ]

        # Tokenize
        batch_dict = self.tokenizer(
            prefixed_texts,
            padding=True,
            truncation=truncate,
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            
            if late_chunking:
                embeddings = outputs.last_hidden_state
            else:
                embeddings = last_token_pool(
                    outputs.last_hidden_state, batch_dict["attention_mask"]
                )

        # Handle dimension truncation
        target_dims = dimensions or self.default_dimensions
        if target_dims < embeddings.shape[-1]:
            embeddings = embeddings[..., :target_dims]

        # L2 normalize if requested
        if normalized:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        # Token count from tokenizer
        token_count = sum(
            batch_dict["attention_mask"][i].sum().item()
            for i in range(len(texts))
        )

        return embeddings, int(token_count)
