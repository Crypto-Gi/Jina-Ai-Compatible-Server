"""
Jina Local API Server - Jina Embeddings V3 Wrapper
==================================================
Wrapper for jinaai/jina-embeddings-v3 model.
Multilingual text embeddings with LoRA adapters for different tasks.
"""

import os
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from app.logging_config import get_logger
from app.models.base import EmbeddingModelWrapper

# Disable flash attention to avoid segfaults on newer GPUs (e.g., RTX 5090)
os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"

logger = get_logger(__name__)


class EmbeddingsV3Wrapper(EmbeddingModelWrapper):
    """Wrapper for jina-embeddings-v3 model."""

    model_id = "jina-embeddings-v3"
    hf_model_id = "jinaai/jina-embeddings-v3"
    default_dimensions = 1024
    max_tokens = 8192
    supports_multimodal = False

    # Task mapping for LoRA adapters
    TASK_MAP = {
        "retrieval.query": "retrieval.query",
        "retrieval.passage": "retrieval.passage",
        "separation": "separation",
        "classification": "classification",
        "text-matching": "text-matching",
        # Aliases for convenience
        "retrieval": "retrieval.query",
        "query": "retrieval.query",
        "passage": "retrieval.passage",
    }

    def load(self, device: str, torch_dtype: torch.dtype = torch.float16) -> None:
        """Load the jina-embeddings-v3 model."""
        logger.info(
            "Loading model",
            model_id=self.model_id,
            hf_model_id=self.hf_model_id,
            device=device,
        )

        self.device = device
        dtype = torch_dtype if device == "cuda" else torch.float32

        # Load tokenizer for late chunking
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
        
        try:
            # Use 'dtype' instead of deprecated 'torch_dtype' per HuggingFace warning
            # Use 'eager' attention since XLMRobertaLoRA doesn't support SDPA yet
            self.model = AutoModel.from_pretrained(
                self.hf_model_id,
                trust_remote_code=True,
                dtype=dtype,
                attn_implementation="eager",
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
                    dtype=torch.float32,
                    attn_implementation="eager",
                )
                self.model.to("cpu")
            else:
                raise e
        self.model.eval()
        self._loaded = True

        logger.info("Model loaded successfully", model_id=self.model_id)

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
        Encode texts to embeddings using jina-embeddings-v3.
        
        Supports:
        - Task-specific LoRA adapters
        - MRL dimension truncation
        - Late chunking for contextual chunk embeddings
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

        # Map task
        internal_task = self.TASK_MAP.get(task, "text-matching") if task else "text-matching"

        if late_chunking:
            # Late chunking: process each text through transformer, then chunk
            return self._encode_late_chunking(
                texts, internal_task, dimensions, normalized, truncate
            )
        else:
            # Standard encoding
            return self._encode_standard(
                texts, internal_task, dimensions, normalized, truncate
            )

    def _encode_standard(
        self,
        texts: list[str],
        task: str,
        dimensions: int | None,
        normalized: bool,
        truncate: bool,
    ) -> tuple[torch.Tensor, int]:
        """Standard encoding using model's encode() method."""
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                task=task,
                max_length=self.max_tokens if truncate else None,
                truncate_dim=dimensions,
            )

        # Convert to tensor if numpy array
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, device=self.device)

        # L2 normalize if requested
        if normalized:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        # Calculate actual token count using tokenizer
        try:
            token_count = sum(
                len(self.tokenizer.encode(text, add_special_tokens=False))
                for text in texts
            )
        except Exception:
            # Fallback to heuristic if tokenizer fails
            token_count = int(sum(len(text.split()) for text in texts) * 1.3)

        return embeddings, token_count

    def _encode_late_chunking(
        self,
        texts: list[str],
        task: str,
        dimensions: int | None,
        normalized: bool,
        truncate: bool,
    ) -> tuple[torch.Tensor, int]:
        """
        Late chunking: concatenate all inputs, process through transformer, then pool per input.
        
        This matches Jina's official late_chunking behavior:
        1. Concatenate ALL input texts into one long document
        2. Pass entire concatenated text through transformer (full context)
        3. Split token embeddings back by original input boundaries
        4. Mean pool each input's tokens to get one embedding per input
        
        Each input's embedding contains contextual information from ALL other inputs.
        
        Returns one embedding per input (not per chunk).
        """
        from app.late_chunking import late_chunking_pooling, ChunkSpan
        
        if not texts:
            return torch.empty(0, dimensions or self.default_dimensions, device=self.device), 0
        
        # Step 1: Concatenate all texts with a separator
        # We use double newline as separator to maintain clear boundaries
        separator = "\n\n"
        concatenated_text = separator.join(texts)
        
        # Step 2: Tokenize the concatenated text
        encoding = self.tokenizer(
            concatenated_text,
            padding=True,
            truncation=truncate,
            max_length=self.max_tokens,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding.get("offset_mapping", None)
        
        total_tokens = attention_mask.sum().item()
        
        # Step 3: Get token-level embeddings using LoRA adapter
        with torch.no_grad():
            # Prepare encoding for model (without offset_mapping)
            model_encoding = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            
            # Get task ID from model's adaptation map for LoRA
            if hasattr(self.model, '_adaptation_map') and task in self.model._adaptation_map:
                task_id = self.model._adaptation_map[task]
                adapter_mask = torch.full(
                    (input_ids.shape[0],), 
                    task_id, 
                    dtype=torch.int32,
                    device=self.device
                )
                outputs = self.model(**model_encoding, adapter_mask=adapter_mask)
            else:
                outputs = self.model(**model_encoding)
            
            if hasattr(outputs, 'last_hidden_state'):
                token_embeddings = outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                token_embeddings = outputs[0]
            else:
                token_embeddings = outputs
        
        # Step 4: Find token boundaries for each original input
        # We need to map back from concatenated text to original inputs
        chunk_spans = []
        current_char_pos = 0
        
        if offset_mapping is not None:
            offset_mapping = offset_mapping[0].tolist()  # Remove batch dim
            
            for text in texts:
                text_start = concatenated_text.find(text, current_char_pos)
                text_end = text_start + len(text)
                
                # Find token indices for this text
                token_start = None
                token_end = None
                
                for idx, (char_start, char_end) in enumerate(offset_mapping):
                    if char_start is None or char_end is None:
                        continue
                    if token_start is None and char_end > text_start:
                        token_start = idx
                    if char_start < text_end:
                        token_end = idx + 1
                
                if token_start is None:
                    token_start = 0
                if token_end is None:
                    token_end = len(offset_mapping)
                
                chunk_spans.append(ChunkSpan(
                    start_token=token_start,
                    end_token=token_end,
                    text=text,
                ))
                
                current_char_pos = text_end + len(separator)
        else:
            # Fallback: estimate token boundaries based on text lengths
            current_token = 1  # Skip [CLS]
            for text in texts:
                text_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
                chunk_spans.append(ChunkSpan(
                    start_token=current_token,
                    end_token=current_token + text_tokens,
                    text=text,
                ))
                current_token += text_tokens + len(self.tokenizer.encode(separator, add_special_tokens=False))
        
        # Step 5: Apply late chunking pooling - one embedding per original input
        embeddings = late_chunking_pooling(
            token_embeddings.squeeze(0),
            attention_mask.squeeze(0),
            chunk_spans,
        )
        
        # Apply dimension truncation if needed
        if dimensions and dimensions < embeddings.shape[-1]:
            embeddings = embeddings[:, :dimensions]
        
        # Normalize
        if normalized:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings, int(total_tokens)

