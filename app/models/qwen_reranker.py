"""
Jina Local API Server - Qwen3 Reranker Wrapper
===============================================
Wrapper for Qwen/Qwen3-Reranker models (0.6B, 4B, 8B).

Specifications (verified from HuggingFace):
- Qwen3-Reranker-0.6B: 0.6B params, 32k context
- Qwen3-Reranker-4B: 4B params, 32k context
- Qwen3-Reranker-8B: 8B params, 32k context
- Languages: 100+
- Requires: transformers>=4.51.0

IMPORTANT: Qwen3-Reranker is a Causal LM that outputs "yes"/"no" token probabilities,
NOT a sequence classification model. It uses a special prompt format with
<Instruct>, <Query>, <Document> tags.
"""

from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.logging_config import get_logger
from app.models.base import RerankerModelWrapper

logger = get_logger(__name__)


class QwenRerankerWrapper(RerankerModelWrapper):
    """
    Wrapper for Qwen3-Reranker models.
    
    Uses AutoModelForCausalLM with special prompt format.
    Computes relevance scores from "yes"/"no" token probabilities.
    Returns normalized scores (0-1 range).
    """

    max_length = 8192  # Use reasonable max for reranking

    # Prompt templates (from official HuggingFace docs)
    PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

    # Model size configurations
    MODEL_CONFIGS = {
        "0.6b": {
            "model_id": "qwen3-reranker-0.6b",
            "hf_model_id": "Qwen/Qwen3-Reranker-0.6B",
        },
        "4b": {
            "model_id": "qwen3-reranker-4b",
            "hf_model_id": "Qwen/Qwen3-Reranker-4B",
        },
        "8b": {
            "model_id": "qwen3-reranker-8b",
            "hf_model_id": "Qwen/Qwen3-Reranker-8B",
        },
    }

    def __init__(self, model_size: str = "0.6b"):
        """
        Initialize Qwen3 reranker wrapper.
        
        Args:
            model_size: One of "0.6b", "4b", "8b"
        """
        super().__init__()
        self.model_size = model_size.lower()

        if self.model_size not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unknown Qwen3 reranker size: {model_size}. "
                f"Valid options: {list(self.MODEL_CONFIGS.keys())}"
            )

        config = self.MODEL_CONFIGS[self.model_size]
        self.model_id = config["model_id"]
        self.hf_model_id = config["hf_model_id"]

        # Token IDs for "yes" and "no" - set after tokenizer loads
        self.token_true_id: int = 0
        self.token_false_id: int = 0
        self.prefix_tokens: list[int] = []
        self.suffix_tokens: list[int] = []

    def load(self, device: str, torch_dtype: torch.dtype = torch.float16) -> None:
        """Load the Qwen3 reranker model (CausalLM)."""
        logger.info(
            "Loading model",
            model_id=self.model_id,
            hf_model_id=self.hf_model_id,
            device=device,
        )

        self.device = device

        # Load tokenizer with left padding (required for Qwen3)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_id,
            padding_side="left",
        )

        # Get token IDs for "yes" and "no"
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")

        # Pre-tokenize prefix and suffix
        self.prefix_tokens = self.tokenizer.encode(self.PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.SUFFIX, add_special_tokens=False)

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_id,
                torch_dtype=torch_dtype if device == "cuda" else torch.float32,
                trust_remote_code=True,
            )
            self.model.to(device)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower() and device == "cuda":
                logger.warning(
                    "GPU out of memory, falling back to CPU",
                    model_id=self.model_id,
                    error=str(e),
                )
                self.device = "cpu"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.hf_model_id,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )
                self.model.to("cpu")
            else:
                raise e

        self.model.eval()
        self._loaded = True
        logger.info("Model loaded successfully", model_id=self.model_id)

    def _format_instruction(
        self,
        instruction: str | None,
        query: str,
        document: str,
    ) -> str:
        """Format the input according to Qwen3-Reranker prompt template."""
        if instruction is None:
            instruction = self.DEFAULT_INSTRUCTION
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"

    def _process_inputs(self, pairs: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize inputs with prefix/suffix tokens."""
        # Tokenize the content (without prefix/suffix)
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation=True,
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )

        # Add prefix and suffix tokens to each input
        for i, input_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + input_ids + self.suffix_tokens

        # Pad the inputs
        inputs = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        )

        # Move to device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)

        return inputs

    def _compute_scores(self, inputs: dict[str, torch.Tensor]) -> list[float]:
        """Compute relevance scores from yes/no token probabilities."""
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get logits for the last token position
            batch_logits = outputs.logits[:, -1, :]

            # Extract logits for "yes" and "no" tokens
            true_logits = batch_logits[:, self.token_true_id]
            false_logits = batch_logits[:, self.token_false_id]

            # Stack and apply log_softmax, then get probability of "yes"
            stacked = torch.stack([false_logits, true_logits], dim=1)
            log_probs = F.log_softmax(stacked, dim=1)
            scores = log_probs[:, 1].exp().cpu().tolist()  # Probability of "yes"

        return scores

    def rerank(
        self,
        query: str,
        documents: list[str | dict[str, Any]],
        top_n: int | None = None,
        return_documents: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Rerank documents by relevance to query.
        
        Uses Qwen3-Reranker's yes/no probability scoring.
        Returns results sorted by score (highest first), with scores in 0-1 range.
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

        # Format query-document pairs with instruction template
        formatted_pairs = [
            self._format_instruction(None, query, doc)
            for doc in texts
        ]

        # Process inputs and compute scores
        inputs = self._process_inputs(formatted_pairs)
        scores = self._compute_scores(inputs)

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
