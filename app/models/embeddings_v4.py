"""
Jina Local API Server - Jina Embeddings V4 Wrapper
==================================================
Wrapper for jinaai/jina-embeddings-v4 model.
Multimodal embeddings (text + image) based on Qwen2.5-VL.

Supports:
- Text and image inputs (multimodal)
- Dense (single-vector) and multi-vector embeddings
- Late chunking for contextual chunk embeddings
- Task-specific adapters: retrieval, text-matching, code
- Flexible dimensions (2048 default, truncatable to 128)
"""

import base64
import io
from typing import Any

import httpx
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from app.logging_config import get_logger
from app.models.base import EmbeddingModelWrapper

logger = get_logger(__name__)


class EmbeddingsV4Wrapper(EmbeddingModelWrapper):
    """Wrapper for jina-embeddings-v4 multimodal model."""

    model_id = "jina-embeddings-v4"
    hf_model_id = "jinaai/jina-embeddings-v4"
    default_dimensions = 2048
    max_tokens = 8192
    supports_multimodal = True

    # Task mapping
    TASK_MAP = {
        "retrieval": "retrieval",
        "text-matching": "text-matching",
        "code": "code",
        # Aliases for v3 compatibility
        "retrieval.query": "retrieval",
        "retrieval.passage": "retrieval",
        # Code task aliases (per Jina API)
        "code.query": "code",
        "code.passage": "code",
    }

    def load(self, device: str, torch_dtype: torch.dtype = torch.float16) -> None:
        """Load the jina-embeddings-v4 model."""
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

        # HTTP client for downloading images
        self._http_client = httpx.Client(timeout=30.0)

        logger.info("Model loaded successfully", model_id=self.model_id)

    def _load_image(self, image_source: str) -> Image.Image:
        """
        Load an image from URL or base64 string.
        
        Args:
            image_source: URL or base64-encoded image data
            
        Returns:
            PIL Image object
        """
        if image_source.startswith(("http://", "https://")):
            # Download from URL
            response = self._http_client.get(image_source)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        else:
            # Assume base64-encoded
            # Handle data URI format: data:image/png;base64,xxxxx
            if image_source.startswith("data:"):
                image_source = image_source.split(",", 1)[1]
            image_data = base64.b64decode(image_source)
            return Image.open(io.BytesIO(image_data)).convert("RGB")

    def encode(
        self,
        inputs: list[str | dict[str, str]],
        task: str | None = None,
        dimensions: int | None = None,
        normalized: bool = True,
        prompt_name: str | None = None,
        late_chunking: bool = False,
        truncate: bool = True,
        return_multivector: bool = False,
        **kwargs: Any,
    ) -> tuple[Any, int]:
        """
        Encode text and/or images to embeddings using jina-embeddings-v4.
        
        Supports:
        - Multimodal input (text and images)
        - Dense embeddings (default) or multi-vector embeddings
        - Late chunking for contextual chunk embeddings
        - Task-specific adapters (retrieval, text-matching, code)
        
        Input formats:
        - Plain strings: "Hello world" (compatible mode)
        - Text objects: {"text": "Hello world"} (v4 mode)
        - Image objects: {"image": "https://..."} or {"image": "base64..."} (v4 mode)
        
        Args:
            inputs: List of text strings or dicts with text/image keys
            task: Task type (retrieval, text-matching, code)
            dimensions: Target embedding dimensions (128-2048)
            normalized: Whether to L2-normalize embeddings
            prompt_name: For retrieval task - "query" or "passage"
            late_chunking: Return contextual chunk embeddings
            truncate: Truncate inputs exceeding max length
            return_multivector: Return NxD multi-vector embeddings per input
            
        Returns:
            Tuple of (embeddings, token_count)
            - Standard: embeddings shape [batch, dims]
            - Multi-vector: embeddings is list of tensors [tokens, dims] per input
            - Late chunking: embeddings shape [total_chunks, dims]
        """
        if not self._loaded:
            raise RuntimeError(f"Model {self.model_id} is not loaded")

        # Separate text and image inputs, tracking original order
        text_inputs: list[str] = []
        image_inputs: list[str | Image.Image] = []
        input_order: list[tuple[str, int]] = []  # (type, index_in_respective_list)

        for inp in inputs:
            if isinstance(inp, str):
                # Compatible mode: plain strings are text
                text_inputs.append(inp)
                input_order.append(("text", len(text_inputs) - 1))
            elif isinstance(inp, dict):
                if "text" in inp:
                    text_inputs.append(inp["text"])
                    input_order.append(("text", len(text_inputs) - 1))
                elif "image" in inp:
                    image_inputs.append(inp["image"])
                    input_order.append(("image", len(image_inputs) - 1))
                else:
                    raise ValueError(f"Invalid input dict for {self.model_id}: {inp}")
            else:
                raise ValueError(f"Invalid input type for {self.model_id}: {type(inp)}")

        # Map task and extract prompt_name if embedded in task
        # Jina API uses "retrieval.query", "retrieval.passage", "code.query", "code.passage"
        # V4 model expects task="retrieval" + prompt_name="query"
        internal_task = task or "text-matching"
        effective_prompt_name = prompt_name
        
        if task and "." in task:
            # Extract prompt_name from task like "retrieval.query" -> ("retrieval", "query")
            parts = task.split(".", 1)
            internal_task = parts[0]  # "retrieval" or "code"
            if effective_prompt_name is None:
                effective_prompt_name = parts[1]  # "query" or "passage"
        else:
            # Map simple task names
            internal_task = self.TASK_MAP.get(task, task) if task else "text-matching"

        # Handle different output modes
        if return_multivector:
            return self._encode_multivector(
                text_inputs, image_inputs, input_order,
                internal_task, effective_prompt_name, dimensions, normalized
            )
        elif late_chunking and text_inputs:
            return self._encode_late_chunking(
                text_inputs, image_inputs, input_order,
                internal_task, effective_prompt_name, dimensions, normalized, truncate
            )
        else:
            return self._encode_standard(
                text_inputs, image_inputs, input_order,
                internal_task, effective_prompt_name, dimensions, normalized
            )

    def _encode_standard(
        self,
        text_inputs: list[str],
        image_inputs: list[str | Image.Image],
        input_order: list[tuple[str, int]],
        task: str,
        prompt_name: str | None,
        dimensions: int | None,
        normalized: bool,
    ) -> tuple[torch.Tensor, int]:
        """Standard encoding returning one embedding per input."""
        embeddings_list: list[torch.Tensor] = []

        with torch.no_grad():
            # Encode text inputs
            text_embeddings = None
            if text_inputs:
                logger.debug(f"Encoding {len(text_inputs)} text inputs with task={task}, prompt_name={prompt_name}")
                text_embeddings = self.model.encode_text(
                    texts=text_inputs,
                    task=task,
                    prompt_name=prompt_name,
                    truncate_dim=dimensions,
                )
                logger.debug(f"encode_text returned type={type(text_embeddings)}")
                # Convert to tensor if needed (handle various return types)
                if isinstance(text_embeddings, list):
                    # Model returns list of embeddings - stack them
                    import numpy as np
                    if len(text_embeddings) > 0:
                        if isinstance(text_embeddings[0], torch.Tensor):
                            text_embeddings = torch.stack(text_embeddings).to(self.device)
                        elif isinstance(text_embeddings[0], np.ndarray):
                            text_embeddings = torch.from_numpy(np.stack(text_embeddings)).to(self.device)
                        else:
                            # List of lists/floats
                            text_embeddings = torch.tensor(text_embeddings, device=self.device)
                    logger.debug(f"Converted list to tensor shape={text_embeddings.shape}")
                elif not isinstance(text_embeddings, torch.Tensor):
                    import numpy as np
                    if isinstance(text_embeddings, np.ndarray):
                        logger.debug(f"Converting numpy array shape={text_embeddings.shape}")
                        text_embeddings = torch.from_numpy(text_embeddings).to(self.device)
                    else:
                        logger.debug(f"Converting other type to tensor")
                        text_embeddings = torch.tensor(text_embeddings, device=self.device)
                else:
                    logger.debug(f"Already tensor, shape={text_embeddings.shape}")
                    text_embeddings = text_embeddings.to(self.device)
                logger.debug(f"text_embeddings final shape={text_embeddings.shape}")

            # Encode image inputs
            image_embeddings = None
            if image_inputs:
                loaded_images = []
                for img in image_inputs:
                    if isinstance(img, str):
                        loaded_images.append(self._load_image(img))
                    else:
                        loaded_images.append(img)

                image_embeddings = self.model.encode_image(
                    images=loaded_images,
                    task=task,
                    truncate_dim=dimensions,
                )
                # Convert to tensor if needed (handle various return types)
                if isinstance(image_embeddings, list):
                    import numpy as np
                    if len(image_embeddings) > 0:
                        if isinstance(image_embeddings[0], torch.Tensor):
                            image_embeddings = torch.stack(image_embeddings).to(self.device)
                        elif isinstance(image_embeddings[0], np.ndarray):
                            image_embeddings = torch.from_numpy(np.stack(image_embeddings)).to(self.device)
                        else:
                            image_embeddings = torch.tensor(image_embeddings, device=self.device)
                elif not isinstance(image_embeddings, torch.Tensor):
                    import numpy as np
                    if isinstance(image_embeddings, np.ndarray):
                        image_embeddings = torch.from_numpy(image_embeddings).to(self.device)
                    else:
                        image_embeddings = torch.tensor(image_embeddings, device=self.device)
                else:
                    image_embeddings = image_embeddings.to(self.device)

            # Reassemble embeddings in original order
            for inp_type, idx in input_order:
                if inp_type == "text" and text_embeddings is not None:
                    embeddings_list.append(text_embeddings[idx])
                elif inp_type == "image" and image_embeddings is not None:
                    embeddings_list.append(image_embeddings[idx])

        # Stack into single tensor
        embeddings = torch.stack(embeddings_list)

        # Handle dimension truncation (if not done by model)
        target_dims = dimensions or self.default_dimensions
        if target_dims < embeddings.shape[-1]:
            embeddings = embeddings[:, :target_dims]

        # L2 normalize if requested
        if normalized:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        # Calculate token count using heuristic (V4 tokenizer may not support simple encode)
        # V4 is based on Qwen2.5-VL which has a different tokenizer interface
        try:
            if hasattr(self.tokenizer, 'encode'):
                text_tokens = 0
                for text in text_inputs:
                    encoded = self.tokenizer.encode(text, add_special_tokens=False)
                    # Handle both list and tensor returns
                    if hasattr(encoded, '__len__'):
                        text_tokens += len(encoded)
                    else:
                        text_tokens += 1
            else:
                text_tokens = int(sum(len(text.split()) * 1.3 for text in text_inputs))
        except Exception:
            # Fallback to heuristic
            text_tokens = int(sum(len(text.split()) * 1.3 for text in text_inputs))
        image_tokens = len(image_inputs) * 576  # Approximate tokens per image
        token_count = text_tokens + image_tokens

        return embeddings, token_count

    def _encode_multivector(
        self,
        text_inputs: list[str],
        image_inputs: list[str | Image.Image],
        input_order: list[tuple[str, int]],
        task: str,
        prompt_name: str | None,
        dimensions: int | None,
        normalized: bool,
    ) -> tuple[list[torch.Tensor], int]:
        """
        Multi-vector encoding returning NxD embeddings per input.
        
        For each input, returns a tensor of shape [N, D] where N is the number
        of tokens in that input. Useful for late-interaction retrieval (ColBERT).
        """
        multivector_list: list[torch.Tensor] = []

        with torch.no_grad():
            # Encode text inputs with multi-vector
            text_multivectors = None
            if text_inputs:
                text_multivectors = self.model.encode_text(
                    texts=text_inputs,
                    task=task,
                    prompt_name=prompt_name,
                    return_multivector=True,
                )

            # Encode image inputs with multi-vector
            image_multivectors = None
            if image_inputs:
                loaded_images = []
                for img in image_inputs:
                    if isinstance(img, str):
                        loaded_images.append(self._load_image(img))
                    else:
                        loaded_images.append(img)

                image_multivectors = self.model.encode_image(
                    images=loaded_images,
                    task=task,
                    return_multivector=True,
                )

            # Reassemble in original order
            for inp_type, idx in input_order:
                if inp_type == "text" and text_multivectors is not None:
                    mv = text_multivectors[idx]
                    if not isinstance(mv, torch.Tensor):
                        mv = torch.tensor(mv, device=self.device)
                    multivector_list.append(mv)
                elif inp_type == "image" and image_multivectors is not None:
                    mv = image_multivectors[idx]
                    if not isinstance(mv, torch.Tensor):
                        mv = torch.tensor(mv, device=self.device)
                    multivector_list.append(mv)

        # Normalize each multi-vector if requested
        if normalized:
            multivector_list = [F.normalize(mv, p=2, dim=-1) for mv in multivector_list]

        # Calculate token count using heuristic (V4 tokenizer may not support simple encode)
        try:
            if hasattr(self.tokenizer, 'encode'):
                text_tokens = 0
                for text in text_inputs:
                    encoded = self.tokenizer.encode(text, add_special_tokens=False)
                    if hasattr(encoded, '__len__'):
                        text_tokens += len(encoded)
                    else:
                        text_tokens += 1
            else:
                text_tokens = int(sum(len(text.split()) * 1.3 for text in text_inputs))
        except Exception:
            text_tokens = int(sum(len(text.split()) * 1.3 for text in text_inputs))
        image_tokens = len(image_inputs) * 576
        token_count = text_tokens + image_tokens

        return multivector_list, token_count

    def _encode_late_chunking(
        self,
        text_inputs: list[str],
        image_inputs: list[str | Image.Image],
        input_order: list[tuple[str, int]],
        task: str,
        prompt_name: str | None,
        dimensions: int | None,
        normalized: bool,
        truncate: bool,
    ) -> tuple[torch.Tensor, int]:
        """
        Late chunking: concatenate all text inputs, process through transformer, then pool per input.
        
        This matches Jina's official late_chunking behavior:
        1. Concatenate ALL text inputs into one long document
        2. Pass entire concatenated text through transformer (full context)
        3. Split token embeddings back by original input boundaries
        4. Mean pool each input's tokens to get one embedding per input
        
        Each text input's embedding contains contextual information from ALL other text inputs.
        Images are processed separately (no late chunking for images).
        
        Returns embeddings in original input order.
        """
        from app.late_chunking import late_chunking_pooling, ChunkSpan

        total_tokens = 0
        text_embeddings_list = []
        
        # Process text inputs with late chunking (concatenated)
        if text_inputs:
            # Step 1: Concatenate all texts with separator
            separator = "\n\n"
            concatenated_text = separator.join(text_inputs)
            
            # Step 2: Get token-level embeddings using VLM's last hidden states
            # IMPORTANT: We use output_vlm_last_hidden_states=True to get 2048-dim token embeddings
            # NOT return_multivector=True which returns 128-dim embeddings for late-interaction retrieval
            # See: https://huggingface.co/jinaai/jina-embeddings-v4/discussions/68
            with torch.no_grad():
                # Use the model's processor to tokenize with proper prefix
                prefix = "Query" if prompt_name == "query" else "Passage"
                processed = self.model.processor.process_texts(
                    [concatenated_text],
                    max_length=self.max_tokens if truncate else None,
                    prefix=prefix,
                )
                batch = {k: v.to(self.device) for k, v in processed.items()}
                
                # Get VLM's last hidden states (2048-dim per token)
                outputs = self.model(
                    **batch,
                    task_label=task,
                    output_vlm_last_hidden_states=True,
                )
                
                # vlm_last_hidden_states shape: [batch, seq_len, 2048]
                token_embeddings = outputs.vlm_last_hidden_states[0]  # Remove batch dim
                    
                if not isinstance(token_embeddings, torch.Tensor):
                    token_embeddings = torch.tensor(token_embeddings, device=self.device)
            
            total_tokens += token_embeddings.shape[0]
            
            # Step 3: Find EXACT token boundaries using offset mapping
            # This matches Jina's official API behavior
            full_text = f"{prefix}: {concatenated_text}"
            encoded = self.tokenizer(
                full_text, 
                return_offsets_mapping=True, 
                add_special_tokens=False
            )
            offset_mapping = encoded["offset_mapping"]
            
            # Calculate character positions for each original text
            prefix_len = len(f"{prefix}: ")
            char_ranges = []
            current_pos = prefix_len
            for text in text_inputs:
                start_char = current_pos
                end_char = current_pos + len(text)
                char_ranges.append((start_char, end_char))
                current_pos = end_char + len(separator)
            
            # Map character ranges to token ranges
            chunk_spans = []
            for i, (start_char, end_char) in enumerate(char_ranges):
                # Find first token that overlaps with this text
                start_token = None
                end_token = len(offset_mapping)
                
                for t, (tok_start, tok_end) in enumerate(offset_mapping):
                    # Token overlaps with text start
                    if tok_end > start_char and start_token is None:
                        start_token = t
                    # Token starts at or after text end
                    if tok_start >= end_char:
                        end_token = t
                        break
                
                # Handle edge case where start_token wasn't found
                if start_token is None:
                    start_token = 0
                
                # Include trailing separator token if it contains part of the text
                # (e.g., ".\n\n" is one token that includes both period and separator)
                if end_token < len(offset_mapping):
                    tok_start, tok_end = offset_mapping[end_token]
                    # If this token starts before our text ends, include it
                    if tok_start < end_char:
                        end_token += 1
                
                chunk_spans.append(ChunkSpan(
                    start_token=start_token,
                    end_token=min(end_token, token_embeddings.shape[0]),
                    text=text_inputs[i],
                ))
            
            # Step 4: Apply late chunking pooling - one embedding per original text input
            attention_mask = torch.ones(token_embeddings.shape[0], device=self.device)
            text_chunk_embeddings = late_chunking_pooling(
                token_embeddings,
                attention_mask,
                chunk_spans,
            )
            
            # Store embeddings for each text input
            for i in range(len(text_inputs)):
                if i < text_chunk_embeddings.shape[0]:
                    text_embeddings_list.append(text_chunk_embeddings[i])
                else:
                    # Fallback: use last embedding if we ran out
                    text_embeddings_list.append(text_chunk_embeddings[-1])

        # Process image inputs (no late chunking for images)
        image_embeddings_list = []
        if image_inputs:
            loaded_images = []
            for img in image_inputs:
                if isinstance(img, str):
                    loaded_images.append(self._load_image(img))
                else:
                    loaded_images.append(img)

            with torch.no_grad():
                image_embeddings = self.model.encode_image(
                    images=loaded_images,
                    task=task,
                    truncate_dim=dimensions,
                )
                if not isinstance(image_embeddings, torch.Tensor):
                    image_embeddings = torch.tensor(image_embeddings, device=self.device)

            for i in range(len(image_inputs)):
                image_embeddings_list.append(image_embeddings[i])
            
            total_tokens += len(image_inputs) * 576

        # Reassemble in original order
        all_embeddings = []
        for inp_type, idx in input_order:
            if inp_type == "text" and idx < len(text_embeddings_list):
                all_embeddings.append(text_embeddings_list[idx])
            elif inp_type == "image" and idx < len(image_embeddings_list):
                all_embeddings.append(image_embeddings_list[idx])

        # Stack into tensor
        if all_embeddings:
            embeddings = torch.stack(all_embeddings)
        else:
            embeddings = torch.empty(0, dimensions or self.default_dimensions, device=self.device)

        # Dimension truncation
        if dimensions and dimensions < embeddings.shape[-1]:
            embeddings = embeddings[:, :dimensions]

        # Normalize
        if normalized:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings, int(total_tokens)

    def unload(self) -> None:
        """Unload model and cleanup resources."""
        if hasattr(self, "_http_client"):
            self._http_client.close()
        super().unload()
