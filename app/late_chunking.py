"""
Jina Local API Server - Late Chunking Module
=============================================
Implements late chunking for contextual chunk embeddings.

Late chunking processes the entire document through the transformer first,
then applies mean pooling to each chunk span to preserve full document context.

Reference: https://jina.ai/news/late-chunking-in-long-context-embedding-models/
"""

import re
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoTokenizer

from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkSpan:
    """Represents a chunk with its token span boundaries."""
    
    start_token: int  # Inclusive start index in token sequence
    end_token: int    # Exclusive end index in token sequence
    text: str         # Original text of the chunk


def chunk_by_sentences(
    text: str,
    tokenizer: AutoTokenizer,
    max_chunk_tokens: int = 256,
) -> list[ChunkSpan]:
    """
    Split text into sentence-based chunks and compute token spans.
    
    This uses a simple sentence boundary regex and groups sentences
    into chunks that don't exceed max_chunk_tokens.
    
    Args:
        text: Full document text
        tokenizer: HuggingFace tokenizer
        max_chunk_tokens: Maximum tokens per chunk
        
    Returns:
        List of ChunkSpan objects with token boundaries
    """
    # Split by sentence boundaries (simple heuristic)
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    # Tokenize the full text to get token offsets
    # We need to track which tokens correspond to which characters
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=False,
    )
    
    tokens = encoding['input_ids']
    offset_mapping = encoding.get('offset_mapping', [])
    
    if not offset_mapping:
        # Fallback: tokenize each sentence separately
        return _chunk_sentences_fallback(sentences, tokenizer, max_chunk_tokens)
    
    # Build chunks by grouping sentences
    chunks: list[ChunkSpan] = []
    current_sentences: list[str] = []
    current_char_start = 0
    
    for sentence in sentences:
        # Find where this sentence starts in the original text
        sent_start = text.find(sentence, current_char_start)
        sent_end = sent_start + len(sentence)
        
        # Find token indices for this sentence
        token_start = None
        token_end = None
        
        for idx, (char_start, char_end) in enumerate(offset_mapping):
            if char_start is None:  # Special tokens
                continue
            if token_start is None and char_start >= sent_start:
                token_start = idx
            if char_end <= sent_end:
                token_end = idx + 1
        
        if token_start is None:
            token_start = 0
        if token_end is None:
            token_end = len(tokens)
        
        sent_tokens = token_end - token_start
        current_tokens = sum(
            len(tokenizer.encode(s, add_special_tokens=False)) 
            for s in current_sentences
        )
        
        if current_tokens + sent_tokens > max_chunk_tokens and current_sentences:
            # Finalize current chunk
            chunk_text = ' '.join(current_sentences)
            chunk_start = text.find(current_sentences[0], current_char_start if chunks else 0)
            chunk_end = chunk_start + len(chunk_text)
            
            # Find token boundaries for the chunk
            chunk_token_start = None
            chunk_token_end = None
            for idx, (char_start, char_end) in enumerate(offset_mapping):
                if char_start is None:
                    continue
                if chunk_token_start is None and char_start >= chunk_start:
                    chunk_token_start = idx
                if char_end <= chunk_end:
                    chunk_token_end = idx + 1
            
            if chunk_token_start is not None and chunk_token_end is not None:
                chunks.append(ChunkSpan(
                    start_token=chunk_token_start,
                    end_token=chunk_token_end,
                    text=chunk_text,
                ))
            
            current_sentences = [sentence]
            current_char_start = sent_start
        else:
            current_sentences.append(sentence)
        
        current_char_start = sent_end
    
    # Add remaining sentences as final chunk
    if current_sentences:
        chunk_text = ' '.join(current_sentences)
        chunk_start = text.find(current_sentences[0])
        chunk_end = chunk_start + len(chunk_text)
        
        chunk_token_start = None
        chunk_token_end = None
        for idx, (char_start, char_end) in enumerate(offset_mapping):
            if char_start is None:
                continue
            if chunk_token_start is None and char_start >= chunk_start:
                chunk_token_start = idx
            if char_end <= chunk_end:
                chunk_token_end = idx + 1
        
        if chunk_token_start is not None and chunk_token_end is not None:
            chunks.append(ChunkSpan(
                start_token=chunk_token_start,
                end_token=chunk_token_end,
                text=chunk_text,
            ))
    
    return chunks


def _chunk_sentences_fallback(
    sentences: list[str],
    tokenizer: AutoTokenizer,
    max_chunk_tokens: int,
) -> list[ChunkSpan]:
    """Fallback chunking when offset mapping is not available."""
    chunks: list[ChunkSpan] = []
    current_sentences: list[str] = []
    current_token_count = 0
    token_pos = 1  # Skip [CLS] token
    
    for sentence in sentences:
        sent_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))
        
        if current_token_count + sent_tokens > max_chunk_tokens and current_sentences:
            # Finalize chunk
            chunk_text = ' '.join(current_sentences)
            chunk_tokens = len(tokenizer.encode(chunk_text, add_special_tokens=False))
            chunks.append(ChunkSpan(
                start_token=token_pos,
                end_token=token_pos + chunk_tokens,
                text=chunk_text,
            ))
            token_pos += chunk_tokens
            current_sentences = [sentence]
            current_token_count = sent_tokens
        else:
            current_sentences.append(sentence)
            current_token_count += sent_tokens
    
    # Add remaining
    if current_sentences:
        chunk_text = ' '.join(current_sentences)
        chunk_tokens = len(tokenizer.encode(chunk_text, add_special_tokens=False))
        chunks.append(ChunkSpan(
            start_token=token_pos,
            end_token=token_pos + chunk_tokens,
            text=chunk_text,
        ))
    
    return chunks


def late_chunking_pooling(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    chunk_spans: list[ChunkSpan],
) -> torch.Tensor:
    """
    Apply mean pooling to each chunk span of token embeddings.
    
    This is the core of late chunking: after getting contextualized token
    embeddings from the full document, we pool each chunk's tokens separately.
    
    Args:
        token_embeddings: [seq_len, hidden_dim] or [batch, seq_len, hidden_dim]
        attention_mask: [seq_len] or [batch, seq_len]
        chunk_spans: List of ChunkSpan objects defining token boundaries
        
    Returns:
        Tensor of shape [num_chunks, hidden_dim] with pooled chunk embeddings
    """
    # Handle both batched and non-batched inputs
    if token_embeddings.dim() == 2:
        token_embeddings = token_embeddings.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
    
    batch_size = token_embeddings.shape[0]
    hidden_dim = token_embeddings.shape[-1]
    num_chunks = len(chunk_spans)
    
    # For batch size > 1, we assume same chunking for all items
    # (typically late chunking is used with batch_size=1 for a single long doc)
    
    chunk_embeddings = torch.zeros(
        batch_size, num_chunks, hidden_dim,
        device=token_embeddings.device,
        dtype=token_embeddings.dtype,
    )
    
    for chunk_idx, span in enumerate(chunk_spans):
        start = span.start_token
        end = min(span.end_token, token_embeddings.shape[1])
        
        if start >= end:
            continue
        
        # Extract chunk tokens
        chunk_tokens = token_embeddings[:, start:end, :]  # [batch, chunk_len, hidden]
        chunk_mask = attention_mask[:, start:end]  # [batch, chunk_len]
        
        # Mean pooling with attention mask
        mask_expanded = chunk_mask.unsqueeze(-1).expand(chunk_tokens.size()).float()
        sum_embeddings = torch.sum(chunk_tokens * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        chunk_embeddings[:, chunk_idx, :] = sum_embeddings / sum_mask
    
    # Remove batch dimension if input was unbatched
    if batch_size == 1:
        chunk_embeddings = chunk_embeddings.squeeze(0)
    
    return chunk_embeddings


def process_late_chunking(
    text: str,
    model: Any,
    tokenizer: AutoTokenizer,
    device: str,
    task: str = "text-matching",
    max_length: int = 8192,
    max_chunk_tokens: int = 256,
    normalized: bool = True,
) -> tuple[torch.Tensor, list[str], int]:
    """
    Full late chunking pipeline for a single document.
    
    Args:
        text: Full document text
        model: The embedding model (must support forward pass with attention mask)
        tokenizer: HuggingFace tokenizer
        device: Device to run on
        task: Task name for task-specific models
        max_length: Maximum total tokens
        max_chunk_tokens: Maximum tokens per chunk
        normalized: Whether to L2-normalize output embeddings
        
    Returns:
        Tuple of (chunk_embeddings, chunk_texts, total_tokens)
    """
    import torch.nn.functional as F
    
    # Step 1: Chunk the text and get spans
    chunks = chunk_by_sentences(text, tokenizer, max_chunk_tokens)
    
    if not chunks:
        # Fallback: treat entire text as one chunk
        chunks = [ChunkSpan(start_token=0, end_token=max_length, text=text)]
    
    # Step 2: Tokenize the full document
    encoding = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}
    
    total_tokens = encoding["attention_mask"].sum().item()
    
    # Step 3: Get token-level embeddings from the model
    with torch.no_grad():
        # Different models have different forward signatures
        if hasattr(model, 'forward'):
            # Try to pass task parameter if model supports it
            try:
                outputs = model(**encoding, task=task)
            except TypeError:
                outputs = model(**encoding)
        else:
            outputs = model(**encoding)
        
        # Get last hidden state (token embeddings)
        if hasattr(outputs, 'last_hidden_state'):
            token_embeddings = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            token_embeddings = outputs[0]
        else:
            token_embeddings = outputs
    
    # Step 4: Apply late chunking pooling
    chunk_embeddings = late_chunking_pooling(
        token_embeddings.squeeze(0),
        encoding["attention_mask"].squeeze(0),
        chunks,
    )
    
    # Step 5: Normalize if requested
    if normalized:
        chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=-1)
    
    chunk_texts = [c.text for c in chunks]
    
    return chunk_embeddings, chunk_texts, int(total_tokens)
