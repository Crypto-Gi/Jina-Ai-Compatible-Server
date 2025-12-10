"""
Jina Local API Server - Embeddings API Endpoint
===============================================
POST /v1/embeddings - Generate embeddings compatible with Jina API.
"""

import base64
import struct
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.logging_config import get_logger
from app.models.base import EmbeddingModelWrapper
from app.schemas.embeddings import (
    EmbeddingData,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingsUsage,
    ErrorResponse,
)

logger = get_logger(__name__)

router = APIRouter(tags=["Embeddings"])


def _normalize_inputs(payload: EmbeddingsRequest) -> list[str | dict[str, str]]:
    """
    Normalize input to a list of items.
    
    Handles:
    - Single string
    - List of strings
    - List of dicts with 'text' or 'image' keys
    """
    inp = payload.input
    
    if isinstance(inp, str):
        return [inp]
    
    # List input
    result: list[str | dict[str, str]] = []
    for item in inp:
        if isinstance(item, str):
            result.append(item)
        elif hasattr(item, "text"):
            result.append({"text": item.text})
        elif hasattr(item, "image"):
            result.append({"image": item.image})
        elif isinstance(item, dict):
            result.append(item)
        else:
            raise ValueError(f"Invalid input item type: {type(item)}")
    
    return result


def _format_embedding(
    embedding: Any, embedding_type: str
) -> list[float] | list[int] | str:
    """
    Format embedding according to requested type.
    
    Args:
        embedding: Tensor or list of embedding values
        embedding_type: 'float', 'binary', or 'base64'
        
    Returns:
        Formatted embedding value
    """
    # Convert to list if tensor
    if hasattr(embedding, "tolist"):
        values = embedding.tolist()
    else:
        values = list(embedding)

    if embedding_type == "base64":
        # Pack as float32 array and encode to base64
        packed = struct.pack(f"{len(values)}f", *values)
        return base64.b64encode(packed).decode("utf-8")
    
    elif embedding_type == "binary":
        # Binary quantization: 1 if positive, 0 otherwise (signed)
        return [1 if v > 0 else 0 for v in values]
    
    elif embedding_type == "ubinary":
        # Unsigned binary - not yet implemented
        # TODO: Implement uint8 bit-packed format matching Jina's spec
        raise ValueError(
            "embedding_type 'ubinary' is not yet supported by this local server. "
            "Use 'float', 'base64', or 'binary' instead."
        )
    
    else:  # float
        return values


@router.post(
    "/embeddings",
    response_model=EmbeddingsResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Model not found"},
        422: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)
async def create_embeddings(request: Request, payload: EmbeddingsRequest) -> EmbeddingsResponse:
    """
    Generate embeddings for the given input(s).
    
    Compatible with the Jina AI Embeddings API.
    
    Supports:
    - Text embeddings with jina-embeddings-v3
    - Multimodal embeddings (text + image) with jina-embeddings-v4
    - Code embeddings with jina-code-embeddings-0.5b/1.5b
    """
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.info(
        "Embeddings request received",
        request_id=request_id,
        model=payload.model,
        input_count=len(payload.input) if isinstance(payload.input, list) else 1,
    )

    # Get model from registry
    registry = request.app.state.registry
    model = registry.get_embedding_model(payload.model)
    
    if model is None:
        logger.warning("Model not found", model=payload.model)
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Model '{payload.model}' not found or not loaded",
                    "type": "invalid_request_error",
                    "param": "model",
                }
            },
        )

    # Validate multimodal input
    inputs = _normalize_inputs(payload)
    has_images = any(isinstance(i, dict) and "image" in i for i in inputs)
    
    if has_images and not model.supports_multimodal:
        raise HTTPException(
            status_code=422,
            detail={
                "error": {
                    "message": f"Model '{payload.model}' does not support image inputs",
                    "type": "invalid_request_error",
                    "param": "input",
                }
            },
        )

    try:
        # Check if this is a multi-vector request (v4 only)
        is_multivector = (
            payload.model == "jina-embeddings-v4" 
            and getattr(payload, "return_multivector", False)
        )

        # Generate embeddings
        embeddings, token_count = model.encode(
            inputs=inputs,
            task=payload.task,
            dimensions=payload.dimensions,
            normalized=payload.normalized,
            prompt_name=payload.prompt_name,
            late_chunking=payload.late_chunking,
            truncate=payload.truncate,
            return_multivector=getattr(payload, "return_multivector", False),
        )

        # Format response
        data = []
        
        if is_multivector and isinstance(embeddings, list):
            # Multi-vector response: list of [N, D] tensors
            # Each input gets one entry with N vectors (one per token)
            for i, mv_embedding in enumerate(embeddings):
                # For multi-vector, we return a list of embeddings per input
                # Each is a list of floats (flattened) or nested list
                if hasattr(mv_embedding, "tolist"):
                    formatted_emb = mv_embedding.tolist()
                else:
                    formatted_emb = list(mv_embedding)
                data.append(
                    EmbeddingData(
                        object="embedding",
                        index=i,
                        embedding=formatted_emb,  # This will be [[float, ...], [float, ...], ...]
                    )
                )
        else:
            # Standard response: [batch, D] tensor
            for i in range(len(embeddings)):
                formatted_emb = _format_embedding(embeddings[i], payload.embedding_type)
                data.append(
                    EmbeddingData(
                        object="embedding",
                        index=i,
                        embedding=formatted_emb,
                    )
                )

        response = EmbeddingsResponse(
            object="list",
            model=payload.model,
            data=data,
            usage=EmbeddingsUsage(
                prompt_tokens=token_count,
                total_tokens=token_count,
            ),
        )

        # Get dimension info for logging
        if is_multivector and isinstance(embeddings, list) and len(embeddings) > 0:
            dims = embeddings[0].shape[-1] if hasattr(embeddings[0], 'shape') else 0
            count = sum(e.shape[0] if hasattr(e, 'shape') else 1 for e in embeddings)
        else:
            dims = len(embeddings[0]) if len(embeddings) > 0 else 0
            count = len(data)

        logger.info(
            "Embeddings generated",
            request_id=request_id,
            model=payload.model,
            count=count,
            dimensions=dims,
            tokens=token_count,
            multivector=is_multivector,
        )

        return response

    except Exception as e:
        logger.error(
            "Embeddings generation failed",
            request_id=request_id,
            model=payload.model,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Embedding generation failed: {str(e)}",
                    "type": "internal_error",
                }
            },
        )
