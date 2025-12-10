"""
Jina Local API Server - Models API Endpoint
==========================================
GET /v1/models - List available models.
"""

import time

from fastapi import APIRouter, Request

from app.logging_config import get_logger
from app.schemas.models import ModelInfo, ModelsListResponse

logger = get_logger(__name__)

router = APIRouter(tags=["Models"])

# Model creation timestamps (approximate release dates)
MODEL_CREATED_TIMESTAMPS = {
    "jina-embeddings-v3": 1696118400,   # Oct 2023
    "jina-embeddings-v4": 1717200000,   # Jun 2024
    "jina-code-embeddings-0.5b": 1730000000,  # Oct 2024
    "jina-code-embeddings-1.5b": 1730000000,  # Oct 2024
    "jina-reranker-v3": 1730000000,     # Oct 2024
}


@router.get("/models", response_model=ModelsListResponse)
async def list_models(request: Request) -> ModelsListResponse:
    """
    List all available models.
    
    Returns information about each loaded model including:
    - Model ID
    - Model type (embedding or reranker)
    - Capabilities (max tokens, dimensions, multimodal support)
    """
    registry = request.app.state.registry
    
    models_data = []
    
    for model_id in registry.loaded_models():
        model = registry.get_model(model_id)
        config = registry.get_model_info(model_id) or {}
        
        # Determine model type
        model_type = getattr(model, "model_type", "embedding")
        
        # Get model properties
        max_tokens = getattr(model, "max_tokens", config.get("max_tokens"))
        dimensions = getattr(model, "default_dimensions", config.get("default_dimensions"))
        supports_multimodal = getattr(model, "supports_multimodal", config.get("supports_multimodal", False))
        tasks = config.get("tasks")
        
        model_info = ModelInfo(
            id=model_id,
            object="model",
            created=MODEL_CREATED_TIMESTAMPS.get(model_id, int(time.time())),
            owned_by="jinaai",
            type=model_type,
            max_tokens=max_tokens,
            dimensions=dimensions,
            supports_multimodal=supports_multimodal,
            tasks=tasks,
        )
        models_data.append(model_info)
    
    logger.debug("Models list requested", model_count=len(models_data))
    
    return ModelsListResponse(
        object="list",
        data=models_data,
    )
