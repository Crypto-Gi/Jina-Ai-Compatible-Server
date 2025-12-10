"""Jina Local API Server - Schemas Package"""

from app.schemas.embeddings import (
    EmbeddingData,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingsUsage,
    ErrorDetail,
    ErrorResponse,
    ImageInput,
    InputItem,
    TextInput,
)
from app.schemas.models import ModelInfo, ModelsListResponse
from app.schemas.rerank import (
    DocumentObject,
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankUsage,
)

__all__ = [
    # Embeddings
    "EmbeddingsRequest",
    "EmbeddingsResponse",
    "EmbeddingData",
    "EmbeddingsUsage",
    "TextInput",
    "ImageInput",
    "InputItem",
    "ErrorDetail",
    "ErrorResponse",
    # Rerank
    "RerankRequest",
    "RerankResponse",
    "RerankResult",
    "RerankUsage",
    "DocumentObject",
    # Models
    "ModelInfo",
    "ModelsListResponse",
]
