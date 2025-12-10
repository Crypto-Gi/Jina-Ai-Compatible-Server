"""
Jina Local API Server - Pydantic Schemas for Models Listing
==========================================================
Request and response models for the /v1/models endpoint.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Information about a single model."""

    id: str = Field(..., description="Model identifier")

    object: Literal["model"] = Field(
        default="model", description="Object type, always 'model'"
    )

    created: int = Field(..., description="Unix timestamp of model creation/availability")

    owned_by: str = Field(
        default="jinaai", description="Organization that owns the model"
    )

    type: Literal["embedding", "reranker"] = Field(
        ..., description="Type of model"
    )

    # Additional metadata
    max_tokens: int | None = Field(
        default=None, description="Maximum input tokens supported"
    )

    dimensions: int | None = Field(
        default=None, description="Default embedding dimensions (embedding models only)"
    )

    supports_multimodal: bool = Field(
        default=False, description="Whether model supports image inputs"
    )

    tasks: list[str] | None = Field(
        default=None, description="Supported task types"
    )


class ModelsListResponse(BaseModel):
    """Response body for GET /v1/models."""

    object: Literal["list"] = Field(
        default="list", description="Object type, always 'list'"
    )

    data: list[ModelInfo] = Field(..., description="List of available models")
