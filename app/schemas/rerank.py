"""
Jina Local API Server - Pydantic Schemas for Reranking
======================================================
Request and response models for the /v1/rerank endpoint.
Compatible with the official Jina API.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


# ============================================
# Request Schema
# ============================================


class RerankRequest(BaseModel):
    """Request body for POST /v1/rerank - Jina API compatible."""

    model: str = Field(
        default="jina-reranker-v3",
        description="Model ID to use for reranking",
        examples=["jina-reranker-v3"],
    )

    query: str = Field(
        ...,
        description="The search query to rank documents against",
        min_length=1,
    )

    documents: list[str | dict[str, Any]] = Field(
        ...,
        description="List of documents to rerank (strings or objects with 'text' field)",
        min_length=1,
    )

    top_n: int | None = Field(
        default=None,
        description="Return only top N documents. If not specified, returns all documents.",
        ge=1,
    )

    return_documents: bool = Field(
        default=True,
        description="Whether to include document text in the response. "
        "Defaults to true per Jina API spec.",
    )


# ============================================
# Response Schema
# ============================================


class DocumentObject(BaseModel):
    """Document object in rerank response."""

    text: str = Field(..., description="The document text")


class RerankResult(BaseModel):
    """Single result in the rerank response."""

    index: int = Field(
        ..., description="Index of the document in the original input list"
    )

    relevance_score: float = Field(
        ..., description="Relevance score (higher = more relevant)"
    )

    document: DocumentObject | None = Field(
        default=None,
        description="Document object (only included if return_documents=true)",
    )


class RerankUsage(BaseModel):
    """Token usage statistics for reranking."""

    total_tokens: int = Field(..., description="Total tokens processed")


class RerankResponse(BaseModel):
    """Response body for POST /v1/rerank - Jina API compatible."""

    model: str = Field(..., description="Model ID used for reranking")

    results: list[RerankResult] = Field(
        ...,
        description="Reranked documents sorted by relevance_score (highest first)",
    )

    usage: RerankUsage = Field(..., description="Token usage statistics")
