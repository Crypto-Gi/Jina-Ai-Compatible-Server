"""
Jina Local API Server - Pydantic Schemas for Embeddings
=======================================================
Request and response models for the /v1/embeddings endpoint.
Compatible with the official Jina API.
"""

from typing import Any, Literal, Union

from pydantic import BaseModel, Field


# ============================================
# Input Types
# ============================================


class TextInput(BaseModel):
    """Text input for embedding."""

    text: str = Field(..., description="Text content to embed")


class ImageInput(BaseModel):
    """Image input for multimodal embedding (v4 only)."""

    image: str = Field(
        ..., description="Image URL or base64-encoded image data"
    )


# Union type for flexible input handling
InputItem = Union[str, TextInput, ImageInput]


# ============================================
# Request Schema
# ============================================


class EmbeddingsRequest(BaseModel):
    """Request body for POST /v1/embeddings - Jina API compatible."""

    model: str = Field(
        ...,
        description="Model ID to use for embedding",
        examples=["jina-embeddings-v3", "jina-embeddings-v4"],
    )

    input: Union[str, list[InputItem]] = Field(
        ...,
        description="Text(s) or multimodal item(s) to embed. Can be a single string, "
        "list of strings, or list of objects with 'text' or 'image' keys.",
        examples=[
            "Hello world",
            ["Hello", "World"],
            [{"text": "Hello"}, {"image": "https://example.com/image.jpg"}],
        ],
    )

    # Embedding options
    # Note: Jina API does not expose a 'normalized' parameter - embeddings are always normalized.
    # We keep this for internal use but it's not part of the official Jina API.
    normalized: bool = Field(
        default=True,
        description="Internal option: Whether to L2-normalize embeddings. "
        "Jina models always return normalized embeddings by default.",
        exclude=True,  # Don't include in OpenAPI schema
    )

    embedding_type: Literal["float", "binary", "base64", "ubinary"] = Field(
        default="float",
        description="Format of returned embeddings: float array, binary (signed 0/1), "
        "base64 (float32 packed), or ubinary (unsigned binary, not yet implemented)",
    )

    late_chunking: bool = Field(
        default=False,
        description="Whether to perform late chunking for long texts. If true, "
        "the model will chunk long texts internally and average the embeddings.",
    )

    truncate: bool = Field(
        default=False,
        description="Whether to truncate texts that exceed the model's maximum input length. "
        "Defaults to false per Jina API spec.",
    )

    # Model-specific options
    task: str | None = Field(
        default=None,
        description="Task-specific embedding variant. Model-dependent: "
        "v3: retrieval.query, retrieval.passage, text-matching, etc. "
        "v4: retrieval, text-matching, code. "
        "code: nl2code, code2code, qa, etc.",
    )

    dimensions: int | None = Field(
        default=None,
        description="Target embedding dimensions (for MRL truncation). "
        "If not specified, uses model's default dimensions.",
        ge=1,
    )

    prompt_name: Literal["query", "passage"] | None = Field(
        default=None,
        description="Prompt variant for v4 retrieval task",
    )

    return_multivector: bool = Field(
        default=False,
        description="For v4 only: Return multi-vector embeddings (NxD where N is token count). "
        "Useful for late-interaction style retrieval like ColBERT.",
    )


# ============================================
# Response Schema
# ============================================


class EmbeddingData(BaseModel):
    """Single embedding in the response."""

    object: Literal["embedding"] = Field(
        default="embedding", description="Object type, always 'embedding'"
    )

    index: int = Field(..., description="Index of the input this embedding corresponds to")

    embedding: Union[list[float], list[int], str] = Field(
        ...,
        description="The embedding vector. Format depends on embedding_type: "
        "float→list[float], binary→list[int], base64→str",
    )


class EmbeddingsUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = Field(..., description="Number of tokens in the input")
    total_tokens: int = Field(..., description="Total tokens processed")


class EmbeddingsResponse(BaseModel):
    """Response body for POST /v1/embeddings - Jina API compatible."""

    object: Literal["list"] = Field(
        default="list", description="Object type, always 'list'"
    )

    model: str = Field(..., description="Model ID used for embedding")

    data: list[EmbeddingData] = Field(
        ..., description="List of embeddings, one per input"
    )

    usage: EmbeddingsUsage = Field(..., description="Token usage statistics")


# ============================================
# Error Response
# ============================================


class ErrorDetail(BaseModel):
    """Error detail object."""

    message: str = Field(..., description="Error message")
    type: str = Field(default="invalid_request_error", description="Error type")
    param: str | None = Field(default=None, description="Parameter that caused error")
    code: str | None = Field(default=None, description="Error code")


class ErrorResponse(BaseModel):
    """Error response format."""

    error: ErrorDetail
