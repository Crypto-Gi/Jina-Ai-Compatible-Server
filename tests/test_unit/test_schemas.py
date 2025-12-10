"""
Unit Tests - Pydantic Schemas
=============================
Tests for request/response schema validation.
"""

import pytest
from pydantic import ValidationError

from app.schemas.embeddings import (
    EmbeddingData,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingsUsage,
    ImageInput,
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


class TestEmbeddingsSchemas:
    """Test embeddings request/response schemas."""

    def test_embeddings_request_string_input(self):
        """Test request with single string input."""
        req = EmbeddingsRequest(
            model="jina-embeddings-v3",
            input="Hello world",
        )
        assert req.model == "jina-embeddings-v3"
        assert req.input == "Hello world"
        assert req.normalized is True  # default
        assert req.embedding_type == "float"  # default

    def test_embeddings_request_list_input(self):
        """Test request with list of strings."""
        req = EmbeddingsRequest(
            model="jina-embeddings-v3",
            input=["Hello", "World"],
        )
        assert len(req.input) == 2

    def test_embeddings_request_with_options(self):
        """Test request with all options."""
        req = EmbeddingsRequest(
            model="jina-embeddings-v4",
            input=["Hello"],
            normalized=False,
            embedding_type="base64",
            task="retrieval",
            dimensions=512,
            prompt_name="query",
        )
        assert req.normalized is False
        assert req.embedding_type == "base64"
        assert req.task == "retrieval"
        assert req.dimensions == 512
        assert req.prompt_name == "query"

    def test_embeddings_request_invalid_embedding_type(self):
        """Test request fails with invalid embedding_type."""
        with pytest.raises(ValidationError):
            EmbeddingsRequest(
                model="jina-embeddings-v3",
                input="Hello",
                embedding_type="invalid",
            )

    def test_embeddings_request_multimodal_input(self):
        """Test request with multimodal input objects."""
        req = EmbeddingsRequest(
            model="jina-embeddings-v4",
            input=[
                TextInput(text="Hello"),
                ImageInput(image="https://example.com/image.jpg"),
            ],
        )
        assert len(req.input) == 2

    def test_embeddings_response(self):
        """Test embeddings response structure."""
        resp = EmbeddingsResponse(
            object="list",
            model="jina-embeddings-v3",
            data=[
                EmbeddingData(
                    object="embedding",
                    index=0,
                    embedding=[0.1, 0.2, 0.3],
                ),
            ],
            usage=EmbeddingsUsage(prompt_tokens=5, total_tokens=5),
        )
        assert resp.object == "list"
        assert resp.model == "jina-embeddings-v3"
        assert len(resp.data) == 1
        assert resp.data[0].index == 0
        assert len(resp.data[0].embedding) == 3


class TestRerankSchemas:
    """Test rerank request/response schemas."""

    def test_rerank_request_minimal(self):
        """Test minimal rerank request."""
        req = RerankRequest(
            query="What is Python?",
            documents=["Python is a programming language", "Java is also a language"],
        )
        assert req.model == "jina-reranker-v3"  # default
        assert req.query == "What is Python?"
        assert len(req.documents) == 2
        assert req.top_n is None
        assert req.return_documents is False

    def test_rerank_request_with_options(self):
        """Test rerank request with all options."""
        req = RerankRequest(
            model="jina-reranker-v3",
            query="test query",
            documents=["doc1", "doc2", "doc3"],
            top_n=2,
            return_documents=True,
        )
        assert req.top_n == 2
        assert req.return_documents is True

    def test_rerank_request_empty_query_fails(self):
        """Test that empty query fails validation."""
        with pytest.raises(ValidationError):
            RerankRequest(
                query="",
                documents=["doc1"],
            )

    def test_rerank_request_empty_documents_fails(self):
        """Test that empty documents list fails validation."""
        with pytest.raises(ValidationError):
            RerankRequest(
                query="test",
                documents=[],
            )

    def test_rerank_response(self):
        """Test rerank response structure."""
        resp = RerankResponse(
            model="jina-reranker-v3",
            results=[
                RerankResult(
                    index=1,
                    relevance_score=0.95,
                    document=DocumentObject(text="Most relevant doc"),
                ),
                RerankResult(
                    index=0,
                    relevance_score=0.75,
                ),
            ],
            usage=RerankUsage(total_tokens=100),
        )
        assert resp.model == "jina-reranker-v3"
        assert len(resp.results) == 2
        assert resp.results[0].relevance_score > resp.results[1].relevance_score
        assert resp.results[0].document is not None
        assert resp.results[1].document is None


class TestModelsSchemas:
    """Test models listing schemas."""

    def test_model_info(self):
        """Test ModelInfo structure."""
        info = ModelInfo(
            id="jina-embeddings-v3",
            object="model",
            created=1696118400,
            owned_by="jinaai",
            type="embedding",
            max_tokens=8192,
            dimensions=1024,
            supports_multimodal=False,
            tasks=["retrieval.query", "retrieval.passage"],
        )
        assert info.id == "jina-embeddings-v3"
        assert info.type == "embedding"
        assert info.supports_multimodal is False

    def test_models_list_response(self):
        """Test ModelsListResponse structure."""
        resp = ModelsListResponse(
            object="list",
            data=[
                ModelInfo(
                    id="jina-embeddings-v3",
                    created=1696118400,
                    type="embedding",
                ),
                ModelInfo(
                    id="jina-reranker-v3",
                    created=1730000000,
                    type="reranker",
                ),
            ],
        )
        assert resp.object == "list"
        assert len(resp.data) == 2
