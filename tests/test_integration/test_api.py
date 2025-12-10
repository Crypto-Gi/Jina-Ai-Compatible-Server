"""
Integration Tests - API Endpoints
=================================
Tests for HTTP API endpoints using TestClient.
These tests use mocked models to avoid loading real models.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import torch

from app.main import app
from app.models.registry import ModelRegistry


@pytest.fixture
def mock_registry():
    """Create a mock model registry with fake models."""
    registry = MagicMock(spec=ModelRegistry)
    registry.is_ready.return_value = True
    registry.loaded_models.return_value = [
        "jina-embeddings-v3",
        "jina-reranker-v3",
    ]
    
    # Mock embedding model
    mock_embedding_model = MagicMock()
    mock_embedding_model.model_id = "jina-embeddings-v3"
    mock_embedding_model.model_type = "embedding"
    mock_embedding_model.supports_multimodal = False
    mock_embedding_model.default_dimensions = 1024
    mock_embedding_model.max_tokens = 8192
    mock_embedding_model.encode.return_value = (
        torch.randn(2, 1024),  # 2 embeddings of 1024 dims
        50,  # token count
    )
    
    # Mock reranker model
    mock_reranker_model = MagicMock()
    mock_reranker_model.model_id = "jina-reranker-v3"
    mock_reranker_model.model_type = "reranker"
    mock_reranker_model.rerank.return_value = (
        [
            {"index": 1, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.75},
        ],
        100,  # token count
    )
    
    def get_model(model_id):
        if model_id == "jina-embeddings-v3":
            return mock_embedding_model
        elif model_id == "jina-reranker-v3":
            return mock_reranker_model
        return None
    
    def get_embedding_model(model_id):
        if model_id == "jina-embeddings-v3":
            return mock_embedding_model
        return None
    
    def get_reranker_model(model_id):
        if model_id == "jina-reranker-v3":
            return mock_reranker_model
        return None
    
    registry.get_model.side_effect = get_model
    registry.get_embedding_model.side_effect = get_embedding_model
    registry.get_reranker_model.side_effect = get_reranker_model
    registry.get_model_info.return_value = None
    
    return registry


@pytest.fixture
def client(mock_registry):
    """Create test client with mocked registry."""
    app.state.registry = mock_registry
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_healthz(self, client):
        """Test /healthz returns 200."""
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_readyz_when_ready(self, client):
        """Test /readyz returns 200 when models are loaded."""
        response = client.get("/readyz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert "models" in data

    def test_readyz_when_not_ready(self, mock_registry):
        """Test /readyz returns 503 when no models are loaded."""
        mock_registry.is_ready.return_value = False
        app.state.registry = mock_registry
        
        with TestClient(app) as client:
            response = client.get("/readyz")
            assert response.status_code == 503
            assert response.json()["status"] == "not_ready"

    def test_root_endpoint(self, client):
        """Test root endpoint returns server info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Jina Local API Server"
        assert "endpoints" in data


class TestEmbeddingsEndpoint:
    """Test /v1/embeddings endpoint."""

    def test_embeddings_single_text(self, client):
        """Test embedding single text."""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "jina-embeddings-v3",
                "input": "Hello world",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert data["model"] == "jina-embeddings-v3"
        assert len(data["data"]) >= 1
        assert data["data"][0]["object"] == "embedding"
        assert "embedding" in data["data"][0]
        assert "usage" in data

    def test_embeddings_batch(self, client):
        """Test embedding multiple texts."""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "jina-embeddings-v3",
                "input": ["Hello", "World"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2
        assert data["data"][0]["index"] == 0
        assert data["data"][1]["index"] == 1

    def test_embeddings_with_task(self, client):
        """Test embedding with task parameter."""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "jina-embeddings-v3",
                "input": ["query text"],
                "task": "retrieval.query",
            },
        )
        assert response.status_code == 200

    def test_embeddings_model_not_found(self, client):
        """Test error when model not found."""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "nonexistent-model",
                "input": "Hello",
            },
        )
        assert response.status_code == 404
        assert "error" in response.json()

    def test_embeddings_missing_input(self, client):
        """Test validation error for missing input."""
        response = client.post(
            "/v1/embeddings",
            json={"model": "jina-embeddings-v3"},
        )
        assert response.status_code == 422

    def test_embeddings_response_has_request_id(self, client):
        """Test that response includes X-Request-ID header."""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "jina-embeddings-v3",
                "input": "Hello",
            },
        )
        assert "X-Request-ID" in response.headers


class TestRerankEndpoint:
    """Test /v1/rerank endpoint."""

    def test_rerank_basic(self, client):
        """Test basic reranking."""
        response = client.post(
            "/v1/rerank",
            json={
                "model": "jina-reranker-v3",
                "query": "What is Python?",
                "documents": [
                    "Python is a programming language",
                    "Java is also a programming language",
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "jina-reranker-v3"
        assert "results" in data
        assert len(data["results"]) == 2
        assert "usage" in data

    def test_rerank_with_top_n(self, client):
        """Test reranking with top_n."""
        response = client.post(
            "/v1/rerank",
            json={
                "model": "jina-reranker-v3",
                "query": "test query",
                "documents": ["doc1", "doc2", "doc3"],
                "top_n": 2,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2

    def test_rerank_results_sorted(self, client):
        """Test that results are sorted by relevance score."""
        response = client.post(
            "/v1/rerank",
            json={
                "model": "jina-reranker-v3",
                "query": "test",
                "documents": ["doc1", "doc2"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        results = data["results"]
        
        # Verify scores are in descending order
        scores = [r["relevance_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_model_not_found(self, client):
        """Test error when reranker model not found."""
        response = client.post(
            "/v1/rerank",
            json={
                "model": "nonexistent-reranker",
                "query": "test",
                "documents": ["doc1"],
            },
        )
        assert response.status_code == 404


class TestModelsEndpoint:
    """Test /v1/models endpoint."""

    def test_list_models(self, client):
        """Test listing available models."""
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        assert len(data["data"]) >= 1

    def test_model_info_structure(self, client):
        """Test model info contains expected fields."""
        response = client.get("/v1/models")
        assert response.status_code == 200
        
        for model in response.json()["data"]:
            assert "id" in model
            assert "object" in model
            assert model["object"] == "model"
            assert "type" in model
            assert model["type"] in ["embedding", "reranker"]
