"""
Unit Tests - Model Loading
==========================
Tests for model wrapper loading functionality.
"""

import pytest
import torch

from app.config import Settings
from app.models.base import EmbeddingModelWrapper, RerankerModelWrapper
from app.models.code_embeddings import CodeEmbeddingsWrapper
from app.models.embeddings_v3 import EmbeddingsV3Wrapper
from app.models.embeddings_v4 import EmbeddingsV4Wrapper
from app.models.reranker import RerankerWrapper
from app.models.registry import ALL_MODEL_IDS, ModelRegistry


class TestModelWrapperInstantiation:
    """Test that model wrappers can be instantiated."""

    def test_embeddings_v3_wrapper_instantiation(self):
        """Test EmbeddingsV3Wrapper can be created."""
        wrapper = EmbeddingsV3Wrapper()
        assert wrapper.model_id == "jina-embeddings-v3"
        assert wrapper.hf_model_id == "jinaai/jina-embeddings-v3"
        assert isinstance(wrapper, EmbeddingModelWrapper)
        assert not wrapper.is_loaded()

    def test_embeddings_v4_wrapper_instantiation(self):
        """Test EmbeddingsV4Wrapper can be created."""
        wrapper = EmbeddingsV4Wrapper()
        assert wrapper.model_id == "jina-embeddings-v4"
        assert wrapper.supports_multimodal is True
        assert isinstance(wrapper, EmbeddingModelWrapper)

    def test_code_embeddings_wrapper_instantiation(self):
        """Test CodeEmbeddingsWrapper can be created with both variants."""
        wrapper_05b = CodeEmbeddingsWrapper("0.5b")
        assert wrapper_05b.model_id == "jina-code-embeddings-0.5b"
        assert wrapper_05b.hf_model_id == "jinaai/jina-code-embeddings-0.5b"

        wrapper_15b = CodeEmbeddingsWrapper("1.5b")
        assert wrapper_15b.model_id == "jina-code-embeddings-1.5b"
        assert wrapper_15b.hf_model_id == "jinaai/jina-code-embeddings-1.5b"

    def test_code_embeddings_wrapper_invalid_variant(self):
        """Test CodeEmbeddingsWrapper raises error for invalid variant."""
        with pytest.raises(ValueError, match="Invalid variant"):
            CodeEmbeddingsWrapper("2b")

    def test_reranker_wrapper_instantiation(self):
        """Test RerankerWrapper can be created."""
        wrapper = RerankerWrapper()
        assert wrapper.model_id == "jina-reranker-v3"
        assert isinstance(wrapper, RerankerModelWrapper)


class TestModelRegistry:
    """Test model registry functionality."""

    def test_registry_instantiation(self):
        """Test ModelRegistry can be created with settings."""
        settings = Settings(device="cpu", models_to_load="all")
        registry = ModelRegistry(settings)
        assert registry.device == "cpu"
        assert len(registry.models) == 0

    def test_registry_get_models_to_load_all(self):
        """Test that 'all' loads all models."""
        settings = Settings(models_to_load="all")
        registry = ModelRegistry(settings)
        models = registry._get_models_to_load()
        assert models == ALL_MODEL_IDS

    def test_registry_get_models_to_load_specific(self):
        """Test specific model selection."""
        settings = Settings(models_to_load="jina-embeddings-v3,jina-reranker-v3")
        registry = ModelRegistry(settings)
        models = registry._get_models_to_load()
        assert "jina-embeddings-v3" in models
        assert "jina-reranker-v3" in models
        assert len(models) == 2

    def test_registry_create_wrapper(self):
        """Test wrapper creation for each model type."""
        settings = Settings(device="cpu")
        registry = ModelRegistry(settings)

        for model_id in ALL_MODEL_IDS:
            wrapper = registry._create_wrapper(model_id)
            assert wrapper.model_id == model_id

    def test_registry_is_not_ready_before_load(self):
        """Test registry is not ready before models are loaded."""
        settings = Settings(device="cpu")
        registry = ModelRegistry(settings)
        assert not registry.is_ready()

    def test_registry_loaded_models_empty(self):
        """Test loaded_models returns empty list before loading."""
        settings = Settings(device="cpu")
        registry = ModelRegistry(settings)
        assert registry.loaded_models() == []


class TestModelWrapperInterface:
    """Test model wrapper interface compliance."""

    def test_embedding_wrapper_has_encode(self):
        """Test embedding wrappers have encode method."""
        wrappers = [
            EmbeddingsV3Wrapper(),
            EmbeddingsV4Wrapper(),
            CodeEmbeddingsWrapper("1.5b"),
        ]
        for wrapper in wrappers:
            assert hasattr(wrapper, "encode")
            assert callable(wrapper.encode)

    def test_reranker_wrapper_has_rerank(self):
        """Test reranker wrapper has rerank method."""
        wrapper = RerankerWrapper()
        assert hasattr(wrapper, "rerank")
        assert callable(wrapper.rerank)

    def test_all_wrappers_have_load(self):
        """Test all wrappers have load method."""
        wrappers = [
            EmbeddingsV3Wrapper(),
            EmbeddingsV4Wrapper(),
            CodeEmbeddingsWrapper("0.5b"),
            RerankerWrapper(),
        ]
        for wrapper in wrappers:
            assert hasattr(wrapper, "load")
            assert callable(wrapper.load)

    def test_all_wrappers_have_is_loaded(self):
        """Test all wrappers have is_loaded method."""
        wrappers = [
            EmbeddingsV3Wrapper(),
            EmbeddingsV4Wrapper(),
            CodeEmbeddingsWrapper("1.5b"),
            RerankerWrapper(),
        ]
        for wrapper in wrappers:
            assert hasattr(wrapper, "is_loaded")
            assert not wrapper.is_loaded()
