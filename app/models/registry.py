"""
Jina Local API Server - Model Registry
=====================================
Central registry for loading and managing models.
"""

from pathlib import Path
from typing import Any

import torch
import yaml

from app.config import Settings
from app.logging_config import get_logger
from app.models.base import BaseModelWrapper, EmbeddingModelWrapper, RerankerModelWrapper
from app.models.code_embeddings import CodeEmbeddingsWrapper
from app.models.embeddings_v3 import EmbeddingsV3Wrapper
from app.models.embeddings_v4 import EmbeddingsV4Wrapper
from app.models.reranker import RerankerWrapper

logger = get_logger(__name__)


# All available models
ALL_MODEL_IDS = [
    "jina-embeddings-v3",
    "jina-embeddings-v4",
    "jina-code-embeddings-0.5b",
    "jina-code-embeddings-1.5b",
    "jina-reranker-v3",
]


class ModelRegistry:
    """
    Central registry for loading and managing Jina models.
    
    Models are loaded at startup and stored in a dictionary keyed by model ID.
    The registry supports selective loading via MODELS_TO_LOAD environment variable.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.models: dict[str, BaseModelWrapper] = {}
        self.model_configs: dict[str, dict[str, Any]] = {}
        
        # Determine device
        self.device = settings.device
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
        
        # Determine torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(settings.torch_dtype, torch.float16)
        
        # Load model configurations
        self._load_model_configs()

    def _load_model_configs(self) -> None:
        """Load model configurations from YAML file."""
        config_path = Path(self.settings.models_config_path)
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
                self.model_configs = config.get("models", {})
        else:
            logger.warning(
                "Models config not found, using defaults",
                path=str(config_path),
            )

    def _create_wrapper(self, model_id: str) -> BaseModelWrapper:
        """
        Factory method to create the appropriate wrapper for a model.
        
        Args:
            model_id: The model identifier
            
        Returns:
            Instantiated model wrapper
        """
        wrapper_map: dict[str, type[BaseModelWrapper] | callable] = {
            "jina-embeddings-v3": EmbeddingsV3Wrapper,
            "jina-embeddings-v4": EmbeddingsV4Wrapper,
            "jina-code-embeddings-0.5b": lambda: CodeEmbeddingsWrapper("0.5b"),
            "jina-code-embeddings-1.5b": lambda: CodeEmbeddingsWrapper("1.5b"),
            "jina-reranker-v3": RerankerWrapper,
        }

        if model_id not in wrapper_map:
            raise ValueError(f"Unknown model: {model_id}")

        factory = wrapper_map[model_id]
        if callable(factory) and not isinstance(factory, type):
            return factory()
        return factory()

    def _get_models_to_load(self) -> list[str]:
        """
        Determine which models to load based on settings.
        
        Returns:
            List of model IDs to load
        """
        explicit_models = self.settings.get_models_to_load()
        
        if not explicit_models:
            # Empty list means "all"
            return ALL_MODEL_IDS
        
        # Validate requested models
        invalid = set(explicit_models) - set(ALL_MODEL_IDS)
        if invalid:
            logger.warning(
                "Unknown models requested, ignoring",
                invalid_models=list(invalid),
            )
        
        return [m for m in explicit_models if m in ALL_MODEL_IDS]

    async def load_models(self) -> None:
        """
        Load all configured models.
        
        This is called during application startup.
        """
        models_to_load = self._get_models_to_load()
        
        logger.info(
            "Starting model loading",
            models=models_to_load,
            device=self.device,
            torch_dtype=str(self.torch_dtype),
        )

        for model_id in models_to_load:
            try:
                wrapper = self._create_wrapper(model_id)
                wrapper.load(self.device, self.torch_dtype)
                self.models[model_id] = wrapper
                logger.info("Model loaded", model_id=model_id)
            except Exception as e:
                logger.error(
                    "Failed to load model",
                    model_id=model_id,
                    error=str(e),
                )
                # Continue loading other models even if one fails
                continue

        logger.info(
            "Model loading complete",
            loaded_count=len(self.models),
            loaded_models=list(self.models.keys()),
        )

    def get_model(self, model_id: str) -> BaseModelWrapper | None:
        """
        Get a loaded model by ID.
        
        Args:
            model_id: The model identifier
            
        Returns:
            The model wrapper if loaded, None otherwise
        """
        return self.models.get(model_id)

    def get_embedding_model(self, model_id: str) -> EmbeddingModelWrapper | None:
        """Get an embedding model by ID."""
        model = self.get_model(model_id)
        if model and isinstance(model, EmbeddingModelWrapper):
            return model
        return None

    def get_reranker_model(self, model_id: str) -> RerankerModelWrapper | None:
        """Get a reranker model by ID."""
        model = self.get_model(model_id)
        if model and isinstance(model, RerankerModelWrapper):
            return model
        return None

    def is_ready(self) -> bool:
        """
        Check if at least one model is loaded and ready.
        """
        return len(self.models) > 0 and all(
            m.is_loaded() for m in self.models.values()
        )

    def loaded_models(self) -> list[str]:
        """Get list of loaded model IDs."""
        return list(self.models.keys())

    def get_model_info(self, model_id: str) -> dict[str, Any] | None:
        """
        Get model configuration info.
        
        Args:
            model_id: The model identifier
            
        Returns:
            Model configuration dict or None
        """
        return self.model_configs.get(model_id)

    def unload_all(self) -> None:
        """Unload all models and free memory."""
        for model_id, model in self.models.items():
            logger.info("Unloading model", model_id=model_id)
            model.unload()
        self.models.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
