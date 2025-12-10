"""
Jina Local API Server - Configuration Module
============================================
Loads configuration from environment variables using pydantic-settings.
"""

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8080, description="Server port")

    # Device settings
    device: Literal["cuda", "cpu"] = Field(
        default="cuda", description="Device to run models on"
    )

    # Model settings
    models_to_load: str = Field(
        default="all",
        description="Comma-separated list of model IDs to load, or 'all'",
    )
    models_config_path: str = Field(
        default="config/models.yaml", description="Path to models configuration YAML"
    )

    # HuggingFace settings
    hf_home: Optional[str] = Field(
        default=None,
        description="HuggingFace cache directory",
    )
    hf_token: Optional[str] = Field(
        default=None,
        description="HuggingFace API token for private models",
    )

    # Performance settings
    max_batch_size: int = Field(
        default=32, description="Maximum batch size for inference"
    )
    enable_flash_attn: bool = Field(
        default=True, description="Enable Flash Attention 2 if available"
    )
    torch_dtype: Literal["float16", "bfloat16", "float32"] = Field(
        default="float16", description="Default torch dtype for models"
    )

    # Logging settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )
    log_format: Literal["json", "console"] = Field(
        default="json", description="Log output format"
    )

    def get_models_to_load(self) -> list[str]:
        """Parse models_to_load string into list of model IDs."""
        if self.models_to_load.lower() == "all":
            return []  # Empty list means load all
        return [m.strip() for m in self.models_to_load.split(",") if m.strip()]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
