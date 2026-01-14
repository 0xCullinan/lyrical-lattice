"""
File: src/core/config.py
Purpose: Configuration management using Pydantic Settings
"""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables.
    
    All settings are loaded from environment variables or .env file.
    See .env.example for complete list of configuration options.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Database (ENV-001)
    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/oronym_db"
    
    # Redis Cache (ENV-002)
    redis_url: str = "redis://localhost:6379/0"
    
    # Model Directories (ENV-003, ENV-004, ENV-005)
    model_dir: str = "data/models"
    corpus_dir: str = "data/corpus"
    index_dir: str = "data/indices"
    
    # File Upload (ENV-006)
    upload_dir: str = "/tmp/audio"
    
    # Logging (ENV-007)
    log_level: str = "INFO"
    
    # API Configuration (ENV-008)
    api_port: int = 8000
    
    # CORS (ENV-009)
    cors_origins: str = "http://localhost:3000,http://localhost:8080"
    
    # Rate Limiting (ENV-010)
    rate_limit_per_minute: int = 100
    
    # File Size Limit (ENV-011)
    max_file_size_mb: int = 10
    
    # Cache TTL (ENV-012)
    cache_ttl_seconds: int = 3600
    
    # Search Configuration (ENV-013, ENV-014)
    beam_width: int = 10
    max_phoneme_sequence_length: int = 100
    
    # Genius API (ENV-015)
    genius_api_key: Optional[str] = None
    
    # Environment
    environment: str = "development"
    
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins string into list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    @property
    def max_file_size_bytes(self) -> int:
        """Get maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return upper_v
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is one of allowed values."""
        valid_envs = {"development", "staging", "production"}
        lower_v = v.lower()
        if lower_v not in valid_envs:
            raise ValueError(f"Invalid environment: {v}. Must be one of {valid_envs}")
        return lower_v


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.
    
    Returns:
        Settings: Application settings singleton.
    """
    return Settings()


# Global settings instance
settings = get_settings()
