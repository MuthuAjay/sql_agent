"""Configuration management for SQL Agent."""

from typing import Literal, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "SQL Agent"
    app_version: str = "0.1.0"
    debug: bool = Field(default=False, alias="DEBUG")
    
    # LLM Configuration
    llm_provider: Literal["openai", "google", "local"] = Field(
        default="openai", alias="LLM_PROVIDER"
    )
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", alias="OPENAI_MODEL")
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    google_model: str = Field(default="gemini-pro", alias="GOOGLE_MODEL")
    
    # Database Configuration
    database_type: Literal["postgresql", "mysql", "sqlite"] = Field(
        default="postgresql", alias="DATABASE_TYPE"
    )
    database_url: str = Field(default="", alias="DATABASE_URL")
    database_pool_size: int = Field(default=10, alias="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, alias="DATABASE_MAX_OVERFLOW")
    
    # Vector Database Configuration
    vector_db_type: Literal["chromadb", "qdrant"] = Field(
        default="chromadb", alias="VECTOR_DB_TYPE"
    )
    vector_db_url: str = Field(default="http://localhost:8000", alias="VECTOR_DB_URL")
    vector_db_collection: str = Field(default="schema_context", alias="VECTOR_DB_COLLECTION")
    chroma_db_path: Optional[str] = Field(default="./chroma_db", alias="CHROMA_DB_PATH")
    
    # MCP Configuration
    mcp_server_host: str = Field(default="localhost", alias="MCP_SERVER_HOST")
    mcp_server_port: int = Field(default=3000, alias="MCP_SERVER_PORT")
    
    # Security Configuration
    rate_limit_per_minute: int = Field(default=100, alias="RATE_LIMIT_PER_MINUTE")
    query_timeout_seconds: int = Field(default=30, alias="QUERY_TIMEOUT_SECONDS")
    max_rows_returned: int = Field(default=1000, alias="MAX_ROWS_RETURNED")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")
    
    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Validate database URL format."""
        if not v:
            raise ValueError("Database URL is required")
        return v
    
    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v: Optional[str], info) -> Optional[str]:
        """Validate OpenAI API key when OpenAI is selected as provider."""
        if info.data.get("llm_provider") == "openai" and not v:
            raise ValueError("OpenAI API key is required when using OpenAI provider")
        return v
    
    @field_validator("google_api_key")
    @classmethod
    def validate_google_key(cls, v: Optional[str], info) -> Optional[str]:
        """Validate Google API key when Google is selected as provider."""
        if info.data.get("llm_provider") == "google" and not v:
            raise ValueError("Google API key is required when using Google provider")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings() 