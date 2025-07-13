"""Enhanced configuration management for SQL Agent."""

import os
from typing import Literal, Optional, Dict, Any, List
from functools import lru_cache
from pydantic import Field, field_validator, computed_field, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Enhanced application settings with environment variable support."""
    
    # Application
    app_name: str = "SQL Agent"
    app_version: str = "1.0.0"
    ENVIRONMENT: str = "development"
    debug: bool = Field(default=False, alias="DEBUG")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_prefix: str = Field(default="/api/v1", alias="API_PREFIX")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"], 
        alias="CORS_ORIGINS"
    )
    
    # LLM Configuration
    llm_provider: Literal["openai", "anthropic", "google", "ollama", "auto"] = Field(
        default="auto", alias="LLM_PROVIDER"
    )
    
    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", alias="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.1, alias="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=2000, alias="OPENAI_MAX_TOKENS")
    
    # Anthropic
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-sonnet-20240229", alias="ANTHROPIC_MODEL")
    
    # Google
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    google_model: str = Field(default="gemini-pro", alias="GOOGLE_MODEL")
    
    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3", alias="OLLAMA_MODEL")
    
    # Database Configuration
    database_type: Literal["postgresql", "mysql", "sqlite", "duckdb"] = Field(
        default="postgresql", alias="DATABASE_TYPE"
    )
    database_url: str = Field(default="", alias="DATABASE_URL")
    database_pool_size: int = Field(default=10, alias="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, alias="DATABASE_MAX_OVERFLOW")
    database_pool_timeout: int = Field(default=30, alias="DATABASE_POOL_TIMEOUT")
    database_pool_recycle: int = Field(default=3600, alias="DATABASE_POOL_RECYCLE")
    
    # Vector Database Configuration
    vector_db_type: Literal["chromadb", "qdrant", "pinecone"] = Field(
        default="chromadb", alias="VECTOR_DB_TYPE"
    )
    vector_db_url: str = Field(default="http://localhost:8000", alias="VECTOR_DB_URL")
    vector_db_collection: str = Field(default="schema_context", alias="VECTOR_DB_COLLECTION")
    chroma_db_path: Optional[str] = Field(default="./chroma_db", alias="CHROMA_DB_PATH")
    embedding_model: str = Field(default="text-embedding-ada-002", alias="EMBEDDING_MODEL")
    
    # Redis Configuration (for caching and sessions)
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    redis_cache_ttl: int = Field(default=3600, alias="REDIS_CACHE_TTL")  # 1 hour
    
    # MCP Configuration
    mcp_server_host: str = Field(default="localhost", alias="MCP_SERVER_HOST")
    mcp_server_port: int = Field(default=8001, alias="MCP_SERVER_PORT")
    mcp_tools_enabled: bool = Field(default=True, alias="MCP_TOOLS_ENABLED")
    
    # Security Configuration
    SECRET_KEY: str = Field(default="dev-secret-key-please-change-this-to-a-long-random-string-1234567890", alias="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    rate_limit_per_minute: int = Field(default=100, alias="RATE_LIMIT_PER_MINUTE")
    ALLOWED_HOSTS: List[str] = Field(default=["*"], alias="ALLOWED_HOSTS")
    
    # Query Configuration
    query_timeout_seconds: int = Field(default=60, alias="QUERY_TIMEOUT_SECONDS")
    max_rows_returned: int = Field(default=10000, alias="MAX_ROWS_RETURNED")
    enable_query_caching: bool = Field(default=True, alias="ENABLE_QUERY_CACHING")
    cache_ttl_seconds: int = Field(default=1800, alias="CACHE_TTL_SECONDS")  # 30 minutes
    
    # Agent Configuration
    orchestrator_workflow_mode: Literal["sequential", "parallel", "adaptive", "custom"] = Field(
        default="adaptive", alias="ORCHESTRATOR_WORKFLOW_MODE"
    )
    orchestrator_execution_strategy: Literal["fast", "comprehensive", "analysis_focused", "visualization_focused"] = Field(
        default="comprehensive", alias="ORCHESTRATOR_EXECUTION_STRATEGY"
    )
    
    # Agent Timeouts
    router_timeout: int = Field(default=30, alias="ROUTER_TIMEOUT")
    sql_timeout: int = Field(default=120, alias="SQL_TIMEOUT")
    analysis_timeout: int = Field(default=180, alias="ANALYSIS_TIMEOUT")
    visualization_timeout: int = Field(default=60, alias="VISUALIZATION_TIMEOUT")
    
    # Circuit Breaker Configuration
    circuit_breaker_failure_threshold: int = Field(default=5, alias="CIRCUIT_BREAKER_FAILURE_THRESHOLD")
    circuit_breaker_recovery_timeout: int = Field(default=300, alias="CIRCUIT_BREAKER_RECOVERY_TIMEOUT")  # 5 minutes
    
    # RAG Configuration
    rag_enabled: bool = Field(default=True, alias="RAG_ENABLED")
    rag_similarity_threshold: float = Field(default=0.6, alias="RAG_SIMILARITY_THRESHOLD")
    rag_max_contexts: int = Field(default=5, alias="RAG_MAX_CONTEXTS")
    rag_chunk_size: int = Field(default=1000, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=200, alias="RAG_CHUNK_OVERLAP")
    
    # Feature Flags
    enable_query_optimization: bool = Field(default=True, alias="ENABLE_QUERY_OPTIMIZATION")
    enable_sql_validation: bool = Field(default=True, alias="ENABLE_SQL_VALIDATION")
    enable_visualization: bool = Field(default=True, alias="ENABLE_VISUALIZATION")
    enable_analysis: bool = Field(default=True, alias="ENABLE_ANALYSIS")
    enable_websockets: bool = Field(default=True, alias="ENABLE_WEBSOCKETS")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: Literal["json", "text"] = Field(default="json", alias="LOG_FORMAT")
    log_file: Optional[str] = Field(default=None, alias="LOG_FILE")
    enable_request_logging: bool = Field(default=True, alias="ENABLE_REQUEST_LOGGING")
    
    # File Upload Configuration
    max_file_size_mb: int = Field(default=50, alias="MAX_FILE_SIZE_MB")
    allowed_file_types: List[str] = Field(
        default=[".csv", ".json", ".xlsx", ".sql"], 
        alias="ALLOWED_FILE_TYPES"
    )
    upload_directory: str = Field(default="./uploads", alias="UPLOAD_DIRECTORY")
    
    # Monitoring Configuration
    enable_metrics: bool = Field(default=True, alias="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, alias="METRICS_PORT")
    health_check_interval: int = Field(default=30, alias="HEALTH_CHECK_INTERVAL")
    
    # Performance Configuration
    max_concurrent_queries: int = Field(default=10, alias="MAX_CONCURRENT_QUERIES")
    query_queue_size: int = Field(default=100, alias="QUERY_QUEUE_SIZE")
    enable_query_parallelization: bool = Field(default=True, alias="ENABLE_QUERY_PARALLELIZATION")
    
    @computed_field
    @property
    def effective_llm_provider(self) -> str:
        """Determine the effective LLM provider based on availability."""
        if self.llm_provider != "auto":
            return self.llm_provider
        
        # Auto-detect based on available API keys and services
        if self.openai_api_key:
            return "openai"
        elif self.anthropic_api_key:
            return "anthropic"
        elif self.google_api_key:
            return "google"
        else:
            return "ollama"
    
    @computed_field
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"
    
    @computed_field
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == "development"
    
    @computed_field
    @property
    def database_config(self) -> Dict[str, Any]:
        """Get database configuration dictionary."""
        return {
            "url": self.database_url,
            "pool_size": self.database_pool_size,
            "max_overflow": self.database_max_overflow,
            "pool_timeout": self.database_pool_timeout,
            "pool_recycle": self.database_pool_recycle,
        }
    
    @computed_field
    @property
    def llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration dictionary."""
        provider = self.effective_llm_provider
        
        base_config = {"provider": provider}
        
        if provider == "openai":
            base_config.update({
                "api_key": self.openai_api_key,
                "model": self.openai_model,
                "temperature": self.openai_temperature,
                "max_tokens": self.openai_max_tokens,
            })
        elif provider == "anthropic":
            base_config.update({
                "api_key": self.anthropic_api_key,
                "model": self.anthropic_model,
            })
        elif provider == "google":
            base_config.update({
                "api_key": self.google_api_key,
                "model": self.google_model,
            })
        elif provider == "ollama":
            base_config.update({
                "base_url": self.ollama_base_url,
                "model": self.ollama_model,
            })
        
        return base_config
    
    @computed_field
    @property
    def agent_timeouts(self) -> Dict[str, int]:
        """Get agent timeout configuration."""
        return {
            "router": self.router_timeout,
            "sql": self.sql_timeout,
            "analysis": self.analysis_timeout,
            "visualization": self.visualization_timeout,
        }
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @field_validator("allowed_file_types", mode="before")
    @classmethod
    def parse_file_types(cls, v):
        """Parse allowed file types from string or list."""
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(",")]
        return v
    
    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Validate database URL format."""
        if not v:
            raise ValueError("Database URL is required")
        return v
    
    @model_validator(mode="after")
    def validate_llm_provider_keys(self):
        """Validate that required API keys are present for selected LLM provider."""
        provider = self.effective_llm_provider
        
        if provider == "openai" and not self.openai_api_key:
            raise ValueError("OpenAI API key is required when using OpenAI provider")
        elif provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("Anthropic API key is required when using Anthropic provider")
        elif provider == "google" and not self.google_api_key:
            raise ValueError("Google API key is required when using Google provider")
        
        return self
    
    @model_validator(mode="after")
    def validate_production_settings(self):
        """Validate production-specific settings."""
        if self.is_production:
            if self.debug:
                raise ValueError("Debug mode should be disabled in production")
            if "*" in self.ALLOWED_HOSTS:
                raise ValueError("Wildcard hosts not allowed in production")
            if not self.enable_metrics:
                raise ValueError("Metrics should be enabled in production")
        
        return self
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration."""
        return {
            "url": self.redis_url,
            "cache_ttl": self.redis_cache_ttl,
        }
    
    def get_vector_db_config(self) -> Dict[str, Any]:
        """Get vector database configuration."""
        config = {
            "type": self.vector_db_type,
            "collection": self.vector_db_collection,
            "embedding_model": self.embedding_model,
        }
        
        if self.vector_db_type == "chromadb":
            config["path"] = self.chroma_db_path
        else:
            config["url"] = self.vector_db_url
        
        return config
    
    def get_orchestrator_config(self) -> Dict[str, Any]:
        """Get orchestrator configuration."""
        return {
            "workflow_mode": self.orchestrator_workflow_mode,
            "execution_strategy": self.orchestrator_execution_strategy,
            "agent_timeouts": self.agent_timeouts,
            "circuit_breaker": {
                "failure_threshold": self.circuit_breaker_failure_threshold,
                "recovery_timeout": self.circuit_breaker_recovery_timeout,
            },
            "performance": {
                "max_concurrent_queries": self.max_concurrent_queries,
                "query_queue_size": self.query_queue_size,
                "enable_parallelization": self.enable_query_parallelization,
            }
        }
    
    def get_rag_config(self) -> Dict[str, Any]:
        """Get RAG configuration."""
        return {
            "enabled": self.rag_enabled,
            "similarity_threshold": self.rag_similarity_threshold,
            "max_contexts": self.rag_max_contexts,
            "chunk_size": self.rag_chunk_size,
            "chunk_overlap": self.rag_chunk_overlap,
        }
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get all feature flags."""
        return {
            "query_optimization": self.enable_query_optimization,
            "sql_validation": self.enable_sql_validation,
            "visualization": self.enable_visualization,
            "analysis": self.enable_analysis,
            "websockets": self.enable_websockets,
            "query_caching": self.enable_query_caching,
            "metrics": self.enable_metrics,
            "rag": self.rag_enabled,
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            "level": self.log_level,
            "format": self.log_format,
            "file": self.log_file,
            "enable_request_logging": self.enable_request_logging,
        }
    
    def export_for_frontend(self) -> Dict[str, Any]:
        """Export safe configuration for frontend."""
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "environment": self.ENVIRONMENT,
            "api_prefix": self.api_prefix,
            "features": self.get_feature_flags(),
            "limits": {
                "max_rows_returned": self.max_rows_returned,
                "query_timeout_seconds": self.query_timeout_seconds,
                "max_file_size_mb": self.max_file_size_mb,
                "allowed_file_types": self.allowed_file_types,
            },
            "llm_provider": self.effective_llm_provider,
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore unknown environment variables


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()


# Environment-specific configuration overrides
class DevelopmentSettings(Settings):
    """Development environment settings."""
    ENVIRONMENT: str = "development"
    debug: bool = True
    log_level: str = "DEBUG"
    enable_request_logging: bool = True


class ProductionSettings(Settings):
    """Production environment settings."""
    ENVIRONMENT: str = "production"
    debug: bool = False
    log_level: str = "INFO"
    cors_origins: List[str] = []  # Must be explicitly set
    ALLOWED_HOSTS: List[str] = []  # Must be explicitly set


def get_environment_settings() -> Settings:
    """Get environment-specific settings."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "staging":
        # Staging inherits from production but allows some dev features
        staging_settings = ProductionSettings()
        staging_settings.ENVIRONMENT = "staging"
        staging_settings.debug = False
        return staging_settings
    else:
        return DevelopmentSettings()


# Validation helper
def validate_configuration() -> None:
    """Validate the current configuration."""
    try:
        settings = get_settings()
        print(f"✅ Configuration valid for {settings.ENVIRONMENT} environment")
        print(f"   LLM Provider: {settings.effective_llm_provider}")
        print(f"   Database: {settings.database_type}")
        print(f"   Features: {sum(settings.get_feature_flags().values())}/{len(settings.get_feature_flags())} enabled")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        raise


if __name__ == "__main__":
    validate_configuration()