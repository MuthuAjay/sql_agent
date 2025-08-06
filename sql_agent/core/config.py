"""Enhanced configuration management for SQL Agent - Phase 4 Production Ready."""

import os
from typing import Literal, Optional, Dict, Any, List
from functools import lru_cache
from pydantic import Field, field_validator, computed_field, model_validator
from pydantic_settings import BaseSettings
import json

class Settings(BaseSettings):
    """Enhanced application settings with production-grade configuration."""
    
    # Application
    app_name: str = "SQL Agent"
    app_version: str = "4.0.0"
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
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")  # Updated to latest model
    openai_temperature: float = Field(default=0.1, alias="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=4000, alias="OPENAI_MAX_TOKENS")  # Increased for complex queries
    
    # Anthropic
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022", alias="ANTHROPIC_MODEL")  # Latest Claude
    
    # Google
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    google_model: str = Field(default="gemini-1.5-pro", alias="GOOGLE_MODEL")  # Latest Gemini
    
    # Ollama Configuration - NEW
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="mistral:7b", alias="OLLAMA_MODEL")
    ollama_timeout: int = Field(default=120, alias="OLLAMA_TIMEOUT")
    ollama_context_window: int = Field(default=4096, alias="OLLAMA_CONTEXT_WINDOW")
    ollama_temperature: float = Field(default=0.1, alias="OLLAMA_TEMPERATURE")
    ollama_num_gpu: int = Field(default=1, alias="OLLAMA_NUM_GPU")
    ollama_num_thread: int = Field(default=8, alias="OLLAMA_NUM_THREAD")
    
    # Hardware Configuration - NEW
    gpu_memory_gb: int = Field(default=16, alias="GPU_MEMORY_GB")
    hardware_profile: Literal["desktop_24gb", "laptop_16gb", "auto"] = Field(default="auto", alias="HARDWARE_PROFILE")
    max_concurrent_llm_requests: int = Field(default=2, alias="MAX_CONCURRENT_LLM_REQUESTS")
    enable_hardware_optimization: bool = Field(default=True, alias="ENABLE_HARDWARE_OPTIMIZATION")
    
    # Schema Analysis Configuration - NEW
    schema_analysis_mode: Literal["instant", "quick", "standard", "deep"] = Field(default="standard", alias="SCHEMA_ANALYSIS_MODE")
    schema_batch_size: int = Field(default=10, alias="SCHEMA_BATCH_SIZE")
    enable_parallel_analysis: bool = Field(default=True, alias="ENABLE_PARALLEL_ANALYSIS")
    analysis_cache_ttl_hours: int = Field(default=24, alias="ANALYSIS_CACHE_TTL_HOURS")
    fingerprint_cache_ttl_hours: int = Field(default=168, alias="FINGERPRINT_CACHE_TTL_HOURS")  # 1 week
    
    # Cache Configuration - NEW
    cache_directory: str = Field(default="./cache", alias="CACHE_DIRECTORY")
    max_cache_size_mb: int = Field(default=2048, alias="MAX_CACHE_SIZE_MB")  # 2GB
    enable_cache_compression: bool = Field(default=True, alias="ENABLE_CACHE_COMPRESSION")
    
    # Database Configuration
    database_type: Literal["postgresql", "mysql", "sqlite", "duckdb"] = Field(
        default="postgresql", alias="DATABASE_TYPE"
    )
    database_url: str = Field(default="", alias="DATABASE_URL")
    database_pool_size: int = Field(default=20, alias="DATABASE_POOL_SIZE")  # Increased for Phase 4
    database_max_overflow: int = Field(default=30, alias="DATABASE_MAX_OVERFLOW")
    database_pool_timeout: int = Field(default=30, alias="DATABASE_POOL_TIMEOUT")
    database_pool_recycle: int = Field(default=3600, alias="DATABASE_POOL_RECYCLE")
    
    # Phase 4: Advanced Database Configuration
    database_pool_pre_ping: bool = Field(default=True, alias="DATABASE_POOL_PRE_PING")
    database_isolation_level: Literal["READ_COMMITTED", "READ_UNCOMMITTED", "REPEATABLE_READ", "SERIALIZABLE"] = Field(
        default="READ_COMMITTED", alias="DATABASE_ISOLATION_LEVEL"
    )
    database_statement_timeout: int = Field(default=300, alias="DATABASE_STATEMENT_TIMEOUT")  # 5 minutes
    database_lock_timeout: int = Field(default=30, alias="DATABASE_LOCK_TIMEOUT")  # 30 seconds
    
    # Vector Database Configuration
    vector_db_type: Literal["chromadb", "qdrant", "pinecone", "weaviate"] = Field(
        default="chromadb", alias="VECTOR_DB_TYPE"
    )
    vector_db_url: str = Field(default="http://localhost:8000", alias="VECTOR_DB_URL")
    vector_db_collection: str = Field(default="schema_context", alias="VECTOR_DB_COLLECTION")
    chroma_db_path: Optional[str] = Field(default="./chroma_db", alias="CHROMA_DB_PATH")
    
    # Phase 4: Enhanced Embedding Configuration
    embedding_model: str = Field(default="text-embedding-3-large", alias="EMBEDDING_MODEL")  # Latest OpenAI
    embedding_dimension: int = Field(default=3072, alias="EMBEDDING_DIMENSION")  # For text-embedding-3-large
    embedding_batch_size: int = Field(default=100, alias="EMBEDDING_BATCH_SIZE")
    embedding_cache_ttl: int = Field(default=86400, alias="EMBEDDING_CACHE_TTL")  # 24 hours
    
    # Redis Configuration (for caching and sessions)
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    redis_cache_ttl: int = Field(default=3600, alias="REDIS_CACHE_TTL")  # 1 hour
    redis_session_ttl: int = Field(default=86400, alias="REDIS_SESSION_TTL")  # 24 hours
    redis_max_connections: int = Field(default=50, alias="REDIS_MAX_CONNECTIONS")
    
    # MCP Configuration
    mcp_server_host: str = Field(default="localhost", alias="MCP_SERVER_HOST")
    mcp_server_port: int = Field(default=8001, alias="MCP_SERVER_PORT")
    mcp_tools_enabled: bool = Field(default=True, alias="MCP_TOOLS_ENABLED")
    
    # Security Configuration
    SECRET_KEY: str = Field(default="dev-secret-key-please-change-this-to-a-long-random-string-1234567890", alias="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    rate_limit_per_minute: int = Field(default=100, alias="RATE_LIMIT_PER_MINUTE")
    ALLOWED_HOSTS: List[str] = Field(default=["*"], alias="ALLOWED_HOSTS")
    
    # Phase 4: Enhanced Security
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    enable_csrf_protection: bool = Field(default=True, alias="ENABLE_CSRF_PROTECTION")
    session_cookie_secure: bool = Field(default=False, alias="SESSION_COOKIE_SECURE")  # True in production
    session_cookie_httponly: bool = Field(default=True, alias="SESSION_COOKIE_HTTPONLY")
    session_cookie_samesite: Literal["strict", "lax", "none"] = Field(default="lax", alias="SESSION_COOKIE_SAMESITE")
    
    # Query Configuration
    query_timeout_seconds: int = Field(default=60, alias="QUERY_TIMEOUT_SECONDS")
    max_rows_returned: int = Field(default=10000, alias="MAX_ROWS_RETURNED")
    enable_query_caching: bool = Field(default=True, alias="ENABLE_QUERY_CACHING")
    cache_ttl_seconds: int = Field(default=1800, alias="CACHE_TTL_SECONDS")  # 30 minutes
    
    # Phase 4: Advanced Query Configuration
    query_complexity_limit: int = Field(default=1000, alias="QUERY_COMPLEXITY_LIMIT")
    enable_query_plan_caching: bool = Field(default=True, alias="ENABLE_QUERY_PLAN_CACHING")
    query_memory_limit_mb: int = Field(default=1024, alias="QUERY_MEMORY_LIMIT_MB")  # 1GB
    enable_query_optimization: bool = Field(default=True, alias="ENABLE_QUERY_OPTIMIZATION")
    
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
    
    # Phase 4: Enhanced Agent Configuration
    agent_retry_attempts: int = Field(default=3, alias="AGENT_RETRY_ATTEMPTS")
    agent_retry_delay: float = Field(default=1.0, alias="AGENT_RETRY_DELAY")
    agent_backoff_multiplier: float = Field(default=2.0, alias="AGENT_BACKOFF_MULTIPLIER")
    enable_agent_failover: bool = Field(default=True, alias="ENABLE_AGENT_FAILOVER")
    
    # Circuit Breaker Configuration
    circuit_breaker_failure_threshold: int = Field(default=5, alias="CIRCUIT_BREAKER_FAILURE_THRESHOLD")
    circuit_breaker_recovery_timeout: int = Field(default=300, alias="CIRCUIT_BREAKER_RECOVERY_TIMEOUT")  # 5 minutes
    circuit_breaker_half_open_max_calls: int = Field(default=3, alias="CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS")
    
    # RAG Configuration
    rag_enabled: bool = Field(default=True, alias="RAG_ENABLED")
    rag_similarity_threshold: float = Field(default=0.6, alias="RAG_SIMILARITY_THRESHOLD")
    rag_max_contexts: int = Field(default=5, alias="RAG_MAX_CONTEXTS")
    rag_chunk_size: int = Field(default=1000, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=200, alias="RAG_CHUNK_OVERLAP")
    
    # Phase 4: Enhanced RAG Configuration
    rag_reranking_enabled: bool = Field(default=True, alias="RAG_RERANKING_ENABLED")
    rag_context_window_tokens: int = Field(default=8000, alias="RAG_CONTEXT_WINDOW_TOKENS")
    rag_index_refresh_interval: int = Field(default=3600, alias="RAG_INDEX_REFRESH_INTERVAL")  # 1 hour
    rag_batch_update_size: int = Field(default=100, alias="RAG_BATCH_UPDATE_SIZE")
    
    # Business Intelligence Configuration
    enable_business_intelligence: bool = Field(default=True, alias="ENABLE_BUSINESS_INTELLIGENCE")
    business_domain_classification: bool = Field(default=True, alias="BUSINESS_DOMAIN_CLASSIFICATION")
    enable_schema_profiling: bool = Field(default=True, alias="ENABLE_SCHEMA_PROFILING")
    schema_profiling_interval: int = Field(default=86400, alias="SCHEMA_PROFILING_INTERVAL")  # 24 hours
    
    # Performance Monitoring Configuration
    enable_performance_monitoring: bool = Field(default=True, alias="ENABLE_PERFORMANCE_MONITORING")
    performance_metrics_retention_days: int = Field(default=30, alias="PERFORMANCE_METRICS_RETENTION_DAYS")
    enable_slow_query_logging: bool = Field(default=True, alias="ENABLE_SLOW_QUERY_LOGGING")
    slow_query_threshold_ms: int = Field(default=1000, alias="SLOW_QUERY_THRESHOLD_MS")  # 1 second
    
    # Data Quality Configuration
    enable_data_quality_checks: bool = Field(default=True, alias="ENABLE_DATA_QUALITY_CHECKS")
    data_quality_check_interval: int = Field(default=3600, alias="DATA_QUALITY_CHECK_INTERVAL")  # 1 hour
    data_quality_threshold: float = Field(default=0.8, alias="DATA_QUALITY_THRESHOLD")
    enable_anomaly_detection: bool = Field(default=True, alias="ENABLE_ANOMALY_DETECTION")
    
    # Feature Flags
    enable_sql_validation: bool = Field(default=True, alias="ENABLE_SQL_VALIDATION")
    enable_visualization: bool = Field(default=True, alias="ENABLE_VISUALIZATION")
    enable_analysis: bool = Field(default=True, alias="ENABLE_ANALYSIS")
    enable_websockets: bool = Field(default=True, alias="ENABLE_WEBSOCKETS")
    enable_real_time_updates: bool = Field(default=True, alias="ENABLE_REAL_TIME_UPDATES")
    
    # Phase 4: Advanced Features
    enable_ai_explanations: bool = Field(default=True, alias="ENABLE_AI_EXPLANATIONS")
    enable_query_suggestions: bool = Field(default=True, alias="ENABLE_QUERY_SUGGESTIONS")
    enable_auto_optimization: bool = Field(default=True, alias="ENABLE_AUTO_OPTIMIZATION")
    enable_collaborative_features: bool = Field(default=True, alias="ENABLE_COLLABORATIVE_FEATURES")
    enable_export_features: bool = Field(default=True, alias="ENABLE_EXPORT_FEATURES")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: Literal["json", "text", "structured"] = Field(default="structured", alias="LOG_FORMAT")
    log_file: Optional[str] = Field(default=None, alias="LOG_FILE")
    enable_request_logging: bool = Field(default=True, alias="ENABLE_REQUEST_LOGGING")
    log_retention_days: int = Field(default=30, alias="LOG_RETENTION_DAYS")
    
    # File Upload Configuration
    max_file_size_mb: int = Field(default=100, alias="MAX_FILE_SIZE_MB")  # Increased for Phase 4
    allowed_file_types: List[str] = Field(
        default=[".csv", ".json", ".xlsx", ".sql", ".parquet", ".jsonl"], 
        alias="ALLOWED_FILE_TYPES"
    )
    upload_directory: str = Field(default="./uploads", alias="UPLOAD_DIRECTORY")
    enable_file_virus_scanning: bool = Field(default=False, alias="ENABLE_FILE_VIRUS_SCANNING")
    
    # Monitoring Configuration
    enable_metrics: bool = Field(default=True, alias="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, alias="METRICS_PORT")
    health_check_interval: int = Field(default=30, alias="HEALTH_CHECK_INTERVAL")
    
    # Phase 4: Enhanced Monitoring
    enable_distributed_tracing: bool = Field(default=True, alias="ENABLE_DISTRIBUTED_TRACING")
    tracing_sample_rate: float = Field(default=0.1, alias="TRACING_SAMPLE_RATE")  # 10% sampling
    enable_alerting: bool = Field(default=True, alias="ENABLE_ALERTING")
    alerting_webhook_url: Optional[str] = Field(default=None, alias="ALERTING_WEBHOOK_URL")
    
    # Performance Configuration
    max_concurrent_queries: int = Field(default=20, alias="MAX_CONCURRENT_QUERIES")  # Increased
    query_queue_size: int = Field(default=200, alias="QUERY_QUEUE_SIZE")  # Increased
    enable_query_parallelization: bool = Field(default=True, alias="ENABLE_QUERY_PARALLELIZATION")
    
    # Phase 4: Advanced Performance
    enable_connection_pooling: bool = Field(default=True, alias="ENABLE_CONNECTION_POOLING")
    enable_result_streaming: bool = Field(default=True, alias="ENABLE_RESULT_STREAMING")
    enable_compression: bool = Field(default=True, alias="ENABLE_COMPRESSION")
    compression_algorithm: Literal["gzip", "brotli", "deflate"] = Field(default="gzip", alias="COMPRESSION_ALGORITHM")
    
    # Backup and Recovery Configuration
    enable_automatic_backups: bool = Field(default=True, alias="ENABLE_AUTOMATIC_BACKUPS")
    backup_interval_hours: int = Field(default=24, alias="BACKUP_INTERVAL_HOURS")
    backup_retention_days: int = Field(default=7, alias="BACKUP_RETENTION_DAYS")
    backup_storage_path: str = Field(default="./backups", alias="BACKUP_STORAGE_PATH")
    
    # Scaling Configuration
    enable_horizontal_scaling: bool = Field(default=False, alias="ENABLE_HORIZONTAL_SCALING")
    cluster_node_id: Optional[str] = Field(default=None, alias="CLUSTER_NODE_ID")
    cluster_discovery_method: Literal["static", "dns", "consul"] = Field(default="static", alias="CLUSTER_DISCOVERY_METHOD")
    load_balancer_algorithm: Literal["round_robin", "least_connections", "weighted"] = Field(
        default="round_robin", alias="LOAD_BALANCER_ALGORITHM"
    )
    
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
    def effective_hardware_profile(self) -> str:
        """Determine effective hardware profile."""
        if self.hardware_profile != "auto":
            return self.hardware_profile
        
        # Auto-detect based on GPU memory
        if self.gpu_memory_gb >= 24:
            return "desktop_24gb"
        elif self.gpu_memory_gb >= 16:
            return "laptop_16gb"
        else:
            return "generic"
    
    @computed_field
    @property
    def ollama_config(self) -> Dict[str, Any]:
        """Get complete Ollama configuration."""
        return {
            "base_url": self.ollama_base_url,
            "model": self.ollama_model,
            "timeout": self.ollama_timeout,
            "context_window": self.ollama_context_window,
            "temperature": self.ollama_temperature,
            "num_gpu": self.ollama_num_gpu,
            "num_thread": self.ollama_num_thread,
            "concurrent_requests": self.max_concurrent_llm_requests
        }
    
    @computed_field
    @property
    def hardware_config(self) -> Dict[str, Any]:
        """Get hardware optimization configuration."""
        return {
            "profile": self.effective_hardware_profile,
            "gpu_memory_gb": self.gpu_memory_gb,
            "max_concurrent_llm": self.max_concurrent_llm_requests,
            "enable_optimization": self.enable_hardware_optimization,
            "analysis_mode": self.schema_analysis_mode,
            "batch_size": self.schema_batch_size,
            "parallel_analysis": self.enable_parallel_analysis
        }
    
    @computed_field
    @property
    def cache_config(self) -> Dict[str, Any]:
        """Get comprehensive cache configuration."""
        return {
            "directory": self.cache_directory,
            "max_size_mb": self.max_cache_size_mb,
            "compression": self.enable_cache_compression,
            "analysis_ttl_hours": self.analysis_cache_ttl_hours,
            "fingerprint_ttl_hours": self.fingerprint_cache_ttl_hours
        }
    
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
    def is_staging(self) -> bool:
        """Check if running in staging environment."""
        return self.ENVIRONMENT == "staging"
    
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
            "pool_pre_ping": self.database_pool_pre_ping,
            "isolation_level": self.database_isolation_level,
            "statement_timeout": self.database_statement_timeout,
            "lock_timeout": self.database_lock_timeout,
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
            base_config.update(self.ollama_config)
        
        return base_config
    
    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_url(cls, v: str) -> str:
        """Validate Ollama base URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Ollama base URL must start with http:// or https://")
        return v
    
    @field_validator("gpu_memory_gb")
    @classmethod
    def validate_gpu_memory(cls, v: int) -> int:
        """Validate GPU memory is reasonable."""
        if v < 4:
            raise ValueError("GPU memory should be at least 4GB for local LLM")
        if v > 80:
            raise ValueError("GPU memory value seems unrealistic (max 80GB)")
        return v
    
    @field_validator("max_concurrent_llm_requests")
    @classmethod
    def validate_concurrent_requests(cls, v: int) -> int:
        """Validate concurrent LLM requests is reasonable."""
        if v < 1:
            raise ValueError("Must allow at least 1 concurrent LLM request")
        if v > 10:
            raise ValueError("Too many concurrent requests may overwhelm local LLM")
        return v
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get all feature flags."""
        return {
            "sql_validation": self.enable_sql_validation,
            "visualization": self.enable_visualization,
            "analysis": self.enable_analysis,
            "websockets": self.enable_websockets,
            "query_caching": self.enable_query_caching,
            "query_optimization": self.enable_query_optimization,
            "metrics": self.enable_metrics,
            "rag": self.rag_enabled,
            "business_intelligence": self.enable_business_intelligence,
            "performance_monitoring": self.enable_performance_monitoring,
            "real_time_updates": self.enable_real_time_updates,
            "ai_explanations": self.enable_ai_explanations,
            "query_suggestions": self.enable_query_suggestions,
            "auto_optimization": self.enable_auto_optimization,
            "collaborative_features": self.enable_collaborative_features,
            "export_features": self.enable_export_features,
        }
    
    def get_hardware_optimized_settings(self) -> Dict[str, Any]:
        """Get settings optimized for detected hardware profile."""
        profile = self.effective_hardware_profile
        
        if profile == "desktop_24gb":
            return {
                "max_concurrent_llm": 4,
                "analysis_mode": "deep",
                "batch_size": 15,
                "recommended_model": "mistral:13b",
                "enable_parallel": True,
                "memory_buffer_mb": 4096
            }
        elif profile == "laptop_16gb":
            return {
                "max_concurrent_llm": 2,
                "analysis_mode": "standard",
                "batch_size": 8,
                "recommended_model": "mistral:7b",
                "enable_parallel": True,
                "memory_buffer_mb": 2048
            }
        else:
            return {
                "max_concurrent_llm": 1,
                "analysis_mode": "quick",
                "batch_size": 5,
                "recommended_model": "mistral:7b",
                "enable_parallel": False,
                "memory_buffer_mb": 1024
            }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Environment-specific configuration overrides
class DevelopmentSettings(Settings):
    """Development environment settings with Ollama optimization."""
    ENVIRONMENT: str = "development"
    debug: bool = True
    log_level: str = "DEBUG"
    enable_request_logging: bool = True
    session_cookie_secure: bool = False
    enable_csrf_protection: bool = False
    enable_distributed_tracing: bool = False
    enable_automatic_backups: bool = False
    max_concurrent_queries: int = 5
    query_queue_size: int = 50
    
    # Ollama optimizations for development
    llm_provider: str = "ollama"
    schema_analysis_mode: str = "quick"
    max_concurrent_llm_requests: int = 2


class StagingSettings(Settings):
    """Staging environment settings."""
    ENVIRONMENT: str = "staging"
    debug: bool = False
    log_level: str = "INFO"
    session_cookie_secure: bool = True
    enable_csrf_protection: bool = True
    enable_distributed_tracing: bool = True
    enable_automatic_backups: bool = True
    enable_performance_monitoring: bool = True
    max_concurrent_queries: int = 15
    query_queue_size: int = 100


class ProductionSettings(Settings):
    """Production environment settings."""
    ENVIRONMENT: str = "production"
    debug: bool = False
    log_level: str = "INFO"
    cors_origins: List[str] = []  # Must be explicitly set
    ALLOWED_HOSTS: List[str] = []  # Must be explicitly set
    session_cookie_secure: bool = True
    enable_csrf_protection: bool = True
    enable_distributed_tracing: bool = True
    enable_automatic_backups: bool = True
    enable_performance_monitoring: bool = True
    enable_compression: bool = True
    enable_connection_pooling: bool = True
    max_concurrent_queries: int = 20
    query_queue_size: int = 200
    log_format: str = "structured"


def get_environment_settings() -> Settings:
    """Get environment-specific settings."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "staging":
        return StagingSettings()
    else:
        return DevelopmentSettings()


def validate_configuration() -> Dict[str, Any]:
    """Validate the current configuration and return status."""
    try:
        settings = get_settings()
        warnings = settings.validate_for_environment()
        
        status = {
            "valid": True,
            "environment": settings.ENVIRONMENT,
            "llm_provider": settings.effective_llm_provider,
            "hardware_profile": settings.effective_hardware_profile,
            "database": settings.database_type,
            "features_enabled": sum(settings.get_feature_flags().values()),
            "total_features": len(settings.get_feature_flags()),
            "warnings": warnings,
            "performance_config": {
                "max_concurrent_queries": settings.max_concurrent_queries,
                "query_timeout": settings.query_timeout_seconds,
                "cache_enabled": settings.enable_query_caching,
                "monitoring_enabled": settings.enable_performance_monitoring,
            },
            "hardware_config": settings.hardware_config,
            "ollama_config": settings.ollama_config if settings.effective_llm_provider == "ollama" else None
        }
        
        print(f"✅ Configuration valid for {settings.ENVIRONMENT} environment")
        print(f"   LLM Provider: {settings.effective_llm_provider}")
        print(f"   Hardware Profile: {settings.effective_hardware_profile}")
        print(f"   Database: {settings.database_type}")
        print(f"   Features: {status['features_enabled']}/{status['total_features']} enabled")
        
        if warnings:
            print(f"   ⚠️  {len(warnings)} warnings:")
            for warning in warnings:
                print(f"      - {warning}")
        
        return status
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return {
            "valid": False,
            "error": str(e),
            "environment": os.getenv("ENVIRONMENT", "unknown")
        }


def validate_for_environment(self) -> List[str]:
    """Validate configuration for specific environment and return warnings."""
    warnings = []
    
    if self.is_production:
        # Production-specific validations
        if self.max_concurrent_queries < 10:
            warnings.append("Consider increasing max_concurrent_queries for production")
        
        if not self.enable_performance_monitoring:
            warnings.append("Performance monitoring should be enabled in production")
        
        if not self.enable_automatic_backups:
            warnings.append("Automatic backups should be enabled in production")
        
        if self.log_level == "DEBUG":
            warnings.append("DEBUG log level not recommended for production")
        
        if not self.enable_compression:
            warnings.append("Response compression should be enabled in production")
        
        if self.effective_llm_provider == "ollama":
            warnings.append("Consider API-based LLM providers for production reliability")
    
    elif self.is_development:
        # Development-specific suggestions
        if self.enable_automatic_backups:
            warnings.append("Automatic backups can be disabled in development")
        
        if self.enable_distributed_tracing:
            warnings.append("Distributed tracing can be disabled in development for performance")
    
    # Ollama-specific validations
    if self.effective_llm_provider == "ollama":
        if self.max_concurrent_llm_requests > self.gpu_memory_gb // 6:
            warnings.append(f"Concurrent LLM requests may exceed GPU memory capacity")
        
        if self.schema_analysis_mode == "deep" and self.gpu_memory_gb < 20:
            warnings.append("Deep analysis mode may be slow on GPUs with <20GB memory")
    
    # General validations
    if self.database_pool_size > self.max_concurrent_queries * 3:
        warnings.append("Database pool size seems too large for concurrent query limit")
    
    if self.rag_enabled and not self.enable_business_intelligence:
        warnings.append("RAG is enabled but business intelligence is disabled")
    
    return warnings


def generate_env_template() -> str:
    """Generate a .env template file with all available options."""
    template = """# SQL Agent Configuration Template
# Copy this file to .env and update with your values

# Application Configuration
ENVIRONMENT=development
DEBUG=false
APP_VERSION=4.0.0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# LLM Configuration
LLM_PROVIDER=ollama
OPENAI_API_KEY=sk-your-openai-key-here
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b
OLLAMA_TIMEOUT=120
OLLAMA_CONTEXT_WINDOW=4096

# Hardware Configuration
GPU_MEMORY_GB=16
HARDWARE_PROFILE=auto
MAX_CONCURRENT_LLM_REQUESTS=2
ENABLE_HARDWARE_OPTIMIZATION=true

# Schema Analysis Configuration
SCHEMA_ANALYSIS_MODE=standard
SCHEMA_BATCH_SIZE=10
ENABLE_PARALLEL_ANALYSIS=true
ANALYSIS_CACHE_TTL_HOURS=24
FINGERPRINT_CACHE_TTL_HOURS=168

# Cache Configuration
CACHE_DIRECTORY=./cache
MAX_CACHE_SIZE_MB=2048
ENABLE_CACHE_COMPRESSION=true

# Database Configuration
DATABASE_TYPE=postgresql
DATABASE_URL=postgresql://username:password@localhost:5432/sql_agent_db
DATABASE_POOL_SIZE=20

# Vector Database Configuration
VECTOR_DB_TYPE=chromadb
CHROMA_DB_PATH=./chroma_db
EMBEDDING_MODEL=text-embedding-3-large

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Security Configuration
SECRET_KEY=your-very-long-secret-key-here-at-least-32-characters
ENABLE_CSRF_PROTECTION=true

# Performance Configuration
MAX_CONCURRENT_QUERIES=20
QUERY_TIMEOUT_SECONDS=60
ENABLE_QUERY_CACHING=true
ENABLE_QUERY_OPTIMIZATION=true

# Business Intelligence Features
ENABLE_BUSINESS_INTELLIGENCE=true
ENABLE_SCHEMA_PROFILING=true
ENABLE_DATA_QUALITY_CHECKS=true

# Monitoring Configuration
ENABLE_METRICS=true
ENABLE_PERFORMANCE_MONITORING=true
LOG_LEVEL=INFO

# Feature Flags
ENABLE_VISUALIZATION=true
ENABLE_ANALYSIS=true
ENABLE_AI_EXPLANATIONS=true
"""
    return template


# Global settings instance
settings = get_settings()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "validate":
            validate_configuration()
        elif command == "template":
            template = generate_env_template()
            with open(".env.template", "w") as f:
                f.write(template)
            print("✅ Generated .env.template file")
        elif command == "export":
            settings = get_settings()
            config = settings.export_for_frontend()
            print(json.dumps(config, indent=2))
        else:
            print("Usage: python config.py [validate|template|export]")
    else:
        validate_configuration()