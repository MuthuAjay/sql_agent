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
    
    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.1", alias="OLLAMA_MODEL")  # Updated version
    
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
    
    @computed_field
    @property
    def security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        return {
            "jwt_algorithm": self.jwt_algorithm,
            "access_token_expire_minutes": self.access_token_expire_minutes,
            "csrf_protection": self.enable_csrf_protection,
            "session_config": {
                "secure": self.session_cookie_secure,
                "httponly": self.session_cookie_httponly,
                "samesite": self.session_cookie_samesite,
            },
            "rate_limiting": {
                "per_minute": self.rate_limit_per_minute,
            }
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
    
    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Validate secret key strength."""
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
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
            if not self.session_cookie_secure:
                raise ValueError("Secure cookies should be enabled in production")
            if self.SECRET_KEY.startswith("dev-"):
                raise ValueError("Production secret key cannot use development default")
        
        return self
    
    @model_validator(mode="after")
    def validate_performance_settings(self):
        """Validate performance-related settings."""
        if self.max_concurrent_queries > 100:
            raise ValueError("Max concurrent queries should not exceed 100")
        
        if self.query_memory_limit_mb > 8192:  # 8GB
            raise ValueError("Query memory limit should not exceed 8GB")
        
        if self.embedding_batch_size > 1000:
            raise ValueError("Embedding batch size should not exceed 1000")
        
        return self
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration."""
        return {
            "url": self.redis_url,
            "cache_ttl": self.redis_cache_ttl,
            "session_ttl": self.redis_session_ttl,
            "max_connections": self.redis_max_connections,
        }
    
    def get_vector_db_config(self) -> Dict[str, Any]:
        """Get vector database configuration."""
        config = {
            "type": self.vector_db_type,
            "collection": self.vector_db_collection,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "batch_size": self.embedding_batch_size,
            "cache_ttl": self.embedding_cache_ttl,
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
            "retry_config": {
                "attempts": self.agent_retry_attempts,
                "delay": self.agent_retry_delay,
                "backoff_multiplier": self.agent_backoff_multiplier,
                "enable_failover": self.enable_agent_failover,
            },
            "circuit_breaker": {
                "failure_threshold": self.circuit_breaker_failure_threshold,
                "recovery_timeout": self.circuit_breaker_recovery_timeout,
                "half_open_max_calls": self.circuit_breaker_half_open_max_calls,
            },
            "performance": {
                "max_concurrent_queries": self.max_concurrent_queries,
                "query_queue_size": self.query_queue_size,
                "enable_parallelization": self.enable_query_parallelization,
                "enable_streaming": self.enable_result_streaming,
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
            "reranking_enabled": self.rag_reranking_enabled,
            "context_window_tokens": self.rag_context_window_tokens,
            "index_refresh_interval": self.rag_index_refresh_interval,
            "batch_update_size": self.rag_batch_update_size,
        }
    
    def get_business_intelligence_config(self) -> Dict[str, Any]:
        """Get business intelligence configuration."""
        return {
            "enabled": self.enable_business_intelligence,
            "domain_classification": self.business_domain_classification,
            "schema_profiling": {
                "enabled": self.enable_schema_profiling,
                "interval": self.schema_profiling_interval,
            },
            "data_quality": {
                "enabled": self.enable_data_quality_checks,
                "check_interval": self.data_quality_check_interval,
                "threshold": self.data_quality_threshold,
                "anomaly_detection": self.enable_anomaly_detection,
            }
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return {
            "metrics": {
                "enabled": self.enable_metrics,
                "port": self.metrics_port,
                "retention_days": self.performance_metrics_retention_days,
            },
            "performance": {
                "enabled": self.enable_performance_monitoring,
                "slow_query_logging": self.enable_slow_query_logging,
                "slow_query_threshold_ms": self.slow_query_threshold_ms,
            },
            "tracing": {
                "enabled": self.enable_distributed_tracing,
                "sample_rate": self.tracing_sample_rate,
            },
            "alerting": {
                "enabled": self.enable_alerting,
                "webhook_url": self.alerting_webhook_url,
            },
            "health_check_interval": self.health_check_interval,
        }
    
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
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            "level": self.log_level,
            "format": self.log_format,
            "file": self.log_file,
            "enable_request_logging": self.enable_request_logging,
            "retention_days": self.log_retention_days,
        }
    
    def get_backup_config(self) -> Dict[str, Any]:
        """Get backup and recovery configuration."""
        return {
            "enabled": self.enable_automatic_backups,
            "interval_hours": self.backup_interval_hours,
            "retention_days": self.backup_retention_days,
            "storage_path": self.backup_storage_path,
        }
    
    def get_scaling_config(self) -> Dict[str, Any]:
        """Get scaling configuration."""
        return {
            "horizontal_scaling": self.enable_horizontal_scaling,
            "node_id": self.cluster_node_id,
            "discovery_method": self.cluster_discovery_method,
            "load_balancer_algorithm": self.load_balancer_algorithm,
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
                "query_complexity_limit": self.query_complexity_limit,
                "query_memory_limit_mb": self.query_memory_limit_mb,
            },
            "llm_provider": self.effective_llm_provider,
            "business_intelligence": {
                "enabled": self.enable_business_intelligence,
                "domain_classification": self.business_domain_classification,
                "data_quality_checks": self.enable_data_quality_checks,
            },
            "performance": {
                "streaming_enabled": self.enable_result_streaming,
                "compression_enabled": self.enable_compression,
                "caching_enabled": self.enable_query_caching,
            }
        }
    
    def get_health_check_config(self) -> Dict[str, Any]:
        """Get health check configuration for monitoring."""
        return {
            "database": {
                "enabled": True,
                "timeout": self.database_pool_timeout,
                "max_retries": 3,
            },
            "llm_provider": {
                "enabled": True,
                "provider": self.effective_llm_provider,
                "timeout": 30,
            },
            "vector_db": {
                "enabled": self.rag_enabled,
                "type": self.vector_db_type,
                "timeout": 10,
            },
            "redis": {
                "enabled": True,
                "timeout": 5,
            },
            "agents": {
                "router": {"timeout": self.router_timeout},
                "sql": {"timeout": self.sql_timeout},
                "analysis": {"timeout": self.analysis_timeout},
                "visualization": {"timeout": self.visualization_timeout},
            }
        }
    
    def get_performance_thresholds(self) -> Dict[str, Any]:
        """Get performance monitoring thresholds."""
        return {
            "response_time": {
                "warning": 1000,  # 1 second
                "critical": 5000,  # 5 seconds
            },
            "query_execution": {
                "warning": self.slow_query_threshold_ms,
                "critical": self.slow_query_threshold_ms * 5,
            },
            "memory_usage": {
                "warning": self.query_memory_limit_mb * 0.8,
                "critical": self.query_memory_limit_mb * 0.95,
            },
            "concurrent_queries": {
                "warning": self.max_concurrent_queries * 0.8,
                "critical": self.max_concurrent_queries * 0.95,
            },
            "cache_hit_rate": {
                "warning": 0.7,  # 70%
                "critical": 0.5,  # 50%
            },
            "error_rate": {
                "warning": 0.05,  # 5%
                "critical": 0.1,   # 10%
            }
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
        
        elif self.is_development:
            # Development-specific suggestions
            if self.enable_automatic_backups:
                warnings.append("Automatic backups can be disabled in development")
            
            if self.enable_distributed_tracing:
                warnings.append("Distributed tracing can be disabled in development for performance")
        
        # General validations
        if self.database_pool_size > self.max_concurrent_queries * 3:
            warnings.append("Database pool size seems too large for concurrent query limit")
        
        if self.rag_enabled and not self.enable_business_intelligence:
            warnings.append("RAG is enabled but business intelligence is disabled")
        
        return warnings
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore unknown environment variables
        json_schema_extra = {
            "examples": [
                {
                    "ENVIRONMENT": "production",
                    "DATABASE_URL": "postgresql://user:pass@localhost:5432/sqldb",
                    "OPENAI_API_KEY": "sk-...",
                    "SECRET_KEY": "your-secret-key-here",
                    "REDIS_URL": "redis://localhost:6379/0",
                    "ENABLE_METRICS": True,
                    "ENABLE_BUSINESS_INTELLIGENCE": True,
                    "LOG_LEVEL": "INFO"
                }
            ]
        }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Environment-specific configuration overrides
class DevelopmentSettings(Settings):
    """Development environment settings."""
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


# Configuration validation and health check
def validate_configuration() -> Dict[str, Any]:
    """Validate the current configuration and return status."""
    try:
        settings = get_settings()
        warnings = settings.validate_for_environment()
        
        status = {
            "valid": True,
            "environment": settings.ENVIRONMENT,
            "llm_provider": settings.effective_llm_provider,
            "database": settings.database_type,
            "features_enabled": sum(settings.get_feature_flags().values()),
            "total_features": len(settings.get_feature_flags()),
            "warnings": warnings,
            "performance_config": {
                "max_concurrent_queries": settings.max_concurrent_queries,
                "query_timeout": settings.query_timeout_seconds,
                "cache_enabled": settings.enable_query_caching,
                "monitoring_enabled": settings.enable_performance_monitoring,
            }
        }
        
        print(f"✅ Configuration valid for {settings.ENVIRONMENT} environment")
        print(f"   LLM Provider: {settings.effective_llm_provider}")
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
API_PREFIX=/api/v1
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# Security Configuration
SECRET_KEY=your-very-long-secret-key-here-at-least-32-characters
ACCESS_TOKEN_EXPIRE_MINUTES=30
RATE_LIMIT_PER_MINUTE=100
ENABLE_CSRF_PROTECTION=true
SESSION_COOKIE_SECURE=false

# Database Configuration
DATABASE_TYPE=postgresql
DATABASE_URL=postgresql://username:password@localhost:5432/sql_agent_db
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# LLM Configuration
LLM_PROVIDER=auto
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-4o
ANTHROPIC_API_KEY=your-anthropic-key-here
GOOGLE_API_KEY=your-google-key-here

# Vector Database Configuration
VECTOR_DB_TYPE=chromadb
CHROMA_DB_PATH=./chroma_db
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSION=3072

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_CACHE_TTL=3600

# Performance Configuration
MAX_CONCURRENT_QUERIES=20
QUERY_TIMEOUT_SECONDS=60
MAX_ROWS_RETURNED=10000
ENABLE_QUERY_CACHING=true
ENABLE_QUERY_OPTIMIZATION=true

# Business Intelligence Features
ENABLE_BUSINESS_INTELLIGENCE=true
BUSINESS_DOMAIN_CLASSIFICATION=true
ENABLE_SCHEMA_PROFILING=true
ENABLE_DATA_QUALITY_CHECKS=true

# Monitoring Configuration
ENABLE_METRICS=true
ENABLE_PERFORMANCE_MONITORING=true
ENABLE_DISTRIBUTED_TRACING=true
LOG_LEVEL=INFO
LOG_FORMAT=structured

# Feature Flags
ENABLE_VISUALIZATION=true
ENABLE_ANALYSIS=true
ENABLE_AI_EXPLANATIONS=true
ENABLE_QUERY_SUGGESTIONS=true
ENABLE_AUTO_OPTIMIZATION=true

# File Upload Configuration
MAX_FILE_SIZE_MB=100
ALLOWED_FILE_TYPES=.csv,.json,.xlsx,.sql,.parquet

# Backup Configuration
ENABLE_AUTOMATIC_BACKUPS=true
BACKUP_INTERVAL_HOURS=24
BACKUP_RETENTION_DAYS=7
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