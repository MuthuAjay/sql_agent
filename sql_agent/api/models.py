"""
Enhanced API Models - Phase 3 Production Intelligence

This module contains enhanced Pydantic models for API requests and responses,
providing validation, serialization, documentation, and business intelligence.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Literal
from enum import Enum
import uuid

from pydantic import BaseModel, Field, validator, root_validator


# Enhanced Enums for Phase 3
class QueryIntent(str, Enum):
    """Query intent types."""
    SQL_GENERATION = "sql_generation"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    SCHEMA_INFO = "schema_info"
    DATA_EXPORT = "data_export"
    OPTIMIZATION = "optimization"
    MONITORING = "monitoring"
    UNKNOWN = "unknown"


class ChartType(str, Enum):
    """Supported chart types."""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    BOX = "box"
    AREA = "area"
    TABLE = "table"
    TREEMAP = "treemap"
    SANKEY = "sankey"
    FUNNEL = "funnel"


class DatabaseType(str, Enum):
    """Supported database types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"
    MSSQL = "mssql"
    ORACLE = "oracle"
    BIGQUERY = "bigquery"
    SNOWFLAKE = "snowflake"
    REDSHIFT = "redshift"
    CLICKHOUSE = "clickhouse"


class QueryStatus(str, Enum):
    """Query execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    CACHED = "cached"
    OPTIMIZED = "optimized"


class AnalysisType(str, Enum):
    """Analysis types."""
    COMPREHENSIVE = "comprehensive"
    STATISTICAL = "statistical"
    BUSINESS = "business"
    QUALITY = "quality"
    TREND = "trend"
    CORRELATION = "correlation"
    ANOMALY = "anomaly"
    FORECASTING = "forecasting"


class BusinessDomain(str, Enum):
    """Business domain classifications."""
    CUSTOMER_MANAGEMENT = "customer_management"
    PRODUCT_CATALOG = "product_catalog"
    ORDER_PROCESSING = "order_processing"
    FINANCIAL = "financial"
    HR_MANAGEMENT = "hr_management"
    MARKETING = "marketing"
    OPERATIONS = "operations"
    LOGISTICS = "logistics"
    COMPLIANCE = "compliance"
    ANALYTICS = "analytics"


class RelationshipType(str, Enum):
    """Database relationship types."""
    FOREIGN_KEY = "foreign_key"
    IMPLICIT = "implicit"
    INFERRED = "inferred"
    BUSINESS_RULE = "business_rule"
    SEMANTIC = "semantic"


class PerformanceLevel(str, Enum):
    """Performance level indicators."""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    CRITICAL = "critical"


class DataQualityLevel(str, Enum):
    """Data quality level indicators."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CRITICAL = "critical"


class OptimizationType(str, Enum):
    """Query optimization types."""
    INDEX_SUGGESTION = "index_suggestion"
    QUERY_REWRITE = "query_rewrite"
    PARTITION_SUGGESTION = "partition_suggestion"
    CACHING = "caching"
    MATERIALIZED_VIEW = "materialized_view"


# Phase 3 New Models
class BusinessDomainInfo(BaseModel):
    """Business domain information."""
    domain_name: BusinessDomain = Field(..., description="Business domain name")
    confidence: float = Field(..., description="Classification confidence", ge=0.0, le=1.0)
    tables: List[str] = Field(..., description="Tables in this domain")
    key_concepts: List[str] = Field(default_factory=list, description="Key business concepts")
    relationships: List[str] = Field(default_factory=list, description="Related domains")
    kpis: List[str] = Field(default_factory=list, description="Relevant KPIs")


class PerformanceMetrics(BaseModel):
    """Performance metrics for queries and tables."""
    execution_time_ms: Optional[float] = Field(None, description="Execution time in milliseconds")
    query_cost: Optional[float] = Field(None, description="Estimated query cost")
    index_usage: Dict[str, float] = Field(default_factory=dict, description="Index usage percentages")
    cardinality_estimates: Dict[str, int] = Field(default_factory=dict, description="Cardinality estimates")
    join_selectivity: Dict[str, float] = Field(default_factory=dict, description="Join selectivity factors")
    cache_hit_rate: Optional[float] = Field(None, description="Cache hit rate", ge=0.0, le=1.0)
    optimization_score: Optional[float] = Field(None, description="Query optimization score", ge=0.0, le=1.0)


class InferredRelationship(BaseModel):
    """Enhanced relationship information with inference."""
    relationship_type: RelationshipType = Field(..., description="Type of relationship")
    source_table: str = Field(..., description="Source table")
    source_column: str = Field(..., description="Source column")
    target_table: str = Field(..., description="Target table")
    target_column: str = Field(..., description="Target column")
    confidence: float = Field(..., description="Inference confidence", ge=0.0, le=1.0)
    cardinality: Literal["1:1", "1:many", "many:1", "many:many"] = Field(..., description="Relationship cardinality")
    join_cost: Optional[float] = Field(None, description="Estimated join cost")
    business_meaning: Optional[str] = Field(None, description="Business meaning of relationship")


class SchemaChange(BaseModel):
    """Schema evolution tracking."""
    change_type: Literal["table_added", "table_removed", "column_added", "column_modified", "index_added", "constraint_added"] = Field(..., description="Type of change")
    table_name: str = Field(..., description="Affected table")
    column_name: Optional[str] = Field(None, description="Affected column")
    timestamp: datetime = Field(..., description="Change timestamp")
    impact_score: float = Field(..., description="Impact score", ge=0.0, le=1.0)
    affected_queries: List[str] = Field(default_factory=list, description="Potentially affected queries")
    migration_required: bool = Field(False, description="Whether migration is required")


class AgentPerformance(BaseModel):
    """Agent performance metrics."""
    agent_name: str = Field(..., description="Agent identifier")
    success_rate: float = Field(..., description="Success rate", ge=0.0, le=1.0)
    avg_response_time: float = Field(..., description="Average response time in seconds")
    error_count: int = Field(..., description="Error count in last period")
    last_24h_queries: int = Field(..., description="Queries processed in last 24 hours")
    accuracy_score: Optional[float] = Field(None, description="Accuracy score", ge=0.0, le=1.0)
    optimization_suggestions: int = Field(0, description="Number of optimization suggestions provided")


class SystemMetrics(BaseModel):
    """System health and performance metrics."""
    cpu_usage: float = Field(..., description="CPU usage percentage", ge=0.0, le=100.0)
    memory_usage: float = Field(..., description="Memory usage percentage", ge=0.0, le=100.0)
    active_connections: int = Field(..., description="Active database connections")
    cache_hit_rate: float = Field(..., description="Cache hit rate", ge=0.0, le=1.0)
    avg_query_time: float = Field(..., description="Average query time in seconds")
    queries_per_second: float = Field(..., description="Queries per second")
    error_rate: float = Field(..., description="Error rate", ge=0.0, le=1.0)
    uptime_seconds: float = Field(..., description="System uptime in seconds")


class DomainInsight(BaseModel):
    """Domain-specific business insights."""
    domain: BusinessDomain = Field(..., description="Business domain")
    insight_type: str = Field(..., description="Type of insight")
    title: str = Field(..., description="Insight title")
    description: str = Field(..., description="Detailed description")
    business_impact: Literal["high", "medium", "low"] = Field(..., description="Business impact level")
    actionable_recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations")
    kpi_impact: Dict[str, float] = Field(default_factory=dict, description="KPI impact estimates")
    data_sources: List[str] = Field(default_factory=list, description="Supporting data sources")
    confidence: float = Field(..., description="Insight confidence", ge=0.0, le=1.0)


class QueryPattern(BaseModel):
    """Query pattern analysis."""
    pattern_type: str = Field(..., description="Pattern classification")
    pattern_description: str = Field(..., description="Pattern description")
    frequency: int = Field(..., description="Pattern frequency")
    avg_performance: float = Field(..., description="Average performance in seconds")
    optimization_potential: float = Field(..., description="Optimization potential score", ge=0.0, le=1.0)
    suggested_improvements: List[str] = Field(default_factory=list, description="Improvement suggestions")
    business_value: Optional[str] = Field(None, description="Business value assessment")


class OptimizationSuggestion(BaseModel):
    """Query and schema optimization suggestions."""
    type: OptimizationType = Field(..., description="Optimization type")
    title: str = Field(..., description="Suggestion title")
    description: str = Field(..., description="Detailed description")
    impact_score: float = Field(..., description="Expected impact score", ge=0.0, le=1.0)
    effort_level: Literal["low", "medium", "high"] = Field(..., description="Implementation effort")
    expected_improvement: str = Field(..., description="Expected improvement description")
    implementation_steps: List[str] = Field(default_factory=list, description="Implementation steps")
    sql_example: Optional[str] = Field(None, description="Example SQL for implementation")


class DataQualityAssessment(BaseModel):
    """Data quality assessment."""
    overall_score: float = Field(..., description="Overall quality score", ge=0.0, le=1.0)
    completeness: float = Field(..., description="Data completeness score", ge=0.0, le=1.0)
    accuracy: float = Field(..., description="Data accuracy score", ge=0.0, le=1.0)
    consistency: float = Field(..., description="Data consistency score", ge=0.0, le=1.0)
    timeliness: float = Field(..., description="Data timeliness score", ge=0.0, le=1.0)
    validity: float = Field(..., description="Data validity score", ge=0.0, le=1.0)
    issues_found: List[str] = Field(default_factory=list, description="Quality issues identified")
    recommendations: List[str] = Field(default_factory=list, description="Quality improvement recommendations")


# Configuration Models
class CacheConfig(BaseModel):
    """Caching configuration."""
    enabled: bool = Field(True, description="Enable caching")
    ttl_seconds: int = Field(1800, description="Time to live in seconds")
    max_size: int = Field(1000, description="Maximum cache size")
    strategy: Literal["lru", "lfu", "fifo"] = Field("lru", description="Cache eviction strategy")
    compression: bool = Field(False, description="Enable cache compression")


class SecurityConfig(BaseModel):
    """Security configuration."""
    rate_limit_per_minute: int = Field(60, description="Rate limit per minute")
    max_query_length: int = Field(10000, description="Maximum query length")
    allowed_functions: List[str] = Field(default_factory=list, description="Allowed SQL functions")
    blocked_keywords: List[str] = Field(default_factory=list, description="Blocked SQL keywords")
    enable_audit_log: bool = Field(True, description="Enable audit logging")
    require_authentication: bool = Field(True, description="Require user authentication")


class MonitoringConfig(BaseModel):
    """Monitoring and alerting configuration."""
    enable_metrics: bool = Field(True, description="Enable metrics collection")
    enable_tracing: bool = Field(True, description="Enable request tracing")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO", description="Log level")
    alert_thresholds: Dict[str, float] = Field(default_factory=dict, description="Alert thresholds")
    metric_retention_days: int = Field(30, description="Metric retention period")


# Base Models (Enhanced)
class BaseRequest(BaseModel):
    """Enhanced base request model with common fields."""
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    trace_id: Optional[str] = Field(None, description="Distributed tracing ID")


class BaseResponse(BaseModel):
    """Enhanced base response model with common fields."""
    request_id: str = Field(..., description="Request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    processing_time: float = Field(..., description="Processing time in seconds")
    cache_hit: bool = Field(False, description="Whether response was cached")
    trace_id: Optional[str] = Field(None, description="Distributed tracing ID")


# Enhanced Request Models
class QueryRequest(BaseRequest):
    """Enhanced request model for natural language queries."""
    query: str = Field(..., description="Natural language query", min_length=1, max_length=2000)
    database_name: Optional[str] = Field(None, description="Target database name")
    max_results: Optional[int] = Field(1000, description="Maximum number of results to return", ge=1, le=50000)
    include_analysis: Optional[bool] = Field(True, description="Include data analysis in response")
    include_visualization: Optional[bool] = Field(False, description="Include visualization in response")
    include_optimization: Optional[bool] = Field(False, description="Include optimization suggestions")
    chart_type: Optional[ChartType] = Field(None, description="Preferred chart type for visualization")
    analysis_type: Optional[AnalysisType] = Field(AnalysisType.COMPREHENSIVE, description="Type of analysis to perform")
    business_context: Optional[BusinessDomain] = Field(None, description="Business context for query")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the query")
    enable_caching: Optional[bool] = Field(True, description="Enable result caching")
    performance_mode: Optional[bool] = Field(False, description="Enable performance optimization mode")
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query is not empty and contains meaningful content."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        
        words = v.strip().split()
        if len(words) < 2:
            raise ValueError("Query must contain at least 2 words")
        
        return v.strip()
    
    @validator('max_results')
    def validate_max_results(cls, v):
        """Validate max_results is reasonable."""
        if v > 50000:
            raise ValueError("max_results cannot exceed 50,000")
        return v


class SQLGenerationRequest(BaseRequest):
    """Enhanced request model for SQL generation."""
    query: str = Field(..., description="Natural language query", min_length=1, max_length=2000)
    database_name: Optional[str] = Field(None, description="Target database name")
    include_explanation: Optional[bool] = Field(True, description="Include SQL explanation")
    include_validation: Optional[bool] = Field(True, description="Include SQL validation")
    include_optimization: Optional[bool] = Field(True, description="Include optimization suggestions")
    optimize: Optional[bool] = Field(False, description="Auto-optimize generated SQL")
    business_context: Optional[BusinessDomain] = Field(None, description="Business context for SQL generation")
    context: Optional[Dict[str, Any]] = Field(None, description="Query context and history")
    target_performance: Optional[PerformanceLevel] = Field(None, description="Target performance level")
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query is not empty."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class SQLExecutionRequest(BaseRequest):
    """Enhanced request model for SQL execution."""
    sql: str = Field(..., description="SQL query to execute", min_length=1)
    database_name: Optional[str] = Field(None, description="Target database name")
    max_results: Optional[int] = Field(1000, description="Maximum number of results to return", ge=1, le=50000)
    timeout: Optional[int] = Field(300, description="Query timeout in seconds", ge=1, le=3600)
    dry_run: Optional[bool] = Field(False, description="Validate query without execution")
    explain_plan: Optional[bool] = Field(False, description="Include execution plan")
    enable_caching: Optional[bool] = Field(True, description="Enable result caching")
    collect_metrics: Optional[bool] = Field(True, description="Collect performance metrics")
    
    @validator('sql')
    def validate_sql(cls, v):
        """Enhanced SQL validation."""
        if not v.strip():
            raise ValueError("SQL cannot be empty")
        
        sql_upper = v.upper().strip()
        
        allowed_start_keywords = ['SELECT', 'WITH', 'EXPLAIN', '(']
        if not any(sql_upper.startswith(keyword) for keyword in allowed_start_keywords):
            raise ValueError("Only SELECT, WITH, and EXPLAIN statements are allowed")
        
        dangerous_keywords = [
            'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 
            'UPDATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE'
        ]
        for keyword in dangerous_keywords:
            if f' {keyword} ' in f' {sql_upper} ':
                raise ValueError(f"SQL contains restricted keyword: {keyword}")
        
        return v.strip()


class AnalysisRequest(BaseRequest):
    """Enhanced request model for data analysis."""
    data: List[Dict[str, Any]] = Field(..., description="Data to analyze")
    query_context: Optional[str] = Field(None, description="Original query context")
    analysis_type: AnalysisType = Field(AnalysisType.COMPREHENSIVE, description="Type of analysis to perform")
    business_context: Optional[BusinessDomain] = Field(None, description="Business context for analysis")
    include_insights: Optional[bool] = Field(True, description="Include business insights")
    include_recommendations: Optional[bool] = Field(True, description="Include recommendations")
    include_anomalies: Optional[bool] = Field(True, description="Include anomaly detection")
    include_quality_assessment: Optional[bool] = Field(True, description="Include data quality assessment")
    confidence_threshold: Optional[float] = Field(0.8, description="Confidence threshold for insights", ge=0.0, le=1.0)
    
    @validator('data')
    def validate_data(cls, v):
        """Validate data is not empty and well-formed."""
        if not v:
            raise ValueError("Data cannot be empty")
        
        if len(v) > 100000:
            raise ValueError("Data cannot exceed 100,000 rows")
        
        if v:
            first_keys = set(v[0].keys())
            for i, row in enumerate(v[1:], 1):
                if set(row.keys()) != first_keys:
                    raise ValueError(f"Row {i} has inconsistent structure")
        
        return v


class VisualizationRequest(BaseRequest):
    """Enhanced request model for data visualization."""
    data: List[Dict[str, Any]] = Field(..., description="Data to visualize")
    chart_type: Optional[ChartType] = Field(None, description="Chart type")
    title: Optional[str] = Field(None, description="Chart title", max_length=100)
    x_axis: Optional[str] = Field(None, description="X-axis column name")
    y_axis: Optional[str] = Field(None, description="Y-axis column name")
    color_by: Optional[str] = Field(None, description="Color grouping column")
    size_by: Optional[str] = Field(None, description="Size mapping column")
    filters: Optional[Dict[str, Any]] = Field(None, description="Data filters")
    aggregation: Optional[str] = Field(None, description="Aggregation method")
    theme: Optional[str] = Field("light", description="Chart theme")
    interactive: Optional[bool] = Field(True, description="Enable interactivity")
    business_context: Optional[BusinessDomain] = Field(None, description="Business context for visualization")
    include_insights: Optional[bool] = Field(True, description="Include visual insights")
    
    @validator('data')
    def validate_data(cls, v):
        """Validate visualization data."""
        if not v:
            raise ValueError("Data cannot be empty")
        
        if len(v) > 50000:
            raise ValueError("Data for visualization cannot exceed 50,000 rows")
        
        return v


# Enhanced Information Models
class ColumnInfo(BaseModel):
    """Enhanced database column information."""
    name: str = Field(..., description="Column name")
    type: str = Field(..., description="Column data type")
    nullable: bool = Field(..., description="Whether column is nullable")
    primary_key: bool = Field(False, description="Whether column is primary key")
    foreign_key: Optional[str] = Field(None, description="Foreign key reference")
    unique: bool = Field(False, description="Whether column has unique constraint")
    indexed: bool = Field(False, description="Whether column is indexed")
    default_value: Optional[Any] = Field(None, description="Default value")
    max_length: Optional[int] = Field(None, description="Maximum length for string types")
    precision: Optional[int] = Field(None, description="Precision for numeric types")
    scale: Optional[int] = Field(None, description="Scale for numeric types")
    description: Optional[str] = Field(None, description="Column description")
    sample_values: Optional[List[Any]] = Field(None, description="Sample values from the column")
    business_concept: Optional[str] = Field(None, description="Business concept classification")
    data_quality_score: Optional[float] = Field(None, description="Data quality score", ge=0.0, le=1.0)
    cardinality: Optional[int] = Field(None, description="Estimated cardinality")
    null_percentage: Optional[float] = Field(None, description="Percentage of null values", ge=0.0, le=100.0)


class IndexInfo(BaseModel):
    """Enhanced database index information."""
    name: str = Field(..., description="Index name")
    columns: List[str] = Field(..., description="Indexed columns")
    unique: bool = Field(False, description="Whether index is unique")
    primary: bool = Field(False, description="Whether index is primary key")
    type: str = Field("btree", description="Index type")
    size: Optional[str] = Field(None, description="Index size")
    usage_frequency: Optional[float] = Field(None, description="Index usage frequency", ge=0.0, le=1.0)
    selectivity: Optional[float] = Field(None, description="Index selectivity", ge=0.0, le=1.0)
    performance_impact: Optional[PerformanceLevel] = Field(None, description="Performance impact level")


class RelationshipInfo(BaseModel):
    """Enhanced database relationship information."""
    type: RelationshipType = Field(..., description="Relationship type")
    source_table: str = Field(..., description="Source table")
    source_column: str = Field(..., description="Source column")
    target_table: str = Field(..., description="Target table")
    target_column: str = Field(..., description="Target column")
    constraint_name: Optional[str] = Field(None, description="Constraint name")
    on_delete: Optional[str] = Field(None, description="On delete action")
    on_update: Optional[str] = Field(None, description="On update action")
    cardinality: Optional[str] = Field(None, description="Relationship cardinality")
    confidence: Optional[float] = Field(None, description="Relationship confidence", ge=0.0, le=1.0)
    business_meaning: Optional[str] = Field(None, description="Business meaning of relationship")


class TableInfo(BaseModel):
    """Enhanced database table information."""
    name: str = Field(..., description="Table name")
    schema_name: Optional[str] = Field(None, description="Schema name")
    columns: List[ColumnInfo] = Field(..., description="Table columns")
    indexes: List[IndexInfo] = Field(default_factory=list, description="Table indexes")
    relationships: List[RelationshipInfo] = Field(default_factory=list, description="Table relationships")
    row_count: Optional[int] = Field(None, description="Approximate row count")
    size: Optional[str] = Field(None, description="Table size")
    description: Optional[str] = Field(None, description="Table description")
    created_at: Optional[datetime] = Field(None, description="Table creation time")
    updated_at: Optional[datetime] = Field(None, description="Last update time")
    primary_keys: List[str] = Field(default_factory=list, description="Primary key columns")
    foreign_keys: List[str] = Field(default_factory=list, description="Foreign key columns")
    
    # Phase 3 enhancements
    business_domains: List[BusinessDomain] = Field(default_factory=list, description="Business domain classifications")
    performance_score: Optional[float] = Field(None, description="Performance score", ge=0.0, le=1.0)
    data_quality_score: Optional[float] = Field(None, description="Data quality score", ge=0.0, le=1.0)
    usage_frequency: Optional[int] = Field(None, description="Usage frequency")
    optimization_suggestions: List[OptimizationSuggestion] = Field(default_factory=list, description="Optimization suggestions")
    inferred_relationships: List[InferredRelationship] = Field(default_factory=list, description="Inferred relationships")
    business_value: Optional[str] = Field(None, description="Business value assessment")


class DatabaseInfo(BaseModel):
    """Enhanced database information."""
    id: str = Field(..., description="Database identifier")
    name: str = Field(..., description="Database name")
    type: DatabaseType = Field(..., description="Database type")
    status: Literal["connected", "disconnected", "connecting", "error"] = Field(..., description="Connection status")
    host: Optional[str] = Field(None, description="Database host")
    port: Optional[int] = Field(None, description="Database port")
    version: Optional[str] = Field(None, description="Database version")
    size: Optional[str] = Field(None, description="Database size")
    table_count: Optional[int] = Field(None, description="Number of tables")
    last_sync: datetime = Field(default_factory=datetime.utcnow, description="Last synchronization time")
    capabilities: List[str] = Field(default_factory=list, description="Database capabilities")
    connection_pool_size: Optional[int] = Field(None, description="Connection pool size")
    
    # Phase 3 enhancements
    business_domains: List[BusinessDomainInfo] = Field(default_factory=list, description="Business domain distribution")
    performance_metrics: Optional[PerformanceMetrics] = Field(None, description="Performance metrics")
    data_quality_assessment: Optional[DataQualityAssessment] = Field(None, description="Data quality assessment")
    schema_changes: List[SchemaChange] = Field(default_factory=list, description="Recent schema changes")


# Enhanced Result Models
class ValidationResult(BaseModel):
    """Enhanced SQL validation result."""
    is_valid: bool = Field(..., description="Whether SQL is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    optimization_suggestions: List[OptimizationSuggestion] = Field(default_factory=list, description="Optimization suggestions")
    estimated_cost: Optional[float] = Field(None, description="Estimated query cost")
    estimated_rows: Optional[int] = Field(None, description="Estimated result rows")
    performance_level: Optional[PerformanceLevel] = Field(None, description="Expected performance level")
    security_score: Optional[float] = Field(None, description="Security score", ge=0.0, le=1.0)


class SQLResult(BaseModel):
    """Enhanced SQL execution result."""
    sql: str = Field(..., description="Executed SQL query")
    data: List[Dict[str, Any]] = Field(..., description="Query results")
    row_count: int = Field(..., description="Number of rows returned")
    total_rows: Optional[int] = Field(None, description="Total rows available (if limited)")
    execution_time: float = Field(..., description="Query execution time in seconds")
    columns: List[str] = Field(..., description="Column names")
    column_types: Optional[Dict[str, str]] = Field(None, description="Column data types")
    explanation: Optional[str] = Field(None, description="SQL explanation")
    query_plan: Optional[Dict[str, Any]] = Field(None, description="Query execution plan")
    cache_hit: bool = Field(False, description="Whether result was served from cache")
    warnings: List[str] = Field(default_factory=list, description="Query warnings")
    
    # Phase 3 enhancements
    performance_metrics: Optional[PerformanceMetrics] = Field(None, description="Performance metrics")
    optimization_suggestions: List[OptimizationSuggestion] = Field(default_factory=list, description="Optimization suggestions")
    business_context: Optional[BusinessDomain] = Field(None, description="Detected business context")
    data_quality_score: Optional[float] = Field(None, description="Result data quality score", ge=0.0, le=1.0)


class StatisticalSummary(BaseModel):
    """Enhanced statistical summary of data."""
    count: int = Field(..., description="Number of records")
    numeric_columns: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Numeric column statistics")
    categorical_columns: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Categorical column statistics")
    missing_values: Dict[str, int] = Field(default_factory=dict, description="Missing values per column")
    data_types: Dict[str, str] = Field(default_factory=dict, description="Data types per column")
    correlations: Optional[Dict[str, Dict[str, float]]] = Field(None, description="Correlation matrix")
    
    # Phase 3 enhancements
    data_quality_score: float = Field(..., description="Overall data quality score", ge=0.0, le=1.0)
    completeness_score: float = Field(..., description="Data completeness score", ge=0.0, le=1.0)
    consistency_score: float = Field(..., description="Data consistency score", ge=0.0, le=1.0)
    outlier_detection: Dict[str, List[Any]] = Field(default_factory=dict, description="Detected outliers per column")
    business_insights: List[str] = Field(default_factory=list, description="Business-relevant insights")


class Insight(BaseModel):
    """Enhanced business insight."""
    type: str = Field(..., description="Insight type")
    title: str = Field(..., description="Insight title")
    description: str = Field(..., description="Insight description")
    confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    impact: str = Field(..., description="Expected impact level")
    supporting_data: Optional[Dict[str, Any]] = Field(None, description="Supporting data")
    
    # Phase 3 enhancements
    business_domain: Optional[BusinessDomain] = Field(None, description="Related business domain")
    actionable: bool = Field(True, description="Whether insight is actionable")
    kpi_impact: Dict[str, float] = Field(default_factory=dict, description="KPI impact estimates")
    recommendation_priority: Optional[Literal["high", "medium", "low"]] = Field(None, description="Recommendation priority")
    implementation_effort: Optional[Literal["low", "medium", "high"]] = Field(None, description="Implementation effort")


class Anomaly(BaseModel):
    """Enhanced data anomaly."""
    type: str = Field(..., description="Anomaly type")
    column: Optional[str] = Field(None, description="Affected column")
    description: str = Field(..., description="Anomaly description")
    severity: str = Field(..., description="Anomaly severity")
    affected_rows: Optional[int] = Field(None, description="Number of affected rows")
    threshold: Optional[float] = Field(None, description="Detection threshold")
    
    # Phase 3 enhancements
    confidence: float = Field(..., description="Detection confidence", ge=0.0, le=1.0)
    business_impact: Optional[str] = Field(None, description="Business impact assessment")
    suggested_actions: List[str] = Field(default_factory=list, description="Suggested corrective actions")
    historical_pattern: Optional[bool] = Field(None, description="Whether this is a historical pattern")


class Trend(BaseModel):
    """Enhanced data trend."""
    type: str = Field(..., description="Trend type")
    column: str = Field(..., description="Column showing trend")
    direction: str = Field(..., description="Trend direction")
    strength: float = Field(..., description="Trend strength", ge=0.0, le=1.0)
    period: Optional[str] = Field(None, description="Trend period")
    description: str = Field(..., description="Trend description")
    
    # Phase 3 enhancements
    statistical_significance: float = Field(..., description="Statistical significance", ge=0.0, le=1.0)
    forecast_confidence: Optional[float] = Field(None, description="Forecast confidence", ge=0.0, le=1.0)
    business_relevance: Optional[str] = Field(None, description="Business relevance assessment")
    seasonality: Optional[bool] = Field(None, description="Whether trend shows seasonality")


class Recommendation(BaseModel):
    """Enhanced actionable recommendation."""
    type: str = Field(..., description="Recommendation type")
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Recommendation description")
    priority: str = Field(..., description="Priority level")
    effort: str = Field(..., description="Implementation effort")
    expected_impact: str = Field(..., description="Expected impact")
    action_items: List[str] = Field(default_factory=list, description="Specific action items")
    
    # Phase 3 enhancements
    business_domain: Optional[BusinessDomain] = Field(None, description="Related business domain")
    roi_estimate: Optional[float] = Field(None, description="Estimated ROI")
    timeline: Optional[str] = Field(None, description="Recommended implementation timeline")
    stakeholders: List[str] = Field(default_factory=list, description="Key stakeholders")
    success_metrics: List[str] = Field(default_factory=list, description="Success measurement metrics")


class AnalysisResult(BaseModel):
    """Enhanced data analysis result."""
    summary: StatisticalSummary = Field(..., description="Statistical summary")
    insights: List[Insight] = Field(default_factory=list, description="Business insights")
    anomalies: List[Anomaly] = Field(default_factory=list, description="Detected anomalies")
    trends: List[Trend] = Field(default_factory=list, description="Trend analysis")
    recommendations: List[Recommendation] = Field(default_factory=list, description="Actionable recommendations")
    data_quality_score: float = Field(..., description="Data quality score (0-1)", ge=0.0, le=1.0)
    confidence_score: float = Field(..., description="Overall confidence score", ge=0.0, le=1.0)
    processing_metadata: Optional[Dict[str, Any]] = Field(None, description="Processing metadata")
    
    # Phase 3 enhancements
    domain_insights: List[DomainInsight] = Field(default_factory=list, description="Domain-specific insights")
    query_patterns: List[QueryPattern] = Field(default_factory=list, description="Identified query patterns")
    optimization_opportunities: List[OptimizationSuggestion] = Field(default_factory=list, description="Optimization opportunities")
    business_value_assessment: Optional[str] = Field(None, description="Business value assessment")


class ChartConfig(BaseModel):
    """Enhanced chart configuration."""
    type: ChartType = Field(..., description="Chart type")
    title: str = Field(..., description="Chart title")
    x_axis: Optional[str] = Field(None, description="X-axis column")
    y_axis: Optional[str] = Field(None, description="Y-axis column")
    color_by: Optional[str] = Field(None, description="Color grouping column")
    size_by: Optional[str] = Field(None, description="Size mapping column")
    aggregation: Optional[str] = Field(None, description="Aggregation method")
    theme: str = Field("light", description="Chart theme")
    interactive: bool = Field(True, description="Enable interactivity")
    responsive: bool = Field(True, description="Enable responsiveness")
    animations: bool = Field(True, description="Enable animations")
    legend: bool = Field(True, description="Show legend")
    grid: bool = Field(True, description="Show grid")
    
    # Phase 3 enhancements
    accessibility_features: bool = Field(True, description="Enable accessibility features")
    export_options: List[str] = Field(default_factory=lambda: ["png", "svg", "pdf"], description="Export format options")
    drill_down_enabled: bool = Field(False, description="Enable drill-down functionality")
    real_time_updates: bool = Field(False, description="Enable real-time data updates")


class VisualizationResult(BaseModel):
    """Enhanced data visualization result."""
    chart_type: ChartType = Field(..., description="Generated chart type")
    chart_config: ChartConfig = Field(..., description="Chart configuration")
    chart_data: Dict[str, Any] = Field(..., description="Chart data")
    title: str = Field(..., description="Chart title")
    description: Optional[str] = Field(None, description="Chart description")
    export_formats: List[str] = Field(default_factory=lambda: ["json", "html", "png", "svg"], description="Available export formats")
    alternative_charts: List[ChartType] = Field(default_factory=list, description="Alternative chart suggestions")
    data_insights: List[str] = Field(default_factory=list, description="Insights from visualization")
    
    # Phase 3 enhancements
    business_context: Optional[BusinessDomain] = Field(None, description="Business context")
    storytelling_elements: List[str] = Field(default_factory=list, description="Data storytelling elements")
    interactive_features: List[str] = Field(default_factory=list, description="Available interactive features")
    performance_metrics: Optional[Dict[str, Any]] = Field(None, description="Rendering performance metrics")


# Enhanced Response Models
class QueryResponse(BaseResponse):
    """Enhanced response model for natural language queries."""
    query: str = Field(..., description="Original query")
    intent: QueryIntent = Field(..., description="Detected query intent")
    confidence: float = Field(..., description="Intent detection confidence", ge=0.0, le=1.0)
    sql_result: Optional[SQLResult] = Field(None, description="SQL execution result")
    analysis_result: Optional[AnalysisResult] = Field(None, description="Data analysis result")
    visualization_result: Optional[VisualizationResult] = Field(None, description="Visualization result")
    suggestions: List[str] = Field(default_factory=list, description="Follow-up suggestions")
    cached: bool = Field(False, description="Whether response was cached")
    
    # Phase 3 enhancements
    business_context: Optional[BusinessDomain] = Field(None, description="Detected business context")
    performance_metrics: Optional[PerformanceMetrics] = Field(None, description="Query performance metrics")
    optimization_suggestions: List[OptimizationSuggestion] = Field(default_factory=list, description="Optimization suggestions")
    related_queries: List[str] = Field(default_factory=list, description="Related query suggestions")
    data_lineage: Optional[List[str]] = Field(None, description="Data lineage information")
    compliance_notes: List[str] = Field(default_factory=list, description="Compliance and governance notes")


class ErrorResponse(BaseModel):
    """Enhanced error response model."""
    error: Dict[str, Any] = Field(..., description="Error details")
    request_id: str = Field(..., description="Request ID for tracing")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    user_message: str = Field(..., description="User-friendly error message")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions to fix the error")
    
    # Phase 3 enhancements
    error_code: Optional[str] = Field(None, description="Specific error code")
    retry_after: Optional[int] = Field(None, description="Retry after seconds")
    documentation_links: List[str] = Field(default_factory=list, description="Relevant documentation links")
    support_context: Optional[Dict[str, Any]] = Field(None, description="Context for support team")


class HealthResponse(BaseModel):
    """Enhanced health check response model."""
    status: str = Field(..., description="Overall health status")
    timestamp: float = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    services: Dict[str, str] = Field(..., description="Service health status")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")
    uptime: Optional[float] = Field(None, description="Uptime in seconds")
    
    # Phase 3 enhancements
    system_metrics: Optional[SystemMetrics] = Field(None, description="System performance metrics")
    agent_performance: List[AgentPerformance] = Field(default_factory=list, description="Agent performance metrics")
    database_health: Dict[str, str] = Field(default_factory=dict, description="Database health status")
    cache_status: Optional[Dict[str, Any]] = Field(None, description="Cache system status")
    security_status: Optional[str] = Field(None, description="Security system status")


class HealthCheckResponse(BaseModel):
    """Simplified health check response."""
    status: str
    timestamp: float
    version: str
    services: Dict[str, Any]


class SchemaResponse(BaseResponse):
    """Enhanced database schema response model."""
    database_name: str = Field(..., description="Database name")
    tables: List[TableInfo] = Field(..., description="Database tables")
    relationships: List[RelationshipInfo] = Field(default_factory=list, description="Cross-table relationships")
    total_tables: int = Field(..., description="Total number of tables")
    total_columns: int = Field(..., description="Total number of columns")
    schema_version: Optional[str] = Field(None, description="Schema version")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last schema update")
    
    # Phase 3 enhancements
    business_domains: List[BusinessDomainInfo] = Field(default_factory=list, description="Business domain analysis")
    inferred_relationships: List[InferredRelationship] = Field(default_factory=list, description="Inferred relationships")
    schema_quality_score: Optional[float] = Field(None, description="Overall schema quality score", ge=0.0, le=1.0)
    optimization_opportunities: List[OptimizationSuggestion] = Field(default_factory=list, description="Schema optimization opportunities")
    performance_insights: Optional[Dict[str, Any]] = Field(None, description="Performance analysis insights")
    compliance_assessment: Optional[Dict[str, Any]] = Field(None, description="Compliance and governance assessment")


class FeedbackRequest(BaseRequest):
    """Enhanced user feedback request model."""
    query_id: str = Field(..., description="Original query ID")
    rating: int = Field(..., description="User rating (1-5)", ge=1, le=5)
    feedback: Optional[str] = Field(None, description="User feedback text", max_length=1000)
    category: Optional[str] = Field(None, description="Feedback category")
    specific_issues: List[str] = Field(default_factory=list, description="Specific issues identified")
    suggestions: Optional[str] = Field(None, description="User suggestions", max_length=500)
    
    # Phase 3 enhancements
    business_context: Optional[BusinessDomain] = Field(None, description="Business context of feedback")
    feature_requests: List[str] = Field(default_factory=list, description="Feature requests")
    accuracy_rating: Optional[int] = Field(None, description="Result accuracy rating (1-5)", ge=1, le=5)
    performance_rating: Optional[int] = Field(None, description="Performance rating (1-5)", ge=1, le=5)
    
    @validator('rating', 'accuracy_rating', 'performance_rating')
    def validate_ratings(cls, v):
        """Validate ratings are within range."""
        if v is not None and not 1 <= v <= 5:
            raise ValueError("Rating must be between 1 and 5")
        return v


# Enhanced Pagination Models
class PaginationParams(BaseModel):
    """Enhanced pagination parameters."""
    page: int = Field(1, description="Page number", ge=1)
    limit: int = Field(20, description="Items per page", ge=1, le=1000)
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: Optional[str] = Field("asc", description="Sort order")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")
    include_total: bool = Field(True, description="Include total count")


class PaginatedResponse(BaseModel):
    """Enhanced paginated response wrapper."""
    items: List[Any] = Field(..., description="Response items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page")
    limit: int = Field(..., description="Items per page")
    pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")
    
    # Phase 3 enhancements
    filters_applied: Optional[Dict[str, Any]] = Field(None, description="Applied filters")
    sort_info: Optional[Dict[str, str]] = Field(None, description="Applied sorting")
    performance_metrics: Optional[Dict[str, float]] = Field(None, description="Query performance metrics")


# Monitoring and Analytics Models
class UsageAnalytics(BaseModel):
    """Usage analytics model."""
    period: str = Field(..., description="Analytics period")
    total_queries: int = Field(..., description="Total queries in period")
    successful_queries: int = Field(..., description="Successful queries")
    failed_queries: int = Field(..., description="Failed queries")
    avg_response_time: float = Field(..., description="Average response time")
    popular_query_types: Dict[str, int] = Field(default_factory=dict, description="Popular query types")
    business_domain_usage: Dict[str, int] = Field(default_factory=dict, description="Usage by business domain")
    user_satisfaction: Optional[float] = Field(None, description="Average user satisfaction rating", ge=1.0, le=5.0)


class AuditLog(BaseModel):
    """Audit log entry model."""
    id: str = Field(..., description="Log entry ID")
    timestamp: datetime = Field(..., description="Event timestamp")
    user_id: Optional[str] = Field(None, description="User identifier")
    action: str = Field(..., description="Action performed")
    resource: str = Field(..., description="Resource accessed")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    success: bool = Field(..., description="Whether action was successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")


# Export configuration
__all__ = [
    # Enums
    "QueryIntent", "ChartType", "DatabaseType", "QueryStatus", "AnalysisType",
    "BusinessDomain", "RelationshipType", "PerformanceLevel", "DataQualityLevel", "OptimizationType",
    
    # Base Models
    "BaseRequest", "BaseResponse",
    
    # Request Models
    "QueryRequest", "SQLGenerationRequest", "SQLExecutionRequest", 
    "AnalysisRequest", "VisualizationRequest",
    
    # Information Models
    "ColumnInfo", "IndexInfo", "RelationshipInfo", "TableInfo", "DatabaseInfo",
    
    # Result Models
    "ValidationResult", "SQLResult", "StatisticalSummary", "Insight", "Anomaly", 
    "Trend", "Recommendation", "AnalysisResult", "ChartConfig", "VisualizationResult",
    
    # Response Models
    "QueryResponse", "ErrorResponse", "HealthResponse", "HealthCheckResponse", 
    "SchemaResponse", "FeedbackRequest",
    
    # Pagination
    "PaginationParams", "PaginatedResponse",
    
    # Phase 3 Models
    "BusinessDomainInfo", "PerformanceMetrics", "InferredRelationship", "SchemaChange",
    "AgentPerformance", "SystemMetrics", "DomainInsight", "QueryPattern", 
    "OptimizationSuggestion", "DataQualityAssessment",
    
    # Configuration Models
    "CacheConfig", "SecurityConfig", "MonitoringConfig",
    
    # Analytics Models
    "UsageAnalytics", "AuditLog"]