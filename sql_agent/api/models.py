"""
Enhanced API Models

This module contains enhanced Pydantic models for API requests and responses,
providing validation, serialization, and documentation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Literal
from enum import Enum
import uuid

from pydantic import BaseModel, Field, validator, root_validator


class QueryIntent(str, Enum):
    """Query intent types."""
    SQL_GENERATION = "sql_generation"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    SCHEMA_INFO = "schema_info"
    DATA_EXPORT = "data_export"
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


class QueryStatus(str, Enum):
    """Query execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class AnalysisType(str, Enum):
    """Analysis types."""
    COMPREHENSIVE = "comprehensive"
    STATISTICAL = "statistical"
    BUSINESS = "business"
    QUALITY = "quality"
    TREND = "trend"
    CORRELATION = "correlation"


# Base Models
class BaseRequest(BaseModel):
    """Base request model with common fields."""
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")


class BaseResponse(BaseModel):
    """Base response model with common fields."""
    request_id: str = Field(..., description="Request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    processing_time: float = Field(..., description="Processing time in seconds")


# Enhanced Request Models
class QueryRequest(BaseRequest):
    """Enhanced request model for natural language queries."""
    query: str = Field(..., description="Natural language query", min_length=1, max_length=2000)
    database_name: Optional[str] = Field(None, description="Target database name")
    max_results: Optional[int] = Field(1000, description="Maximum number of results to return", ge=1, le=50000)
    include_analysis: Optional[bool] = Field(True, description="Include data analysis in response")
    include_visualization: Optional[bool] = Field(False, description="Include visualization in response")
    chart_type: Optional[ChartType] = Field(None, description="Preferred chart type for visualization")
    analysis_type: Optional[AnalysisType] = Field(AnalysisType.COMPREHENSIVE, description="Type of analysis to perform")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the query")
    session_id: Optional[str] = Field(None, description="Session identifier for context")
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query is not empty and contains meaningful content."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        
        # Check for minimum meaningful content
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
    optimize: Optional[bool] = Field(False, description="Optimize generated SQL")
    context: Optional[Dict[str, Any]] = Field(None, description="Query context and history")
    
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
    
    @validator('sql')
    def validate_sql(cls, v):
        """Enhanced SQL validation."""
        if not v.strip():
            raise ValueError("SQL cannot be empty")
        
        # Basic security checks
        sql_upper = v.upper().strip()
        
        # Allow SELECT, WITH, and EXPLAIN statements
        allowed_start_keywords = ['SELECT', 'WITH', 'EXPLAIN', '(']
        if not any(sql_upper.startswith(keyword) for keyword in allowed_start_keywords):
            raise ValueError("Only SELECT, WITH, and EXPLAIN statements are allowed")
        
        # Block potentially dangerous keywords
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
    include_insights: Optional[bool] = Field(True, description="Include business insights")
    include_recommendations: Optional[bool] = Field(True, description="Include recommendations")
    include_anomalies: Optional[bool] = Field(True, description="Include anomaly detection")
    confidence_threshold: Optional[float] = Field(0.8, description="Confidence threshold for insights", ge=0.0, le=1.0)
    
    @validator('data')
    def validate_data(cls, v):
        """Validate data is not empty and well-formed."""
        if not v:
            raise ValueError("Data cannot be empty")
        
        if len(v) > 100000:
            raise ValueError("Data cannot exceed 100,000 rows")
        
        # Check that all rows have consistent structure
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


class IndexInfo(BaseModel):
    """Database index information."""
    name: str = Field(..., description="Index name")
    columns: List[str] = Field(..., description="Indexed columns")
    unique: bool = Field(False, description="Whether index is unique")
    primary: bool = Field(False, description="Whether index is primary key")
    type: str = Field("btree", description="Index type")
    size: Optional[str] = Field(None, description="Index size")


class RelationshipInfo(BaseModel):
    """Database relationship information."""
    type: str = Field(..., description="Relationship type")
    source_table: str = Field(..., description="Source table")
    source_column: str = Field(..., description="Source column")
    target_table: str = Field(..., description="Target table")
    target_column: str = Field(..., description="Target column")
    constraint_name: Optional[str] = Field(None, description="Constraint name")
    on_delete: Optional[str] = Field(None, description="On delete action")
    on_update: Optional[str] = Field(None, description="On update action")


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


# Enhanced Result Models
class ValidationResult(BaseModel):
    """SQL validation result."""
    is_valid: bool = Field(..., description="Whether SQL is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    estimated_cost: Optional[float] = Field(None, description="Estimated query cost")
    estimated_rows: Optional[int] = Field(None, description="Estimated result rows")


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


class StatisticalSummary(BaseModel):
    """Statistical summary of data."""
    count: int = Field(..., description="Number of records")
    numeric_columns: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Numeric column statistics")
    categorical_columns: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Categorical column statistics")
    missing_values: Dict[str, int] = Field(default_factory=dict, description="Missing values per column")
    data_types: Dict[str, str] = Field(default_factory=dict, description="Data types per column")
    correlations: Optional[Dict[str, Dict[str, float]]] = Field(None, description="Correlation matrix")


class Insight(BaseModel):
    """Business insight."""
    type: str = Field(..., description="Insight type")
    title: str = Field(..., description="Insight title")
    description: str = Field(..., description="Insight description")
    confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    impact: str = Field(..., description="Expected impact level")
    supporting_data: Optional[Dict[str, Any]] = Field(None, description="Supporting data")


class Anomaly(BaseModel):
    """Data anomaly."""
    type: str = Field(..., description="Anomaly type")
    column: Optional[str] = Field(None, description="Affected column")
    description: str = Field(..., description="Anomaly description")
    severity: str = Field(..., description="Anomaly severity")
    affected_rows: Optional[int] = Field(None, description="Number of affected rows")
    threshold: Optional[float] = Field(None, description="Detection threshold")


class Trend(BaseModel):
    """Data trend."""
    type: str = Field(..., description="Trend type")
    column: str = Field(..., description="Column showing trend")
    direction: str = Field(..., description="Trend direction")
    strength: float = Field(..., description="Trend strength", ge=0.0, le=1.0)
    period: Optional[str] = Field(None, description="Trend period")
    description: str = Field(..., description="Trend description")


class Recommendation(BaseModel):
    """Actionable recommendation."""
    type: str = Field(..., description="Recommendation type")
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Recommendation description")
    priority: str = Field(..., description="Priority level")
    effort: str = Field(..., description="Implementation effort")
    expected_impact: str = Field(..., description="Expected impact")
    action_items: List[str] = Field(default_factory=list, description="Specific action items")


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


class ChartConfig(BaseModel):
    """Chart configuration."""
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


class ErrorResponse(BaseModel):
    """Enhanced error response model."""
    error: Dict[str, Any] = Field(..., description="Error details")
    request_id: str = Field(..., description="Request ID for tracing")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    user_message: str = Field(..., description="User-friendly error message")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions to fix the error")


class HealthResponse(BaseModel):
    """Enhanced health check response model."""
    status: str = Field(..., description="Overall health status")
    timestamp: float = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    services: Dict[str, str] = Field(..., description="Service health status")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")
    uptime: Optional[float] = Field(None, description="Uptime in seconds")


class HealthCheckResponse(BaseModel):
    status: str
    timestamp: float
    version: str
    services: Dict[str, Any]


class SchemaResponse(BaseModel):
    """Enhanced database schema response model."""
    database_name: str = Field(..., description="Database name")
    tables: List[TableInfo] = Field(..., description="Database tables")
    relationships: List[RelationshipInfo] = Field(default_factory=list, description="Cross-table relationships")
    total_tables: int = Field(..., description="Total number of tables")
    total_columns: int = Field(..., description="Total number of columns")
    schema_version: Optional[str] = Field(None, description="Schema version")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last schema update")


class FeedbackRequest(BaseRequest):
    """Enhanced user feedback request model."""
    query_id: str = Field(..., description="Original query ID")
    rating: int = Field(..., description="User rating (1-5)", ge=1, le=5)
    feedback: Optional[str] = Field(None, description="User feedback text", max_length=1000)
    category: Optional[str] = Field(None, description="Feedback category")
    specific_issues: List[str] = Field(default_factory=list, description="Specific issues identified")
    suggestions: Optional[str] = Field(None, description="User suggestions", max_length=500)
    
    @validator('rating')
    def validate_rating(cls, v):
        """Validate rating is within range."""
        if not 1 <= v <= 5:
            raise ValueError("Rating must be between 1 and 5")
        return v


# Pagination Models
class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(1, description="Page number", ge=1)
    limit: int = Field(20, description="Items per page", ge=1, le=1000)
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: Optional[str] = Field("asc", description="Sort order")


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    items: List[Any] = Field(..., description="Response items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page")
    limit: int = Field(..., description="Items per page")
    pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")


# Export configuration
__all__ = [
    "QueryIntent", "ChartType", "DatabaseType", "QueryStatus", "AnalysisType",
    "BaseRequest", "BaseResponse",
    "QueryRequest", "SQLGenerationRequest", "SQLExecutionRequest", 
    "AnalysisRequest", "VisualizationRequest",
    "ColumnInfo", "IndexInfo", "RelationshipInfo", "TableInfo", "DatabaseInfo",
    "ValidationResult", "SQLResult", "StatisticalSummary", "Insight", "Anomaly", 
    "Trend", "Recommendation", "AnalysisResult", "ChartConfig", "VisualizationResult",
    "QueryResponse", "ErrorResponse", "HealthResponse", "SchemaResponse", "FeedbackRequest",
    "PaginationParams", "PaginatedResponse"
]