"""
API Models

This module contains Pydantic models for API requests and responses,
providing validation, serialization, and documentation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Literal
from enum import Enum

from pydantic import BaseModel, Field, validator


class QueryIntent(str, Enum):
    """Query intent types."""
    SQL_GENERATION = "sql_generation"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    SCHEMA_INFO = "schema_info"
    UNKNOWN = "unknown"


class ChartType(str, Enum):
    """Supported chart types."""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    TABLE = "table"


class QueryRequest(BaseModel):
    """Request model for natural language queries."""
    query: str = Field(..., description="Natural language query", min_length=1, max_length=1000)
    database_name: Optional[str] = Field(None, description="Target database name")
    max_results: Optional[int] = Field(100, description="Maximum number of results to return", ge=1, le=10000)
    include_analysis: Optional[bool] = Field(True, description="Include data analysis in response")
    include_visualization: Optional[bool] = Field(False, description="Include visualization in response")
    chart_type: Optional[ChartType] = Field(None, description="Preferred chart type for visualization")
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query is not empty and contains meaningful content."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class SQLGenerationRequest(BaseModel):
    """Request model for SQL generation."""
    query: str = Field(..., description="Natural language query", min_length=1, max_length=1000)
    database_name: Optional[str] = Field(None, description="Target database name")
    include_explanation: Optional[bool] = Field(True, description="Include SQL explanation")
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query is not empty."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class SQLExecutionRequest(BaseModel):
    """Request model for SQL execution."""
    sql: str = Field(..., description="SQL query to execute", min_length=1)
    database_name: Optional[str] = Field(None, description="Target database name")
    max_results: Optional[int] = Field(100, description="Maximum number of results to return", ge=1, le=10000)
    
    @validator('sql')
    def validate_sql(cls, v):
        """Basic SQL validation."""
        if not v.strip():
            raise ValueError("SQL cannot be empty")
        # Basic check for potentially dangerous operations
        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
        upper_sql = v.upper()
        for keyword in dangerous_keywords:
            if keyword in upper_sql:
                raise ValueError(f"SQL contains potentially dangerous keyword: {keyword}")
        return v.strip()


class AnalysisRequest(BaseModel):
    """Request model for data analysis."""
    data: List[Dict[str, Any]] = Field(..., description="Data to analyze")
    query_context: Optional[str] = Field(None, description="Original query context")
    analysis_type: Optional[str] = Field("comprehensive", description="Type of analysis to perform")
    
    @validator('data')
    def validate_data(cls, v):
        """Validate data is not empty."""
        if not v:
            raise ValueError("Data cannot be empty")
        return v


class VisualizationRequest(BaseModel):
    """Request model for data visualization."""
    data: List[Dict[str, Any]] = Field(..., description="Data to visualize")
    chart_type: Optional[ChartType] = Field(None, description="Chart type")
    title: Optional[str] = Field(None, description="Chart title")
    x_axis: Optional[str] = Field(None, description="X-axis column name")
    y_axis: Optional[str] = Field(None, description="Y-axis column name")
    
    @validator('data')
    def validate_data(cls, v):
        """Validate data is not empty."""
        if not v:
            raise ValueError("Data cannot be empty")
        return v


class ColumnInfo(BaseModel):
    """Database column information."""
    name: str = Field(..., description="Column name")
    type: str = Field(..., description="Column data type")
    nullable: bool = Field(..., description="Whether column is nullable")
    primary_key: bool = Field(False, description="Whether column is primary key")
    foreign_key: Optional[str] = Field(None, description="Foreign key reference")


class TableInfo(BaseModel):
    """Database table information."""
    name: str = Field(..., description="Table name")
    columns: List[ColumnInfo] = Field(..., description="Table columns")
    row_count: Optional[int] = Field(None, description="Approximate row count")
    description: Optional[str] = Field(None, description="Table description")


class SQLResult(BaseModel):
    """SQL execution result."""
    sql: str = Field(..., description="Executed SQL query")
    data: List[Dict[str, Any]] = Field(..., description="Query results")
    row_count: int = Field(..., description="Number of rows returned")
    execution_time: float = Field(..., description="Query execution time in seconds")
    columns: List[str] = Field(..., description="Column names")
    explanation: Optional[str] = Field(None, description="SQL explanation")


class AnalysisResult(BaseModel):
    """Data analysis result."""
    summary: Dict[str, Any] = Field(..., description="Statistical summary")
    insights: List[str] = Field(..., description="Business insights")
    anomalies: List[Dict[str, Any]] = Field(..., description="Detected anomalies")
    trends: List[Dict[str, Any]] = Field(..., description="Trend analysis")
    recommendations: List[str] = Field(..., description="Actionable recommendations")
    data_quality_score: float = Field(..., description="Data quality score (0-1)")


class VisualizationResult(BaseModel):
    """Data visualization result."""
    chart_type: ChartType = Field(..., description="Generated chart type")
    chart_config: Dict[str, Any] = Field(..., description="Chart configuration")
    chart_data: Dict[str, Any] = Field(..., description="Chart data")
    title: str = Field(..., description="Chart title")
    description: Optional[str] = Field(None, description="Chart description")
    export_formats: List[str] = Field(..., description="Available export formats")


class QueryResponse(BaseModel):
    """Response model for natural language queries."""
    query: str = Field(..., description="Original query")
    intent: QueryIntent = Field(..., description="Detected query intent")
    sql_result: Optional[SQLResult] = Field(None, description="SQL execution result")
    analysis_result: Optional[AnalysisResult] = Field(None, description="Data analysis result")
    visualization_result: Optional[VisualizationResult] = Field(None, description="Visualization result")
    processing_time: float = Field(..., description="Total processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: Dict[str, Any] = Field(..., description="Error details")
    request_id: str = Field(..., description="Request ID for tracing")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Overall health status")
    timestamp: float = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    services: Dict[str, str] = Field(..., description="Service health status")


class SchemaResponse(BaseModel):
    """Database schema response model."""
    database_name: str = Field(..., description="Database name")
    tables: List[TableInfo] = Field(..., description="Database tables")
    relationships: List[Dict[str, Any]] = Field(..., description="Table relationships")
    total_tables: int = Field(..., description="Total number of tables")
    total_columns: int = Field(..., description="Total number of columns")


class FeedbackRequest(BaseModel):
    """User feedback request model."""
    query_id: str = Field(..., description="Original query ID")
    rating: int = Field(..., description="User rating (1-5)", ge=1, le=5)
    feedback: Optional[str] = Field(None, description="User feedback text")
    category: Optional[str] = Field(None, description="Feedback category")
    
    @validator('rating')
    def validate_rating(cls, v):
        """Validate rating is within range."""
        if not 1 <= v <= 5:
            raise ValueError("Rating must be between 1 and 5")
        return v 


class DatabaseInfo(BaseModel):
    id: str
    name: str
    type: Literal["postgresql", "mysql", "sqlite", "mongodb"]
    status: Literal["connected", "disconnected", "connecting", "error"]
    lastSync: str 