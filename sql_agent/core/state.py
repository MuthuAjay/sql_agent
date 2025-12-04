"""State management for SQL Agent."""

from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from datetime import datetime
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from sql_agent.fraud.models import FraudAnalysisReport


class QueryResult(BaseModel):
    """Result of a SQL query execution."""
    
    data: List[Dict[str, Any]] = Field(default_factory=list)
    columns: List[str] = Field(default_factory=list)
    row_count: int = Field(default=0)
    execution_time: float = Field(default=0.0)
    sql_query: str = Field(default="")
    error: Optional[str] = Field(default=None)
    is_validated: bool = Field(default=False)


class SchemaContext(BaseModel):
    """Database schema context for RAG retrieval."""
    
    table_name: str
    column_name: Optional[str] = None
    data_type: Optional[str] = None
    description: Optional[str] = None
    sample_values: List[str] = Field(default_factory=list)
    relationships: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = Field(default=None)


class AnalysisResult(BaseModel):
    """Result of query analysis."""
    
    insights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    data_quality_score: Optional[float] = Field(default=None)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)


class VisualizationConfig(BaseModel):
    """Configuration for data visualization."""
    
    chart_type: str = Field(default="bar")  # bar, line, pie, scatter, etc.
    x_axis: Optional[str] = Field(default=None)
    y_axis: Optional[str] = Field(default=None)
    title: Optional[str] = Field(default=None)
    color_scheme: Optional[str] = Field(default=None)
    config: Dict[str, Any] = Field(default_factory=dict)


class AgentState(BaseModel):
    """State shared between all agents in the workflow."""
    
    # Core state
    query: str = Field(description="Original user query")
    session_id: str = Field(description="Unique session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Agent routing
    current_agent: Optional[str] = Field(default=None)
    agent_history: List[str] = Field(default_factory=list)
    
    # Context and data
    schema_context: List[SchemaContext] = Field(default_factory=list)
    database_name: Optional[str] = Field(default=None)
    
    # Query processing
    generated_sql: Optional[str] = Field(default=None)
    query_result: Optional[QueryResult] = Field(default=None)
    analysis_result: Optional[AnalysisResult] = Field(default=None)
    visualization_config: Optional[VisualizationConfig] = Field(default=None)
    fraud_analysis_result: Optional['FraudAnalysisReport'] = Field(default=None, description="Fraud detection analysis result")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Performance tracking
    start_time: Optional[datetime] = Field(default=None)
    end_time: Optional[datetime] = Field(default=None)
    processing_time: Optional[float] = Field(default=None)
    
    def add_error(self, error: str) -> None:
        """Add an error to the state."""
        self.errors.append(error)
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the state."""
        self.warnings.append(warning)
    
    def set_current_agent(self, agent_name: str) -> None:
        """Set the current agent and add to history."""
        self.current_agent = agent_name
        self.agent_history.append(agent_name)
    
    def is_complete(self) -> bool:
        """Check if the workflow is complete."""
        return (
            self.generated_sql is not None
            and self.query_result is not None
            and self.query_result.error is None
        )
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0 or (
            self.query_result is not None and self.query_result.error is not None
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current state."""
        return {
            "session_id": self.session_id,
            "query": self.query,
            "current_agent": self.current_agent,
            "agent_history": self.agent_history,
            "has_sql": self.generated_sql is not None,
            "has_results": self.query_result is not None,
            "has_analysis": self.analysis_result is not None,
            "has_visualization": self.visualization_config is not None,
            "has_fraud_analysis": self.fraud_analysis_result is not None,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "processing_time": self.processing_time,
        } 