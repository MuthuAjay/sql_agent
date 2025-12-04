"""Pydantic models for fraud detection."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FraudCategory(str, Enum):
    """Categories of fraud patterns."""
    TRANSACTION_ANOMALY = "transaction_anomaly"
    DATA_QUALITY_RISK = "data_quality_risk"
    ACCESS_PATTERN_RISK = "access_pattern_risk"
    RELATIONSHIP_INTEGRITY = "relationship_integrity"
    TEMPORAL_ANOMALY = "temporal_anomaly"
    STATISTICAL_ANOMALY = "statistical_anomaly"
    SCHEMA_VULNERABILITY = "schema_vulnerability"


class FraudScenario(BaseModel):
    """Individual fraud scenario detection."""
    scenario_id: str = Field(description="Unique identifier for this scenario")
    category: FraudCategory = Field(description="Category of fraud")
    risk_level: RiskLevel = Field(description="Severity of the risk")
    title: str = Field(description="Short title of the scenario")
    description: str = Field(description="Detailed description of the fraud scenario")
    reasoning: str = Field(description="Why this is a fraud risk")
    affected_columns: List[str] = Field(default_factory=list, description="Columns involved")
    detection_sql: Optional[str] = Field(None, description="SQL query to detect this fraud")
    prevention_recommendations: List[str] = Field(default_factory=list, description="How to prevent")
    detection_difficulty: str = Field(default="medium", description="easy, medium, hard")
    impact_severity: str = Field(default="medium", description="low, medium, high, critical")
    likelihood: float = Field(default=0.5, ge=0.0, le=1.0, description="Probability of occurrence")
    real_world_examples: List[str] = Field(default_factory=list, description="Real-world cases")
    compliance_violations: List[str] = Field(default_factory=list, description="Compliance issues")
    confidence_score: float = Field(default=0.7, ge=0.0, le=1.0, description="Detection confidence")


class SchemaVulnerability(BaseModel):
    """Schema-level vulnerability that enables fraud."""
    vulnerability_id: str = Field(description="Unique identifier")
    vulnerability_type: str = Field(description="Type of vulnerability")
    severity: RiskLevel = Field(description="Severity level")
    description: str = Field(description="What is vulnerable")
    affected_columns: List[str] = Field(default_factory=list)
    affected_tables: List[str] = Field(default_factory=list)
    remediation: str = Field(description="How to fix")
    sql_fix: Optional[str] = Field(None, description="SQL to fix the vulnerability")
    exploitability: str = Field(default="medium", description="How easy to exploit")


class DataQualityIssue(BaseModel):
    """Data quality issue that may indicate or enable fraud."""
    issue_id: str = Field(description="Unique identifier")
    issue_type: str = Field(description="Type of data quality issue")
    severity: RiskLevel = Field(description="Severity level")
    description: str = Field(description="Description of the issue")
    affected_columns: List[str] = Field(default_factory=list)
    affected_rows_estimate: Optional[int] = Field(None, description="Estimated affected rows")
    example_sql: Optional[str] = Field(None, description="SQL to find examples")
    impact: str = Field(description="Impact on fraud detection")
    remediation: str = Field(description="How to fix")


class StatisticalAnomaly(BaseModel):
    """Statistical anomaly detected in data."""
    anomaly_id: str = Field(description="Unique identifier")
    anomaly_type: str = Field(description="Type of anomaly")
    severity: RiskLevel = Field(description="Severity level")
    description: str = Field(description="Description of anomaly")
    affected_column: str = Field(description="Column with anomaly")
    statistical_test: str = Field(description="Test used to detect")
    test_statistic: Optional[float] = Field(None, description="Test result value")
    p_value: Optional[float] = Field(None, description="Statistical p-value")
    threshold: Optional[float] = Field(None, description="Detection threshold")
    sample_values: List[Any] = Field(default_factory=list, description="Example anomalous values")
    detection_sql: Optional[str] = Field(None, description="SQL to detect this anomaly")


class FraudAnalysisReport(BaseModel):
    """Complete fraud analysis report for a table."""
    report_id: str = Field(description="Unique report identifier")
    table_name: str = Field(description="Table analyzed")
    database_name: str = Field(description="Database name")
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    analysis_depth: str = Field(default="standard", description="quick, standard, deep")
    overall_risk_score: float = Field(ge=0.0, le=1.0, description="Overall risk score 0-1")
    total_scenarios: int = Field(default=0, description="Total fraud scenarios found")
    critical_count: int = Field(default=0, description="Critical risk scenarios")
    high_risk_count: int = Field(default=0, description="High risk scenarios")
    medium_risk_count: int = Field(default=0, description="Medium risk scenarios")
    low_risk_count: int = Field(default=0, description="Low risk scenarios")

    fraud_scenarios: List[FraudScenario] = Field(default_factory=list)
    schema_vulnerabilities: List[SchemaVulnerability] = Field(default_factory=list)
    data_quality_issues: List[DataQualityIssue] = Field(default_factory=list)
    statistical_anomalies: List[StatisticalAnomaly] = Field(default_factory=list)

    recommended_immediate_actions: List[str] = Field(default_factory=list)
    long_term_recommendations: List[str] = Field(default_factory=list)

    table_schema_summary: Dict[str, Any] = Field(default_factory=dict)
    analysis_metadata: Dict[str, Any] = Field(default_factory=dict)

    estimated_analysis_time: float = Field(default=0.0, description="Time taken in seconds")
    llm_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="LLM reasoning confidence")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "table_name": self.table_name,
            "overall_risk_score": self.overall_risk_score,
            "total_scenarios": self.total_scenarios,
            "risk_breakdown": {
                "critical": self.critical_count,
                "high": self.high_risk_count,
                "medium": self.medium_risk_count,
                "low": self.low_risk_count
            },
            "vulnerabilities": len(self.schema_vulnerabilities),
            "data_quality_issues": len(self.data_quality_issues),
            "statistical_anomalies": len(self.statistical_anomalies),
            "analysis_time": self.estimated_analysis_time
        }

    def get_high_priority_items(self) -> Dict[str, List]:
        """Get critical and high priority items only."""
        return {
            "critical_scenarios": [
                s for s in self.fraud_scenarios
                if s.risk_level == RiskLevel.CRITICAL
            ],
            "high_risk_scenarios": [
                s for s in self.fraud_scenarios
                if s.risk_level == RiskLevel.HIGH
            ],
            "critical_vulnerabilities": [
                v for v in self.schema_vulnerabilities
                if v.severity == RiskLevel.CRITICAL
            ]
        }


class FraudDetectionRequest(BaseModel):
    """Request model for fraud detection."""
    table_name: str = Field(description="Table to analyze")
    database_name: Optional[str] = Field(None, description="Database name")
    analysis_depth: str = Field(default="standard", description="quick, standard, deep")
    include_detection_sql: bool = Field(default=True, description="Generate SQL queries")
    focus_categories: Optional[List[FraudCategory]] = Field(None, description="Specific categories")
    industry_context: Optional[str] = Field(None, description="Industry context for analysis")


class FraudDetectionResponse(BaseModel):
    """Response model for fraud detection."""
    success: bool = Field(description="Whether analysis succeeded")
    report: Optional[FraudAnalysisReport] = Field(None, description="Analysis report")
    error: Optional[str] = Field(None, description="Error message if failed")
    summary: Dict[str, Any] = Field(default_factory=dict, description="Quick summary")
