"""Fraud detection module for SQL Agent."""

from .models import (
    FraudScenario,
    FraudAnalysisReport,
    SchemaVulnerability,
    DataQualityIssue,
    StatisticalAnomaly,
    FraudCategory,
    RiskLevel
)
from .patterns import FraudPatternLibrary
from .reporting import FraudReportGenerator

__all__ = [
    "FraudScenario",
    "FraudAnalysisReport",
    "SchemaVulnerability",
    "DataQualityIssue",
    "StatisticalAnomaly",
    "FraudCategory",
    "RiskLevel",
    "FraudPatternLibrary",
    "FraudReportGenerator"
]
