"""Structured logging for SQL Agent."""

import sys
import structlog
from typing import Any, Dict, Optional
from ..core.config import settings


def configure_logging() -> None:
    """Configure structured logging."""
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if settings.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def log_request_id(logger: structlog.BoundLogger, request_id: str) -> structlog.BoundLogger:
    """Add request ID to logger context."""
    return logger.bind(request_id=request_id)


def log_agent_decision(
    logger: structlog.BoundLogger,
    agent: str,
    decision: str,
    reasoning: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Log an agent decision with structured data."""
    log_data = {
        "agent": agent,
        "decision": decision,
        "reasoning": reasoning,
        **(metadata or {}),
    }
    logger.info("agent_decision", **log_data)


def log_query_execution(
    logger: structlog.BoundLogger,
    sql: str,
    execution_time: float,
    row_count: int,
    error: Optional[str] = None,
) -> None:
    """Log query execution details."""
    log_data = {
        "sql": sql,
        "execution_time": execution_time,
        "row_count": row_count,
        "error": error,
    }
    logger.info("query_executed", **log_data)


def log_performance_metric(
    logger: structlog.BoundLogger,
    metric_name: str,
    value: float,
    unit: str = "seconds",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a performance metric."""
    log_data = {
        "metric_name": metric_name,
        "value": value,
        "unit": unit,
        **(metadata or {}),
    }
    logger.info("performance_metric", **log_data)


# Configure logging on module import
configure_logging() 