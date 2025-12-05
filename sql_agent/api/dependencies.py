# sql_agent/api/dependencies.py
from fastapi import Depends, HTTPException, status
from typing import Optional, Annotated, Dict, Any
import structlog

from sql_agent.core.database import DatabaseManager
from sql_agent.agents.orchestrator import AgentOrchestrator

logger = structlog.get_logger(__name__)

# Global instances (initialized in main.py)
_database_manager: Optional[DatabaseManager] = None
_orchestrator: Optional[AgentOrchestrator] = None
_fraud_detectors: Optional[Dict[str, Any]] = None
_fraud_report_generator: Optional[Any] = None
_enriched_cache: Optional[Any] = None
_enrichment_service: Optional[Any] = None
_schema_processor: Optional[Any] = None

def set_global_instances(db_manager: DatabaseManager, orchestrator: Optional[AgentOrchestrator]):
    """Set global instances for dependency injection."""
    global _database_manager, _orchestrator
    _database_manager = db_manager
    _orchestrator = orchestrator

def set_fraud_instances(detectors: Dict[str, Any], report_generator: Any):
    """Set fraud detection instances for dependency injection."""
    global _fraud_detectors, _fraud_report_generator
    _fraud_detectors = detectors
    _fraud_report_generator = report_generator

def set_schema_cache_instances(enriched_cache: Any, enrichment_service: Any, schema_processor: Any):
    """Set schema cache instances for dependency injection."""
    global _enriched_cache, _enrichment_service, _schema_processor
    _enriched_cache = enriched_cache
    _enrichment_service = enrichment_service
    _schema_processor = schema_processor

async def get_db_manager() -> DatabaseManager:
    """Dependency to get database manager instance."""
    if _database_manager is None:
        logger.error("Database manager not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database manager not available"
        )
    return _database_manager

async def get_orchestrator() -> AgentOrchestrator:
    """Dependency to get orchestrator instance."""
    if _orchestrator is None:
        logger.error("Agent orchestrator not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent orchestrator not available"
        )
    return _orchestrator

async def get_fraud_detectors() -> Dict[str, Any]:
    """Dependency to get fraud detectors."""
    if _fraud_detectors is None:
        logger.warning("Fraud detectors not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fraud detection service not available"
        )
    return _fraud_detectors

async def get_fraud_report_generator() -> Any:
    """Dependency to get fraud report generator."""
    if _fraud_report_generator is None:
        logger.warning("Fraud report generator not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fraud report generator not available"
        )
    return _fraud_report_generator

def get_enriched_cache() -> Any:
    """Dependency to get enriched schema cache instance."""
    if _enriched_cache is None:
        logger.error("Enriched schema cache not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Enriched schema cache not available"
        )
    return _enriched_cache

def get_enrichment_service() -> Any:
    """Dependency to get schema enrichment service instance."""
    if _enrichment_service is None:
        logger.error("Schema enrichment service not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Schema enrichment service not available"
        )
    return _enrichment_service

def get_schema_processor() -> Any:
    """Dependency to get schema processor instance."""
    if _schema_processor is None:
        logger.error("Schema processor not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Schema processor not available"
        )
    return _schema_processor

# Type aliases for cleaner dependency injection
DatabaseManagerDep = Annotated[DatabaseManager, Depends(get_db_manager)]
OrchestratorDep = Annotated[AgentOrchestrator, Depends(get_orchestrator)]
FraudDetectorsDep = Annotated[Dict[str, Any], Depends(get_fraud_detectors)]
FraudReportGeneratorDep = Annotated[Any, Depends(get_fraud_report_generator)]
EnrichedCacheDep = Annotated[Any, Depends(get_enriched_cache)]
EnrichmentServiceDep = Annotated[Any, Depends(get_enrichment_service)]
SchemaProcessorDep = Annotated[Any, Depends(get_schema_processor)]
