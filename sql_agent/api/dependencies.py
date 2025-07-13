# sql_agent/api/dependencies.py
from fastapi import Depends, HTTPException, status
from typing import Optional, Annotated
import structlog

from sql_agent.core.database import DatabaseManager
from sql_agent.agents.orchestrator import AgentOrchestrator

logger = structlog.get_logger(__name__)

# Global instances (initialized in main.py)
_database_manager: Optional[DatabaseManager] = None
_orchestrator: Optional[AgentOrchestrator] = None

def set_global_instances(db_manager: DatabaseManager, orchestrator: Optional[AgentOrchestrator]):
    """Set global instances for dependency injection."""
    global _database_manager, _orchestrator
    _database_manager = db_manager
    _orchestrator = orchestrator

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

# Type aliases for cleaner dependency injection
DatabaseManagerDep = Annotated[DatabaseManager, Depends(get_db_manager)]
OrchestratorDep = Annotated[AgentOrchestrator, Depends(get_orchestrator)]
