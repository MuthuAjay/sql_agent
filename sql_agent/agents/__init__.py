"""Multi-agent system for SQL Agent - Phase 2 Enhanced Intelligence."""

from .base import BaseAgent
from .router import RouterAgent
from .sql import SQLAgent
from .analysis import AnalysisAgent
from .viz import VisualizationAgent
from .orchestrator import AgentOrchestrator

# Use enhanced router by default, fallback to original if needed
# Fallback to original router (Phase 1)
DefaultRouterAgent = RouterAgent

__all__ = [
    "BaseAgent",
    "RouterAgent",
    "EnhancedRouterAgent", 
    "DefaultRouterAgent",
    "SQLAgent",
    "AnalysisAgent",
    "VisualizationAgent",
    "AgentOrchestrator",
]

# Version information
__version__ = "2.0"
__phase__ = "Intelligence Layer"