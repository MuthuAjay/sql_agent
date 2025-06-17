"""SQL Agent - AI-powered SQL Agent with multi-agent architecture, RAG, and MCP integration."""

__version__ = "0.1.0"
__author__ = "SQL Agent Team"

from .core.config import Settings
from .core.state import AgentState

__all__ = ["Settings", "AgentState"] 