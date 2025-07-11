"""MCP server implementation for SQL Agent."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from ..core.database import db_manager, DatabaseManager
from ..utils.logging import get_logger

logger = get_logger("mcp.server")


@dataclass
class AppContext:
    db: DatabaseManager


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context."""
    logger.info("mcp_lifespan_startup")
    # Initialize database manager
    await db_manager.initialize()
    try:
        yield AppContext(db=db_manager)
    finally:
        # Cleanup on shutdown
        await db_manager.close()
        logger.info("mcp_lifespan_shutdown")


mcp_server = FastMCP("SQL Agent", lifespan=app_lifespan)