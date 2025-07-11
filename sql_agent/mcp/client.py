"""MCP client for SQL Agent."""

import asyncio
from typing import Any, Dict, List
from mcp.client.session import ClientSession
from ..utils.logging import get_logger

logger = get_logger("mcp.client")

class MCPClient:
    """Client for interacting with the SQL Agent MCP server."""

    def __init__(self, client_session: ClientSession):
        self.client = client_session

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools from the MCP server."""
        logger.info("listing_mcp_tools")
        try:
            tools = await self.client.list_tools()
            return [tool.dict() for tool in tools]
        except Exception as e:
            logger.error("mcp_list_tools_failed", error=str(e))
            return []

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a specific tool on the MCP server."""
        logger.info("calling_mcp_tool", tool_name=tool_name, arguments=arguments)
        try:
            result = await self.client.call_tool(tool_name, arguments)
            return result.dict()
        except Exception as e:
            logger.error("mcp_client_tool_call_failed", tool_name=tool_name, error=str(e))
            return {"error": str(e)}