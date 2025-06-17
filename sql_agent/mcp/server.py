"""MCP server implementation for SQL Agent."""

import asyncio
import json
from typing import Any, Dict, List, Optional
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    TextContent,
    Tool,
    ToolResult
)

from .tools import DatabaseTools, SchemaTools, VisualizationTools
from ..utils.logging import get_logger


class MCPServer:
    """MCP server for SQL Agent tools."""
    
    def __init__(self):
        self.logger = get_logger("mcp.server")
        
        # Initialize tools
        self.database_tools = DatabaseTools()
        self.schema_tools = SchemaTools()
        self.visualization_tools = VisualizationTools()
        
        # Create MCP server
        self.server = Server("sql-agent")
        
        # Register tools
        self._register_tools()
    
    def _register_tools(self):
        """Register all available tools with the MCP server."""
        
        # Database tools
        self.server.list_tools = self._list_tools
        self.server.call_tool = self._call_tool
        
        # Tool definitions
        self.tools = {
            # Database tools
            "execute_query": Tool(
                name="execute_query",
                description="Execute a SQL query and return results",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "The SQL query to execute"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of rows to return",
                            "default": 100
                        }
                    },
                    "required": ["sql"]
                }
            ),
            "get_sample_data": Tool(
                name="get_sample_data",
                description="Get sample data from a table",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of rows to return",
                            "default": 10
                        }
                    },
                    "required": ["table_name"]
                }
            ),
            "validate_sql": Tool(
                name="validate_sql",
                description="Validate a SQL query without executing it",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "The SQL query to validate"
                        }
                    },
                    "required": ["sql"]
                }
            ),
            
            # Schema tools
            "get_tables": Tool(
                name="get_tables",
                description="Get list of all tables in the database",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            "get_columns": Tool(
                name="get_columns",
                description="Get column information for a specific table",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table"
                        }
                    },
                    "required": ["table_name"]
                }
            ),
            "search_schema": Tool(
                name="search_schema",
                description="Search schema for tables and columns matching a keyword",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "Keyword to search for"
                        }
                    },
                    "required": ["keyword"]
                }
            ),
            "get_relationships": Tool(
                name="get_relationships",
                description="Get table relationships and foreign keys",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            
            # Visualization tools
            "create_chart": Tool(
                name="create_chart",
                description="Create a chart from data",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "chart_type": {
                            "type": "string",
                            "description": "Type of chart to create",
                            "enum": ["bar", "line", "pie", "scatter", "histogram"]
                        },
                        "data": {
                            "type": "string",
                            "description": "JSON data for the chart"
                        },
                        "title": {
                            "type": "string",
                            "description": "Chart title"
                        }
                    },
                    "required": ["chart_type", "data"]
                }
            ),
            "get_chart_types": Tool(
                name="get_chart_types",
                description="Get available chart types and their descriptions",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            "export_chart": Tool(
                name="export_chart",
                description="Export chart in various formats",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "chart_config": {
                            "type": "string",
                            "description": "Chart configuration JSON"
                        },
                        "format": {
                            "type": "string",
                            "description": "Export format",
                            "enum": ["json", "html", "png", "svg"],
                            "default": "json"
                        }
                    },
                    "required": ["chart_config"]
                }
            ),
            "analyze_data_for_visualization": Tool(
                name="analyze_data_for_visualization",
                description="Analyze data and suggest appropriate chart types",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "string",
                            "description": "JSON data to analyze"
                        }
                    },
                    "required": ["data"]
                }
            )
        }
    
    async def _list_tools(self, request: ListToolsRequest) -> ListToolsResult:
        """List all available tools."""
        self.logger.info("mcp_list_tools_requested")
        
        tools = list(self.tools.values())
        return ListToolsResult(tools=tools)
    
    async def _call_tool(self, request: CallToolRequest) -> CallToolResult:
        """Call a specific tool."""
        tool_name = request.name
        arguments = request.arguments
        
        self.logger.info("mcp_tool_called", tool_name=tool_name, arguments=arguments)
        
        try:
            # Route to appropriate tool handler
            if tool_name in ["execute_query", "get_sample_data", "validate_sql"]:
                result = await self._handle_database_tool(tool_name, arguments)
            elif tool_name in ["get_tables", "get_columns", "search_schema", "get_relationships"]:
                result = await self._handle_schema_tool(tool_name, arguments)
            elif tool_name in ["create_chart", "get_chart_types", "export_chart", "analyze_data_for_visualization"]:
                result = await self._handle_visualization_tool(tool_name, arguments)
            else:
                result = TextContent(
                    type="text",
                    text=f"Unknown tool: {tool_name}"
                )
            
            return CallToolResult(
                content=[result]
            )
            
        except Exception as e:
            self.logger.error("mcp_tool_call_failed", tool_name=tool_name, error=str(e))
            
            error_result = TextContent(
                type="text",
                text=f"Tool execution failed: {str(e)}"
            )
            
            return CallToolResult(
                content=[error_result]
            )
    
    async def _handle_database_tool(self, tool_name: str, arguments: Dict[str, Any]) -> TextContent:
        """Handle database-related tool calls."""
        if tool_name == "execute_query":
            sql = arguments.get("sql", "")
            limit = arguments.get("limit", 100)
            return await self.database_tools.execute_query(sql, limit)
        
        elif tool_name == "get_sample_data":
            table_name = arguments.get("table_name", "")
            limit = arguments.get("limit", 10)
            return await self.database_tools.get_sample_data(table_name, limit)
        
        elif tool_name == "validate_sql":
            sql = arguments.get("sql", "")
            return await self.database_tools.validate_sql(sql)
        
        else:
            return TextContent(
                type="text",
                text=f"Unknown database tool: {tool_name}"
            )
    
    async def _handle_schema_tool(self, tool_name: str, arguments: Dict[str, Any]) -> TextContent:
        """Handle schema-related tool calls."""
        if tool_name == "get_tables":
            return await self.schema_tools.get_tables()
        
        elif tool_name == "get_columns":
            table_name = arguments.get("table_name", "")
            return await self.schema_tools.get_columns(table_name)
        
        elif tool_name == "search_schema":
            keyword = arguments.get("keyword", "")
            return await self.schema_tools.search_schema(keyword)
        
        elif tool_name == "get_relationships":
            return await self.schema_tools.get_relationships()
        
        else:
            return TextContent(
                type="text",
                text=f"Unknown schema tool: {tool_name}"
            )
    
    async def _handle_visualization_tool(self, tool_name: str, arguments: Dict[str, Any]) -> TextContent:
        """Handle visualization-related tool calls."""
        if tool_name == "create_chart":
            chart_type = arguments.get("chart_type", "")
            data = arguments.get("data", "")
            title = arguments.get("title", "")
            return await self.visualization_tools.create_chart(chart_type, data, title)
        
        elif tool_name == "get_chart_types":
            return await self.visualization_tools.get_chart_types()
        
        elif tool_name == "export_chart":
            chart_config = arguments.get("chart_config", "")
            format_type = arguments.get("format", "json")
            return await self.visualization_tools.export_chart(chart_config, format_type)
        
        elif tool_name == "analyze_data_for_visualization":
            data = arguments.get("data", "")
            return await self.visualization_tools.analyze_data_for_visualization(data)
        
        else:
            return TextContent(
                type="text",
                text=f"Unknown visualization tool: {tool_name}"
            )
    
    async def start(self):
        """Start the MCP server."""
        self.logger.info("mcp_server_starting")
        
        try:
            # Initialize database connection
            from ..core.database import db_manager
            await db_manager.initialize()
            self.logger.info("mcp_server_database_connected")
            
            # Start the server
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="sql-agent",
                        server_version="1.0.0",
                        capabilities=self.server.get_capabilities(
                            notification_options=None,
                            experimental_capabilities=None
                        )
                    )
                )
                
        except Exception as e:
            self.logger.error("mcp_server_start_failed", error=str(e))
            raise
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get information about available tools."""
        return {
            "database_tools": [
                "execute_query",
                "get_sample_data", 
                "validate_sql"
            ],
            "schema_tools": [
                "get_tables",
                "get_columns",
                "search_schema",
                "get_relationships"
            ],
            "visualization_tools": [
                "create_chart",
                "get_chart_types",
                "export_chart",
                "analyze_data_for_visualization"
            ],
            "total_tools": len(self.tools)
        }


async def main():
    """Main entry point for the MCP server."""
    server = MCPServer()
    await server.start()


if __name__ == "__main__":
    asyncio.run(main()) 