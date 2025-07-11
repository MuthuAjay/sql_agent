"""MCP tools for SQL Agent."""

import json
from typing import Any, Dict, List, Optional
from ..utils.logging import get_logger
from .server import mcp_server

logger = get_logger("mcp.tools")


# Helper to get db_manager from context
def get_db_manager():
    ctx = mcp_server.get_context()
    return ctx.request_context.lifespan_context.db


# --- Database Tools ---

@mcp_server.tool(
    name="execute_query",
    namespace="database",
    description="Execute a SQL query and return results.",
    inputSchema={
        "type": "object",
        "properties": {
            "sql": {"type": "string", "description": "The SQL query to execute."},
            "limit": {"type": "integer", "description": "Maximum number of rows to return.", "default": 100}
        },
        "required": ["sql"]
    }
)
async def execute_query(sql: str, limit: int = 100) -> Dict[str, Any]:
    """Execute a SQL query and return results."""
    db_manager = get_db_manager()
    logger.info("executing_mcp_query", sql=sql)
    is_valid, error = await db_manager.validate_query(sql)
    if not is_valid:
        return {"error": f"Query validation failed: {error}"}
    
    result = await db_manager.execute_query(sql, timeout=30)
    if result.error:
        return {"error": f"Query execution failed: {result.error}"}
    
    return {
        "row_count": result.row_count,
        "execution_time": f"{result.execution_time:.2f}s",
        "data": result.data[:limit],
    }

@mcp_server.tool(
    name="get_sample_data",
    namespace="database",
    description="Get sample data from a table.",
    inputSchema={
        "type": "object",
        "properties": {
            "table_name": {"type": "string", "description": "The name of the table to get sample data from."},
            "limit": {"type": "integer", "description": "Maximum number of rows to return.", "default": 10}
        },
        "required": ["table_name"]
    }
)
async def get_sample_data(table_name: str, limit: int = 10) -> Dict[str, Any]:
    """Get sample data from a table."""
    db_manager = get_db_manager()
    logger.info("getting_mcp_sample_data", table_name=table_name)
    try:
        data = await db_manager.get_sample_data(table_name, limit)
        return {"table": table_name, "data": data}
    except Exception as e:
        logger.error("mcp_get_sample_data_failed", error=str(e))
        return {"error": str(e)}

@mcp_server.tool(
    name="validate_sql",
    namespace="database",
    description="Validate a SQL query without executing it.",
    inputSchema={
        "type": "object",
        "properties": {
            "sql": {"type": "string", "description": "The SQL query to validate."}
        },
        "required": ["sql"]
    }
)
async def validate_sql(sql: str) -> Dict[str, Any]:
    """Validate a SQL query without executing it."""
    db_manager = get_db_manager()
    logger.info("validating_mcp_sql", sql=sql)
    is_valid, error = await db_manager.validate_query(sql)
    return {"is_valid": is_valid, "error": error}

# --- Schema Tools ---

@mcp_server.tool(
    name="get_tables",
    namespace="schema",
    description="Get list of all tables in the database.",
    inputSchema={
        "type": "object",
        "properties": {}
    }
)
async def get_tables() -> Dict[str, Any]:
    """Get list of all tables in the database."""
    db_manager = get_db_manager()
    logger.info("getting_mcp_tables")
    try:
        schema_info = await db_manager.get_schema_info()
        tables = [
            {"name": name, "columns": len(info.get("columns", []))}
            for name, info in schema_info.items()
        ]
        return {"tables": tables}
    except Exception as e:
        logger.error("mcp_get_tables_failed", error=str(e))
        return {"error": str(e)}

@mcp_server.tool(
    name="get_columns",
    namespace="schema",
    description="Get column information for a specific table.",
    inputSchema={
        "type": "object",
        "properties": {
            "table_name": {"type": "string", "description": "The name of the table to get column information for."}
        },
        "required": ["table_name"]
    }
)
async def get_columns(table_name: str) -> Dict[str, Any]:
    """Get column information for a specific table."""
    db_manager = get_db_manager()
    logger.info("getting_mcp_columns", table_name=table_name)
    try:
        schema_info = await db_manager.get_schema_info()
        if table_name not in schema_info:
            return {"error": f"Table '{table_name}' not found"}
        
        return {"table": table_name, "columns": schema_info[table_name].get("columns", [])}
    except Exception as e:
        logger.error("mcp_get_columns_failed", error=str(e))
        return {"error": str(e)}

@mcp_server.tool(
    name="search_schema",
    namespace="schema",
    description="Search schema for tables and columns matching a keyword.",
    inputSchema={
        "type": "object",
        "properties": {
            "keyword": {"type": "string", "description": "The keyword to search for in the schema."}
        },
        "required": ["keyword"]
    }
)
async def search_schema(keyword: str) -> Dict[str, Any]:
    """Search schema for tables and columns matching a keyword."""
    db_manager = get_db_manager()
    logger.info("searching_mcp_schema", keyword=keyword)
    try:
        schema_info = await db_manager.get_schema_info()
        keyword_lower = keyword.lower()
        matches = []
        for table, info in schema_info.items():
            if keyword_lower in table.lower():
                matches.append({"type": "table", "name": table})
            for col in info.get("columns", []):
                if keyword_lower in col.get("column_name", "").lower():
                    matches.append({"type": "column", "name": f"{table}.{col['column_name']}"})
        return {"matches": matches}
    except Exception as e:
        logger.error("mcp_search_schema_failed", error=str(e))
        return {"error": str(e)}

@mcp_server.tool(
    name="get_relationships",
    namespace="schema",
    description="Get potential table relationships based on naming conventions.",
    inputSchema={
        "type": "object",
        "properties": {}
    }
)
async def get_relationships() -> Dict[str, Any]:
    """Get potential table relationships based on naming conventions."""
    db_manager = get_db_manager()
    logger.info("getting_mcp_relationships")
    try:
        schema_info = await db_manager.get_schema_info()
        relationships = []
        for table, info in schema_info.items():
            for col in info.get("columns", []):
                col_name = col.get("column_name", "")
                if col_name.endswith("_id"):
                    relationships.append({"potential_fk": f"{table}.{col_name}"})
        return {"relationships": relationships}
    except Exception as e:
        logger.error("mcp_get_relationships_failed", error=str(e))
        return {"error": str(e)}

# --- Visualization Tools ---

@mcp_server.tool(
    name="create_chart",
    namespace="visualization",
    description="Create a chart configuration from data.",
    inputSchema={
        "type": "object",
        "properties": {
            "chart_type": {"type": "string", "description": "The type of chart to create (e.g., bar, line, pie)."},
            "data": {"type": "string", "description": "The data for the chart in JSON string format."},
            "title": {"type": "string", "description": "The title of the chart.", "default": ""}
        },
        "required": ["chart_type", "data"]
    }
)
async def create_chart(chart_type: str, data: str, title: str = "") -> Dict[str, Any]:
    """Create a chart configuration from data."""
    logger.info("creating_mcp_chart", chart_type=chart_type)
    try:
        chart_data = json.loads(data)
        valid_types = ["bar", "line", "pie", "scatter", "histogram"]
        if chart_type not in valid_types:
            return {"error": f"Invalid chart type. Supported: {valid_types}"}
        
        return {
            "chart_type": chart_type,
            "title": title or f"{chart_type.title()} Chart",
            "data": chart_data,
        }
    except json.JSONDecodeError:
        return {"error": "Invalid data format. Please provide valid JSON."}
    except Exception as e:
        logger.error("mcp_create_chart_failed", error=str(e))
        return {"error": str(e)}

@mcp_server.tool(
    name="get_chart_types",
    namespace="visualization",
    description="Get available chart types and their descriptions.",
    inputSchema={
        "type": "object",
        "properties": {}
    }
)
async def get_chart_types() -> Dict[str, Any]:
    """Get available chart types and their descriptions."""
    logger.info("getting_mcp_chart_types")
    return {
        "chart_types": {
            "bar": "Compares values across categories.",
            "line": "Shows trends over time.",
            "pie": "Shows proportions of a whole.",
            "scatter": "Shows relationships between two variables.",
            "histogram": "Shows the distribution of a single variable.",
        }
    }

@mcp_server.tool(
    name="export_chart",
    namespace="visualization",
    description="Export chart in various formats (currently supports JSON).",
    inputSchema={
        "type": "object",
        "properties": {
            "chart_config": {"type": "string", "description": "The chart configuration in JSON string format."},
            "format": {"type": "string", "description": "The export format (e.g., json, html, png, svg).", "default": "json"}
        },
        "required": ["chart_config"]
    }
)
async def export_chart(chart_config: str, format: str = "json") -> Dict[str, Any]:
    """Export chart in various formats (currently supports JSON)."""
    logger.info("exporting_mcp_chart", format=format)
    try:
        config = json.loads(chart_config)
        if format == "json":
            return {"format": "json", "config": config}
        # In a real implementation, you would generate HTML, PNG, etc.
        return {"error": f"Format '{format}' not yet implemented."}
    except json.JSONDecodeError:
        return {"error": "Invalid chart configuration format."}
    except Exception as e:
        logger.error("mcp_export_chart_failed", error=str(e))
        return {"error": str(e)}

@mcp_server.tool(
    name="analyze_data_for_visualization",
    namespace="visualization",
    description="Analyze data and suggest appropriate chart types.",
    inputSchema={
        "type": "object",
        "properties": {
            "data": {"type": "string", "description": "The data to analyze in JSON string format."}
        },
        "required": ["data"]
    }
)
async def analyze_data_for_visualization(data: str) -> Dict[str, Any]:
    """Analyze data and suggest appropriate chart types."""
    logger.info("analyzing_mcp_data_for_viz")
    try:
        chart_data = json.loads(data)
        if not isinstance(chart_data, list) or not chart_data:
            return {"error": "Data must be a non-empty list of objects."}
        
        keys = chart_data[0].keys()
        suggestions = []
        if len(keys) >= 2:
            suggestions.extend(["bar", "line", "scatter"])
        if len(keys) == 1:
            suggestions.extend(["histogram", "pie"])
            
        return {"suggestions": suggestions, "columns": list(keys)}
    except json.JSONDecodeError:
        return {"error": "Invalid data format. Please provide valid JSON."}
    except Exception as e:
        logger.error("mcp_analyze_data_failed", error=str(e))
        return {"error": str(e)}