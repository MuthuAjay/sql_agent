"""MCP client for SQL Agent."""

import asyncio
import json
from typing import Any, Dict, List, Optional
from ..utils.logging import get_logger


class MCPClient:
    """Client for interacting with the SQL Agent MCP server."""
    
    def __init__(self, server_process=None):
        self.logger = get_logger("mcp.client")
        self.server_process = server_process
        self.tools_cache = {}
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools from the MCP server."""
        try:
            # This would normally communicate with the MCP server
            # For now, return the known tools
            tools = [
                {
                    "name": "execute_query",
                    "description": "Execute a SQL query and return results",
                    "category": "database"
                },
                {
                    "name": "get_sample_data",
                    "description": "Get sample data from a table",
                    "category": "database"
                },
                {
                    "name": "validate_sql",
                    "description": "Validate a SQL query without executing it",
                    "category": "database"
                },
                {
                    "name": "get_tables",
                    "description": "Get list of all tables in the database",
                    "category": "schema"
                },
                {
                    "name": "get_columns",
                    "description": "Get column information for a specific table",
                    "category": "schema"
                },
                {
                    "name": "search_schema",
                    "description": "Search schema for tables and columns matching a keyword",
                    "category": "schema"
                },
                {
                    "name": "get_relationships",
                    "description": "Get table relationships and foreign keys",
                    "category": "schema"
                },
                {
                    "name": "create_chart",
                    "description": "Create a chart from data",
                    "category": "visualization"
                },
                {
                    "name": "get_chart_types",
                    "description": "Get available chart types and their descriptions",
                    "category": "visualization"
                },
                {
                    "name": "export_chart",
                    "description": "Export chart in various formats",
                    "category": "visualization"
                },
                {
                    "name": "analyze_data_for_visualization",
                    "description": "Analyze data and suggest appropriate chart types",
                    "category": "visualization"
                }
            ]
            
            self.tools_cache = {tool["name"]: tool for tool in tools}
            return tools
            
        except Exception as e:
            self.logger.error("mcp_list_tools_failed", error=str(e))
            return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a specific tool on the MCP server."""
        try:
            self.logger.info("mcp_client_calling_tool", tool_name=tool_name, arguments=arguments)
            
            # This would normally communicate with the MCP server
            # For now, simulate tool execution
            result = await self._simulate_tool_execution(tool_name, arguments)
            
            return result
            
        except Exception as e:
            self.logger.error("mcp_client_tool_call_failed", tool_name=tool_name, error=str(e))
            return f"Tool execution failed: {str(e)}"
    
    async def _simulate_tool_execution(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Simulate tool execution for testing purposes."""
        from ..core.database import db_manager
        
        try:
            if tool_name == "execute_query":
                sql = arguments.get("sql", "")
                limit = arguments.get("limit", 100)
                
                # Validate and execute query
                is_valid, error = await db_manager.validate_query(sql)
                if not is_valid:
                    return f"Query validation failed: {error}"
                
                result = await db_manager.execute_query(sql, timeout=30)
                if result.error:
                    return f"Query execution failed: {result.error}"
                
                # Format results
                if result.data:
                    formatted_data = json.dumps(result.data[:limit], indent=2)
                    return f"Query executed successfully.\n\nRows: {result.row_count}\nExecution time: {result.execution_time:.2f}s\n\nResults:\n{formatted_data}"
                else:
                    return f"Query executed successfully.\n\nRows: {result.row_count}\nExecution time: {result.execution_time:.2f}s\n\nNo data returned."
            
            elif tool_name == "get_sample_data":
                table_name = arguments.get("table_name", "")
                limit = arguments.get("limit", 10)
                
                data = await db_manager.get_sample_data(table_name, limit)
                if data:
                    formatted_data = json.dumps(data, indent=2)
                    return f"Sample data from {table_name}:\n\n{formatted_data}"
                else:
                    return f"No data found in table {table_name}"
            
            elif tool_name == "validate_sql":
                sql = arguments.get("sql", "")
                
                is_valid, error = await db_manager.validate_query(sql)
                if is_valid:
                    return f"SQL query is valid: {sql}"
                else:
                    return f"SQL query validation failed: {error}\n\nQuery: {sql}"
            
            elif tool_name == "get_tables":
                schema_info = await db_manager.get_schema_info()
                
                if schema_info:
                    table_list = []
                    for table_name, table_info in schema_info.items():
                        columns = table_info.get("columns", [])
                        column_count = len(columns)
                        table_list.append(f"- {table_name} ({column_count} columns)")
                    
                    return f"Database tables:\n\n" + "\n".join(table_list)
                else:
                    return "No tables found in the database"
            
            elif tool_name == "get_columns":
                table_name = arguments.get("table_name", "")
                schema_info = await db_manager.get_schema_info()
                
                if table_name not in schema_info:
                    return f"Table '{table_name}' not found in the database"
                
                table_info = schema_info[table_name]
                columns = table_info.get("columns", [])
                
                if columns:
                    column_info = []
                    for col in columns:
                        col_name = col.get("column_name", "")
                        data_type = col.get("data_type", "")
                        nullable = col.get("is_nullable", "")
                        default = col.get("column_default", "")
                        
                        column_info.append(f"- {col_name}: {data_type} (nullable: {nullable})")
                        if default:
                            column_info[-1] += f" (default: {default})"
                    
                    return f"Columns in table '{table_name}':\n\n" + "\n".join(column_info)
                else:
                    return f"No columns found in table '{table_name}'"
            
            elif tool_name == "search_schema":
                keyword = arguments.get("keyword", "")
                schema_info = await db_manager.get_schema_info()
                keyword_lower = keyword.lower()
                
                matches = []
                
                for table_name, table_info in schema_info.items():
                    # Check table name
                    if keyword_lower in table_name.lower():
                        matches.append(f"Table: {table_name}")
                    
                    # Check column names
                    columns = table_info.get("columns", [])
                    for col in columns:
                        col_name = col.get("column_name", "")
                        if keyword_lower in col_name.lower():
                            matches.append(f"Column: {table_name}.{col_name}")
                
                if matches:
                    return f"Schema search results for '{keyword}':\n\n" + "\n".join(matches)
                else:
                    return f"No matches found for '{keyword}' in the schema"
            
            elif tool_name == "get_relationships":
                schema_info = await db_manager.get_schema_info()
                
                relationships = []
                
                for table_name, table_info in schema_info.items():
                    columns = table_info.get("columns", [])
                    for col in columns:
                        col_name = col.get("column_name", "")
                        # Look for foreign key patterns
                        if col_name.endswith("_id") or "foreign" in col_name.lower():
                            relationships.append(f"Potential FK: {table_name}.{col_name}")
                
                if relationships:
                    return f"Potential table relationships:\n\n" + "\n".join(relationships)
                else:
                    return "No obvious relationships found in the schema"
            
            elif tool_name == "create_chart":
                chart_type = arguments.get("chart_type", "")
                data = arguments.get("data", "")
                title = arguments.get("title", "")
                
                # Validate chart type
                valid_types = ["bar", "line", "pie", "scatter", "histogram"]
                if chart_type not in valid_types:
                    return f"Invalid chart type. Supported types: {', '.join(valid_types)}"
                
                # Parse data
                try:
                    chart_data = json.loads(data)
                except json.JSONDecodeError:
                    return "Invalid data format. Please provide valid JSON data."
                
                # Generate chart configuration
                chart_config = {
                    "type": chart_type,
                    "data": chart_data,
                    "title": title or f"{chart_type.title()} Chart",
                    "config": {
                        "responsive": True,
                        "displayModeBar": True
                    }
                }
                
                return f"Chart configuration generated:\n\n{json.dumps(chart_config, indent=2)}"
            
            elif tool_name == "get_chart_types":
                chart_types = {
                    "bar": "Bar chart for categorical data comparison",
                    "line": "Line chart for time series or trends",
                    "pie": "Pie chart for proportions and percentages",
                    "scatter": "Scatter plot for correlation analysis",
                    "histogram": "Histogram for data distribution"
                }
                
                chart_info = []
                for chart_type, description in chart_types.items():
                    chart_info.append(f"- {chart_type}: {description}")
                
                return "Available chart types:\n\n" + "\n".join(chart_info)
            
            elif tool_name == "export_chart":
                chart_config = arguments.get("chart_config", "")
                format_type = arguments.get("format", "json")
                
                # Parse chart configuration
                try:
                    config = json.loads(chart_config)
                except json.JSONDecodeError:
                    return "Invalid chart configuration format."
                
                # Validate export format
                valid_formats = ["json", "html", "png", "svg"]
                if format_type not in valid_formats:
                    return f"Invalid export format. Supported formats: {', '.join(valid_formats)}"
                
                if format_type == "json":
                    return f"Chart exported as JSON:\n\n{json.dumps(config, indent=2)}"
                elif format_type == "html":
                    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{config.get('title', 'Chart')}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div id="chart"></div>
    <script>
        var data = {json.dumps(config.get('data', []))};
        var layout = {{
            title: '{config.get('title', 'Chart')}'
        }};
        Plotly.newPlot('chart', data, layout);
    </script>
</body>
</html>
"""
                    return f"Chart exported as HTML:\n\n{html_content}"
                else:
                    return f"Export format '{format_type}' not yet implemented. Returning JSON format."
            
            elif tool_name == "analyze_data_for_visualization":
                data = arguments.get("data", "")
                
                # Parse data
                try:
                    chart_data = json.loads(data)
                except json.JSONDecodeError:
                    return "Invalid data format. Please provide valid JSON data."
                
                # Analyze data structure
                if isinstance(chart_data, list) and len(chart_data) > 0:
                    sample = chart_data[0]
                    if isinstance(sample, dict):
                        keys = list(sample.keys())
                        
                        # Simple analysis based on data structure
                        suggestions = []
                        
                        if len(keys) >= 2:
                            suggestions.append("- Bar chart: Good for comparing categories")
                            suggestions.append("- Line chart: Good for trends over time")
                            suggestions.append("- Scatter plot: Good for correlation analysis")
                        
                        if len(keys) == 1:
                            suggestions.append("- Histogram: Good for data distribution")
                            suggestions.append("- Pie chart: Good for proportions")
                        
                        return f"Data analysis:\n\nData structure: {len(chart_data)} rows, {len(keys)} columns\nColumns: {', '.join(keys)}\n\nSuggested chart types:\n" + "\n".join(suggestions)
                
                return "Unable to analyze data structure. Please provide data in a list of objects format."
            
            else:
                return f"Unknown tool: {tool_name}"
                
        except Exception as e:
            self.logger.error("mcp_simulation_failed", tool_name=tool_name, error=str(e))
            return f"Tool execution failed: {str(e)}"
    
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
            "total_tools": len(self.tools_cache)
        }
    
    async def test_all_tools(self) -> Dict[str, Any]:
        """Test all available tools with sample data."""
        self.logger.info("mcp_client_testing_all_tools")
        
        results = {}
        
        # Test database tools
        results["execute_query"] = await self.call_tool("execute_query", {
            "sql": "SELECT * FROM customers LIMIT 5"
        })
        
        results["get_sample_data"] = await self.call_tool("get_sample_data", {
            "table_name": "customers",
            "limit": 3
        })
        
        results["validate_sql"] = await self.call_tool("validate_sql", {
            "sql": "SELECT * FROM customers WHERE id = 1"
        })
        
        # Test schema tools
        results["get_tables"] = await self.call_tool("get_tables", {})
        
        results["get_columns"] = await self.call_tool("get_columns", {
            "table_name": "customers"
        })
        
        results["search_schema"] = await self.call_tool("search_schema", {
            "keyword": "customer"
        })
        
        results["get_relationships"] = await self.call_tool("get_relationships", {})
        
        # Test visualization tools
        sample_data = json.dumps([
            {"name": "John", "revenue": 1000},
            {"name": "Jane", "revenue": 2000},
            {"name": "Bob", "revenue": 1500}
        ])
        
        results["create_chart"] = await self.call_tool("create_chart", {
            "chart_type": "bar",
            "data": sample_data,
            "title": "Customer Revenue"
        })
        
        results["get_chart_types"] = await self.call_tool("get_chart_types", {})
        
        results["analyze_data_for_visualization"] = await self.call_tool("analyze_data_for_visualization", {
            "data": sample_data
        })
        
        return results 