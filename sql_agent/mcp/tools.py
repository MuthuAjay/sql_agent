"""MCP tools for SQL Agent."""

import json
from typing import Any, Dict, List, Optional
from mcp.types import TextContent, ImageContent, EmbeddedResource
from ..core.database import db_manager
from ..core.state import QueryResult, SchemaContext
from ..utils.logging import get_logger


class DatabaseTools:
    """Database-related MCP tools."""
    
    def __init__(self):
        self.logger = get_logger("mcp.database_tools")
    
    async def execute_query(self, sql: str, limit: int = 100) -> TextContent:
        """Execute a SQL query and return results."""
        try:
            # Validate query
            is_valid, error = await db_manager.validate_query(sql)
            if not is_valid:
                return TextContent(
                    type="text",
                    text=f"Query validation failed: {error}"
                )
            
            # Execute query
            result = await db_manager.execute_query(sql, timeout=30)
            
            if result.error:
                return TextContent(
                    type="text",
                    text=f"Query execution failed: {result.error}"
                )
            
            # Format results
            if result.data:
                # Convert to JSON for display
                formatted_data = json.dumps(result.data[:limit], indent=2)
                return TextContent(
                    type="text",
                    text=f"Query executed successfully.\n\nRows: {result.row_count}\nExecution time: {result.execution_time:.2f}s\n\nResults:\n{formatted_data}"
                )
            else:
                return TextContent(
                    type="text",
                    text=f"Query executed successfully.\n\nRows: {result.row_count}\nExecution time: {result.execution_time:.2f}s\n\nNo data returned."
                )
                
        except Exception as e:
            self.logger.error("mcp_execute_query_failed", error=str(e))
            return TextContent(
                type="text",
                text=f"Query execution failed: {str(e)}"
            )
    
    async def get_sample_data(self, table_name: str, limit: int = 10) -> TextContent:
        """Get sample data from a table."""
        try:
            data = await db_manager.get_sample_data(table_name, limit)
            
            if data:
                formatted_data = json.dumps(data, indent=2)
                return TextContent(
                    type="text",
                    text=f"Sample data from {table_name}:\n\n{formatted_data}"
                )
            else:
                return TextContent(
                    type="text",
                    text=f"No data found in table {table_name}"
                )
                
        except Exception as e:
            self.logger.error("mcp_get_sample_data_failed", error=str(e))
            return TextContent(
                type="text",
                text=f"Failed to get sample data: {str(e)}"
            )
    
    async def validate_sql(self, sql: str) -> TextContent:
        """Validate a SQL query without executing it."""
        try:
            is_valid, error = await db_manager.validate_query(sql)
            
            if is_valid:
                return TextContent(
                    type="text",
                    text=f"SQL query is valid: {sql}"
                )
            else:
                return TextContent(
                    type="text",
                    text=f"SQL query validation failed: {error}\n\nQuery: {sql}"
                )
                
        except Exception as e:
            self.logger.error("mcp_validate_sql_failed", error=str(e))
            return TextContent(
                type="text",
                text=f"SQL validation failed: {str(e)}"
            )


class SchemaTools:
    """Schema-related MCP tools."""
    
    def __init__(self):
        self.logger = get_logger("mcp.schema_tools")
    
    async def get_tables(self) -> TextContent:
        """Get list of all tables in the database."""
        try:
            schema_info = await db_manager.get_schema_info()
            
            if schema_info:
                table_list = []
                for table_name, table_info in schema_info.items():
                    columns = table_info.get("columns", [])
                    column_count = len(columns)
                    table_list.append(f"- {table_name} ({column_count} columns)")
                
                return TextContent(
                    type="text",
                    text=f"Database tables:\n\n" + "\n".join(table_list)
                )
            else:
                return TextContent(
                    type="text",
                    text="No tables found in the database"
                )
                
        except Exception as e:
            self.logger.error("mcp_get_tables_failed", error=str(e))
            return TextContent(
                type="text",
                text=f"Failed to get tables: {str(e)}"
            )
    
    async def get_columns(self, table_name: str) -> TextContent:
        """Get column information for a specific table."""
        try:
            schema_info = await db_manager.get_schema_info()
            
            if table_name not in schema_info:
                return TextContent(
                    type="text",
                    text=f"Table '{table_name}' not found in the database"
                )
            
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
                
                return TextContent(
                    type="text",
                    text=f"Columns in table '{table_name}':\n\n" + "\n".join(column_info)
                )
            else:
                return TextContent(
                    type="text",
                    text=f"No columns found in table '{table_name}'"
                )
                
        except Exception as e:
            self.logger.error("mcp_get_columns_failed", error=str(e))
            return TextContent(
                type="text",
                text=f"Failed to get columns: {str(e)}"
            )
    
    async def search_schema(self, keyword: str) -> TextContent:
        """Search schema for tables and columns matching a keyword."""
        try:
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
                return TextContent(
                    type="text",
                    text=f"Schema search results for '{keyword}':\n\n" + "\n".join(matches)
                )
            else:
                return TextContent(
                    type="text",
                    text=f"No matches found for '{keyword}' in the schema"
                )
                
        except Exception as e:
            self.logger.error("mcp_search_schema_failed", error=str(e))
            return TextContent(
                type="text",
                text=f"Schema search failed: {str(e)}"
            )
    
    async def get_relationships(self) -> TextContent:
        """Get table relationships and foreign keys."""
        try:
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
                return TextContent(
                    type="text",
                    text=f"Potential table relationships:\n\n" + "\n".join(relationships)
                )
            else:
                return TextContent(
                    type="text",
                    text="No obvious relationships found in the schema"
                )
                
        except Exception as e:
            self.logger.error("mcp_get_relationships_failed", error=str(e))
            return TextContent(
                type="text",
                text=f"Failed to get relationships: {str(e)}"
            )


class VisualizationTools:
    """Visualization-related MCP tools."""
    
    def __init__(self):
        self.logger = get_logger("mcp.visualization_tools")
    
    async def create_chart(self, chart_type: str, data: str, title: str = "") -> TextContent:
        """Create a chart from data."""
        try:
            # Parse data (assuming JSON format)
            try:
                chart_data = json.loads(data)
            except json.JSONDecodeError:
                return TextContent(
                    type="text",
                    text="Invalid data format. Please provide valid JSON data."
                )
            
            # Validate chart type
            valid_types = ["bar", "line", "pie", "scatter", "histogram"]
            if chart_type not in valid_types:
                return TextContent(
                    type="text",
                    text=f"Invalid chart type. Supported types: {', '.join(valid_types)}"
                )
            
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
            
            # Return chart configuration as JSON
            return TextContent(
                type="text",
                text=f"Chart configuration generated:\n\n{json.dumps(chart_config, indent=2)}"
            )
            
        except Exception as e:
            self.logger.error("mcp_create_chart_failed", error=str(e))
            return TextContent(
                type="text",
                text=f"Failed to create chart: {str(e)}"
            )
    
    async def get_chart_types(self) -> TextContent:
        """Get available chart types and their descriptions."""
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
        
        return TextContent(
            type="text",
            text="Available chart types:\n\n" + "\n".join(chart_info)
        )
    
    async def export_chart(self, chart_config: str, format: str = "json") -> TextContent:
        """Export chart in various formats."""
        try:
            # Parse chart configuration
            try:
                config = json.loads(chart_config)
            except json.JSONDecodeError:
                return TextContent(
                    type="text",
                    text="Invalid chart configuration format."
                )
            
            # Validate export format
            valid_formats = ["json", "html", "png", "svg"]
            if format not in valid_formats:
                return TextContent(
                    type="text",
                    text=f"Invalid export format. Supported formats: {', '.join(valid_formats)}"
                )
            
            # For now, return the configuration in the requested format
            if format == "json":
                return TextContent(
                    type="text",
                    text=f"Chart exported as JSON:\n\n{json.dumps(config, indent=2)}"
                )
            elif format == "html":
                # Generate simple HTML chart
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
                return TextContent(
                    type="text",
                    text=f"Chart exported as HTML:\n\n{html_content}"
                )
            else:
                return TextContent(
                    type="text",
                    text=f"Export format '{format}' not yet implemented. Returning JSON format."
                )
                
        except Exception as e:
            self.logger.error("mcp_export_chart_failed", error=str(e))
            return TextContent(
                type="text",
                text=f"Failed to export chart: {str(e)}"
            )
    
    async def analyze_data_for_visualization(self, data: str) -> TextContent:
        """Analyze data and suggest appropriate chart types."""
        try:
            # Parse data
            try:
                chart_data = json.loads(data)
            except json.JSONDecodeError:
                return TextContent(
                    type="text",
                    text="Invalid data format. Please provide valid JSON data."
                )
            
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
                    
                    return TextContent(
                        type="text",
                        text=f"Data analysis:\n\nData structure: {len(chart_data)} rows, {len(keys)} columns\nColumns: {', '.join(keys)}\n\nSuggested chart types:\n" + "\n".join(suggestions)
                    )
            
            return TextContent(
                type="text",
                text="Unable to analyze data structure. Please provide data in a list of objects format."
            )
            
        except Exception as e:
            self.logger.error("mcp_analyze_data_failed", error=str(e))
            return TextContent(
                type="text",
                text=f"Failed to analyze data: {str(e)}"
            ) 