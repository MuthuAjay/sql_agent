"""Tests for MCP integration."""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from sql_agent.mcp import MCPClient
from sql_agent.mcp.tools import DatabaseTools, SchemaTools, VisualizationTools


class TestDatabaseTools:
    """Test the DatabaseTools class."""
    
    @pytest.fixture
    def database_tools(self):
        """Create a DatabaseTools instance."""
        return DatabaseTools()
    
    @pytest.mark.asyncio
    async def test_execute_query_success(self, database_tools):
        """Test successful query execution."""
        with patch('sql_agent.mcp.tools.db_manager') as mock_db:
            mock_db.validate_query.return_value = (True, None)
            mock_db.execute_query.return_value = Mock(
                data=[{"id": 1, "name": "John"}],
                row_count=1,
                execution_time=0.1,
                error=None
            )
            
            result = await database_tools.execute_query("SELECT * FROM customers", 10)
            
            assert result.type == "text"
            assert "Query executed successfully" in result.text
            assert "Rows: 1" in result.text
    
    @pytest.mark.asyncio
    async def test_execute_query_validation_failure(self, database_tools):
        """Test query execution with validation failure."""
        with patch('sql_agent.mcp.tools.db_manager') as mock_db:
            mock_db.validate_query.return_value = (False, "Invalid SQL")
            
            result = await database_tools.execute_query("INVALID SQL", 10)
            
            assert result.type == "text"
            assert "Query validation failed" in result.text
    
    @pytest.mark.asyncio
    async def test_execute_query_execution_failure(self, database_tools):
        """Test query execution with execution failure."""
        with patch('sql_agent.mcp.tools.db_manager') as mock_db:
            mock_db.validate_query.return_value = (True, None)
            mock_db.execute_query.return_value = Mock(
                data=[],
                row_count=0,
                execution_time=0.0,
                error="Table not found"
            )
            
            result = await database_tools.execute_query("SELECT * FROM nonexistent", 10)
            
            assert result.type == "text"
            assert "Query execution failed" in result.text
    
    @pytest.mark.asyncio
    async def test_get_sample_data_success(self, database_tools):
        """Test successful sample data retrieval."""
        with patch('sql_agent.mcp.tools.db_manager') as mock_db:
            mock_db.get_sample_data.return_value = [
                {"id": 1, "name": "John"},
                {"id": 2, "name": "Jane"}
            ]
            
            result = await database_tools.get_sample_data("customers", 5)
            
            assert result.type == "text"
            assert "Sample data from customers" in result.text
            assert "John" in result.text
    
    @pytest.mark.asyncio
    async def test_get_sample_data_no_data(self, database_tools):
        """Test sample data retrieval with no data."""
        with patch('sql_agent.mcp.tools.db_manager') as mock_db:
            mock_db.get_sample_data.return_value = []
            
            result = await database_tools.get_sample_data("empty_table", 5)
            
            assert result.type == "text"
            assert "No data found" in result.text
    
    @pytest.mark.asyncio
    async def test_validate_sql_success(self, database_tools):
        """Test successful SQL validation."""
        with patch('sql_agent.mcp.tools.db_manager') as mock_db:
            mock_db.validate_query.return_value = (True, None)
            
            result = await database_tools.validate_sql("SELECT * FROM customers")
            
            assert result.type == "text"
            assert "SQL query is valid" in result.text
    
    @pytest.mark.asyncio
    async def test_validate_sql_failure(self, database_tools):
        """Test SQL validation failure."""
        with patch('sql_agent.mcp.tools.db_manager') as mock_db:
            mock_db.validate_query.return_value = (False, "Syntax error")
            
            result = await database_tools.validate_sql("INVALID SQL")
            
            assert result.type == "text"
            assert "SQL query validation failed" in result.text


class TestSchemaTools:
    """Test the SchemaTools class."""
    
    @pytest.fixture
    def schema_tools(self):
        """Create a SchemaTools instance."""
        return SchemaTools()
    
    @pytest.mark.asyncio
    async def test_get_tables_success(self, schema_tools):
        """Test successful table listing."""
        with patch('sql_agent.mcp.tools.db_manager') as mock_db:
            mock_db.get_schema_info.return_value = {
                "customers": {"columns": [{"column_name": "id"}, {"column_name": "name"}]},
                "orders": {"columns": [{"column_name": "id"}, {"column_name": "customer_id"}]}
            }
            
            result = await schema_tools.get_tables()
            
            assert result.type == "text"
            assert "customers (2 columns)" in result.text
            assert "orders (2 columns)" in result.text
    
    @pytest.mark.asyncio
    async def test_get_tables_no_tables(self, schema_tools):
        """Test table listing with no tables."""
        with patch('sql_agent.mcp.tools.db_manager') as mock_db:
            mock_db.get_schema_info.return_value = {}
            
            result = await schema_tools.get_tables()
            
            assert result.type == "text"
            assert "No tables found" in result.text
    
    @pytest.mark.asyncio
    async def test_get_columns_success(self, schema_tools):
        """Test successful column listing."""
        with patch('sql_agent.mcp.tools.db_manager') as mock_db:
            mock_db.get_schema_info.return_value = {
                "customers": {
                    "columns": [
                        {"column_name": "id", "data_type": "integer", "is_nullable": "NO"},
                        {"column_name": "name", "data_type": "varchar", "is_nullable": "YES"}
                    ]
                }
            }
            
            result = await schema_tools.get_columns("customers")
            
            assert result.type == "text"
            assert "id: integer" in result.text
            assert "name: varchar" in result.text
    
    @pytest.mark.asyncio
    async def test_get_columns_table_not_found(self, schema_tools):
        """Test column listing for non-existent table."""
        with patch('sql_agent.mcp.tools.db_manager') as mock_db:
            mock_db.get_schema_info.return_value = {}
            
            result = await schema_tools.get_columns("nonexistent")
            
            assert result.type == "text"
            assert "not found" in result.text
    
    @pytest.mark.asyncio
    async def test_search_schema_success(self, schema_tools):
        """Test successful schema search."""
        with patch('sql_agent.mcp.tools.db_manager') as mock_db:
            mock_db.get_schema_info.return_value = {
                "customers": {
                    "columns": [
                        {"column_name": "customer_id"},
                        {"column_name": "name"}
                    ]
                },
                "customer_orders": {
                    "columns": [
                        {"column_name": "order_id"}
                    ]
                }
            }
            
            result = await schema_tools.search_schema("customer")
            
            assert result.type == "text"
            assert "Table: customers" in result.text
            assert "Table: customer_orders" in result.text
            assert "Column: customers.customer_id" in result.text
    
    @pytest.mark.asyncio
    async def test_search_schema_no_matches(self, schema_tools):
        """Test schema search with no matches."""
        with patch('sql_agent.mcp.tools.db_manager') as mock_db:
            mock_db.get_schema_info.return_value = {
                "users": {"columns": [{"column_name": "id"}]}
            }
            
            result = await schema_tools.search_schema("customer")
            
            assert result.type == "text"
            assert "No matches found" in result.text
    
    @pytest.mark.asyncio
    async def test_get_relationships_success(self, schema_tools):
        """Test successful relationship detection."""
        with patch('sql_agent.mcp.tools.db_manager') as mock_db:
            mock_db.get_schema_info.return_value = {
                "customers": {
                    "columns": [
                        {"column_name": "id"},
                        {"column_name": "name"}
                    ]
                },
                "orders": {
                    "columns": [
                        {"column_name": "id"},
                        {"column_name": "customer_id"}
                    ]
                }
            }
            
            result = await schema_tools.get_relationships()
            
            assert result.type == "text"
            assert "Potential FK: orders.customer_id" in result.text
    
    @pytest.mark.asyncio
    async def test_get_relationships_none_found(self, schema_tools):
        """Test relationship detection with no relationships."""
        with patch('sql_agent.mcp.tools.db_manager') as mock_db:
            mock_db.get_schema_info.return_value = {
                "users": {
                    "columns": [
                        {"column_name": "id"},
                        {"column_name": "name"}
                    ]
                }
            }
            
            result = await schema_tools.get_relationships()
            
            assert result.type == "text"
            assert "No obvious relationships found" in result.text


class TestVisualizationTools:
    """Test the VisualizationTools class."""
    
    @pytest.fixture
    def visualization_tools(self):
        """Create a VisualizationTools instance."""
        return VisualizationTools()
    
    @pytest.mark.asyncio
    async def test_create_chart_success(self, visualization_tools):
        """Test successful chart creation."""
        data = json.dumps([
            {"name": "John", "value": 100},
            {"name": "Jane", "value": 200}
        ])
        
        result = await visualization_tools.create_chart("bar", data, "Test Chart")
        
        assert result.type == "text"
        assert "Chart configuration generated" in result.text
        assert "bar" in result.text
    
    @pytest.mark.asyncio
    async def test_create_chart_invalid_type(self, visualization_tools):
        """Test chart creation with invalid chart type."""
        data = json.dumps([{"name": "John", "value": 100}])
        
        result = await visualization_tools.create_chart("invalid_type", data)
        
        assert result.type == "text"
        assert "Invalid chart type" in result.text
    
    @pytest.mark.asyncio
    async def test_create_chart_invalid_data(self, visualization_tools):
        """Test chart creation with invalid data format."""
        result = await visualization_tools.create_chart("bar", "invalid json", "Test")
        
        assert result.type == "text"
        assert "Invalid data format" in result.text
    
    @pytest.mark.asyncio
    async def test_get_chart_types(self, visualization_tools):
        """Test chart types listing."""
        result = await visualization_tools.get_chart_types()
        
        assert result.type == "text"
        assert "bar" in result.text
        assert "line" in result.text
        assert "pie" in result.text
        assert "scatter" in result.text
        assert "histogram" in result.text
    
    @pytest.mark.asyncio
    async def test_export_chart_json(self, visualization_tools):
        """Test chart export in JSON format."""
        config = json.dumps({
            "type": "bar",
            "data": [{"x": ["A", "B"], "y": [1, 2]}],
            "title": "Test"
        })
        
        result = await visualization_tools.export_chart(config, "json")
        
        assert result.type == "text"
        assert "Chart exported as JSON" in result.text
    
    @pytest.mark.asyncio
    async def test_export_chart_html(self, visualization_tools):
        """Test chart export in HTML format."""
        config = json.dumps({
            "type": "bar",
            "data": [{"x": ["A", "B"], "y": [1, 2]}],
            "title": "Test"
        })
        
        result = await visualization_tools.export_chart(config, "html")
        
        assert result.type == "text"
        assert "Chart exported as HTML" in result.text
        assert "<!DOCTYPE html>" in result.text
    
    @pytest.mark.asyncio
    async def test_export_chart_invalid_format(self, visualization_tools):
        """Test chart export with invalid format."""
        config = json.dumps({"type": "bar", "data": []})
        
        result = await visualization_tools.export_chart(config, "invalid")
        
        assert result.type == "text"
        assert "Invalid export format" in result.text
    
    @pytest.mark.asyncio
    async def test_export_chart_invalid_config(self, visualization_tools):
        """Test chart export with invalid configuration."""
        result = await visualization_tools.export_chart("invalid json", "json")
        
        assert result.type == "text"
        assert "Invalid chart configuration format" in result.text
    
    @pytest.mark.asyncio
    async def test_analyze_data_for_visualization_success(self, visualization_tools):
        """Test successful data analysis for visualization."""
        data = json.dumps([
            {"name": "John", "value": 100},
            {"name": "Jane", "value": 200}
        ])
        
        result = await visualization_tools.analyze_data_for_visualization(data)
        
        assert result.type == "text"
        assert "Data analysis" in result.text
        assert "Bar chart" in result.text
        assert "Line chart" in result.text
    
    @pytest.mark.asyncio
    async def test_analyze_data_for_visualization_invalid_data(self, visualization_tools):
        """Test data analysis with invalid data format."""
        result = await visualization_tools.analyze_data_for_visualization("invalid json")
        
        assert result.type == "text"
        assert "Invalid data format" in result.text
    
    @pytest.mark.asyncio
    async def test_analyze_data_for_visualization_single_column(self, visualization_tools):
        """Test data analysis with single column data."""
        data = json.dumps([{"value": 100}, {"value": 200}])
        
        result = await visualization_tools.analyze_data_for_visualization(data)
        
        assert result.type == "text"
        assert "Histogram" in result.text
        assert "Pie chart" in result.text


class TestMCPClient:
    """Test the MCPClient class."""
    
    @pytest.fixture
    def mcp_client(self):
        """Create an MCPClient instance."""
        return MCPClient()
    
    @pytest.mark.asyncio
    async def test_list_tools(self, mcp_client):
        """Test tool listing."""
        tools = await mcp_client.list_tools()
        
        assert len(tools) > 0
        assert any(tool["name"] == "execute_query" for tool in tools)
        assert any(tool["name"] == "get_tables" for tool in tools)
        assert any(tool["name"] == "create_chart" for tool in tools)
    
    @pytest.mark.asyncio
    async def test_call_tool_execute_query(self, mcp_client):
        """Test calling execute_query tool."""
        with patch('sql_agent.mcp.client.db_manager') as mock_db:
            mock_db.validate_query.return_value = (True, None)
            mock_db.execute_query.return_value = Mock(
                data=[{"id": 1, "name": "John"}],
                row_count=1,
                execution_time=0.1,
                error=None
            )
            
            result = await mcp_client.call_tool("execute_query", {
                "sql": "SELECT * FROM customers",
                "limit": 10
            })
            
            assert "Query executed successfully" in result
    
    @pytest.mark.asyncio
    async def test_call_tool_get_tables(self, mcp_client):
        """Test calling get_tables tool."""
        with patch('sql_agent.mcp.client.db_manager') as mock_db:
            mock_db.get_schema_info.return_value = {
                "customers": {"columns": [{"column_name": "id"}]}
            }
            
            result = await mcp_client.call_tool("get_tables", {})
            
            assert "customers" in result
    
    @pytest.mark.asyncio
    async def test_call_tool_create_chart(self, mcp_client):
        """Test calling create_chart tool."""
        data = json.dumps([{"name": "John", "value": 100}])
        
        result = await mcp_client.call_tool("create_chart", {
            "chart_type": "bar",
            "data": data,
            "title": "Test Chart"
        })
        
        assert "Chart configuration generated" in result
    
    @pytest.mark.asyncio
    async def test_call_tool_unknown(self, mcp_client):
        """Test calling unknown tool."""
        result = await mcp_client.call_tool("unknown_tool", {})
        
        assert "Unknown tool" in result
    
    def test_get_tool_info(self, mcp_client):
        """Test tool information retrieval."""
        tool_info = mcp_client.get_tool_info()
        
        assert "database_tools" in tool_info
        assert "schema_tools" in tool_info
        assert "visualization_tools" in tool_info
        assert "total_tools" in tool_info
        assert tool_info["total_tools"] > 0
    
    @pytest.mark.asyncio
    async def test_test_all_tools(self, mcp_client):
        """Test the test_all_tools method."""
        with patch('sql_agent.mcp.client.db_manager') as mock_db:
            mock_db.validate_query.return_value = (True, None)
            mock_db.execute_query.return_value = Mock(
                data=[{"id": 1, "name": "John"}],
                row_count=1,
                execution_time=0.1,
                error=None
            )
            mock_db.get_schema_info.return_value = {
                "customers": {"columns": [{"column_name": "id"}]}
            }
            mock_db.get_sample_data.return_value = [{"id": 1, "name": "John"}]
            
            results = await mcp_client.test_all_tools()
            
            assert "execute_query" in results
            assert "get_sample_data" in results
            assert "get_tables" in results
            assert "create_chart" in results
            assert len(results) > 0 