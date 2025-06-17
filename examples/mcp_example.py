#!/usr/bin/env python3
"""Example demonstrating the MCP integration."""

import asyncio
import json
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sql_agent.mcp import MCPClient
from sql_agent.core.config import settings
from sql_agent.core.database import db_manager
from sql_agent.utils.logging import get_logger


async def main():
    """Main example function."""
    logger = get_logger("mcp_example")
    
    print("🚀 SQL Agent MCP Integration Example")
    print("=" * 50)
    
    # Initialize database (if available)
    try:
        await db_manager.initialize()
        print("✅ Database connected")
    except Exception as e:
        print(f"⚠️  Database not available: {e}")
        print("   Continuing with mock data...")
    
    # Create MCP client
    print("\n🔧 Initializing MCP client...")
    client = MCPClient()
    
    # List available tools
    print("\n📋 Available MCP Tools:")
    tools = await client.list_tools()
    
    # Group tools by category
    tool_categories = {}
    for tool in tools:
        category = tool.get("category", "other")
        if category not in tool_categories:
            tool_categories[category] = []
        tool_categories[category].append(tool)
    
    for category, category_tools in tool_categories.items():
        print(f"\n{category.upper()} Tools:")
        for tool in category_tools:
            print(f"   - {tool['name']}: {tool['description']}")
    
    print(f"\n📊 Total tools available: {len(tools)}")
    
    # Test database tools
    print(f"\n🗄️  Testing Database Tools:")
    print("-" * 30)
    
    # Test execute_query
    print("\n1. Testing execute_query:")
    result = await client.call_tool("execute_query", {
        "sql": "SELECT * FROM customers LIMIT 3",
        "limit": 5
    })
    print(f"Result: {result[:200]}..." if len(result) > 200 else f"Result: {result}")
    
    # Test get_sample_data
    print("\n2. Testing get_sample_data:")
    result = await client.call_tool("get_sample_data", {
        "table_name": "customers",
        "limit": 2
    })
    print(f"Result: {result[:200]}..." if len(result) > 200 else f"Result: {result}")
    
    # Test validate_sql
    print("\n3. Testing validate_sql:")
    result = await client.call_tool("validate_sql", {
        "sql": "SELECT name, email FROM customers WHERE id = 1"
    })
    print(f"Result: {result}")
    
    # Test schema tools
    print(f"\n📋 Testing Schema Tools:")
    print("-" * 30)
    
    # Test get_tables
    print("\n1. Testing get_tables:")
    result = await client.call_tool("get_tables", {})
    print(f"Result: {result}")
    
    # Test get_columns
    print("\n2. Testing get_columns:")
    result = await client.call_tool("get_columns", {
        "table_name": "customers"
    })
    print(f"Result: {result}")
    
    # Test search_schema
    print("\n3. Testing search_schema:")
    result = await client.call_tool("search_schema", {
        "keyword": "customer"
    })
    print(f"Result: {result}")
    
    # Test get_relationships
    print("\n4. Testing get_relationships:")
    result = await client.call_tool("get_relationships", {})
    print(f"Result: {result}")
    
    # Test visualization tools
    print(f"\n📊 Testing Visualization Tools:")
    print("-" * 30)
    
    # Sample data for visualization
    sample_data = json.dumps([
        {"name": "John Doe", "revenue": 15000, "region": "North"},
        {"name": "Jane Smith", "revenue": 22000, "region": "South"},
        {"name": "Bob Johnson", "revenue": 18000, "region": "East"},
        {"name": "Alice Brown", "revenue": 12000, "region": "West"},
        {"name": "Charlie Wilson", "revenue": 25000, "region": "North"}
    ])
    
    # Test create_chart
    print("\n1. Testing create_chart:")
    result = await client.call_tool("create_chart", {
        "chart_type": "bar",
        "data": sample_data,
        "title": "Customer Revenue by Region"
    })
    print(f"Result: {result[:300]}..." if len(result) > 300 else f"Result: {result}")
    
    # Test get_chart_types
    print("\n2. Testing get_chart_types:")
    result = await client.call_tool("get_chart_types", {})
    print(f"Result: {result}")
    
    # Test analyze_data_for_visualization
    print("\n3. Testing analyze_data_for_visualization:")
    result = await client.call_tool("analyze_data_for_visualization", {
        "data": sample_data
    })
    print(f"Result: {result}")
    
    # Test export_chart
    print("\n4. Testing export_chart:")
    chart_config = json.dumps({
        "type": "bar",
        "data": [
            {"x": ["North", "South", "East", "West"], "y": [40000, 22000, 18000, 12000], "type": "bar"}
        ],
        "title": "Revenue by Region"
    })
    result = await client.call_tool("export_chart", {
        "chart_config": chart_config,
        "format": "html"
    })
    print(f"Result: {result[:400]}..." if len(result) > 400 else f"Result: {result}")
    
    # Test error handling
    print(f"\n⚠️  Testing Error Handling:")
    print("-" * 30)
    
    # Test invalid SQL
    print("\n1. Testing invalid SQL:")
    result = await client.call_tool("execute_query", {
        "sql": "SELECT * FROM nonexistent_table"
    })
    print(f"Result: {result}")
    
    # Test invalid chart type
    print("\n2. Testing invalid chart type:")
    result = await client.call_tool("create_chart", {
        "chart_type": "invalid_type",
        "data": sample_data
    })
    print(f"Result: {result}")
    
    # Test invalid data format
    print("\n3. Testing invalid data format:")
    result = await client.call_tool("create_chart", {
        "chart_type": "bar",
        "data": "invalid json data"
    })
    print(f"Result: {result}")
    
    # Show tool information
    print(f"\n📊 Tool Information:")
    print("-" * 30)
    tool_info = client.get_tool_info()
    print(f"Database tools: {len(tool_info['database_tools'])}")
    print(f"Schema tools: {len(tool_info['schema_tools'])}")
    print(f"Visualization tools: {len(tool_info['visualization_tools'])}")
    print(f"Total tools: {tool_info['total_tools']}")
    
    # Demonstrate tool chaining
    print(f"\n🔗 Tool Chaining Example:")
    print("-" * 30)
    
    print("\nChaining: get_tables → get_columns → create_chart")
    
    # Step 1: Get tables
    tables_result = await client.call_tool("get_tables", {})
    print(f"1. Tables available: {tables_result[:100]}...")
    
    # Step 2: Get columns for customers table
    columns_result = await client.call_tool("get_columns", {
        "table_name": "customers"
    })
    print(f"2. Customer table columns: {columns_result[:100]}...")
    
    # Step 3: Create a chart with sample data
    chart_result = await client.call_tool("create_chart", {
        "chart_type": "pie",
        "data": json.dumps([
            {"name": "Active", "value": 75},
            {"name": "Inactive", "value": 25}
        ]),
        "title": "Customer Status Distribution"
    })
    print(f"3. Chart created: {chart_result[:100]}...")
    
    print(f"\n🎉 MCP Integration Example Completed!")
    print("=" * 50)
    
    # Summary
    print(f"\n📋 Summary:")
    print(f"   ✅ MCP client initialized")
    print(f"   ✅ {len(tools)} tools available")
    print(f"   ✅ Database tools tested")
    print(f"   ✅ Schema tools tested")
    print(f"   ✅ Visualization tools tested")
    print(f"   ✅ Error handling tested")
    print(f"   ✅ Tool chaining demonstrated")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main()) 