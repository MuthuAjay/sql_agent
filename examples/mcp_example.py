#!/usr/bin/env python3
"""Example demonstrating the MCP integration."""

import asyncio
import json
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sql_agent.mcp.client import MCPClient
from sql_agent.core.config import settings
from sql_agent.core.database import db_manager
from sql_agent.utils.logging import get_logger
from mcp.client.session import ClientSession
from mcp.client.websocket import websocket_client


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
    # Connect to the FastAPI server where MCP is exposed via WebSocket
    websocket_url = f"ws://localhost:8000/mcp"
    
    async with websocket_client(websocket_url) as (read_stream, write_stream):
        mcp_client_raw = ClientSession(read_stream, write_stream)
        await mcp_client_raw.initialize()

        client = MCPClient(client_session=mcp_client_raw)
        
        # List available tools
        print("\n📋 Available MCP Tools:")
        tools = await client.list_tools()
        
        # Group tools by namespace
        tool_namespaces = {}
        for tool_item in tools:
            namespace = tool_item.get("namespace", "default")
            if namespace not in tool_namespaces:
                tool_namespaces[namespace] = []
            tool_namespaces[namespace].append(tool_item)
        
        for namespace, namespace_tools in tool_namespaces.items():
            print(f"\n{namespace.upper()} Tools:")
            for tool_item in namespace_tools:
                print(f"   - {tool_item['name']}: {tool_item['description']}")
        
        print(f"\n📊 Total tools available: {len(tools)}")
        
        # Test database tools
        print(f"\n🗄️  Testing Database Tools:")
        print("-" * 30)
        
        # Test execute_query
        print("\n1. Testing execute_query:")
        result = await client.call_tool("database.execute_query", {
            "sql": "SELECT * FROM customers LIMIT 3",
        })
        print(f"Result: {result}")
        
        # Test get_sample_data
        print("\n2. Testing get_sample_data:")
        result = await client.call_tool("database.get_sample_data", {
            "table_name": "customers",
            "limit": 2
        })
        print(f"Result: {result}")
        
        # Test validate_sql
        print("\n3. Testing validate_sql:")
        result = await client.call_tool("database.validate_sql", {
            "sql": "SELECT name, email FROM customers WHERE id = 1"
        })
        print(f"Result: {result}")
        
        # Test schema tools
        print(f"\n📋 Testing Schema Tools:")
        print("-" * 30)
        
        # Test get_tables
        print("\n1. Testing get_tables:")
        result = await client.call_tool("schema.get_tables", {})
        print(f"Result: {result}")
        
        # Test get_columns
        print("\n2. Testing get_columns:")
        result = await client.call_tool("schema.get_columns", {
            "table_name": "customers"
        })
        print(f"Result: {result}")
        
        # Test search_schema
        print("\n3. Testing search_schema:")
        result = await client.call_tool("schema.search_schema", {
            "keyword": "customer"
        })
        print(f"Result: {result}")
        
        # Test get_relationships
        print("\n4. Testing get_relationships:")
        result = await client.call_tool("schema.get_relationships", {})
        print(f"Result: {result}")
        
        # Test visualization tools
        print(f"\n📊 Testing Visualization Tools:")
        print("-" * 30)
        
        # Sample data for visualization
        sample_data = json.dumps([
            {"name": "John Doe", "revenue": 15000, "region": "North"},
            {"name": "Jane Smith", "revenue": 22000, "region": "South"},
        ])
        
        # Test create_chart
        print("\n1. Testing create_chart:")
        result = await client.call_tool("visualization.create_chart", {
            "chart_type": "bar",
            "data": sample_data,
            "title": "Customer Revenue by Region"
        })
        print(f"Result: {result}")
        
        # Test get_chart_types
        print("\n2. Testing get_chart_types:")
        result = await client.call_tool("visualization.get_chart_types", {})
        print(f"Result: {result}")
        
        print(f"\n🎉 MCP Integration Example Completed!")
        print("=" * 50)

if __name__ == "__main__":
    # Run the example
    asyncio.run(main())