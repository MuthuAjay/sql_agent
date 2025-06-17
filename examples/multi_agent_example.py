#!/usr/bin/env python3
"""Example demonstrating the multi-agent system."""

import asyncio
import json
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sql_agent.agents import AgentOrchestrator
from sql_agent.core.config import settings
from sql_agent.core.database import db_manager
from sql_agent.utils.logging import get_logger


async def main():
    """Main example function."""
    logger = get_logger("example")
    
    print("🚀 SQL Agent Multi-Agent System Example")
    print("=" * 50)
    
    # Initialize database (if available)
    try:
        await db_manager.initialize()
        print("✅ Database connected")
    except Exception as e:
        print(f"⚠️  Database not available: {e}")
        print("   Continuing with mock data...")
    
    # Create orchestrator
    print("\n🤖 Initializing agents...")
    orchestrator = AgentOrchestrator()
    
    # Show agent information
    agent_info = orchestrator.get_agent_info()
    print(f"✅ Loaded {len(agent_info)} agents:")
    for name, info in agent_info.items():
        print(f"   - {name}: {info['description']}")
    
    # Example queries
    example_queries = [
        "Show me the top 5 customers by revenue",
        "Analyze the customer data trends",
        "Create a bar chart of sales by region",
        "How many orders do we have?",
        "What are the most popular products?"
    ]
    
    print(f"\n📝 Testing {len(example_queries)} example queries...")
    
    for i, query in enumerate(example_queries, 1):
        print(f"\n--- Example {i}: {query} ---")
        
        try:
            # Process the query
            result = await orchestrator.process_query(query)
            
            # Display results
            print(f"Session ID: {result.session_id}")
            print(f"Processing time: {result.processing_time:.2f}s")
            
            # Show routing decision
            if "routing" in result.metadata:
                routing = result.metadata["routing"]
                print(f"Primary agent: {routing.get('primary_agent', 'unknown')}")
                print(f"Confidence: {routing.get('confidence', 0):.2f}")
                print(f"Reasoning: {routing.get('reasoning', 'N/A')}")
            
            # Show SQL generation
            if result.generated_sql:
                print(f"Generated SQL: {result.generated_sql}")
            
            # Show query results
            if result.query_result:
                print(f"Query results: {result.query_result.row_count} rows")
                if result.query_result.data:
                    print("Sample data:")
                    for j, row in enumerate(result.query_result.data[:3]):
                        print(f"   Row {j+1}: {row}")
            
            # Show analysis
            if result.analysis_result:
                print(f"Analysis insights: {len(result.analysis_result.insights)}")
                for insight in result.analysis_result.insights[:2]:
                    print(f"   - {insight}")
            
            # Show visualization
            if result.visualization_config:
                viz = result.visualization_config
                print(f"Visualization: {viz.chart_type} chart")
                print(f"   X-axis: {viz.x_axis}")
                print(f"   Y-axis: {viz.y_axis}")
                print(f"   Title: {viz.title}")
            
            # Show errors if any
            if result.errors:
                print(f"⚠️  Errors: {len(result.errors)}")
                for error in result.errors:
                    print(f"   - {error}")
            
            print("✅ Query processed successfully")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            logger.error("example_query_failed", query=query, error=str(e))
    
    # Show workflow information
    print(f"\n📊 Workflow Information:")
    workflow_info = orchestrator.get_workflow_info()
    print(f"   Agents: {', '.join(workflow_info['agents'])}")
    print(f"   Default flow: {' → '.join(workflow_info['default_flow'])}")
    print(f"   LLM Provider: {workflow_info['llm_provider']}")
    
    # Health check
    print(f"\n🏥 Health Check:")
    health_status = await orchestrator.health_check()
    print(f"   Orchestrator: {health_status['orchestrator']}")
    print(f"   LLM Provider: {health_status['llm_provider']}")
    for agent, status in health_status['agents'].items():
        print(f"   {agent.capitalize()}: {status}")
    
    print(f"\n🎉 Example completed!")
    print("=" * 50)


if __name__ == "__main__":
    # Run the example
    asyncio.run(main()) 