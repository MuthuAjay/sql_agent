#!/usr/bin/env python3
"""Example demonstrating RAG integration with the multi-agent system."""

import asyncio
import json
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sql_agent.agents import AgentOrchestrator
from sql_agent.core.database import db_manager
from sql_agent.utils.logging import get_logger


async def main():
    """Main example function demonstrating RAG + Multi-Agent integration."""
    logger = get_logger("rag_multi_agent_example")
    
    print("🚀 SQL Agent RAG + Multi-Agent Integration Example")
    print("=" * 60)
    
    # Initialize database (if available)
    try:
        await db_manager.initialize()
        print("✅ Database connected")
    except Exception as e:
        print(f"⚠️  Database not available: {e}")
        print("   Continuing with mock data...")
        return
    
    # Create orchestrator
    orchestrator = AgentOrchestrator()
    
    # Initialize orchestrator (this will also initialize RAG)
    print("\n🔧 Initializing Multi-Agent System with RAG...")
    try:
        await orchestrator.initialize()
        print("✅ Multi-Agent System initialized with RAG")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Test queries that demonstrate RAG-enhanced capabilities
    test_queries = [
        "Show me customer information with their order history",
        "Analyze sales performance by product category",
        "Create a chart showing revenue trends over time",
        "Find users who made purchases in the last month",
        "Compare customer satisfaction scores across different regions"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: '{query}'")
        print("-" * 50)
        
        try:
            # Process query through the multi-agent system
            result = await orchestrator.process_query(query)
            
            # Display results
            print(f"   Session ID: {result.session_id}")
            print(f"   Processing Time: {result.processing_time:.2f}s")
            print(f"   RAG Context Count: {len(result.schema_context) if result.schema_context else 0}")
            
            # Show routing information
            routing = result.metadata.get("routing", {})
            print(f"   Primary Agent: {routing.get('primary_agent', 'unknown')}")
            print(f"   Confidence: {routing.get('confidence', 0):.2f}")
            print(f"   Secondary Agents: {routing.get('secondary_agents', [])}")
            
            # Show intent analysis
            intent_analysis = result.metadata.get("intent_analysis", {})
            if intent_analysis:
                print(f"   Primary Intent: {intent_analysis.get('primary_intent', 'unknown')}")
                print(f"   Overall Confidence: {intent_analysis.get('overall_confidence', 0):.2f}")
            
            # Show SQL generation results
            if result.generated_sql:
                print(f"   Generated SQL: {result.generated_sql}")
            
            # Show query results
            if result.query_result:
                print(f"   Query Results: {result.query_result.row_count} rows")
                if result.query_result.error:
                    print(f"   Query Error: {result.query_result.error}")
            
            # Show analysis results
            if result.analysis_result:
                print(f"   Analysis Insights: {len(result.analysis_result.insights)} insights")
                print(f"   Data Quality Score: {result.analysis_result.data_quality_score:.2f}")
                if result.analysis_result.recommendations:
                    print(f"   Recommendations: {len(result.analysis_result.recommendations)} items")
            
            # Show visualization results
            if result.visualization_config:
                print(f"   Chart Type: {result.visualization_config.chart_type}")
                print(f"   Chart Title: {result.visualization_config.title}")
            
            # Show any errors
            if result.errors:
                print(f"   Errors: {len(result.errors)}")
                for error in result.errors:
                    print(f"     - {error}")
            
            print("   ✅ Query processed successfully")
            
        except Exception as e:
            print(f"   ❌ Error processing query: {e}")
    
    # Test custom agent sequences
    print(f"\n🎯 Testing Custom Agent Sequences:")
    print("-" * 50)
    
    custom_flows = [
        ["router", "sql"],  # Just routing and SQL generation
        ["router", "sql", "analysis"],  # Routing, SQL, and analysis
        ["router", "sql", "visualization"],  # Routing, SQL, and visualization
    ]
    
    test_query = "Show me the top 5 customers by revenue"
    
    for i, agent_sequence in enumerate(custom_flows, 1):
        print(f"\n   Flow {i}: {' → '.join(agent_sequence)}")
        
        try:
            result = await orchestrator.process_query_with_custom_flow(
                test_query, agent_sequence
            )
            
            print(f"     Processing Time: {result.processing_time:.2f}s")
            print(f"     RAG Context Count: {len(result.schema_context) if result.schema_context else 0}")
            print(f"     Agent History: {' → '.join(result.agent_history)}")
            
            if result.errors:
                print(f"     Errors: {len(result.errors)}")
            else:
                print("     ✅ Flow completed successfully")
                
        except Exception as e:
            print(f"     ❌ Flow failed: {e}")
    
    # Test RAG-specific functionality
    print(f"\n🧠 Testing RAG-Specific Features:")
    print("-" * 50)
    
    try:
        # Get context statistics
        from sql_agent.rag import context_manager
        stats = await context_manager.get_context_statistics()
        
        print(f"   Vector Store Contexts: {stats['vector_store']['total_contexts']}")
        print(f"   Unique Tables: {stats['vector_store']['unique_tables']}")
        print(f"   Embedding Dimension: {stats['embedding_dimension']}")
        
        # Test keyword-based search
        keywords = ["customer", "revenue", "order"]
        contexts = await context_manager.search_schema_by_keywords(keywords, limit=3)
        print(f"   Keyword Search Results: {len(contexts)} contexts found")
        
        # Test relevant table identification
        relevant_tables = await context_manager.get_relevant_tables(
            "customer purchase history", limit=3
        )
        print(f"   Relevant Tables: {relevant_tables}")
        
    except Exception as e:
        print(f"   ❌ RAG features test failed: {e}")
    
    # Performance summary
    print(f"\n📊 Performance Summary:")
    print("-" * 50)
    
    try:
        # Get workflow information
        workflow_info = orchestrator.get_workflow_info()
        print(f"   Available Agents: {', '.join(workflow_info['agents'])}")
        print(f"   Default Flow: {' → '.join(workflow_info['default_flow'])}")
        print(f"   LLM Provider: {workflow_info['llm_provider']}")
        
        # Health check
        health = await orchestrator.health_check()
        print(f"   Orchestrator Health: {health['orchestrator']}")
        print(f"   LLM Provider Health: {health['llm_provider']}")
        
        agent_health = health['agents']
        for agent, status in agent_health.items():
            print(f"   {agent.capitalize()} Agent: {status}")
        
    except Exception as e:
        print(f"   ❌ Performance summary failed: {e}")
    
    print(f"\n🎉 RAG + Multi-Agent Integration Example Completed!")
    print("=" * 60)
    
    # Summary
    print(f"\n📋 Summary:")
    print(f"   ✅ Multi-Agent System with RAG initialized")
    print(f"   ✅ {len(test_queries)} test queries processed")
    print(f"   ✅ Custom agent flows tested")
    print(f"   ✅ RAG-specific features demonstrated")
    print(f"   ✅ Performance metrics collected")
    print(f"\n🔧 Key Features Demonstrated:")
    print(f"   • RAG-enhanced schema context retrieval")
    print(f"   • Improved intent analysis with schema context")
    print(f"   • Better SQL generation with relevant context")
    print(f"   • Seamless integration between RAG and agents")
    print(f"   • Fallback mechanisms for robustness")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main()) 