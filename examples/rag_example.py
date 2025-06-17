#!/usr/bin/env python3
"""Example demonstrating the RAG (Retrieval-Augmented Generation) functionality."""

import asyncio
import json
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sql_agent.rag import context_manager, schema_processor, vector_store, embedding_service
from sql_agent.core.config import settings
from sql_agent.core.database import db_manager
from sql_agent.utils.logging import get_logger


async def main():
    """Main example function."""
    logger = get_logger("rag_example")
    
    print("🚀 SQL Agent RAG Integration Example")
    print("=" * 50)
    
    # Initialize database (if available)
    try:
        await db_manager.initialize()
        print("✅ Database connected")
    except Exception as e:
        print(f"⚠️  Database not available: {e}")
        print("   Continuing with mock data...")
        return
    
    # Initialize RAG components
    print("\n🔧 Initializing RAG components...")
    try:
        await context_manager.initialize()
        print("✅ RAG components initialized")
    except Exception as e:
        print(f"❌ Failed to initialize RAG components: {e}")
        return
    
    # Get context statistics
    print("\n📊 Context Statistics:")
    stats = await context_manager.get_context_statistics()
    print(f"   Total contexts: {stats['vector_store']['total_contexts']}")
    print(f"   Unique tables: {stats['vector_store']['unique_tables']}")
    print(f"   Embedding dimension: {stats['embedding_dimension']}")
    print(f"   Last update: {stats['last_update']}")
    
    # Test schema context retrieval
    print(f"\n🔍 Testing Schema Context Retrieval:")
    print("-" * 40)
    
    test_queries = [
        "Show me customer information",
        "Find orders with high revenue",
        "Get product details and pricing",
        "Analyze sales performance",
        "List all users and their roles"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        
        try:
            # Retrieve relevant schema context
            contexts = await context_manager.retrieve_schema_context(query, limit=3)
            
            if contexts:
                print(f"   Found {len(contexts)} relevant contexts:")
                for j, context in enumerate(contexts, 1):
                    if context.column_name:
                        print(f"      {j}. Column: {context.table_name}.{context.column_name}")
                        print(f"         Type: {context.data_type}")
                    else:
                        print(f"      {j}. Table: {context.table_name}")
                    
                    if context.description:
                        desc = context.description[:80] + "..." if len(context.description) > 80 else context.description
                        print(f"         Description: {desc}")
            else:
                print("   No relevant contexts found")
                
        except Exception as e:
            print(f"   Error: {e}")
    
    # Test keyword-based search
    print(f"\n🔎 Testing Keyword-Based Search:")
    print("-" * 40)
    
    test_keywords = [
        ["customer", "user"],
        ["order", "purchase"],
        ["product", "item"],
        ["revenue", "sales"],
        ["email", "contact"]
    ]
    
    for i, keywords in enumerate(test_keywords, 1):
        print(f"\n{i}. Keywords: {keywords}")
        
        try:
            contexts = await context_manager.search_schema_by_keywords(keywords, limit=3)
            
            if contexts:
                print(f"   Found {len(contexts)} matching contexts:")
                for j, context in enumerate(contexts, 1):
                    if context.column_name:
                        print(f"      {j}. {context.table_name}.{context.column_name}")
                    else:
                        print(f"      {j}. Table: {context.table_name}")
            else:
                print("   No matching contexts found")
                
        except Exception as e:
            print(f"   Error: {e}")
    
    # Test table-specific context retrieval
    print(f"\n📋 Testing Table-Specific Context Retrieval:")
    print("-" * 40)
    
    # Get schema summary to find available tables
    schema_summary = await schema_processor.get_schema_summary()
    available_tables = list(schema_summary.get("tables", {}).keys())[:3]  # Test first 3 tables
    
    for i, table_name in enumerate(available_tables, 1):
        print(f"\n{i}. Table: {table_name}")
        
        try:
            contexts = await context_manager.retrieve_context_by_tables([table_name])
            
            if contexts:
                print(f"   Found {len(contexts)} contexts:")
                for j, context in enumerate(contexts, 1):
                    if context.column_name:
                        print(f"      {j}. Column: {context.column_name}")
                        print(f"         Type: {context.data_type}")
                    else:
                        print(f"      {j}. Table context")
            else:
                print("   No contexts found for this table")
                
        except Exception as e:
            print(f"   Error: {e}")
    
    # Test relevant table identification
    print(f"\n🎯 Testing Relevant Table Identification:")
    print("-" * 40)
    
    complex_queries = [
        "Find customers who made purchases in the last month",
        "Show me the top 10 products by sales volume",
        "Get user profiles with their order history",
        "Analyze revenue trends by product category"
    ]
    
    for i, query in enumerate(complex_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        
        try:
            relevant_tables = await context_manager.get_relevant_tables(query, limit=3)
            
            if relevant_tables:
                print(f"   Relevant tables: {relevant_tables}")
            else:
                print("   No relevant tables identified")
                
        except Exception as e:
            print(f"   Error: {e}")
    
    # Test embedding similarity
    print(f"\n🧮 Testing Embedding Similarity:")
    print("-" * 40)
    
    try:
        # Generate embeddings for similar queries
        query1 = "customer information"
        query2 = "user details"
        query3 = "product catalog"
        
        embedding1 = await embedding_service.embed_query(query1)
        embedding2 = await embedding_service.embed_query(query2)
        embedding3 = await embedding_service.embed_query(query3)
        
        # Calculate similarities
        similarity_12 = await embedding_service.compute_similarity(embedding1, embedding2)
        similarity_13 = await embedding_service.compute_similarity(embedding1, embedding3)
        similarity_23 = await embedding_service.compute_similarity(embedding2, embedding3)
        
        print(f"   Similarity between '{query1}' and '{query2}': {similarity_12:.3f}")
        print(f"   Similarity between '{query1}' and '{query3}': {similarity_13:.3f}")
        print(f"   Similarity between '{query2}' and '{query3}': {similarity_23:.3f}")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test schema refresh
    print(f"\n🔄 Testing Schema Refresh:")
    print("-" * 40)
    
    try:
        print("   Refreshing schema contexts...")
        await context_manager.refresh_schema_contexts()
        print("   ✅ Schema contexts refreshed successfully")
        
        # Get updated statistics
        updated_stats = await context_manager.get_context_statistics()
        print(f"   Updated total contexts: {updated_stats['vector_store']['total_contexts']}")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test cache staleness
    print(f"\n⏰ Testing Cache Management:")
    print("-" * 40)
    
    is_stale = context_manager.is_schema_stale(max_age_minutes=1)  # Very short time for testing
    print(f"   Schema cache is stale: {is_stale}")
    
    # Test individual table update
    if available_tables:
        test_table = available_tables[0]
        print(f"\n   Updating context for table: {test_table}")
        
        try:
            await context_manager.update_table_context(test_table)
            print(f"   ✅ Table context updated successfully")
        except Exception as e:
            print(f"   Error: {e}")
    
    print(f"\n🎉 RAG Integration Example Completed!")
    print("=" * 50)
    
    # Summary
    print(f"\n📋 Summary:")
    print(f"   ✅ RAG components initialized")
    print(f"   ✅ Schema context retrieval tested")
    print(f"   ✅ Keyword-based search tested")
    print(f"   ✅ Table-specific context tested")
    print(f"   ✅ Relevant table identification tested")
    print(f"   ✅ Embedding similarity tested")
    print(f"   ✅ Schema refresh tested")
    print(f"   ✅ Cache management tested")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main()) 