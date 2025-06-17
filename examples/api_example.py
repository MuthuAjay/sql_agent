"""
API Example Script

This script demonstrates how to use the SQL Agent API endpoints
for natural language to SQL conversion, analysis, and visualization.
"""

import asyncio
import json
import time
from typing import Dict, Any

import httpx


class SQLAgentAPI:
    """Client for the SQL Agent API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    async def process_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Process a natural language query."""
        payload = {
            "query": query,
            **kwargs
        }
        response = await self.client.post(
            f"{self.base_url}/api/v1/query",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def simple_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Process a simple query (SQL only)."""
        payload = {
            "query": query,
            **kwargs
        }
        response = await self.client.post(
            f"{self.base_url}/api/v1/query/simple",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def generate_sql(self, query: str, **kwargs) -> Dict[str, Any]:
        """Generate SQL from natural language."""
        payload = {
            "query": query,
            **kwargs
        }
        response = await self.client.post(
            f"{self.base_url}/api/v1/sql/generate",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def execute_sql(self, sql: str, **kwargs) -> Dict[str, Any]:
        """Execute SQL query."""
        payload = {
            "sql": sql,
            **kwargs
        }
        response = await self.client.post(
            f"{self.base_url}/api/v1/sql/execute",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def analyze_data(self, data: list, **kwargs) -> Dict[str, Any]:
        """Analyze data."""
        payload = {
            "data": data,
            **kwargs
        }
        response = await self.client.post(
            f"{self.base_url}/api/v1/analysis/analyze",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def create_visualization(self, data: list, **kwargs) -> Dict[str, Any]:
        """Create visualization."""
        payload = {
            "data": data,
            **kwargs
        }
        response = await self.client.post(
            f"{self.base_url}/api/v1/visualization/create",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def get_schema(self, database_name: str = None) -> Dict[str, Any]:
        """Get database schema."""
        params = {}
        if database_name:
            params["database_name"] = database_name
        
        response = await self.client.get(
            f"{self.base_url}/api/v1/schema",
            params=params
        )
        response.raise_for_status()
        return response.json()


async def main():
    """Main example function."""
    api = SQLAgentAPI()
    
    try:
        print("🚀 SQL Agent API Example")
        print("=" * 50)
        
        # Check API health
        print("\n1. Checking API health...")
        health = await api.health_check()
        print(f"✅ API Status: {health['status']}")
        print(f"📊 Services: {json.dumps(health['services'], indent=2)}")
        
        # Example 1: Simple query
        print("\n2. Processing simple query...")
        simple_result = await api.simple_query(
            "Show me the first 5 users from the users table"
        )
        print(f"✅ Query: {simple_result['query']}")
        print(f"🎯 Intent: {simple_result['intent']}")
        print(f"📝 SQL: {simple_result['sql']}")
        print(f"📊 Results: {len(simple_result['data'])} rows")
        print(f"⏱️  Processing time: {simple_result['processing_time']:.2f}s")
        
        # Example 2: Full query with analysis
        print("\n3. Processing full query with analysis...")
        full_result = await api.process_query(
            "What are the top 10 products by sales?",
            include_analysis=True,
            include_visualization=True
        )
        print(f"✅ Query: {full_result['query']}")
        print(f"🎯 Intent: {full_result['intent']}")
        
        if full_result.get('sql_result'):
            sql_result = full_result['sql_result']
            print(f"📝 SQL: {sql_result['sql']}")
            print(f"📊 Results: {sql_result['row_count']} rows")
            print(f"⏱️  Execution time: {sql_result['execution_time']:.2f}s")
        
        if full_result.get('analysis_result'):
            analysis = full_result['analysis_result']
            print(f"🧠 Insights: {len(analysis['insights'])} insights")
            print(f"📈 Data quality score: {analysis['data_quality_score']:.2f}")
            print(f"💡 Recommendations: {len(analysis['recommendations'])} recommendations")
        
        if full_result.get('visualization_result'):
            viz = full_result['visualization_result']
            print(f"📊 Chart type: {viz['chart_type']}")
            print(f"📝 Title: {viz['title']}")
        
        print(f"⏱️  Total processing time: {full_result['processing_time']:.2f}s")
        
        # Example 3: SQL generation only
        print("\n4. Generating SQL only...")
        sql_result = await api.generate_sql(
            "Find all orders from last month with total amount greater than $100"
        )
        print(f"✅ Query: {sql_result['query']}")
        print(f"📝 Generated SQL: {sql_result['generated_sql']}")
        print(f"📖 Explanation: {sql_result['explanation']}")
        
        # Example 4: Data analysis
        print("\n5. Analyzing sample data...")
        sample_data = [
            {"product": "Laptop", "sales": 150, "revenue": 75000},
            {"product": "Phone", "sales": 200, "revenue": 60000},
            {"product": "Tablet", "sales": 75, "revenue": 22500},
            {"product": "Monitor", "sales": 100, "revenue": 30000}
        ]
        
        analysis_result = await api.analyze_data(
            sample_data,
            query_context="Analyze product sales performance"
        )
        print(f"🧠 Insights: {len(analysis_result['insights'])} insights")
        print(f"📈 Data quality score: {analysis_result['data_quality_score']:.2f}")
        print(f"💡 Recommendations: {len(analysis_result['recommendations'])} recommendations")
        
        # Example 5: Visualization
        print("\n6. Creating visualization...")
        viz_result = await api.create_visualization(
            sample_data,
            chart_type="bar",
            title="Product Sales Performance",
            x_axis="product",
            y_axis="sales"
        )
        print(f"📊 Chart type: {viz_result['chart_type']}")
        print(f"📝 Title: {viz_result['title']}")
        print(f"📤 Export formats: {viz_result['export_formats']}")
        
        # Example 6: Schema information
        print("\n7. Getting schema information...")
        try:
            schema = await api.get_schema()
            print(f"📊 Database: {schema['database_name']}")
            print(f"📋 Tables: {schema['total_tables']}")
            print(f"📝 Columns: {schema['total_columns']}")
            print(f"🔗 Relationships: {len(schema['relationships'])}")
        except Exception as e:
            print(f"⚠️  Schema not available: {e}")
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure the SQL Agent API is running on http://localhost:8000")
        print("   You can start it with: uvicorn sql_agent.api.main:app --reload")
    
    finally:
        await api.close()


if __name__ == "__main__":
    asyncio.run(main()) 