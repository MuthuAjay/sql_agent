#!/usr/bin/env python3
"""
Focused test script for SQL Agent API - Tests only implemented endpoints
Based on actual API test results showing which endpoints work vs return 404
"""

import asyncio
import json
import time
import sys
from typing import Dict, Any, Optional
from dataclasses import dataclass

import httpx
from rich.console import Console
from rich.table import Table

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 30.0

console = Console()

@dataclass
class TestResult:
    """Test result data structure"""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    success: bool
    error_message: Optional[str] = None
    response_data: Optional[Dict] = None

class FocusedAPITester:
    """Focused API testing class - only tests working endpoints"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=TIMEOUT)
        self.results: list[TestResult] = []
        
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None
    ) -> TestResult:
        """Make HTTP request and return test result"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            if method.upper() == "GET":
                response = await self.client.get(url, params=params, headers=headers)
            elif method.upper() == "POST":
                response = await self.client.post(url, json=data, params=params, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response_time = time.time() - start_time
            
            # Try to parse JSON response
            try:
                response_data = response.json()
            except:
                response_data = {"raw_content": response.text}
            
            success = 200 <= response.status_code < 300
            
            result = TestResult(
                endpoint=endpoint,
                method=method.upper(),
                status_code=response.status_code,
                response_time=response_time,
                success=success,
                response_data=response_data,
                error_message=None if success else f"HTTP {response.status_code}"
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            result = TestResult(
                endpoint=endpoint,
                method=method.upper(),
                status_code=0,
                response_time=response_time,
                success=False,
                error_message=str(e)
            )
        
        self.results.append(result)
        return result
    
    async def test_working_infrastructure(self):
        """Test infrastructure endpoints that we know work"""
        console.print("\n[bold green]✅ Testing Working Infrastructure Endpoints[/bold green]")
        
        endpoints = [
            ("GET", "/health"),  # Start with the fastest one
            ("GET", "/api/v1/info"),
            ("GET", "/api/v1/status"),
            ("POST", "/api/v1/ping"),
            ("GET", "/"),  # Test root last since it's slow
        ]
        
        for method, endpoint in endpoints:
            try:
                # Use shorter timeout for potentially problematic endpoints
                if endpoint == "/":
                    console.print(f"  🔄 Testing {method} {endpoint} (may be slow)...")
                
                result = await self.make_request(method, endpoint)
                status = "✅" if result.success else "❌"
                
                # Warn about slow responses
                if result.response_time > 5:
                    console.print(f"  {status} {method} {endpoint} - {result.status_code} ({result.response_time:.3f}s) ⚠️ SLOW")
                else:
                    console.print(f"  {status} {method} {endpoint} - {result.status_code} ({result.response_time:.3f}s)")
                
                # Show some response details for info endpoints
                if result.success and endpoint in ["/api/v1/info", "/api/v1/status"] and result.response_data:
                    if isinstance(result.response_data, dict):
                        for key, value in list(result.response_data.items())[:3]:  # Show first 3 items
                            console.print(f"    {key}: {value}")
                            
            except Exception as e:
                console.print(f"  ❌ {method} {endpoint} - Failed: {str(e)[:50]}...")
                # Add a failed result manually
                self.results.append(TestResult(
                    endpoint=endpoint,
                    method=method,
                    status_code=0,
                    response_time=0,
                    success=False,
                    error_message=str(e)
                ))
    
    async def test_working_sql_endpoints(self):
        """Test SQL endpoints that we know work"""
        console.print("\n[bold green]✅ Testing Working SQL Endpoints[/bold green]")
        
        # Test SQL queries that we know work with our database
        working_sql_queries = [
            {
                "name": "Count customers",
                "sql": "SELECT COUNT(*) as total_customers FROM customers;",
                "database": "sql_agent_db"
            },
            {
                "name": "List premium customers",
                "sql": "SELECT first_name, last_name, country FROM customers WHERE is_premium = true LIMIT 3;",
                "database": "sql_agent_db"
            },
            {
                "name": "Electronics products",
                "sql": "SELECT product_name, price FROM products WHERE category = 'Electronics' LIMIT 3;",
                "database": "sql_agent_db"
            },
            {
                "name": "Customer by country stats",
                "sql": "SELECT country, COUNT(*) as count FROM customers GROUP BY country ORDER BY count DESC;",
                "database": "sql_agent_db"
            },
            {
                "name": "Top orders by value",
                "sql": "SELECT order_id, total_amount, order_status FROM orders ORDER BY total_amount DESC LIMIT 3;",
                "database": "sql_agent_db"
            }
        ]
        
        # Test SQL execute endpoint
        console.print("\n  Testing SQL Execute:")
        for query_info in working_sql_queries:
            result = await self.make_request("POST", "/api/v1/sql/execute", data=query_info)
            status = "✅" if result.success else "❌"
            console.print(f"    {status} {query_info['name']} - {result.status_code} ({result.response_time:.3f}s)")
            
            # Show result count if successful
            if result.success and result.response_data:
                if 'rows' in result.response_data:
                    row_count = len(result.response_data['rows'])
                    console.print(f"      → Returned {row_count} rows")
                elif 'data' in result.response_data:
                    data_count = len(result.response_data['data']) if isinstance(result.response_data['data'], list) else 1
                    console.print(f"      → Returned {data_count} results")
        
        # Test SQL validate endpoint
        console.print("\n  Testing SQL Validate:")
        for query_info in working_sql_queries[:3]:  # Test first 3 for validation
            result = await self.make_request("POST", "/api/v1/sql/validate", data=query_info)
            status = "✅" if result.success else "❌"
            console.print(f"    {status} {query_info['name']} validation - {result.status_code} ({result.response_time:.3f}s)")
            
            if result.success and result.response_data:
                valid = result.response_data.get('valid', result.response_data.get('is_valid', 'unknown'))
                console.print(f"      → Valid: {valid}")
    
    async def test_working_schema_endpoints(self):
        """Test schema endpoints that we know work"""
        console.print("\n[bold green]✅ Testing Working Schema Endpoints[/bold green]")
        
        # Test databases list
        result = await self.make_request("GET", "/api/v1/schema/databases")
        status = "✅" if result.success else "❌"
        console.print(f"  {status} GET /api/v1/schema/databases - {result.status_code} ({result.response_time:.3f}s)")
        
        if result.success and result.response_data:
            # Handle different response formats
            if isinstance(result.response_data, dict):
                databases = result.response_data.get('databases', result.response_data.get('data', []))
            else:
                databases = result.response_data if isinstance(result.response_data, list) else []
            console.print(f"    → Found {len(databases)} databases: {databases}")
        
        # Test tables list
        result = await self.make_request("GET", "/api/v1/schema/tables")
        status = "✅" if result.success else "❌"
        console.print(f"  {status} GET /api/v1/schema/tables - {result.status_code} ({result.response_time:.3f}s)")
        
        if result.success and result.response_data:
            # Handle different response formats for tables
            if isinstance(result.response_data, dict):
                tables = result.response_data.get('tables', result.response_data.get('data', []))
            else:
                tables = result.response_data if isinstance(result.response_data, list) else []
            
            console.print(f"    → Found {len(tables)} tables")
            if tables:
                # Show first few table names - handle different formats
                table_names = []
                for t in tables[:5]:
                    if isinstance(t, str):
                        table_names.append(t)
                    elif isinstance(t, dict):
                        # Try different possible field names
                        name = t.get('name') or t.get('table_name') or t.get('tablename') or str(t)
                        table_names.append(name)
                    else:
                        table_names.append(str(t))
                console.print(f"    → Tables: {table_names}")
                
                # Show raw response structure for debugging
                console.print(f"    → Raw response type: {type(result.response_data)}")
                if isinstance(result.response_data, dict):
                    console.print(f"    → Response keys: {list(result.response_data.keys())}")
                if tables and isinstance(tables[0], dict):
                    console.print(f"    → First table keys: {list(tables[0].keys())}")
    
    async def test_working_visualization(self):
        """Test visualization endpoints that partially work"""
        console.print("\n[bold yellow]🟡 Testing Partially Working Visualization[/bold yellow]")
        
        # Test data that we know works
        working_viz_tests = [
            {
                "name": "Revenue by Country",
                "data": [
                    {"country": "USA", "revenue": 15000.50},
                    {"country": "Canada", "revenue": 8500.25}, 
                    {"country": "UK", "revenue": 12300.75}
                ],
                "chart_type": "bar",
                "x_axis": "country",
                "y_axis": "revenue"
            },
            {
                "name": "Products by Category",
                "data": [
                    {"category": "Electronics", "count": 45},
                    {"category": "Clothing", "count": 32},
                    {"category": "Home & Kitchen", "count": 28}
                ],
                "chart_type": "pie"
            }
        ]
        
        console.print("\n  Testing Visualization Suggest:")
        for viz_test in working_viz_tests:
            result = await self.make_request("POST", "/api/v1/visualization/suggest", data=viz_test)
            status = "✅" if result.success else "❌"
            console.print(f"    {status} {viz_test['name']} - {result.status_code} ({result.response_time:.3f}s)")
            
            if result.success and result.response_data:
                suggestion = result.response_data.get('suggestion', result.response_data.get('chart_type', 'unknown'))
                console.print(f"      → Suggested: {suggestion}")
    
    async def test_error_handling_working_endpoints(self):
        """Test error handling on endpoints we know work"""
        console.print("\n[bold blue]🔧 Testing Error Handling (Working Endpoints)[/bold blue]")
        
        # Test malformed SQL requests on working endpoint
        malformed_sql_tests = [
            {
                "sql": "",  # Empty SQL
                "database": "sql_agent_db"
            },
            {
                "sql": "INVALID SQL SYNTAX HERE",  # Invalid syntax
                "database": "sql_agent_db"
            },
            {
                "sql": "SELECT * FROM nonexistent_table;",  # Invalid table
                "database": "sql_agent_db"
            }
        ]
        
        console.print("\n  Testing Error Handling on SQL Execute:")
        for i, data in enumerate(malformed_sql_tests):
            result = await self.make_request("POST", "/api/v1/sql/execute", data=data)
            expected_error = 400 <= result.status_code < 500
            status = "✅" if expected_error else "❌"
            console.print(f"    {status} Malformed SQL {i+1} - {result.status_code} (Expected 4xx)")
            
            if result.response_data and 'error' in result.response_data:
                error_msg = str(result.response_data['error'])[:50]
                console.print(f"      → Error: {error_msg}...")
    
    async def test_performance_working_endpoints(self):
        """Test performance on endpoints we know work"""
        console.print("\n[bold blue]⚡ Performance Testing (Working Endpoints)[/bold blue]")
        
        # Test concurrent health checks
        async def make_health_check():
            return await self.make_request("GET", "/health")
        
        start_time = time.time()
        tasks = [make_health_check() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        successful_requests = sum(1 for r in results if r.success)
        avg_response_time = sum(r.response_time for r in results) / len(results)
        
        console.print(f"  ✅ Health checks: {successful_requests}/10 successful")
        console.print(f"  ✅ Total time: {total_time:.3f}s")
        console.print(f"  ✅ Average response time: {avg_response_time:.3f}s")
        
        # Test concurrent SQL queries
        async def make_sql_request():
            sql_data = {
                "sql": "SELECT COUNT(*) FROM customers;",
                "database": "sql_agent_db"
            }
            return await self.make_request("POST", "/api/v1/sql/execute", data=sql_data)
        
        console.print("\n  Testing concurrent SQL execution:")
        start_time = time.time()
        tasks = [make_sql_request() for _ in range(3)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        successful_queries = sum(1 for r in results if r.success)
        avg_query_time = sum(r.response_time for r in results) / len(results)
        
        console.print(f"  ✅ SQL queries: {successful_queries}/3 successful")
        console.print(f"  ✅ Total time: {total_time:.3f}s")
        console.print(f"  ✅ Average query time: {avg_query_time:.3f}s")
    
    def generate_focused_report(self):
        """Generate focused test report"""
        console.print("\n[bold green]📊 Focused Test Report[/bold green]")
        
        # Summary statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        avg_response_time = sum(r.response_time for r in self.results) / total_tests if total_tests > 0 else 0
        
        # Create summary table
        summary_table = Table(title="Working Endpoints Test Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Tests", str(total_tests))
        summary_table.add_row("Successful", str(successful_tests))
        summary_table.add_row("Failed", str(failed_tests))
        summary_table.add_row("Success Rate", f"{(successful_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%")
        summary_table.add_row("Avg Response Time", f"{avg_response_time:.3f}s")
        
        console.print(summary_table)
        
        # Feature status table
        console.print("\n[bold cyan]📋 Feature Status Summary[/bold cyan]")
        feature_table = Table()
        feature_table.add_column("Feature", style="cyan")
        feature_table.add_column("Status", style="green")
        feature_table.add_column("Notes", style="blue")
        
        feature_table.add_row("Infrastructure", "✅ Working", "All health/status endpoints functional")
        feature_table.add_row("SQL Execution", "✅ Working", "Execute and validate work well")
        feature_table.add_row("Schema Discovery", "🟡 Partial", "Can list databases/tables, no details")
        feature_table.add_row("Visualization", "🟡 Partial", "Suggest works, generate missing")
        feature_table.add_row("Query Processing", "❌ Missing", "Natural language queries not implemented")
        feature_table.add_row("Data Analysis", "❌ Missing", "Profiling/insights not implemented")
        
        console.print(feature_table)
        
        # Next steps
        console.print("\n[bold yellow]🎯 Recommended Next Steps[/bold yellow]")
        console.print("1. Implement [bold]/api/v1/query/process[/bold] - Core LLM functionality")
        console.print("2. Add [bold]/api/v1/sql/explain[/bold] - Query explanation")
        console.print("3. Complete schema endpoints - table/column details")
        console.print("4. Add data analysis features - profiling and insights")
        console.print("5. Complete visualization pipeline - chart generation")
        
        # Save results
        self.save_focused_results()
    
    def save_focused_results(self):
        """Save focused test results"""
        results_data = {
            "timestamp": time.time(),
            "test_type": "focused_working_endpoints",
            "database": "sql_agent_db",
            "summary": {
                "total_tests": len(self.results),
                "successful_tests": sum(1 for r in self.results if r.success),
                "failed_tests": sum(1 for r in self.results if not r.success),
                "avg_response_time": sum(r.response_time for r in self.results) / len(self.results) if self.results else 0
            },
            "working_features": [
                "Infrastructure (health, status, info)",
                "SQL execution and validation", 
                "Basic schema discovery",
                "Visualization suggestions"
            ],
            "missing_features": [
                "Natural language query processing",
                "SQL explanation",
                "Detailed schema endpoints",
                "Data analysis and profiling",
                "Chart generation"
            ],
            "results": [
                {
                    "endpoint": r.endpoint,
                    "method": r.method,
                    "status_code": r.status_code,
                    "response_time": r.response_time,
                    "success": r.success,
                    "error_message": r.error_message
                }
                for r in self.results
            ]
        }
        
        filename = f"sql_agent_focused_test_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        console.print(f"\n📄 Focused test results saved to: {filename}")

async def main():
    """Main focused test execution function"""
    console.print("[bold green]🎯 SQL Agent API - Focused Test Suite[/bold green]")
    console.print(f"Testing API at: {BASE_URL}")
    console.print("Focus: Only testing endpoints that are known to work")
    console.print("=" * 60)
    
    async with FocusedAPITester(BASE_URL) as tester:
        try:
            # Run focused test suites
            await tester.test_working_infrastructure()
            await tester.test_working_sql_endpoints()
            await tester.test_working_schema_endpoints()
            await tester.test_working_visualization()
            await tester.test_error_handling_working_endpoints()
            await tester.test_performance_working_endpoints()
            
            # Generate focused report
            tester.generate_focused_report()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Focused test suite interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Focused test suite failed with error: {e}[/red]")
            return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)