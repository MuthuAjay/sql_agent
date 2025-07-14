#!/usr/bin/env python3
"""
Comprehensive test script for SQL Agent API

This script tests all major endpoints and functionality of the SQL Agent API,
including health checks, query processing, SQL execution, analysis, and visualization.
Updated to work with the sql_agent_db schema (customers, products, orders, order_items, employee_performance).
"""

import asyncio
import json
import time
import sys
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID
from rich import print as rprint

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

class APITester:
    """Main API testing class"""
    
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
            elif method.upper() == "PUT":
                response = await self.client.put(url, json=data, params=params, headers=headers)
            elif method.upper() == "DELETE":
                response = await self.client.delete(url, params=params, headers=headers)
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
    
    async def test_root_endpoints(self):
        """Test basic root endpoints"""
        console.print("\n[bold blue]Testing Root Endpoints[/bold blue]")
        
        endpoints = [
            ("GET", "/"),
            ("GET", "/health"),
            ("GET", "/api/v1/info"),
            ("GET", "/api/v1/status"),
            ("POST", "/api/v1/ping"),
        ]
        
        for method, endpoint in endpoints:
            result = await self.make_request(method, endpoint)
            status = "✅" if result.success else "❌"
            console.print(f"{status} {method} {endpoint} - {result.status_code} ({result.response_time:.3f}s)")
    
    async def test_query_endpoints(self):
        """Test query processing endpoints with realistic e-commerce queries"""
        console.print("\n[bold blue]Testing Query Endpoints[/bold blue]")
        
        # Test realistic business queries for the e-commerce database
        test_queries = [
            {
                "query": "Show me all customers from USA",
                "database": "sql_agent_db"
            },
            {
                "query": "What are the top 5 products by price?",
                "database": "sql_agent_db"
            },
            {
                "query": "How many orders were placed in June 2024?",
                "database": "sql_agent_db"
            },
            {
                "query": "Which customers have the highest account balance?",
                "database": "sql_agent_db"
            },
            {
                "query": "Show me premium customers who haven't logged in recently",
                "database": "sql_agent_db"
            },
            {
                "query": "What's the total revenue from all orders?",
                "database": "sql_agent_db"
            },
            {
                "query": "Which products are running low on stock?",
                "database": "sql_agent_db"
            },
            {
                "query": "Show me employee performance by department",
                "database": "sql_agent_db"
            }
        ]
        
        for query_data in test_queries:
            result = await self.make_request("POST", "/api/v1/query/process", data=query_data)
            status = "✅" if result.success else "❌"
            query_short = query_data["query"][:50] + "..." if len(query_data["query"]) > 50 else query_data["query"]
            console.print(f"{status} Query: '{query_short}' - {result.status_code} ({result.response_time:.3f}s)")
            
            if not result.success and result.response_data:
                console.print(f"   Error: {result.response_data.get('error', {}).get('detail', 'Unknown error')}")
    
    async def test_sql_endpoints(self):
        """Test SQL execution endpoints with actual database tables"""
        console.print("\n[bold blue]Testing SQL Endpoints[/bold blue]")
        
        # SQL queries that work with our actual database schema
        test_sqls = [
            {
                "sql": "SELECT COUNT(*) as total_customers FROM customers;",
                "database": "sql_agent_db"
            },
            {
                "sql": "SELECT * FROM products WHERE category = 'Electronics' LIMIT 5;",
                "database": "sql_agent_db"
            },
            {
                "sql": "SELECT c.first_name, c.last_name, c.country FROM customers c WHERE c.is_premium = true LIMIT 3;",
                "database": "sql_agent_db"
            },
            {
                "sql": "SELECT p.product_name, p.price, p.stock_quantity FROM products p WHERE p.stock_quantity < p.min_stock_level;",
                "database": "sql_agent_db"
            },
            {
                "sql": "SELECT o.order_id, o.total_amount, o.order_status FROM orders o ORDER BY o.total_amount DESC LIMIT 5;",
                "database": "sql_agent_db"
            },
            {
                "sql": "SELECT department, AVG(performance_score) as avg_score FROM employee_performance GROUP BY department;",
                "database": "sql_agent_db"
            },
            {
                "sql": "SELECT c.country, COUNT(*) as customer_count FROM customers c GROUP BY c.country ORDER BY customer_count DESC;",
                "database": "sql_agent_db"
            }
        ]
        
        endpoints = [
            "/api/v1/sql/execute",
            "/api/v1/sql/validate", 
            "/api/v1/sql/explain"
        ]
        
        for endpoint in endpoints:
            console.print(f"\n  Testing {endpoint}:")
            for i, sql_data in enumerate(test_sqls[:3]):  # Test first 3 queries for each endpoint
                result = await self.make_request("POST", endpoint, data=sql_data)
                status = "✅" if result.success else "❌"
                sql_short = sql_data["sql"][:60] + "..." if len(sql_data["sql"]) > 60 else sql_data["sql"]
                console.print(f"    {status} SQL {i+1}: '{sql_short}' - {result.status_code} ({result.response_time:.3f}s)")
                
                if not result.success and result.response_data:
                    console.print(f"      Error: {result.response_data.get('error', {}).get('detail', 'Unknown error')}")
    
    async def test_analysis_endpoints(self):
        """Test data analysis endpoints with actual tables"""
        console.print("\n[bold blue]Testing Analysis Endpoints[/bold blue]")
        
        # Test analysis on each of our main tables
        test_analyses = [
            {
                "table": "customers",
                "database": "sql_agent_db",
                "analysis_type": "profile"
            },
            {
                "table": "products", 
                "database": "sql_agent_db",
                "analysis_type": "profile"
            },
            {
                "table": "orders",
                "database": "sql_agent_db", 
                "analysis_type": "statistics"
            },
            {
                "table": "employee_performance",
                "database": "sql_agent_db",
                "analysis_type": "insights"
            }
        ]
        
        endpoints = [
            "/api/v1/analysis/profile",
            "/api/v1/analysis/statistics", 
            "/api/v1/analysis/insights"
        ]
        
        for endpoint in endpoints:
            console.print(f"\n  Testing {endpoint}:")
            for analysis_data in test_analyses:
                result = await self.make_request("POST", endpoint, data=analysis_data)
                status = "✅" if result.success else "❌"
                console.print(f"    {status} Table: {analysis_data['table']} - {result.status_code} ({result.response_time:.3f}s)")
                
                if not result.success and result.response_data:
                    console.print(f"      Error: {result.response_data.get('error', {}).get('detail', 'Unknown error')}")
    
    async def test_visualization_endpoints(self):
        """Test visualization endpoints with realistic e-commerce data"""
        console.print("\n[bold blue]Testing Visualization Endpoints[/bold blue]")
        
        # Sample data that could come from our database queries
        visualization_tests = [
            {
                "name": "Revenue by Country",
                "data": [
                    {"country": "USA", "revenue": 15000.50},
                    {"country": "Canada", "revenue": 8500.25}, 
                    {"country": "UK", "revenue": 12300.75},
                    {"country": "Germany", "revenue": 9800.00},
                    {"country": "France", "revenue": 7200.30}
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
                    {"category": "Home & Kitchen", "count": 28},
                    {"category": "Sports", "count": 18},
                    {"category": "Accessories", "count": 15}
                ],
                "chart_type": "pie"
            },
            {
                "name": "Order Status Distribution", 
                "data": [
                    {"status": "delivered", "count": 6},
                    {"status": "shipped", "count": 2},
                    {"status": "processing", "count": 2}
                ],
                "chart_type": "doughnut"
            },
            {
                "name": "Employee Performance by Department",
                "data": [
                    {"department": "Sales", "avg_score": 3.8},
                    {"department": "Engineering", "avg_score": 4.3},
                    {"department": "Marketing", "avg_score": 4.1},
                    {"department": "Customer Service", "avg_score": 4.0}
                ],
                "chart_type": "line"
            }
        ]
        
        endpoints = [
            "/api/v1/visualization/suggest",
            "/api/v1/visualization/generate"
        ]
        
        for endpoint in endpoints:
            console.print(f"\n  Testing {endpoint}:")
            for viz_test in visualization_tests:
                result = await self.make_request("POST", endpoint, data=viz_test)
                status = "✅" if result.success else "❌"
                console.print(f"    {status} {viz_test['name']} - {result.status_code} ({result.response_time:.3f}s)")
                
                if not result.success and result.response_data:
                    console.print(f"      Error: {result.response_data.get('error', {}).get('detail', 'Unknown error')}")
    
    async def test_schema_endpoints(self):
        """Test schema management endpoints with actual database"""
        console.print("\n[bold blue]Testing Schema Endpoints[/bold blue]")
        
        # Test GET endpoints
        get_endpoints = [
            "/api/v1/schema/databases",
            "/api/v1/schema/tables", 
            "/api/v1/schema/columns"
        ]
        
        for endpoint in get_endpoints:
            result = await self.make_request("GET", endpoint)
            status = "✅" if result.success else "❌"
            console.print(f"{status} GET {endpoint} - {result.status_code} ({result.response_time:.3f}s)")
            
            if result.success and result.response_data:
                # Log some details about what was returned
                if 'databases' in result.response_data:
                    db_count = len(result.response_data.get('databases', []))
                    console.print(f"    Found {db_count} databases")
                elif 'tables' in result.response_data:
                    table_count = len(result.response_data.get('tables', []))
                    console.print(f"    Found {table_count} tables")
        
        # Test specific database schema
        result = await self.make_request("GET", "/api/v1/schema/database/sql_agent_db")
        status = "✅" if result.success else "❌"
        console.print(f"{status} GET /api/v1/schema/database/sql_agent_db - {result.status_code} ({result.response_time:.3f}s)")
        
        # Test specific table schemas
        tables = ["customers", "products", "orders", "order_items", "employee_performance"]
        for table in tables:
            result = await self.make_request("GET", f"/api/v1/schema/table/{table}")
            status = "✅" if result.success else "❌"
            console.print(f"    {status} GET /api/v1/schema/table/{table} - {result.status_code} ({result.response_time:.3f}s)")
    
    async def test_business_query_scenarios(self):
        """Test complex business scenarios that combine multiple operations"""
        console.print("\n[bold blue]Testing Business Query Scenarios[/bold blue]")
        
        scenarios = [
            {
                "name": "Customer Analytics Dashboard",
                "queries": [
                    "How many customers do we have by country?",
                    "What's the average account balance of premium vs regular customers?",
                    "Who are our top 10 customers by loyalty points?"
                ]
            },
            {
                "name": "Sales Performance Analysis", 
                "queries": [
                    "What's our total revenue this month?",
                    "Which products have the highest profit margins?",
                    "What's the average order value by customer segment?"
                ]
            },
            {
                "name": "Inventory Management",
                "queries": [
                    "Which products are below minimum stock levels?",
                    "What's the inventory value by category?",
                    "Which brands are performing best in sales?"
                ]
            },
            {
                "name": "HR Analytics",
                "queries": [
                    "What's the average performance score by department?",
                    "Which employees exceeded their sales targets?",
                    "How does training hours correlate with performance?"
                ]
            }
        ]
        
        for scenario in scenarios:
            console.print(f"\n  {scenario['name']}:")
            for query in scenario['queries']:
                query_data = {
                    "query": query,
                    "database": "sql_agent_db"
                }
                result = await self.make_request("POST", "/api/v1/query/process", data=query_data)
                status = "✅" if result.success else "❌"
                console.print(f"    {status} {query} - {result.status_code} ({result.response_time:.3f}s)")
    
    async def test_error_handling(self):
        """Test error handling and edge cases"""
        console.print("\n[bold blue]Testing Error Handling[/bold blue]")
        
        # Test invalid endpoints
        invalid_endpoints = [
            ("GET", "/api/v1/nonexistent"),
            ("POST", "/api/v1/invalid/endpoint"),
        ]
        
        for method, endpoint in invalid_endpoints:
            result = await self.make_request(method, endpoint)
            status = "✅" if result.status_code == 404 else "❌"
            console.print(f"{status} {method} {endpoint} - {result.status_code} (Expected 404)")
        
        # Test malformed SQL requests
        malformed_sql_tests = [
            {
                "sql": "",  # Empty SQL
                "database": "sql_agent_db"
            },
            {
                "sql": "SELECT * FROM nonexistent_table;",  # Invalid table
                "database": "sql_agent_db"
            },
            {
                "sql": "INVALID SQL SYNTAX HERE",  # Invalid syntax
                "database": "sql_agent_db"
            },
            {
                "sql": "DROP TABLE customers;",  # Potentially dangerous SQL
                "database": "sql_agent_db"
            }
        ]
        
        console.print("\n  Testing malformed SQL requests:")
        for i, data in enumerate(malformed_sql_tests):
            result = await self.make_request("POST", "/api/v1/sql/execute", data=data)
            status = "✅" if 400 <= result.status_code < 500 or result.status_code == 403 else "❌"
            console.print(f"    {status} Malformed SQL {i+1} - {result.status_code}")
            
        # Test malformed query requests
        malformed_query_tests = [
            {"query": None, "database": "sql_agent_db"},  # Null query
            {"query": "", "database": "sql_agent_db"},    # Empty query
            {"query": "Valid query", "database": ""},     # Empty database
            {"query": "Valid query"},                     # Missing database
        ]
        
        console.print("\n  Testing malformed query requests:")
        for i, data in enumerate(malformed_query_tests):
            result = await self.make_request("POST", "/api/v1/query/process", data=data)
            status = "✅" if 400 <= result.status_code < 500 else "❌"
            console.print(f"    {status} Malformed Query {i+1} - {result.status_code}")
    
    async def test_performance(self):
        """Test API performance with concurrent requests"""
        console.print("\n[bold blue]Testing Performance[/bold blue]")
        
        # Test concurrent health checks
        async def make_health_check():
            return await self.make_request("GET", "/health")
        
        start_time = time.time()
        tasks = [make_health_check() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        successful_requests = sum(1 for r in results if r.success)
        avg_response_time = sum(r.response_time for r in results) / len(results)
        
        console.print(f"✅ Concurrent health checks: {successful_requests}/10 successful")
        console.print(f"✅ Total time: {total_time:.3f}s")
        console.print(f"✅ Average response time: {avg_response_time:.3f}s")
        
        # Test concurrent query processing
        async def make_query_request():
            query_data = {
                "query": "SELECT COUNT(*) FROM customers",
                "database": "sql_agent_db"
            }
            return await self.make_request("POST", "/api/v1/query/process", data=query_data)
        
        console.print("\n  Testing concurrent query processing:")
        start_time = time.time()
        tasks = [make_query_request() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        successful_queries = sum(1 for r in results if r.success)
        avg_query_time = sum(r.response_time for r in results) / len(results)
        
        console.print(f"✅ Concurrent queries: {successful_queries}/5 successful") 
        console.print(f"✅ Total time: {total_time:.3f}s")
        console.print(f"✅ Average query time: {avg_query_time:.3f}s")
    
    def generate_report(self):
        """Generate test report"""
        console.print("\n[bold green]Test Report[/bold green]")
        
        # Summary statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        avg_response_time = sum(r.response_time for r in self.results) / total_tests if total_tests > 0 else 0
        
        # Create summary table
        summary_table = Table(title="Test Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Tests", str(total_tests))
        summary_table.add_row("Successful", str(successful_tests))
        summary_table.add_row("Failed", str(failed_tests))
        summary_table.add_row("Success Rate", f"{(successful_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%")
        summary_table.add_row("Avg Response Time", f"{avg_response_time:.3f}s")
        
        console.print(summary_table)
        
        # Endpoint success rate breakdown
        endpoint_stats = {}
        for result in self.results:
            key = f"{result.method} {result.endpoint}"
            if key not in endpoint_stats:
                endpoint_stats[key] = {"total": 0, "success": 0}
            endpoint_stats[key]["total"] += 1
            if result.success:
                endpoint_stats[key]["success"] += 1
        
        if endpoint_stats:
            console.print("\n[bold cyan]Endpoint Success Rates[/bold cyan]")
            endpoint_table = Table()
            endpoint_table.add_column("Endpoint", style="cyan")
            endpoint_table.add_column("Success Rate", style="green")
            endpoint_table.add_column("Total Requests", style="blue")
            
            for endpoint, stats in sorted(endpoint_stats.items()):
                success_rate = (stats["success"] / stats["total"]) * 100
                color = "green" if success_rate == 100 else "yellow" if success_rate >= 80 else "red"
                endpoint_table.add_row(
                    endpoint,
                    f"[{color}]{success_rate:.1f}%[/{color}]",
                    str(stats["total"])
                )
            
            console.print(endpoint_table)
        
        # Failed tests details
        if failed_tests > 0:
            console.print("\n[bold red]Failed Tests[/bold red]")
            failed_table = Table()
            failed_table.add_column("Method", style="cyan")
            failed_table.add_column("Endpoint", style="cyan")
            failed_table.add_column("Status Code", style="red")
            failed_table.add_column("Error", style="red")
            
            for result in self.results:
                if not result.success:
                    error_msg = result.error_message or "Unknown error"
                    if result.response_data and 'error' in result.response_data:
                        error_msg = result.response_data['error'].get('detail', error_msg)
                    
                    failed_table.add_row(
                        result.method,
                        result.endpoint,
                        str(result.status_code),
                        error_msg[:50] + "..." if len(error_msg) > 50 else error_msg
                    )
            
            console.print(failed_table)
        
        # Save detailed results to file
        self.save_results_to_file()
    
    def save_results_to_file(self):
        """Save test results to JSON file"""
        results_data = {
            "timestamp": time.time(),
            "database": "sql_agent_db",
            "tables": ["customers", "products", "orders", "order_items", "employee_performance"],
            "summary": {
                "total_tests": len(self.results),
                "successful_tests": sum(1 for r in self.results if r.success),
                "failed_tests": sum(1 for r in self.results if not r.success),
                "avg_response_time": sum(r.response_time for r in self.results) / len(self.results) if self.results else 0
            },
            "results": [
                {
                    "endpoint": r.endpoint,
                    "method": r.method,
                    "status_code": r.status_code,
                    "response_time": r.response_time,
                    "success": r.success,
                    "error_message": r.error_message,
                    "response_data": r.response_data
                }
                for r in self.results
            ]
        }
        
        filename = f"sql_agent_test_results_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        console.print(f"\n📄 Detailed results saved to: {filename}")

async def main():
    """Main test execution function"""
    console.print("[bold green]SQL Agent API Test Suite[/bold green]")
    console.print(f"Testing API at: {BASE_URL}")
    console.print("Database: sql_agent_db")
    console.print("Tables: customers, products, orders, order_items, employee_performance")
    console.print("=" * 70)
    
    async with APITester(BASE_URL) as tester:
        try:
            # Run all test suites
            await tester.test_root_endpoints()
            await tester.test_query_endpoints()
            await tester.test_sql_endpoints()
            await tester.test_analysis_endpoints()
            await tester.test_visualization_endpoints()
            await tester.test_schema_endpoints()
            await tester.test_business_query_scenarios()
            await tester.test_error_handling()
            await tester.test_performance()
            
            # Generate report
            tester.generate_report()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Test suite interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Test suite failed with error: {e}[/red]")
            return 1
    
    return 0

def run_specific_test(test_name: str):
    """Run a specific test suite"""
    test_functions = {
        "root": "test_root_endpoints",
        "query": "test_query_endpoints", 
        "sql": "test_sql_endpoints",
        "analysis": "test_analysis_endpoints",
        "viz": "test_visualization_endpoints",
        "schema": "test_schema_endpoints",
        "business": "test_business_query_scenarios",
        "errors": "test_error_handling",
        "performance": "test_performance"
    }
    
    if test_name not in test_functions:
        console.print(f"[red]Unknown test: {test_name}[/red]")
        console.print(f"Available tests: {', '.join(test_functions.keys())}")
        return 1
    
    async def run_single_test():
        async with APITester(BASE_URL) as tester:
            method = getattr(tester, test_functions[test_name])
            await method()
            tester.generate_report()
    
    return asyncio.run(run_single_test())

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        exit_code = run_specific_test(test_name)
    else:
        # Run all tests
        exit_code = asyncio.run(main())
    
    sys.exit(exit_code)