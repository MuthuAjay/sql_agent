#!/usr/bin/env python3
"""
Enhanced Test Suite for SQL Agent API - Phase 3 Comprehensive Validation

This test suite validates:
- Core functionality with business intelligence
- Performance under load with schema complexity
- Production readiness with monitoring
- Business domain accuracy with real scenarios
- Advanced features like optimization and analytics
"""

import asyncio
import json
import time
import sys
import statistics
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import random

import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID
from rich.panel import Panel

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 60.0
MAX_CONCURRENT_REQUESTS = 10

console = Console()

@dataclass
class TestResult:
    """Enhanced test result with business context"""
    endpoint: str
    method: str
    test_category: str
    business_domain: Optional[str]
    status_code: int
    response_time: float
    success: bool
    error_message: Optional[str] = None
    response_data: Optional[Dict] = None
    validation_passed: bool = True
    performance_rating: str = "good"  # excellent, good, average, poor
    business_accuracy: Optional[float] = None  # 0-1 score

@dataclass
class PerformanceMetrics:
    """Performance analysis metrics"""
    min_time: float
    max_time: float
    avg_time: float
    median_time: float
    p95_time: float
    requests_per_second: float
    success_rate: float

@dataclass
class BusinessScenario:
    """Business scenario test case"""
    name: str
    domain: str
    description: str
    natural_query: str
    expected_tables: List[str]
    expected_insights: List[str]
    complexity: str  # simple, medium, complex

class EnhancedAPITester:
    """Enhanced API testing with production validation"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=TIMEOUT)
        self.results: List[TestResult] = []
        self.start_time = time.time()
        
        # Business scenarios for testing
        self.business_scenarios = [
            BusinessScenario(
                name="Customer Revenue Analysis",
                domain="customer_management", 
                description="Analyze customer revenue trends by geography",
                natural_query="Show me the top 10 customers by revenue in each country",
                expected_tables=["customers", "orders"],
                expected_insights=["revenue_distribution", "geographic_patterns"],
                complexity="medium"
            ),
            BusinessScenario(
                name="Product Performance",
                domain="product_catalog",
                description="Identify best-selling products",
                natural_query="Which products are selling the best this quarter?",
                expected_tables=["products", "order_items", "orders"],
                expected_insights=["top_products", "sales_trends"],
                complexity="medium"
            ),
            BusinessScenario(
                name="Financial KPI Dashboard",
                domain="financial",
                description="Calculate key financial metrics",
                natural_query="What is our total revenue, average order value, and profit margin?",
                expected_tables=["orders", "order_items", "products"],
                expected_insights=["revenue_metrics", "profitability"],
                complexity="complex"
            )
        ]
        
        # Performance thresholds
        self.performance_thresholds = {
            "health_check": 1.0,      # Health checks should be < 1s
            "sql_execution": 5.0,     # SQL queries should be < 5s
            "schema_discovery": 10.0, # Schema operations < 10s
            "llm_processing": 30.0,   # LLM operations < 30s
            "analysis": 15.0,         # Data analysis < 15s
            "visualization": 8.0      # Visualization < 8s
        }
    
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
        headers: Optional[Dict] = None,
        test_category: str = "general",
        business_domain: Optional[str] = None,
        expected_response_time: float = 30.0
    ) -> TestResult:
        """Enhanced request with business context and validation"""
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
            
            # Parse response
            try:
                response_data = response.json()
            except:
                response_data = {"raw_content": response.text}
            
            success = 200 <= response.status_code < 300
            
            # Validate response structure for specific endpoints
            validation_passed = self._validate_response_structure(endpoint, response_data)
            
            # Assess performance
            performance_rating = self._assess_performance(test_category, response_time)
            
            # Calculate business accuracy if applicable
            business_accuracy = self._assess_business_accuracy(
                endpoint, response_data, business_domain
            )
            
            result = TestResult(
                endpoint=endpoint,
                method=method.upper(),
                test_category=test_category,
                business_domain=business_domain,
                status_code=response.status_code,
                response_time=response_time,
                success=success,
                response_data=response_data,
                validation_passed=validation_passed,
                performance_rating=performance_rating,
                business_accuracy=business_accuracy,
                error_message=None if success else f"HTTP {response.status_code}"
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            result = TestResult(
                endpoint=endpoint,
                method=method.upper(),
                test_category=test_category,
                business_domain=business_domain,
                status_code=0,
                response_time=response_time,
                success=False,
                validation_passed=False,
                performance_rating="poor",
                error_message=str(e)
            )
        
        self.results.append(result)
        return result
    
    def _validate_response_structure(self, endpoint: str, response_data: Any) -> bool:
        """Validate response structure matches expected schema"""
        # Handle list responses (like databases endpoint)
        if isinstance(response_data, list):
            return True  # Lists are valid responses
        
        if not isinstance(response_data, dict):
            return False
        
        # Define expected structures for different endpoints
        expected_structures = {
            "/health": ["status"],
            "/api/v1/info": ["version"],
            "/api/v1/sql/execute": ["data"],
            "/api/v1/sql/validate": ["is_valid"],
            "/api/v1/schema/databases": [],  # Can be list or dict
            "/api/v1/schema/tables": [],     # Can vary
            "/api/v1/query/process": ["sql_result", "intent"],
        }
        
        # Check if endpoint has expected structure
        for pattern, required_fields in expected_structures.items():
            if pattern in endpoint:
                return all(field in response_data for field in required_fields)
        
        return True  # Default to valid if no specific structure expected
    
    def _assess_performance(self, test_category: str, response_time: float) -> str:
        """Assess performance rating based on category and time"""
        threshold = self.performance_thresholds.get(test_category, 30.0)
        
        if response_time <= threshold * 0.5:
            return "excellent"
        elif response_time <= threshold:
            return "good"
        elif response_time <= threshold * 2:
            return "average"
        else:
            return "poor"
    
    def _assess_business_accuracy(
        self, 
        endpoint: str, 
        response_data: Dict, 
        business_domain: Optional[str]
    ) -> Optional[float]:
        """Assess business accuracy of the response"""
        if not business_domain or not isinstance(response_data, dict):
            return None
        
        try:
            accuracy_score = 0.8  # Base score
            
            # Check for business domain awareness
            if "business_context" in response_data:
                business_context = response_data["business_context"]
                if business_context == business_domain:
                    accuracy_score += 0.1
            
            # Check for domain-specific insights
            if "insights" in response_data:
                insights = response_data["insights"]
                if insights:  # Check if insights is not None/empty
                    domain_keywords = {
                        "financial": ["revenue", "profit", "cost", "margin"],
                        "customer_management": ["customer", "retention", "churn"],
                        "product_catalog": ["product", "inventory", "category"],
                        "hr_management": ["employee", "performance", "department"]
                    }
                    
                    keywords = domain_keywords.get(business_domain, [])
                    if keywords:
                        insights_text = str(insights).lower()
                        matches = sum(1 for keyword in keywords if keyword in insights_text)
                        accuracy_score += (matches / len(keywords)) * 0.1
            
            return min(1.0, accuracy_score)
        
        except Exception as e:
            # If any error occurs in accuracy assessment, return None
            return None
    
    async def test_infrastructure_health(self):
        """Test infrastructure and health endpoints"""
        console.print("\n[bold green]🏥 Testing Infrastructure Health[/bold green]")
        
        health_endpoints = [
            ("GET", "/health", "Health check"),
            ("GET", "/api/v1/info", "API information"),
            ("GET", "/api/v1/status", "System status"),
            ("POST", "/api/v1/ping", "Ping test"),
        ]
        
        for method, endpoint, description in health_endpoints:
            result = await self.make_request(
                method, endpoint, 
                test_category="health_check",
                business_domain=None
            )
            
            status = "✅" if result.success else "❌"
            perf = self._get_performance_emoji(result.performance_rating)
            
            console.print(f"  {status} {perf} {description} - {result.status_code} ({result.response_time:.3f}s)")
            
            # Show key health metrics
            if result.success and result.response_data:
                if "version" in result.response_data:
                    console.print(f"    Version: {result.response_data['version']}")
                if "uptime" in result.response_data:
                    console.print(f"    Uptime: {result.response_data['uptime']}")
                if "services" in result.response_data:
                    services = result.response_data["services"]
                    if isinstance(services, dict):
                        healthy_services = sum(1 for s in services.values() if s == "healthy")
                        console.print(f"    Services: {healthy_services}/{len(services)} healthy")
    
    async def test_enhanced_sql_capabilities(self):
        """Test SQL capabilities with business scenarios"""
        console.print("\n[bold green]🗄️ Testing Enhanced SQL Capabilities[/bold green]")
        
        # Test SQL execution with business context
        business_sql_tests = [
            {
                "name": "Customer Analytics",
                "sql": "SELECT country, COUNT(*) as customers, AVG(account_balance) as avg_balance FROM customers WHERE is_premium = true GROUP BY country ORDER BY customers DESC;",
                "database": "sql_agent_db",
                "business_context": "customer_management"
            },
            {
                "name": "Revenue Analysis", 
                "sql": "SELECT DATE_TRUNC('month', order_date) as month, SUM(total_amount) as revenue FROM orders WHERE order_status = 'completed' GROUP BY month ORDER BY month;",
                "database": "sql_agent_db",
                "business_context": "financial"
            },
            {
                "name": "Product Performance",
                "sql": "SELECT p.category, COUNT(oi.item_id) as items_sold, SUM(oi.line_total) as revenue FROM products p JOIN order_items oi ON p.product_id = oi.product_id GROUP BY p.category ORDER BY revenue DESC;",
                "database": "sql_agent_db", 
                "business_context": "product_catalog"
            }
        ]
        
        console.print("\n  Testing SQL Execution with Business Context:")
        for test in business_sql_tests:
            result = await self.make_request(
                "POST", "/api/v1/sql/execute",
                data=test,
                test_category="sql_execution",
                business_domain=test["business_context"]
            )
            
            status = "✅" if result.success else "❌"
            perf = self._get_performance_emoji(result.performance_rating)
            validation = "✓" if result.validation_passed else "✗"
            
            console.print(f"    {status} {perf} {validation} {test['name']} - {result.status_code} ({result.response_time:.3f}s)")
            
            if result.success and result.response_data:
                # Analyze result quality
                data = result.response_data.get("data", [])
                console.print(f"      → {len(data)} rows returned")
                
                # Check for business insights
                insights = result.response_data.get("insights")
                if insights and isinstance(insights, list):
                    insights_count = len(insights)
                    console.print(f"      → {insights_count} business insights generated")
                
                # Check for optimization suggestions
                optimization_suggestions = result.response_data.get("optimization_suggestions")
                if optimization_suggestions and isinstance(optimization_suggestions, list):
                    suggestions = len(optimization_suggestions)
                    console.print(f"      → {suggestions} optimization suggestions")
        
        # Test SQL validation
        console.print("\n  Testing SQL Validation:")
        validation_tests = [
            {
                "sql": "SELECT * FROM customers WHERE country = 'USA';",
                "database": "sql_agent_db"
            },
            {
                "sql": "SELECT INVALID SYNTAX;",
                "database": "sql_agent_db"
            }
        ]
        
        for test in validation_tests:
            result = await self.make_request(
                "POST", "/api/v1/sql/validate",
                data=test,
                test_category="sql_validation"
            )
            
            status = "✅" if result.success else "❌"
            console.print(f"    {status} SQL Validation - {result.status_code} ({result.response_time:.3f}s)")
            
            if result.success and result.response_data:
                is_valid = result.response_data.get("is_valid", False)
                console.print(f"      → Valid: {is_valid}")
    
    async def test_schema_intelligence(self):
        """Test schema discovery and intelligence features"""
        console.print("\n[bold green]🗂️ Testing Schema Intelligence[/bold green]")
        
        # Test enhanced schema endpoints
        schema_tests = [
            ("GET", "/api/v1/schema/databases", "Database listing"),
            ("GET", "/api/v1/schema/tables", "Table discovery"),
            ("GET", "/api/v1/schema/relationships", "Relationship mapping"),
        ]
        
        for method, endpoint, description in schema_tests:
            try:
                result = await self.make_request(
                    method, endpoint,
                    test_category="schema_discovery"
                )
                
                status = "✅" if result.success else "❌"
                perf = self._get_performance_emoji(result.performance_rating)
                
                console.print(f"  {status} {perf} {description} - {result.status_code} ({result.response_time:.3f}s)")
                
                if result.success and result.response_data:
                    # Handle both list and dict response formats
                    if isinstance(result.response_data, list):
                        # Direct list response (like databases endpoint)
                        data_list = result.response_data
                        console.print(f"    → {len(data_list)} items found")
                        
                        # Show first item details if available
                        if data_list and isinstance(data_list[0], dict):
                            first_item = data_list[0]
                            if "name" in first_item:
                                console.print(f"    → First item: {first_item['name']}")
                            if "type" in first_item:
                                console.print(f"    → Type: {first_item['type']}")
                    
                    elif isinstance(result.response_data, dict):
                        # Dictionary response with nested data
                        tables = result.response_data.get("tables")
                        if tables and isinstance(tables, list):
                            console.print(f"    → {len(tables)} tables discovered")
                        
                        relationships = result.response_data.get("relationships")
                        if relationships and isinstance(relationships, list):
                            console.print(f"    → {len(relationships)} relationships found")
                        
                        # Check for business domains
                        business_domains = result.response_data.get("business_domains")
                        if business_domains and isinstance(business_domains, list):
                            console.print(f"    → {len(business_domains)} business domains identified")
                    
                    else:
                        console.print(f"    → Unexpected response format: {type(result.response_data)}")
                
                elif result.success:
                    console.print(f"    → Empty response")
                else:
                    error_msg = result.error_message or "Unknown error"
                    console.print(f"    → Error: {error_msg}")
                    
            except Exception as e:
                console.print(f"  ❌ {description} - Exception: {str(e)[:50]}...")
                # Add a failed result manually for reporting
                failed_result = TestResult(
                    endpoint=endpoint,
                    method=method,
                    test_category="schema_discovery",
                    business_domain=None,
                    status_code=0,
                    response_time=0,
                    success=False,
                    validation_passed=False,
                    performance_rating="poor",
                    error_message=str(e)
                )
                self.results.append(failed_result)
    
    async def test_natural_language_processing(self):
        """Test natural language query processing"""
        console.print("\n[bold green]🧠 Testing Natural Language Processing[/bold green]")
        
        # Test business scenarios
        for scenario in self.business_scenarios:
            console.print(f"\n  Testing: {scenario.name} ({scenario.domain})")
            
            query_data = {
                "query": scenario.natural_query,
                "database_name": "sql_agent_db",
                "include_analysis": True,
                "business_context": scenario.domain
            }
            
            result = await self.make_request(
                "POST", "/api/v1/query/process",
                data=query_data,
                test_category="llm_processing",
                business_domain=scenario.domain
            )
            
            status = "✅" if result.success else "❌"
            perf = self._get_performance_emoji(result.performance_rating)
            accuracy = self._get_accuracy_emoji(result.business_accuracy)
            
            console.print(f"    {status} {perf} {accuracy} Processing - {result.status_code} ({result.response_time:.3f}s)")
            
            if result.success and result.response_data:
                # Analyze NLP quality
                if "intent" in result.response_data:
                    intent = result.response_data["intent"]
                    confidence = result.response_data.get("confidence", 0)
                    console.print(f"      → Intent: {intent} (confidence: {confidence:.2f})")
                
                if "sql_result" in result.response_data:
                    sql_data = result.response_data["sql_result"]
                    if sql_data and isinstance(sql_data, dict) and "data" in sql_data:
                        rows = len(sql_data["data"])
                        console.print(f"      → Generated SQL returned {rows} rows")
                
                # Check table selection accuracy
                if "selected_tables" in result.response_data:
                    selected = result.response_data["selected_tables"]
                    if selected and isinstance(selected, list):
                        selected_set = set(selected)
                        expected_set = set(scenario.expected_tables)
                        
                        correct_tables = len(selected_set & expected_set)
                        total_expected = len(expected_set)
                        table_accuracy = correct_tables / total_expected if total_expected > 0 else 0
                        
                        console.print(f"      → Table selection: {correct_tables}/{total_expected} correct ({table_accuracy:.1%})")
                        console.print(f"      → Selected: {list(selected_set)}")
                        console.print(f"      → Expected: {list(expected_set)}")
                
                # Check business insights
                if "analysis_result" in result.response_data:
                    analysis = result.response_data["analysis_result"]
                    if analysis and isinstance(analysis, dict):
                        if "insights" in analysis:
                            insights = analysis["insights"]
                            if insights and isinstance(insights, list):
                                insights_count = len(insights)
                                console.print(f"      → {insights_count} business insights generated")
                        
                        if "domain_insights" in analysis:
                            domain_insights = analysis["domain_insights"]
                            if domain_insights and isinstance(domain_insights, list):
                                domain_insights_count = len(domain_insights)
                                console.print(f"      → {domain_insights_count} domain-specific insights")
            else:
                error_msg = result.error_message or "Unknown error"
                console.print(f"      ❌ Query processing failed: {error_msg}")
    
    async def test_performance_load(self):
        """Test system performance under load"""
        console.print("\n[bold blue]⚡ Testing Performance Under Load[/bold blue]")
        
        # Concurrent health checks
        console.print("\n  Testing Concurrent Health Checks (20 requests):")
        start_time = time.time()
        
        async def health_check():
            return await self.make_request("GET", "/health", test_category="health_check")
        
        health_tasks = [health_check() for _ in range(20)]
        health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
        
        health_time = time.time() - start_time
        health_successful = sum(1 for r in health_results if hasattr(r, 'success') and r.success)
        health_rps = len(health_results) / health_time if health_time > 0 else 0
        
        console.print(f"    ✅ {health_successful}/20 successful")
        console.print(f"    ⚡ {health_rps:.1f} requests/second")
        console.print(f"    ⏱️ Total time: {health_time:.2f}s")
        
        # Concurrent SQL queries
        console.print("\n  Testing Concurrent SQL Queries (5 requests):")
        
        async def sql_query():
            data = {
                "sql": "SELECT country, COUNT(*) FROM customers GROUP BY country;",
                "database": "sql_agent_db"
            }
            return await self.make_request("POST", "/api/v1/sql/execute", data=data, test_category="sql_execution")
        
        start_time = time.time()
        sql_tasks = [sql_query() for _ in range(5)]
        sql_results = await asyncio.gather(*sql_tasks, return_exceptions=True)
        
        sql_time = time.time() - start_time
        sql_successful = sum(1 for r in sql_results if hasattr(r, 'success') and r.success)
        sql_rps = len(sql_results) / sql_time if sql_time > 0 else 0
        
        console.print(f"    ✅ {sql_successful}/5 successful")
        console.print(f"    ⚡ {sql_rps:.1f} requests/second")
        console.print(f"    ⏱️ Total time: {sql_time:.2f}s")
    
    async def test_error_handling_resilience(self):
        """Test error handling and system resilience"""
        console.print("\n[bold red]🛡️ Testing Error Handling & Resilience[/bold red]")
        
        # Test malformed requests
        error_tests = [
            {
                "name": "Empty SQL query",
                "endpoint": "/api/v1/sql/execute",
                "data": {"sql": "", "database": "sql_agent_db"},
                "expected_status": 400
            },
            {
                "name": "Invalid SQL syntax",
                "endpoint": "/api/v1/sql/execute", 
                "data": {"sql": "INVALID SQL SYNTAX HERE", "database": "sql_agent_db"},
                "expected_status": 400
            },
            {
                "name": "Non-existent table",
                "endpoint": "/api/v1/sql/execute",
                "data": {"sql": "SELECT * FROM non_existent_table;", "database": "sql_agent_db"},
                "expected_status": 400
            }
        ]
        
        for test in error_tests:
            try:
                result = await self.make_request(
                    "POST", test["endpoint"],
                    data=test["data"],
                    test_category="error_handling"
                )
                status_code = result.status_code
                response_time = result.response_time
                
                # Check if error was handled correctly
                expected_error = test["expected_status"] <= status_code < 500
                status = "✅" if expected_error else "❌"
                
                console.print(f"  {status} {test['name']} - {status_code} ({response_time:.3f}s)")
                
            except Exception as e:
                console.print(f"  ❌ {test['name']} - Failed: {str(e)[:50]}...")
    
    def _get_performance_emoji(self, rating: str) -> str:
        """Get emoji for performance rating"""
        emoji_map = {
            "excellent": "🚀",
            "good": "✅", 
            "average": "🟡",
            "poor": "🐌"
        }
        return emoji_map.get(rating, "❓")
    
    def _get_accuracy_emoji(self, accuracy: Optional[float]) -> str:
        """Get emoji for business accuracy"""
        if accuracy is None:
            return "❓"
        elif accuracy >= 0.9:
            return "🎯"
        elif accuracy >= 0.7:
            return "👍"
        elif accuracy >= 0.5:
            return "👌"
        else:
            return "👎"
    
    def calculate_performance_metrics(self, results: List[TestResult]) -> PerformanceMetrics:
        """Calculate performance metrics from results"""
        if not results:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0)
        
        response_times = [r.response_time for r in results]
        successful_results = [r for r in results if r.success]
        
        return PerformanceMetrics(
            min_time=min(response_times),
            max_time=max(response_times),
            avg_time=statistics.mean(response_times),
            median_time=statistics.median(response_times),
            p95_time=statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times),
            requests_per_second=len(results) / (time.time() - self.start_time),
            success_rate=len(successful_results) / len(results)
        )
    
    def generate_comprehensive_report(self):
        """Generate comprehensive test report with business intelligence"""
        console.print("\n" + "="*80)
        console.print(Panel.fit("📊 Comprehensive Test Report - Phase 3", style="bold green"))
        
        # Calculate metrics
        total_time = time.time() - self.start_time
        performance_metrics = self.calculate_performance_metrics(self.results)
        
        # Overall summary
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        
        summary_table = Table(title="Overall Test Summary", show_header=True)
        summary_table.add_column("Metric", style="cyan", width=20)
        summary_table.add_column("Value", style="green", width=15)
        summary_table.add_column("Assessment", style="yellow", width=20)
        
        summary_table.add_row("Total Tests", str(total_tests), "")
        summary_table.add_row("Successful", str(successful_tests), "✅ Good" if successful_tests/total_tests > 0.8 else "⚠️ Needs work")
        summary_table.add_row("Failed", str(failed_tests), "✅ Good" if failed_tests < total_tests*0.2 else "❌ Too many")
        summary_table.add_row("Success Rate", f"{(successful_tests/total_tests)*100:.1f}%", "✅ Excellent" if successful_tests/total_tests > 0.9 else "🟡 Good")
        summary_table.add_row("Total Duration", f"{total_time:.2f}s", "")
        summary_table.add_row("Avg Response Time", f"{performance_metrics.avg_time:.3f}s", "✅ Fast" if performance_metrics.avg_time < 2 else "🟡 Acceptable")
        
        console.print(summary_table)
        
        # Recommendations
        console.print("\n")
        recommendations_panel = Panel(
            "[bold yellow]🎯 Priority Recommendations for Production:[/bold yellow]\n\n"
            "1. [bold red]HIGH PRIORITY:[/bold red]\n"
            "   • Implement missing data analysis endpoints (/api/v1/analysis/*)\n"
            "   • Add comprehensive schema detail endpoints\n"
            "   • Implement business domain classification\n\n"
            "2. [bold orange1]MEDIUM PRIORITY:[/bold orange1]\n"
            "   • Complete visualization pipeline\n"
            "   • Add performance monitoring endpoints\n"
            "   • Enhance error reporting with business context\n\n"
            "3. [bold green]ENHANCEMENT:[/bold green]\n"
            "   • Add query optimization suggestions\n"
            "   • Implement advanced analytics features\n"
            "   • Add real-time monitoring dashboard",
            title="Next Steps",
            border_style="bright_blue"
        )
        console.print(recommendations_panel)
        
        # Save results
        self.save_results(performance_metrics)
    
    def save_results(self, performance_metrics: PerformanceMetrics):
        """Save test results"""
        results_data = {
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "test_type": "comprehensive_phase3_validation",
                "version": "3.0",
                "database": "sql_agent_db",
                "total_duration": time.time() - self.start_time
            },
            "summary": {
                "total_tests": len(self.results),
                "successful_tests": sum(1 for r in self.results if r.success),
                "failed_tests": sum(1 for r in self.results if not r.success),
                "success_rate": sum(1 for r in self.results if r.success) / len(self.results) if self.results else 0,
            },
            "performance_metrics": asdict(performance_metrics),
            "detailed_results": [asdict(result) for result in self.results]
        }
        
        # Save to file
        filename = f"sql_agent_comprehensive_test_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        console.print(f"\n📄 Test results saved to: [bold cyan]{filename}[/bold cyan]")

async def main():
    """Main test execution with comprehensive validation"""
    console.print(Panel.fit("Enhanced SQL Agent API Test Suite", style="bold green"))
    console.print(f"🎯 Testing API at: [bold cyan]{BASE_URL}[/bold cyan]")
    console.print(f"📊 Focus: Phase 3 production validation with business intelligence")
    console.print("=" * 80)
    
    async with EnhancedAPITester(BASE_URL) as tester:
        try:
            # Run comprehensive test suite
            await tester.test_infrastructure_health()
            await tester.test_enhanced_sql_capabilities()
            # Test schema intelligence with extra safety
            try:
                await tester.test_schema_intelligence()
            except Exception as e:
                console.print(f"\n[yellow]⚠️ Schema intelligence tests failed: {str(e)[:100]}...[/yellow]")
                console.print("[yellow]Continuing with other tests...[/yellow]")
            
            # Test natural language processing with extra safety
            try:
                await tester.test_natural_language_processing()
            except Exception as e:
                console.print(f"\n[yellow]⚠️ Natural language processing tests failed: {str(e)[:100]}...[/yellow]")
                console.print("[yellow]Continuing with other tests...[/yellow]")
            
            await tester.test_performance_load()
            await tester.test_error_handling_resilience()
            
            # Generate comprehensive report
            tester.generate_comprehensive_report()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Test suite interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Test suite failed with error: {e}[/red]")
            import traceback
            console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
            
            # Still try to generate a report with whatever results we have
            try:
                tester.generate_comprehensive_report()
            except:
                console.print("[yellow]Could not generate final report[/yellow]")
            
            return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)