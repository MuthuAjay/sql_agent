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
    performance_rating: str = "good" # excellent, good, average, poor
    business_accuracy: Optional[float] = None # 0-1 score
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
    complexity: str # simple, medium, complex


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
            ),
            BusinessScenario(
                name="Employee Performance",
                domain="hr_management",
                description="Analyze employee performance metrics",
                natural_query="Show me employee performance scores by department",
                expected_tables=["employee_performance"],
                expected_insights=["performance_distribution", "department_comparison"],
                complexity="simple"
            ),
            BusinessScenario(
                name="Order Processing Efficiency",
                domain="operations",
                description="Analyze order processing and fulfillment",
                natural_query="How long does it take to process orders by payment method?",
                expected_tables=["orders"],
                expected_insights=["processing_time", "payment_efficiency"],
                complexity="medium"
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

    def _validate_response_structure(self, endpoint: str, response_data: Dict) -> bool:
        """Validate response structure matches expected schema"""
        if not isinstance(response_data, dict):
            return False

        # Define expected structures for different endpoints
        expected_structures = {
            "/health": ["status"],
            "/api/v1/info": ["version"],
            "/api/v1/sql/execute": ["data", "columns"],
            "/api/v1/sql/validate": ["is_valid"],
            "/api/v1/schema/tables": ["tables"],
            "/api/v1/query/process": ["sql_result", "intent"],
            "/api/v1/analysis/profile": ["summary", "insights"],
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

        accuracy_score = 0.8  # Base score

        # Check for business domain awareness
        if "business_context" in response_data:
            if response_data["business_context"] == business_domain:
                accuracy_score += 0.1

        # Check for domain-specific insights
        if "insights" in response_data:
            domain_keywords = {
                "financial": ["revenue", "profit", "cost", "margin"],
                "customer_management": ["customer", "retention", "churn"],
                "product_catalog": ["product", "inventory", "category"],
                "hr_management": ["employee", "performance", "department"]
            }

            keywords = domain_keywords.get(business_domain, [])
            if keywords:
                insights_text = str(response_data["insights"]).lower()
                matches = sum(1 for keyword in keywords if keyword in insights_text)
                accuracy_score += (matches / len(keywords)) * 0.1

        return min(1.0, accuracy_score)

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
                "business_context": "customer_management",
                "expected_insights": ["geographic_distribution", "premium_customer_value"]
            },
            {
                "name": "Revenue Analysis",
                "sql": "SELECT DATE_TRUNC('month', order_date) as month, SUM(total_amount) as revenue FROM orders WHERE order_status = 'completed' GROUP BY month ORDER BY month;",
                "database": "sql_agent_db",
                "business_context": "financial",
                "expected_insights": ["revenue_trends", "seasonality"]
            },
            {
                "name": "Product Performance",
                "sql": "SELECT p.category, COUNT(oi.item_id) as items_sold, SUM(oi.line_total) as revenue FROM products p JOIN order_items oi ON p.product_id = oi.product_id GROUP BY p.category ORDER BY revenue DESC;",
                "database": "sql_agent_db",
                "business_context": "product_catalog",
                "expected_insights": ["category_performance", "sales_volume"]
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
                if "insights" in result.response_data:
                    insights_count = len(result.response_data["insights"])
                    console.print(f"      → {insights_count} business insights generated")

                # Check for optimization suggestions
                if "optimization_suggestions" in result.response_data:
                    suggestions = len(result.response_data["optimization_suggestions"])
                    console.print(f"      → {suggestions} optimization suggestions")

        # Test SQL validation and explanation
        console.print("\n  Testing SQL Validation & Explanation:")
        validation_tests = [
            {
                "sql": "SELECT * FROM customers WHERE country = 'USA';",
                "database": "sql_agent_db",
                "include_explanation": True,
                "include_optimization": True
            },
            {
                "sql": "SELECT INVALID SYNTAX;",
                "database": "sql_agent_db",
                "include_explanation": False
            }
        ]

        for test in validation_tests:
            # Test validation
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

                if "optimization_suggestions" in result.response_data:
                    suggestions = len(result.response_data["optimization_suggestions"])
                    console.print(f"      → {suggestions} optimization suggestions")

    async def test_schema_intelligence(self):
        """Test schema discovery and intelligence features"""
        console.print("\n[bold green]🗂️ Testing Schema Intelligence[/bold green]")

        # Test enhanced schema endpoints
        schema_tests = [
            ("GET", "/api/v1/schema/databases", "Database listing"),
            ("GET", "/api/v1/schema/tables", "Table discovery"),
            ("GET", "/api/v1/schema/tables/customers", "Table details"),
            ("GET", "/api/v1/schema/relationships", "Relationship mapping"),
            ("GET", "/api/v1/schema/business-domains", "Business domain classification"),
        ]

        for method, endpoint, description in schema_tests:
            result = await self.make_request(
                method, endpoint,
                test_category="schema_discovery"
            )

            status = "✅" if result.success else "❌"
            perf = self._get_performance_emoji(result.performance_rating)

            console.print(f"  {status} {perf} {description} - {result.status_code} ({result.response_time:.3f}s)")

            if result.success and result.response_data:
                # Analyze schema intelligence
                if "business_domains" in result.response_data:
                    domains = len(result.response_data["business_domains"])
                    console.print(f"    → {domains} business domains identified")

                if "tables" in result.response_data:
                    tables = result.response_data["tables"]
                    if isinstance(tables, list):
                        console.print(f"    → {len(tables)} tables discovered")

                        # Check for enhanced metadata
                        if tables and isinstance(tables, dict):
                            first_table = tables
                            metadata_fields = ["business_domains", "performance_score", "data_quality_score"]
                            enhanced_count = sum(1 for field in metadata_fields if field in first_table)
                            console.print(f"    → {enhanced_count}/{len(metadata_fields)} enhanced metadata fields")

                if "relationships" in result.response_data:
                    relationships = result.response_data["relationships"]
                    if isinstance(relationships, list):
                        console.print(f"    → {len(relationships)} relationships found")

                        # Check for inferred relationships
                        inferred = sum(1 for r in relationships if r.get("type") == "inferred")
                        console.print(f"    → {inferred} inferred relationships")

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
                "include_optimization": True,
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
                    if "data" in sql_data:
                        rows = len(sql_data["data"])
                        console.print(f"      → Generated SQL returned {rows} rows")

                # Check table selection accuracy
                if "selected_tables" in result.response_data:
                    selected = set(result.response_data["selected_tables"])
                    expected = set(scenario.expected_tables)

                    correct_tables = len(selected & expected)
                    total_expected = len(expected)
                    table_accuracy = correct_tables / total_expected if total_expected > 0 else 0

                    console.print(f"      → Table selection: {correct_tables}/{total_expected} correct ({table_accuracy:.1%})")
                    console.print(f"      → Selected: {list(selected)}")
                    console.print(f"      → Expected: {list(expected)}")

                # Check business insights
                if "analysis_result" in result.response_data:
                    analysis = result.response_data["analysis_result"]
                    if "insights" in analysis:
                        insights_count = len(analysis["insights"])
                        console.print(f"      → {insights_count} business insights generated")

                    if "domain_insights" in analysis:
                        domain_insights = len(analysis["domain_insights"])
                        console.print(f"      → {domain_insights} domain-specific insights")

    async def test_advanced_analytics(self):
        """Test advanced analytics and data profiling"""
        console.print("\n[bold green]📊 Testing Advanced Analytics[/bold green]")

        # Test data profiling
        profiling_tests = [
            {
                "table_name": "customers",
                "columns": ["account_balance", "country", "is_premium"],
                "business_context": "customer_management"
            },
            {
                "table_name": "orders",
                "columns": ["total_amount", "order_status", "payment_method"],
                "business_context": "financial"
            }
        ]

        console.print("\n  Testing Data Profiling:")
        for test in profiling_tests:
            result = await self.make_request(
                "POST", "/api/v1/analysis/profile",
                data=test,
                test_category="analysis",
                business_domain=test["business_context"]
            )

            status = "✅" if result.success else "❌"
            perf = self._get_performance_emoji(result.performance_rating)

            console.print(f"    {status} {perf} {test['table_name']} profiling - {result.status_code} ({result.response_time:.3f}s)")

            if result.success and result.response_data:
                # Analyze profiling quality
                if "summary" in result.response_data:
                    summary = result.response_data["summary"]
                    if "data_quality_score" in summary:
                        quality = summary["data_quality_score"]
                        console.print(f"      → Data quality score: {quality:.2f}")

                if "insights" in result.response_data:
                    insights = len(result.response_data["insights"])
                    console.print(f"      → {insights} generated")

                if "anomalies" in result.response_data:
                    anomalies = len(result.response_data["anomalies"])
                    console.print(f"      → {anomalies} anomalies detected")

        # Test trend analysis
        console.print("\n  Testing Trend Analysis:")
        trend_test = {
            "sql": "SELECT DATE_TRUNC('month', order_date) as month, SUM(total_amount) as revenue FROM orders GROUP BY month ORDER BY month;",
            "database": "sql_agent_db",
            "analysis_type": "trend",
            "include_forecasting": True
        }

        result = await self.make_request(
            "POST", "/api/v1/analysis/trends",
            data=trend_test,
            test_category="analysis",
            business_domain="financial"
        )

        status = "✅" if result.success else "❌"
        perf = self._get_performance_emoji(result.performance_rating)

        console.print(f"    {status} {perf} Revenue trend analysis - {result.status_code} ({result.response_time:.3f}s)")

        if result.success and result.response_data:
            if "trends" in result.response_data:
                trends = len(result.response_data["trends"])
                console.print(f"      → {trends} trends identified")

            if "forecast" in result.response_data:
                console.print(f"      → Forecasting enabled")

    async def test_visualization_intelligence(self):
        """Test intelligent visualization capabilities"""
        console.print("\n[bold green]📈 Testing Visualization Intelligence[/bold green]")

        # Test visualization suggestions with business context
        viz_tests = [
            {
                "name": "Revenue by Country",
                "data": [
                    {"country": "USA", "revenue": 150000.50, "customers": 245},
                    {"country": "Canada", "revenue": 85000.25, "customers": 156},
                    {"country": "UK", "revenue": 123000.75, "customers": 189}
                ],
                "business_context": "financial",
                "include_insights": True
            },
            {
                "name": "Product Category Performance",
                "data": [
                    {"category": "Electronics", "sales": 450, "revenue": 89000},
                    {"category": "Clothing", "sales": 320, "revenue": 45000},
                    {"category": "Home & Kitchen", "sales": 280, "revenue": 38000}
                ],
                "business_context": "product_catalog",
                "include_insights": True
            }
        ]

        for test in viz_tests:
            # Test visualization suggestion
            result = await self.make_request(
                "POST", "/api/v1/visualization/suggest",
                data=test,
                test_category="visualization",
                business_domain=test["business_context"]
            )

            status = "✅" if result.success else "❌"
            perf = self._get_performance_emoji(result.performance_rating)

            console.print(f"  {status} {perf} {test['name']} suggestion - {result.status_code} ({result.response_time:.3f}s)")

            if result.success and result.response_data:
                if "chart_type" in result.response_data:
                    chart_type = result.response_data["chart_type"]
                    console.print(f"    → Suggested: {chart_type}")

                if "alternative_charts" in result.response_data:
                    alternatives = len(result.response_data["alternative_charts"])
                    console.print(f"    → {alternatives} alternative suggestions")

                if "business_insights" in result.response_data:
                    insights = len(result.response_data["business_insights"])
                    console.print(f"    → {insights} business insights")

            # Test chart generation
            if result.success:
                generate_data = {
                    **test,
                    "chart_type": result.response_data.get("chart_type", "bar")
                }

                gen_result = await self.make_request(
                    "POST", "/api/v1/visualization/generate",
                    data=generate_data,
                    test_category="visualization"
                )

                gen_status = "✅" if gen_result.success else "❌"
                console.print(f"    {gen_status} Chart generation - {gen_result.status_code} ({gen_result.response_time:.3f}s)")

    async def test_performance_load(self):
        """Test system performance under load"""
        console.print("\n[bold blue]⚡ Testing Performance Under Load[/bold blue]")

        # Concurrent health checks
        console.print("\n  Testing Concurrent Health Checks (50 requests):")
        start_time = time.time()

        async def health_check():
            return await self.make_request("GET", "/health", test_category="health_check")

        health_tasks = [health_check() for _ in range(50)]
        health_results = await asyncio.gather(*health_tasks, return_exceptions=True)

        health_time = time.time() - start_time
        health_successful = sum(1 for r in health_results if hasattr(r, 'success') and r.success)
        health_rps = len(health_results) / health_time

        console.print(f"    ✅ {health_successful}/50 successful")
        console.print(f"    ⚡ {health_rps:.1f} requests/second")
        console.print(f"    ⏱️ Total time: {health_time:.2f}s")

        # Concurrent SQL queries
        console.print("\n  Testing Concurrent SQL Queries (10 requests):")

        async def sql_query():
            data = {
                "sql": "SELECT country, COUNT(*) FROM customers GROUP BY country;",
                "database": "sql_agent_db"
            }
            return await self.make_request("POST", "/api/v1/sql/execute", data=data, test_category="sql_execution")

        start_time = time.time()
        sql_tasks = [sql_query() for _ in range(10)]
        sql_results = await asyncio.gather(*sql_tasks, return_exceptions=True)

        sql_time = time.time() - start_time
        sql_successful = sum(1 for r in sql_results if hasattr(r, 'success') and r.success)
        sql_rps = len(sql_results) / sql_time

        console.print(f"    ✅ {sql_successful}/10 successful")
        console.print(f"    ⚡ {sql_rps:.1f} requests/second")
        console.print(f"    ⏱️ Total time: {sql_time:.2f}s")

        # Memory stress test (if available)
        console.print("\n  Testing Large Data Processing:")
        large_data_test = {
            "sql": "SELECT * FROM customers;",  # Get all customer data
            "database": "sql_agent_db",
            "max_results": 10000,
            "include_analysis": True
        }

        result = await self.make_request(
            "POST", "/api/v1/sql/execute",
            data=large_data_test,
            test_category="sql_execution"
        )

        status = "✅" if result.success else "❌"
        perf = self._get_performance_emoji(result.performance_rating)

        console.print(f"    {status} {perf} Large data processing - {result.status_code} ({result.response_time:.3f}s)")
        if result.success and result.response_data:
            rows = len(result.response_data.get("data", []))
            console.print(f"      → Processed {rows} rows")

    def _get_performance_emoji(self, rating: str) -> str:
        """Get an emoji for a performance rating"""
        return {"excellent": "🚀", "good": "✅", "average": "🤔", "poor": "🐌"}.get(rating, "❓")

    def _get_accuracy_emoji(self, accuracy: Optional[float]) -> str:
        """Get an emoji for a business accuracy score"""
        if accuracy is None:
            return ""
        if accuracy >= 0.9:
            return "🎯"
        if accuracy >= 0.7:
            return "👍"
        return "🤷"

    async def run_tests(self):
        """Run all test suites"""
        console.print(Panel("[bold yellow]Enhanced SQL Agent API Test Suite[/bold yellow]", expand=False))
        await self.test_infrastructure_health()
        await self.test_enhanced_sql_capabilities()
        await self.test_schema_intelligence()
        await self.test_natural_language_processing()
        await self.test_advanced_analytics()
        await self.test_visualization_intelligence()
        await self.test_performance_load()
        console.print("\n[bold green]✅ All tests completed.[/bold green]")

    def display_summary(self):
        """Display a summary of the test results"""
        console.print(Panel("[bold yellow]Test Execution Summary[/bold yellow]", expand=False))

        # Overall stats
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        total_time = time.time() - self.start_time

        summary_text = (
            f"Total Tests: {total_tests}\n"
            f"Successful: [green]{successful_tests}[/green]\n"
            f"Failed: [red]{failed_tests}[/red]\n"
            f"Success Rate: {success_rate:.2f}%\n"
            f"Total Duration: {total_time:.2f}s"
        )
        console.print(Panel(summary_text, title="Overall Results", expand=False))

        # Detailed results table
        table = Table(title="Detailed Test Results")
        table.add_column("Category", style="cyan")
        table.add_column("Endpoint", style="magenta")
        table.add_column("Method", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Time (s)", justify="right", style="blue")
        table.add_column("Perf", justify="center")
        table.add_column("Validation", justify="center")
        table.add_column("Accuracy", justify="center")
        table.add_column("Error", style="red")

        for result in self.results:
            status_color = "green" if result.success else "red"
            table.add_row(
                result.test_category,
                result.endpoint,
                result.method,
                f"[{status_color}]{result.status_code}[/{status_color}]",
                f"{result.response_time:.3f}",
                self._get_performance_emoji(result.performance_rating),
                "✅" if result.validation_passed else "❌",
                self._get_accuracy_emoji(result.business_accuracy),
                result.error_message or ""
            )
        console.print(table)
        
        
async def main():
    """Main function to run the API tester"""
    tester = EnhancedAPITester()
    try:
        await tester.run_tests()
    except httpx.ConnectError as e:
        console.print(f"[bold red]Connection Error:[/bold red] Could not connect to {BASE_URL}.")
        console.print("Please ensure the API server is running.")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
    finally:
        tester.display_summary()
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Test suite interrupted by user.[/bold yellow]")
        sys.exit(0)