#!/usr/bin/env python3
"""
SQL Agent Interactive Test Script

This script provides a comprehensive test interface for your SQL Agent system.
You can:
1. Test database connectivity
2. View available databases
3. List tables in a database
4. Explore table schemas
5. Get sample data
6. Ask natural language queries
7. Test the orchestrator functionality

Usage:
    python interactive_test.py
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.json import JSON
from rich.progress import Progress, SpinnerColumn, TextColumn


class SQLAgentTester:
    """Interactive tester for SQL Agent API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.console = Console()
        self.current_database = None
        self.session_id = f"test_{int(time.time())}"
        
    async def main_menu(self):
        """Main interactive menu."""
        self.console.print("\n[bold blue]🤖 SQL Agent Interactive Test Suite[/bold blue]")
        self.console.print(f"[dim]Base URL: {self.base_url}[/dim]")
        self.console.print(f"[dim]Session ID: {self.session_id}[/dim]\n")
        
        while True:
            self.console.print("\n[bold]📋 Main Menu:[/bold]")
            options = {
                "1": "🔍 Test System Health",
                "2": "🗄️  List Databases",
                "3": "📊 Select Database & List Tables", 
                "4": "🔎 Explore Table Schema",
                "5": "📈 Get Sample Data",
                "6": "💬 Ask Natural Language Query",
                "7": "🔧 Test Orchestrator",
                "8": "📋 Run Full Test Suite",
                "9": "🚀 Performance Test",
                "0": "❌ Exit"
            }
            
            for key, desc in options.items():
                self.console.print(f"  {key}. {desc}")
            
            choice = Prompt.ask("\n[bold]Choose an option[/bold]", choices=list(options.keys()))
            
            try:
                if choice == "1":
                    await self.test_system_health()
                elif choice == "2":
                    await self.list_databases()
                elif choice == "3":
                    await self.select_database_and_list_tables()
                elif choice == "4":
                    await self.explore_table_schema()
                elif choice == "5":
                    await self.get_sample_data()
                elif choice == "6":
                    await self.ask_natural_language_query()
                elif choice == "7":
                    await self.test_orchestrator()
                elif choice == "8":
                    await self.run_full_test_suite()
                elif choice == "9":
                    await self.performance_test()
                elif choice == "0":
                    self.console.print("\n[bold green]👋 Goodbye![/bold green]")
                    break
                    
            except KeyboardInterrupt:
                self.console.print("\n[yellow]⚠️  Operation cancelled[/yellow]")
            except Exception as e:
                self.console.print(f"\n[bold red]❌ Error: {str(e)}[/bold red]")
    
    async def test_system_health(self):
        """Test system health and connectivity."""
        self.console.print("\n[bold]🔍 Testing System Health...[/bold]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            # Test main API health
            task1 = progress.add_task("Testing main API health...", total=None)
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.base_url}/health", timeout=10)
                    health_data = response.json()
                
                progress.update(task1, description="✅ Main API health")
                self.display_health_status(health_data)
                
            except Exception as e:
                progress.update(task1, description="❌ Main API health failed")
                self.console.print(f"[red]Main API Error: {e}[/red]")
                return
            
            # Test query service health
            task2 = progress.add_task("Testing query service...", total=None)
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.base_url}/api/v1/query/health", timeout=10)
                    query_health = response.json()
                
                progress.update(task2, description="✅ Query service health")
                self.display_service_health("Query Service", query_health)
                
            except Exception as e:
                progress.update(task2, description="❌ Query service failed")
                self.console.print(f"[yellow]Query Service Error: {e}[/yellow]")
            
            # Test schema service
            task3 = progress.add_task("Testing schema service...", total=None)
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.base_url}/api/v1/schema/stats", timeout=10)
                    schema_stats = response.json()
                
                progress.update(task3, description="✅ Schema service health")
                self.display_service_health("Schema Service", schema_stats)
                
            except Exception as e:
                progress.update(task3, description="❌ Schema service failed")
                self.console.print(f"[yellow]Schema Service Error: {e}[/yellow]")
    
    def display_health_status(self, health_data: Dict):
        """Display health status in a nice format."""
        status = health_data.get("status", "unknown")
        color = "green" if status == "healthy" else "red" if status == "unhealthy" else "yellow"
        
        panel_content = f"[{color}]Status: {status.upper()}[/{color}]\n"
        panel_content += f"Version: {health_data.get('version', 'unknown')}\n"
        panel_content += f"Timestamp: {health_data.get('timestamp', 'unknown')}\n\n"
        
        services = health_data.get("services", {})
        panel_content += "[bold]Services:[/bold]\n"
        for service, service_status in services.items():
            service_color = "green" if "healthy" in str(service_status) else "red"
            panel_content += f"  • {service}: [{service_color}]{service_status}[/{service_color}]\n"
        
        self.console.print(Panel(panel_content, title="🏥 System Health", border_style="blue"))
    
    def display_service_health(self, service_name: str, health_data: Dict):
        """Display individual service health."""
        self.console.print(f"\n[bold]{service_name}:[/bold]")
        self.console.print(JSON.from_data(health_data))
    
    async def list_databases(self):
        """List available databases."""
        self.console.print("\n[bold]🗄️  Listing Available Databases...[/bold]")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/v1/schema/databases", timeout=15)
                response.raise_for_status()
                databases = response.json()
            
            if not databases:
                self.console.print("[yellow]No databases found[/yellow]")
                return
            
            table = Table(title="Available Databases")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="magenta")
            table.add_column("Type", style="green")
            table.add_column("Status", style="yellow")
            table.add_column("Last Sync", style="blue")
            
            for db in databases:
                table.add_row(
                    db.get("id", "unknown"),
                    db.get("name", "unknown"),
                    db.get("type", "unknown"),
                    db.get("status", "unknown"),
                    str(db.get("lastSync", "never"))
                )
            
            self.console.print(table)
            
        except Exception as e:
            self.console.print(f"[bold red]❌ Failed to list databases: {e}[/bold red]")
    
    async def select_database_and_list_tables(self):
        """Select a database and list its tables."""
        self.console.print("\n[bold]📊 Database & Table Explorer[/bold]")
        
        # First, list available databases
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/v1/schema/databases", timeout=15)
                databases = response.json()
            
            if not databases:
                self.console.print("[yellow]No databases available[/yellow]")
                return
            
            # Show databases
            self.console.print("\n[bold]Available Databases:[/bold]")
            for i, db in enumerate(databases):
                self.console.print(f"  {i+1}. {db.get('name', 'unknown')} ({db.get('status', 'unknown')})")
            
            # Let user select or use default
            if len(databases) == 1:
                selected_db = databases[0]
                self.console.print(f"\n[green]Using database: {selected_db.get('name')}[/green]")
            else:
                db_choice = Prompt.ask("Select database number", default="1")
                try:
                    selected_db = databases[int(db_choice) - 1]
                except (ValueError, IndexError):
                    selected_db = databases[0]
            
            self.current_database = selected_db.get("id", "default")
            
            # Now list tables
            await self.list_tables_for_database(self.current_database)
            
        except Exception as e:
            self.console.print(f"[bold red]❌ Error: {e}[/bold red]")
    
    async def list_tables_for_database(self, database_id: str):
        """List tables for a specific database."""
        self.console.print(f"\n[bold]📋 Tables in Database: {database_id}[/bold]")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/v1/schema/tables", timeout=20)
                response.raise_for_status()
                data = response.json()
                
            print(data)
            
            if isinstance(data, list):
                tables = data
            else:
                tables = data.get("tables", [])
            
            if not tables:
                self.console.print("[yellow]No tables found in database[/yellow]")
                return
            
            table = Table(title=f"Tables in {database_id}")
            table.add_column("Name", style="cyan")
            table.add_column("Columns", style="magenta")
            table.add_column("Rows", style="green")
            table.add_column("Size", style="yellow")
            table.add_column("Description", style="blue")
            
            for tbl in tables:
                row_count = tbl.get("row_count")
                row_display = str(row_count) if row_count is not None else "unknown"
                
                size_bytes = tbl.get("size_bytes")
                size_display = self.format_bytes(size_bytes) if size_bytes else "unknown"
                
                description = tbl.get("description") or "Table without description"
                if len(description) > 50:
                    description = description[:47] + "..."
                
                table.add_row(
                    tbl.get("name", "unknown"),
                    str(tbl.get("column_count", 0)),
                    row_display,
                    size_display,
                    description or "No description"
                )
            
            self.console.print(table)
            
            # Show additional info
            self.console.print(f"\n[dim]Total tables: {len(tables)}[/dim]")
            if data.get("extraction_method"):
                self.console.print(f"[dim]Extraction method: {data.get('extraction_method')}[/dim]")
            
        except Exception as e:
            self.console.print(f"[bold red]❌ Failed to list tables: {e}[/bold red]")
    
    def format_bytes(self, bytes_val: int) -> str:
        """Format bytes into human readable format."""
        if bytes_val < 1024:
            return f"{bytes_val}B"
        elif bytes_val < 1024**2:
            return f"{bytes_val/1024:.1f}KB"
        elif bytes_val < 1024**3:
            return f"{bytes_val/(1024**2):.1f}MB"
        else:
            return f"{bytes_val/(1024**3):.1f}GB"
    
    async def explore_table_schema(self):
        """Explore detailed schema of a specific table."""
        if not self.current_database:
            self.console.print("[yellow]Please select a database first (option 3)[/yellow]")
            return
        
        table_name = Prompt.ask("\n[bold]Enter table name to explore[/bold]")
        
        self.console.print(f"\n[bold]🔎 Exploring Table: {table_name}[/bold]")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/schema/tables/{table_name}",
                    timeout=15
                )
                response.raise_for_status()
                table_info = response.json()
            
            # Display table info
            self.display_table_schema(table_info)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                self.console.print(f"[red]Table '{table_name}' not found[/red]")
            else:
                self.console.print(f"[red]HTTP Error {e.response.status_code}: {e.response.text}[/red]")
        except Exception as e:
            self.console.print(f"[bold red]❌ Error exploring table: {e}[/bold red]")
    
    def display_table_schema(self, table_info: Dict):
        """Display table schema in a formatted way."""
        table_name = table_info.get("name", "unknown")
        description = table_info.get("description", "No description available")
        row_count = table_info.get("row_count")
        
        # Table header info
        header_info = f"[bold cyan]Table:[/bold cyan] {table_name}\n"
        header_info += f"[bold cyan]Description:[/bold cyan] {description}\n"
        if row_count is not None:
            header_info += f"[bold cyan]Rows:[/bold cyan] {row_count:,}\n"
        
        self.console.print(Panel(header_info, title="📊 Table Information", border_style="cyan"))
        
        # Columns table
        columns = table_info.get("columns", [])
        if columns:
            col_table = Table(title="Columns")
            col_table.add_column("Name", style="cyan")
            col_table.add_column("Type", style="magenta")
            col_table.add_column("Nullable", style="yellow")
            col_table.add_column("Key Type", style="green")
            col_table.add_column("Foreign Key", style="blue")
            
            for col in columns:
                key_type = ""
                if col.get("primary_key"):
                    key_type = "PRIMARY"
                
                fk = col.get("foreign_key", "")
                if fk and len(fk) > 20:
                    fk = fk[:17] + "..."
                
                col_table.add_row(
                    col.get("name", "unknown"),
                    col.get("type", "unknown"),
                    "YES" if col.get("nullable", True) else "NO",
                    key_type,
                    fk or "-"
                )
            
            self.console.print(col_table)
    
    async def get_sample_data(self):
        """Get sample data from a table."""
        if not self.current_database:
            self.console.print("[yellow]Please select a database first (option 3)[/yellow]")
            return
        
        table_name = Prompt.ask("\n[bold]Enter table name for sample data[/bold]")
        limit = Prompt.ask("Number of rows to sample", default="5")
        
        try:
            limit = int(limit)
        except ValueError:
            limit = 5
        
        self.console.print(f"\n[bold]📈 Sample Data from {table_name}[/bold]")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/schema/sample/{table_name}?limit={limit}",
                    timeout=15
                )
                response.raise_for_status()
                sample_data = response.json()
            
            self.display_sample_data(sample_data)
            
        except Exception as e:
            self.console.print(f"[bold red]❌ Error getting sample data: {e}[/bold red]")
    
    def display_sample_data(self, sample_data: Dict):
        """Display sample data in a table format."""
        data = sample_data.get("data", {})
        columns = data.get("columns", [])
        rows = data.get("rows", [])
        
        if not columns or not rows:
            self.console.print("[yellow]No sample data available[/yellow]")
            return
        
        table = Table(title=f"Sample Data from {sample_data.get('table_name')}")
        
        # Add columns
        for col in columns:
            table.add_column(str(col), style="cyan")
        
        # Add rows
        for row in rows:
            # Convert all values to strings and handle None
            str_row = [str(val) if val is not None else "NULL" for val in row]
            table.add_row(*str_row)
        
        self.console.print(table)
        self.console.print(f"\n[dim]Showing {len(rows)} of {sample_data.get('total_returned', len(rows))} rows[/dim]")
    
    async def ask_natural_language_query(self):
        """Ask a natural language query to the SQL Agent."""
        self.console.print("\n[bold]💬 Natural Language Query Interface[/bold]")
        self.console.print("[dim]Examples: 'Show me all customers', 'What are the top products by price?', 'Analyze customer orders'[/dim]")
        
        query = Prompt.ask("\n[bold]Enter your question[/bold]")
        
        if not query.strip():
            self.console.print("[yellow]Please enter a valid question[/yellow]")
            return
        
        # Query options
        include_analysis = Confirm.ask("Include data analysis?", default=True)
        include_visualization = Confirm.ask("Include visualization suggestions?", default=False)
        
        self.console.print(f"\n[bold]🤖 Processing: '{query}'[/bold]")
        
        request_data = {
            "query": query,
            "session_id": self.session_id,
            "database_name": self.current_database,
            "include_analysis": include_analysis,
            "include_visualization": include_visualization,
            "max_results": 100
        }
        
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console
                ) as progress:
                    task = progress.add_task("Processing your query...", total=None)
                    
                    response = await client.post(
                        f"{self.base_url}/api/v1/query/process",
                        json=request_data
                    )
                    
                    progress.update(task, description="✅ Query processed")
                
                response.raise_for_status()
                result = response.json()
                
                self.display_query_result(result)
                
        except Exception as e:
            self.console.print(f"[bold red]❌ Query failed: {e}[/bold red]")
    
    def display_query_result(self, result: Dict):
        """Display the result of a natural language query."""
        intent = result.get("intent", "unknown")
        confidence = result.get("confidence", 0.0)
        
        # Show intent and confidence
        intent_info = f"[bold]Intent:[/bold] {intent}\n"
        intent_info += f"[bold]Confidence:[/bold] {confidence:.2%}\n"
        intent_info += f"[bold]Processing Time:[/bold] {result.get('processing_time', 0):.3f}s"
        
        self.console.print(Panel(intent_info, title="🧠 Query Analysis", border_style="blue"))
        
        # Show SQL result if available
        sql_result = result.get("sql_result")
        if sql_result:
            self.display_sql_result(sql_result)
        
        # Show analysis result if available
        analysis_result = result.get("analysis_result")
        if analysis_result:
            self.display_analysis_result(analysis_result)
        
        # Show suggestions
        suggestions = result.get("suggestions", [])
        if suggestions:
            self.console.print("\n[bold]💡 Suggestions:[/bold]")
            for i, suggestion in enumerate(suggestions, 1):
                self.console.print(f"  {i}. {suggestion}")
    
    def display_sql_result(self, sql_result: Dict):
        """Display SQL execution result."""
        sql_query = sql_result.get("sql", "")
        data = sql_result.get("data", [])
        row_count = sql_result.get("row_count", 0)
        execution_time = sql_result.get("execution_time", 0)
        
        # Show SQL query
        self.console.print(f"\n[bold]🔍 Generated SQL:[/bold]")
        self.console.print(Panel(sql_query, title="SQL Query", border_style="green"))
        
        # Show results
        if data:
            if row_count <= 10:
                # Show all data for small results
                table = Table(title=f"Query Results ({row_count} rows)")
                
                if data:
                    columns = list(data[0].keys())
                    for col in columns:
                        table.add_column(str(col), style="cyan")
                    
                    for row in data:
                        str_row = [str(row.get(col, "NULL")) for col in columns]
                        table.add_row(*str_row)
                    
                    self.console.print(table)
            else:
                # Show summary for large results
                self.console.print(f"\n[bold]📊 Results Summary:[/bold]")
                self.console.print(f"  • Total rows: {row_count:,}")
                self.console.print(f"  • Execution time: {execution_time:.3f}s")
                
                if data:
                    columns = list(data[0].keys())
                    self.console.print(f"  • Columns: {', '.join(columns)}")
                    
                    # Show first few rows
                    table = Table(title="First 5 rows")
                    for col in columns:
                        table.add_column(str(col), style="cyan")
                    
                    for row in data[:5]:
                        str_row = [str(row.get(col, "NULL")) for col in columns]
                        table.add_row(*str_row)
                    
                    self.console.print(table)
        else:
            self.console.print("[yellow]No data returned[/yellow]")
    
    def display_analysis_result(self, analysis_result: Dict):
        """Display data analysis result."""
        self.console.print(f"\n[bold]📈 Data Analysis:[/bold]")
        
        summary = analysis_result.get("summary", {})
        if summary:
            self.console.print(f"  • Record count: {summary.get('count', 0):,}")
            
            # Numeric columns
            numeric_cols = summary.get("numeric_columns", {})
            if numeric_cols:
                self.console.print(f"  • Numeric columns: {len(numeric_cols)}")
            
            # Categorical columns  
            categorical_cols = summary.get("categorical_columns", {})
            if categorical_cols:
                self.console.print(f"  • Categorical columns: {len(categorical_cols)}")
        
        # Insights
        insights = analysis_result.get("insights", [])
        if insights:
            self.console.print(f"\n[bold]💡 Key Insights:[/bold]")
            for insight in insights[:3]:  # Show top 3 insights
                self.console.print(f"  • {insight.get('description', 'No description')}")
    
    async def test_orchestrator(self):
        """Test the orchestrator functionality specifically."""
        self.console.print("\n[bold]🔧 Testing Orchestrator Components[/bold]")
        
        # Test simple query endpoint (works without orchestrator)
        self.console.print("\n[bold]1. Testing Simple Query Endpoint:[/bold]")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/query/simple",
                    json={
                        "query": "Show me all customers",
                        "database_name": self.current_database,
                        "max_results": 5
                    },
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                
                self.console.print(f"[green]✅ Simple query successful[/green]")
                self.console.print(f"Intent: {result.get('intent')}")
                self.console.print(f"SQL: {result.get('sql', '')[:100]}...")
                
        except Exception as e:
            self.console.print(f"[red]❌ Simple query failed: {e}[/red]")
        
        # Test query validation
        self.console.print("\n[bold]2. Testing Query Validation:[/bold]")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/query/validate",
                    json={
                        "query": "Show me customer information",
                        "database_name": self.current_database
                    },
                    timeout=15
                )
                response.raise_for_status()
                validation = response.json()
                
                self.console.print(f"[green]✅ Validation successful[/green]")
                self.console.print(f"Valid: {validation.get('is_valid')}")
                if validation.get('suggestions'):
                    self.console.print(f"Suggestions: {len(validation.get('suggestions', []))}")
                
        except Exception as e:
            self.console.print(f"[red]❌ Validation failed: {e}[/red]")
        
        # Test orchestrator status
        self.console.print("\n[bold]3. Testing Query Service Status:[/bold]")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/v1/query/status", timeout=10)
                response.raise_for_status()
                status = response.json()
                
                self.console.print("[green]✅ Status retrieved[/green]")
                self.console.print(JSON.from_data(status, indent=2))
                
        except Exception as e:
            self.console.print(f"[red]❌ Status check failed: {e}[/red]")
    
    async def run_full_test_suite(self):
        """Run a comprehensive test of all functionality."""
        self.console.print("\n[bold]📋 Running Full Test Suite[/bold]")
        
        tests = [
            ("System Health", self.test_system_health),
            ("Database Listing", self.list_databases),
            ("Schema Extraction", self._test_schema_extraction),
            ("Sample Data", self._test_sample_data),
            ("Natural Language Query", self._test_nl_query),
            ("Orchestrator Components", self.test_orchestrator)
        ]
        
        results = []
        
        for test_name, test_func in tests:
            self.console.print(f"\n[bold]Running: {test_name}[/bold]")
            try:
                await test_func()
                results.append((test_name, "✅ PASSED"))
                self.console.print(f"[green]✅ {test_name} completed[/green]")
            except Exception as e:
                results.append((test_name, f"❌ FAILED: {str(e)[:50]}"))
                self.console.print(f"[red]❌ {test_name} failed: {e}[/red]")
        
        # Show summary
        self.console.print("\n[bold]📊 Test Suite Results:[/bold]")
        result_table = Table(title="Test Results")
        result_table.add_column("Test", style="cyan")
        result_table.add_column("Result", style="white")
        
        for test_name, result in results:
            result_table.add_row(test_name, result)
        
        self.console.print(result_table)
        
        passed = len([r for r in results if "PASSED" in r[1]])
        total = len(results)
        self.console.print(f"\n[bold]Summary: {passed}/{total} tests passed ({passed/total*100:.1f}%)[/bold]")
    
    async def _test_schema_extraction(self):
        """Test schema extraction specifically."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/api/v1/schema/tables", timeout=20)
            response.raise_for_status()
            
            data = response.json()
            tables = data.get("tables", [])
            
            if not tables:
                raise Exception("No tables found")
            
            # Test getting info for first table
            table_name = tables[0].get("name")
            response = await client.get(f"{self.base_url}/api/v1/schema/tables/{table_name}", timeout=15)
            response.raise_for_status()
    
    async def _test_sample_data(self):
        """Test sample data retrieval."""
        async with httpx.AsyncClient() as client:
            # Get tables first
            response = await client.get(f"{self.base_url}/api/v1/schema/tables", timeout=20)
            response.raise_for_status()
            
            data = response.json()
            tables = data.get("tables", [])
            
            if not tables:
                raise Exception("No tables found for sample data test")
            
            # Test sample data for first table
            table_name = tables[0].get("name")
            response = await client.get(f"{self.base_url}/api/v1/schema/sample/{table_name}?limit=3", timeout=15)
            response.raise_for_status()
    
    async def _test_nl_query(self):
        """Test natural language query processing."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/query/simple",
                json={
                    "query": "Show me data from the first table",
                    "database_name": self.current_database or "default",
                    "max_results": 3
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            if not result.get("query"):
                raise Exception("No query result returned")
    
    async def performance_test(self):
        """Run performance tests on the SQL Agent."""
        self.console.print("\n[bold]🚀 Performance Testing[/bold]")
        
        # Test concurrent queries
        num_concurrent = Prompt.ask("Number of concurrent queries", default="5")
        try:
            num_concurrent = int(num_concurrent)
        except ValueError:
            num_concurrent = 5
        
        self.console.print(f"\n[bold]Testing {num_concurrent} concurrent queries...[/bold]")
        
        async def single_query(query_id: int):
            """Single query for performance testing."""
            start_time = time.time()
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/api/v1/query/simple",
                        json={
                            "query": f"Test query {query_id}",
                            "database_name": self.current_database or "default",
                            "max_results": 5
                        },
                        timeout=30
                    )
                    response.raise_for_status()
                    end_time = time.time()
                    return {
                        "query_id": query_id,
                        "success": True,
                        "response_time": end_time - start_time,
                        "status_code": response.status_code
                    }
            except Exception as e:
                end_time = time.time()
                return {
                    "query_id": query_id,
                    "success": False,
                    "response_time": end_time - start_time,
                    "error": str(e)
                }
        
        # Run concurrent queries
        start_time = time.time()
        tasks = [single_query(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Analyze results
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        avg_response_time = sum(r["response_time"] for r in successful) / len(successful) if successful else 0
        min_response_time = min(r["response_time"] for r in successful) if successful else 0
        max_response_time = max(r["response_time"] for r in successful) if successful else 0
        
        # Display results
        perf_table = Table(title="Performance Test Results")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="green")
        
        perf_table.add_row("Total Queries", str(num_concurrent))
        perf_table.add_row("Successful", str(len(successful)))
        perf_table.add_row("Failed", str(len(failed)))
        perf_table.add_row("Success Rate", f"{len(successful)/num_concurrent*100:.1f}%")
        perf_table.add_row("Total Time", f"{total_time:.3f}s")
        perf_table.add_row("Queries/Second", f"{num_concurrent/total_time:.2f}")
        perf_table.add_row("Avg Response Time", f"{avg_response_time:.3f}s")
        perf_table.add_row("Min Response Time", f"{min_response_time:.3f}s")
        perf_table.add_row("Max Response Time", f"{max_response_time:.3f}s")
        
        self.console.print(perf_table)
        
        # Show failures if any
        if failed:
            self.console.print(f"\n[bold red]Failed Queries:[/bold red]")
            for fail in failed:
                self.console.print(f"  Query {fail['query_id']}: {fail.get('error', 'Unknown error')}")


def main():
    """Main entry point for the interactive tester."""
    import sys
    
    # Check if rich is available
    try:
        import rich
        import httpx
    except ImportError as e:
        print(f"Missing required dependency: {e}")
        print("Install with: pip install rich httpx")
        sys.exit(1)
    
    # Get base URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    # Create and run tester
    tester = SQLAgentTester(base_url)
    
    try:
        asyncio.run(tester.main_menu())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()