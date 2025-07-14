"""
Simplified Agent orchestrator for SQL Agent - Phase 1 Focus on Table Selection

This version focuses on the core requirement: Natural Language → Table Selection → SQL Generation
Removes over-engineering while maintaining essential functionality.
"""

import uuid
import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from .base import BaseAgent
from .router import RouterAgent
from .sql import SQLAgent
from ..core.state import AgentState, QueryResult
from ..core.llm import LLMFactory
from ..core.database import db_manager
from ..utils.logging import get_logger


@dataclass
class TableSelectionResult:
    """Result of table selection process."""
    selected_tables: List[str]
    confidence: float
    reasoning: str
    schema_context: Dict[str, Any]


class AgentOrchestrator:
    """Simplified orchestrator focused on table selection and SQL generation."""
    
    def __init__(self, llm_provider_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger("orchestrator")
        self.config = config or {}
        
        # Create LLM provider
        try:
            self.llm_provider = LLMFactory.create_provider(llm_provider_name)
            self.logger.info("llm_provider_created", provider=llm_provider_name)
        except Exception as e:
            self.logger.error("llm_provider_creation_failed", error=str(e))
            raise
        
        # Initialize core agents only
        self.router = RouterAgent(self.llm_provider)
        self.sql_agent = SQLAgent(self.llm_provider)
        
        # Analysis and visualization agents (optional for now)
        self.analysis_agent = None
        self.visualization_agent = None
        
        # Table selection state
        self._available_tables: Dict[str, Dict[str, Any]] = {}
        self._schema_cache: Dict[str, Dict[str, Any]] = {}
        
        # Simple stats
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_processing_time": 0.0
        }
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the orchestrator - simplified version."""
        try:
            self.logger.info("initializing_orchestrator")
            
            # Test LLM provider
            await self._test_llm_provider()
            self.logger.info("llm_provider_tested")
            
            # Initialize database manager if available
            try:
                if hasattr(db_manager, 'initialize') and callable(db_manager.initialize):
                    await db_manager.initialize()
                    self.logger.info("database_manager_initialized")
            except Exception as e:
                self.logger.warning("database_manager_init_failed", error=str(e))
                # Continue without DB manager
            
            # Initialize core agents
            for agent_name, agent in [("router", self.router), ("sql", self.sql_agent)]:
                try:
                    if hasattr(agent, 'initialize') and callable(agent.initialize):
                        await agent.initialize()
                    self.logger.info(f"{agent_name}_agent_initialized")
                except Exception as e:
                    self.logger.warning(f"{agent_name}_agent_init_failed", error=str(e))
            
            self._initialized = True
            self.logger.info("orchestrator_initialized_successfully")
            
        except Exception as e:
            self.logger.error("orchestrator_initialization_failed", error=str(e), exc_info=True)
            raise
    
    async def process_query(
        self, 
        query: str, 
        database_name: Optional[str] = None, 
        context: Optional[Dict[str, Any]] = None
    ) -> AgentState:
        """
        Main query processing method - simplified for table selection focus.
        
        Flow:
        1. Get available tables for database
        2. Use LLM to select relevant tables
        3. Generate SQL with selected tables
        4. Execute SQL
        5. Return results
        """
        start_time = time.time()
        session_id = str(uuid.uuid4())
        
        # Create initial state
        state = AgentState(
            query=query,
            session_id=session_id,
            database_name=database_name or "sql_agent_db",  # Default database
            start_time=datetime.utcnow()
        )
        
        # Add context if provided
        if context:
            state.metadata.update(context)
        
        self.logger.info(
            "processing_query_start",
            session_id=session_id,
            query=query[:100],  # Truncate for logging
            database_name=state.database_name
        )
        
        self.stats["total_queries"] += 1
        
        try:
            # Ensure orchestrator is initialized
            if not self._initialized:
                await self.initialize()
            
            # Step 1: Get available tables
            await self._load_database_schema(state)
            
            # Step 2: Select relevant tables using LLM
            table_selection = await self._select_tables_for_query(state)
            state.metadata["table_selection"] = table_selection
            
            # Step 3: Route query (simplified routing)
            await self._simple_route_query(state)
            
            # Step 4: Generate and execute SQL
            state = await self._run_sql_agent(state)
            
            # Step 5: Optional analysis/visualization (if requested)
            if context and context.get("include_analysis"):
                state = await self._run_optional_analysis(state)
            
            if context and context.get("include_visualization"):
                state = await self._run_optional_visualization(state)
            
            # Finalize state
            state.end_time = datetime.utcnow()
            state.processing_time = (state.end_time - state.start_time).total_seconds()
            
            # Update stats
            if state.has_errors():
                self.stats["failed_queries"] += 1
            else:
                self.stats["successful_queries"] += 1
            
            # Update average processing time
            total = self.stats["total_queries"]
            current_avg = self.stats["avg_processing_time"]
            self.stats["avg_processing_time"] = ((current_avg * (total - 1)) + state.processing_time) / total
            
            self.logger.info(
                "processing_query_complete",
                session_id=session_id,
                processing_time=state.processing_time,
                has_errors=state.has_errors(),
                selected_tables=table_selection.selected_tables if table_selection else [],
                has_sql_result=bool(state.query_result)
            )
            
            return state
            
        except Exception as e:
            # Handle errors gracefully
            processing_time = time.time() - start_time
            self.stats["failed_queries"] += 1
            
            self.logger.error(
                "processing_query_failed",
                session_id=session_id,
                error=str(e),
                processing_time=processing_time,
                exc_info=True
            )
            
            state.add_error(f"Query processing failed: {str(e)}")
            state.end_time = datetime.utcnow()
            state.processing_time = processing_time
            
            return state
    
    async def _load_database_schema(self, state: AgentState) -> None:
        """Load and cache database schema information."""
        database_name = state.database_name
        
        # Check cache first
        if database_name in self._schema_cache:
            self.logger.debug("schema_cache_hit", database=database_name)
            state.metadata["schema_info"] = self._schema_cache[database_name]
            return
        
        try:
            self.logger.info("loading_database_schema", database=database_name)
            
            # Get schema from database manager
            schema_info = await self._get_schema_from_database(database_name)
            
            # Cache the schema
            self._schema_cache[database_name] = schema_info
            state.metadata["schema_info"] = schema_info
            
            self.logger.info(
                "schema_loaded",
                database=database_name,
                table_count=len(schema_info.get("tables", []))
            )
            
        except Exception as e:
            self.logger.error(
                "schema_loading_failed",
                database=database_name,
                error=str(e)
            )
            # Use fallback schema or continue without schema
            state.metadata["schema_info"] = {"tables": [], "error": str(e)}
    
    async def _get_schema_from_database(self, database_name: str) -> Dict[str, Any]:
        """Get schema information dynamically from database."""
        try:
            # First try: Use database manager's schema extraction
            if hasattr(db_manager, 'get_database_schema') and callable(db_manager.get_database_schema):
                schema = await db_manager.get_database_schema(database_name)
                if schema and schema.get("tables"):
                    self.logger.info("schema_from_db_manager", database=database_name, table_count=len(schema["tables"]))
                    return schema
            
            # Second try: Direct database introspection
            schema = await self._introspect_database_schema(database_name)
            if schema and schema.get("tables"):
                self.logger.info("schema_from_introspection", database=database_name, table_count=len(schema["tables"]))
                return schema
            
            # Third try: Check if RAG/vector store has schema information
            schema = await self._get_schema_from_rag(database_name)
            if schema and schema.get("tables"):
                self.logger.info("schema_from_rag", database=database_name, table_count=len(schema["tables"]))
                return schema
            
            # If all methods fail, return empty but valid structure
            self.logger.warning("no_schema_available", database=database_name)
            return {
                "database_name": database_name,
                "tables": [],
                "error": "No schema extraction method succeeded"
            }
            
        except Exception as e:
            self.logger.error("get_schema_failed", database=database_name, error=str(e))
            return {
                "database_name": database_name, 
                "tables": [], 
                "error": str(e)
            }
    
    async def _select_tables_for_query(self, state: AgentState) -> TableSelectionResult:
        """
        Core table selection logic using LLM.
        
        This is where the magic happens - LLM analyzes the query and available tables
        to select the most relevant ones.
        """
        query = state.query
        schema_info = state.metadata.get("schema_info", {})
        tables = schema_info.get("tables", [])
        
        if not tables:
            self.logger.warning("no_tables_available", session_id=state.session_id)
            return TableSelectionResult(
                selected_tables=[],
                confidence=0.0,
                reasoning="No tables available in database schema",
                schema_context={}
            )
        
        try:
            # Create prompt for table selection
            table_descriptions = []
            for table in tables:
                table_desc = f"- {table['name']}: {table.get('description', 'No description')}"
                if table.get('columns'):
                    columns_str = ", ".join(table['columns'][:5])  # Show first 5 columns
                    table_desc += f" (Columns: {columns_str})"
                table_descriptions.append(table_desc)
            
            tables_text = "\n".join(table_descriptions)
            
            selection_prompt = f"""
You are a database expert. Analyze this natural language query and select the most relevant tables.

QUERY: {query}

AVAILABLE TABLES:
{tables_text}

Please select the tables needed to answer this query. Respond in JSON format:
{{
    "selected_tables": ["table1", "table2"],
    "confidence": 0.9,
    "reasoning": "Explanation of why these tables were selected"
}}

Guidelines:
- Select only tables that are directly needed
- For customer queries, use 'customers' table
- For product queries, use 'products' table  
- For sales/order queries, use 'orders' and possibly 'order_items'
- For employee queries, use 'employee_performance' table
- If joins are needed, include all related tables
- Be conservative - better to include fewer relevant tables than many irrelevant ones
"""
            
            # Call LLM for table selection
            messages = [{"role": "user", "content": selection_prompt}]
            response = await self.llm_provider.generate(messages)
            
            # Parse LLM response
            import json
            try:
                # Extract JSON from response
                response_text = response.strip()
                if "```json" in response_text:
                    # Extract JSON from code block
                    start = response_text.find("```json") + 7
                    end = response_text.find("```", start)
                    response_text = response_text[start:end].strip()
                elif "```" in response_text:
                    # Extract from generic code block
                    start = response_text.find("```") + 3
                    end = response_text.rfind("```")
                    response_text = response_text[start:end].strip()
                
                selection_data = json.loads(response_text)
                
                selected_tables = selection_data.get("selected_tables", [])
                confidence = selection_data.get("confidence", 0.5)
                reasoning = selection_data.get("reasoning", "LLM table selection")
                
                # Validate selected tables exist
                available_table_names = [t["name"] for t in tables]
                valid_tables = self._validate_and_filter_tables(selected_tables, available_table_names)
                
                # Create schema context for selected tables
                schema_context = {}
                for table in tables:
                    if table["name"] in valid_tables:
                        schema_context[table["name"]] = table
                
                result = TableSelectionResult(
                    selected_tables=valid_tables,
                    confidence=confidence,
                    reasoning=reasoning,
                    schema_context=schema_context
                )
                
                self.logger.info(
                    "table_selection_complete",
                    session_id=state.session_id,
                    selected_tables=valid_tables,
                    confidence=confidence,
                    reasoning=reasoning[:100]  # Truncate for logging
                )
                
                return result
                
            except json.JSONDecodeError as e:
                self.logger.warning(
                    "table_selection_json_parse_failed",
                    session_id=state.session_id,
                    error=str(e),
                    response=response[:200]
                )
                # Fallback to keyword-based selection
                return self._fallback_table_selection(query, tables)
        
        except Exception as e:
            self.logger.error(
                "table_selection_failed",
                session_id=state.session_id,
                error=str(e)
            )
            # Fallback to keyword-based selection
            return self._fallback_table_selection(query, tables)
    
    def _fallback_table_selection(self, query: str, tables: List[Dict]) -> TableSelectionResult:
        """Dynamic fallback table selection using semantic analysis."""
        try:
            # Extract semantic concepts from query
            query_concepts = self._extract_query_concepts(query)
            
            # Score tables based on semantic relevance
            table_scores = []
            for table in tables:
                score = self._calculate_table_relevance_score(query_concepts, table)
                if score > 0:
                    table_scores.append((table["name"], score))
            
            # Sort by relevance score
            table_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select tables above relevance threshold
            threshold = 0.3
            selected_tables = [name for name, score in table_scores if score > threshold]
            
            # If no tables meet threshold, select top scoring tables
            if not selected_tables and table_scores:
                # Select top 3 or all if fewer than 3
                top_count = min(3, len(table_scores))
                selected_tables = [name for name, score in table_scores[:top_count]]
            
            # If still no tables, this means empty database
            if not selected_tables:
                return TableSelectionResult(
                    selected_tables=[],
                    confidence=0.0,
                    reasoning="No relevant tables found in database schema",
                    schema_context={}
                )
            
            # Build reasoning
            reasoning_parts = []
            for name, score in table_scores:
                if name in selected_tables:
                    reasoning_parts.append(f"{name} (relevance: {score:.2f})")
            
            reasoning = f"Semantic analysis selected tables: {', '.join(reasoning_parts)}"
            
            # Calculate overall confidence
            if table_scores:
                avg_score = sum(score for _, score in table_scores[:len(selected_tables)]) / len(selected_tables)
                confidence = min(0.8, avg_score)  # Cap at 0.8 for fallback method
            else:
                confidence = 0.5
            
            # Create schema context
            schema_context = {}
            for table in tables:
                if table["name"] in selected_tables:
                    schema_context[table["name"]] = table
            
            return TableSelectionResult(
                selected_tables=selected_tables,
                confidence=confidence,
                reasoning=reasoning,
                schema_context=schema_context
            )
            
        except Exception as e:
            self.logger.error("fallback_table_selection_failed", error=str(e))
            # Last resort: return all tables
            all_table_names = [table["name"] for table in tables]
            schema_context = {table["name"]: table for table in tables}
            
            return TableSelectionResult(
                selected_tables=all_table_names,
                confidence=0.2,
                reasoning=f"Error in fallback selection, using all tables: {str(e)}",
                schema_context=schema_context
            )
    
    def _extract_query_concepts(self, query: str) -> Dict[str, List[str]]:
        """Extract semantic concepts from natural language query."""
        query_lower = query.lower()
        
        concepts = {
            "entities": [],
            "actions": [],
            "attributes": [],
            "relationships": [],
            "temporal": [],
            "aggregations": []
        }
        
        # Entity patterns (what the query is about)
        entity_patterns = {
            "customer": ["customer", "client", "user", "buyer", "account holder"],
            "product": ["product", "item", "merchandise", "goods", "inventory"],
            "order": ["order", "purchase", "transaction", "sale", "booking"],
            "employee": ["employee", "staff", "worker", "personnel", "team member"],
            "payment": ["payment", "transaction", "billing", "invoice"],
            "shipping": ["shipping", "delivery", "shipment", "logistics"]
        }
        
        # Action patterns (what to do)
        action_patterns = {
            "retrieve": ["show", "list", "get", "find", "display", "retrieve"],
            "analyze": ["analyze", "examine", "study", "investigate", "insights"],
            "compare": ["compare", "contrast", "versus", "difference"],
            "count": ["count", "number", "how many", "total"],
            "calculate": ["calculate", "compute", "sum", "average", "mean"]
        }
        
        # Attribute patterns (what properties)
        attribute_patterns = {
            "financial": ["price", "cost", "revenue", "profit", "amount", "balance"],
            "temporal": ["date", "time", "when", "period", "duration"],
            "geographic": ["location", "region", "country", "city", "address"],
            "performance": ["performance", "rating", "score", "efficiency"],
            "status": ["status", "state", "condition", "active", "inactive"]
        }
        
        # Extract concepts
        for concept_type, patterns in {
            "entities": entity_patterns,
            "actions": action_patterns, 
            "attributes": attribute_patterns
        }.items():
            for concept_name, keywords in patterns.items():
                if any(keyword in query_lower for keyword in keywords):
                    concepts[concept_type].append(concept_name)
        
        # Extract aggregation patterns
        aggregation_keywords = ["sum", "count", "average", "max", "min", "total", "group by"]
        concepts["aggregations"] = [kw for kw in aggregation_keywords if kw in query_lower]
        
        # Extract temporal patterns
        temporal_keywords = ["today", "yesterday", "last", "this month", "year", "recent"]
        concepts["temporal"] = [kw for kw in temporal_keywords if kw in query_lower]
        
        return concepts
    
    def _calculate_table_relevance_score(self, query_concepts: Dict[str, List[str]], table: Dict) -> float:
        """Calculate relevance score between query concepts and table."""
        score = 0.0
        max_score = 1.0
        
        table_name = table.get("name", "").lower()
        table_description = table.get("description", "").lower()
        table_columns = [col.lower() for col in table.get("columns", [])]
        
        # Score based on table name matching entities
        for entity in query_concepts.get("entities", []):
            if entity in table_name:
                score += 0.4  # High weight for table name match
            elif any(entity in col for col in table_columns):
                score += 0.2  # Medium weight for column match
        
        # Score based on description relevance
        for entity in query_concepts.get("entities", []):
            if entity in table_description:
                score += 0.2
        
        # Score based on attribute matches in columns
        for attribute in query_concepts.get("attributes", []):
            matching_columns = [col for col in table_columns if attribute in col]
            if matching_columns:
                score += 0.1 * len(matching_columns)
        
        # Boost score for common business entity tables
        business_entities = ["customer", "product", "order", "employee", "payment"]
        for entity in business_entities:
            if entity in table_name and entity in [e for e in query_concepts.get("entities", [])]:
                score += 0.3
        
        # Normalize score
        return min(score, max_score)
    
    def _validate_and_filter_tables(self, selected_tables: List[str], available_tables: List[str]) -> List[str]:
        """Validate selected tables exist and filter invalid ones."""
        valid_tables = []
        
        for table in selected_tables:
            # Exact match
            if table in available_tables:
                valid_tables.append(table)
                continue
            
            # Fuzzy matching for slight variations
            table_lower = table.lower()
            for available in available_tables:
                available_lower = available.lower()
                
                # Exact case-insensitive match
                if table_lower == available_lower:
                    valid_tables.append(available)
                    break
                
                # Partial match (be careful with this)
                elif (len(table_lower) > 3 and table_lower in available_lower) or \
                     (len(available_lower) > 3 and available_lower in table_lower):
                    valid_tables.append(available)
                    break
        
        return valid_tables
    
    async def _introspect_database_schema(self, database_name: str) -> Dict[str, Any]:
        """Dynamically introspect database schema using SQL queries."""
        try:
            from ..core.database import db_manager
            
            # Get database connection
            connection = await db_manager.get_connection(database_name)
            if not connection:
                self.logger.warning("no_database_connection", database=database_name)
                return {"database_name": database_name, "tables": []}
            
            # PostgreSQL schema introspection queries
            tables_query = """
                SELECT 
                    table_name,
                    COALESCE(obj_description(c.oid), 'No description available') as table_comment
                FROM information_schema.tables t
                LEFT JOIN pg_class c ON c.relname = t.table_name
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """
            
            columns_query = """
                SELECT 
                    table_name,
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    COALESCE(col_description(pgc.oid, cols.ordinal_position), '') as column_comment
                FROM information_schema.columns cols
                LEFT JOIN pg_class pgc ON pgc.relname = cols.table_name
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position;
            """
            
            foreign_keys_query = """
                SELECT
                    kcu.table_name as table_name,
                    kcu.column_name as column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                    AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = 'public';
            """
            
            # Execute queries
            async with connection.cursor() as cursor:
                # Get tables
                await cursor.execute(tables_query)
                table_rows = await cursor.fetchall()
                
                # Get columns
                await cursor.execute(columns_query)
                column_rows = await cursor.fetchall()
                
                # Get foreign keys
                await cursor.execute(foreign_keys_query)
                fk_rows = await cursor.fetchall()
            
            # Process results into schema structure
            tables_info = {}
            
            # Process tables
            for row in table_rows:
                table_name = row[0]
                table_comment = row[1] if len(row) > 1 else "No description available"
                
                tables_info[table_name] = {
                    "name": table_name,
                    "description": table_comment,
                    "columns": [],
                    "column_details": {},
                    "foreign_keys": []
                }
            
            # Process columns
            for row in column_rows:
                table_name = row[0]
                column_name = row[1]
                data_type = row[2]
                is_nullable = row[3]
                column_default = row[4]
                column_comment = row[5] if len(row) > 5 else ""
                
                if table_name in tables_info:
                    tables_info[table_name]["columns"].append(column_name)
                    tables_info[table_name]["column_details"][column_name] = {
                        "type": data_type,
                        "nullable": is_nullable == "YES",
                        "default": column_default,
                        "comment": column_comment
                    }
            
            # Process foreign keys
            for row in fk_rows:
                table_name = row[0]
                column_name = row[1]
                foreign_table = row[2]
                foreign_column = row[3]
                
                if table_name in tables_info:
                    tables_info[table_name]["foreign_keys"].append({
                        "column": column_name,
                        "references_table": foreign_table,
                        "references_column": foreign_column
                    })
            
            # Convert to final format
            schema = {
                "database_name": database_name,
                "tables": list(tables_info.values()),
                "extraction_method": "database_introspection",
                "extraction_timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info(
                "schema_introspection_complete",
                database=database_name,
                table_count=len(schema["tables"]),
                total_columns=sum(len(t["columns"]) for t in schema["tables"])
            )
            
            return schema
            
        except Exception as e:
            self.logger.error("schema_introspection_failed", database=database_name, error=str(e))
            return {"database_name": database_name, "tables": [], "error": f"Introspection failed: {str(e)}"}
    
    async def _get_schema_from_rag(self, database_name: str) -> Dict[str, Any]:
        """Get schema information from RAG/vector store."""
        try:
            # Try to import and use RAG components
            from ..rag import context_manager
            
            if hasattr(context_manager, 'get_database_schema'):
                schema = await context_manager.get_database_schema(database_name)
                if schema and schema.get("tables"):
                    schema["extraction_method"] = "rag_vector_store"
                    return schema
            
            # Alternative: query vector store for schema information
            if hasattr(context_manager, 'search_schema'):
                schema_docs = await context_manager.search_schema(
                    query=f"database schema for {database_name}",
                    limit=100
                )
                
                if schema_docs:
                    # Process schema documents into structured format
                    schema = self._process_rag_schema_docs(schema_docs, database_name)
                    if schema.get("tables"):
                        return schema
            
            return {"database_name": database_name, "tables": []}
            
        except ImportError:
            self.logger.debug("rag_components_not_available", database=database_name)
            return {"database_name": database_name, "tables": []}
        except Exception as e:
            self.logger.warning("rag_schema_retrieval_failed", database=database_name, error=str(e))
            return {"database_name": database_name, "tables": []}
    
    def _process_rag_schema_docs(self, schema_docs: List[Dict], database_name: str) -> Dict[str, Any]:
        """Process RAG schema documents into structured schema format."""
        try:
            tables_info = {}
            
            for doc in schema_docs:
                # Extract table information from RAG documents
                # This would depend on how your RAG system stores schema info
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                
                table_name = metadata.get("table_name")
                if table_name:
                    if table_name not in tables_info:
                        tables_info[table_name] = {
                            "name": table_name,
                            "description": metadata.get("description", ""),
                            "columns": metadata.get("columns", []),
                            "column_details": metadata.get("column_details", {}),
                            "foreign_keys": metadata.get("foreign_keys", [])
                        }
            
            return {
                "database_name": database_name,
                "tables": list(tables_info.values()),
                "extraction_method": "rag_processed",
                "extraction_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("rag_schema_processing_failed", error=str(e))
            return {"database_name": database_name, "tables": []}
    
    async def _simple_route_query(self, state: AgentState) -> None:
        """Simplified query routing - mainly just set intent to SQL generation."""
        try:
            # Use router agent if available
            if self.router:
                router_state = await self.router.run(state)
                state.metadata.update(router_state.metadata)
            else:
                # Default routing
                state.metadata["routing"] = {
                    "primary_agent": "sql",
                    "confidence": 0.8,
                    "reasoning": "Default SQL generation routing"
                }
            
        except Exception as e:
            self.logger.warning("routing_failed", error=str(e), session_id=state.session_id)
            # Fallback routing
            state.metadata["routing"] = {
                "primary_agent": "sql",
                "confidence": 0.5,
                "reasoning": f"Fallback routing due to error: {str(e)}"
            }
    
    async def _run_sql_agent(self, state: AgentState) -> AgentState:
        """Run SQL generation and execution."""
        try:
            self.logger.info("running_sql_agent", session_id=state.session_id)
            
            # Add table selection context to state for SQL agent
            table_selection = state.metadata.get("table_selection")
            if table_selection:
                state.metadata["selected_tables"] = table_selection.selected_tables
                state.metadata["table_schemas"] = table_selection.schema_context
            
            # Run SQL agent
            result_state = await self.sql_agent.run(state)
            
            self.logger.info(
                "sql_agent_complete",
                session_id=state.session_id,
                has_sql_result=bool(result_state.query_result),
                has_errors=result_state.has_errors()
            )
            
            return result_state
            
        except Exception as e:
            self.logger.error("sql_agent_failed", error=str(e), session_id=state.session_id)
            state.add_error(f"SQL agent failed: {str(e)}")
            return state
    
    async def _run_optional_analysis(self, state: AgentState) -> AgentState:
        """Run analysis agent if available and requested."""
        try:
            if self.analysis_agent and state.query_result and state.query_result.data:
                self.logger.info("running_analysis_agent", session_id=state.session_id)
                result_state = await self.analysis_agent.run(state)
                return result_state
            else:
                self.logger.info("skipping_analysis", session_id=state.session_id, reason="agent_not_available_or_no_data")
                return state
                
        except Exception as e:
            self.logger.warning("analysis_agent_failed", error=str(e), session_id=state.session_id)
            state.add_error(f"Analysis agent failed: {str(e)}")
            return state
    
    async def _run_optional_visualization(self, state: AgentState) -> AgentState:
        """Run visualization agent if available and requested."""
        try:
            if self.visualization_agent and state.query_result and state.query_result.data:
                self.logger.info("running_visualization_agent", session_id=state.session_id)
                result_state = await self.visualization_agent.run(state)
                return result_state
            else:
                self.logger.info("skipping_visualization", session_id=state.session_id, reason="agent_not_available_or_no_data")
                return state
                
        except Exception as e:
            self.logger.warning("visualization_agent_failed", error=str(e), session_id=state.session_id)
            state.add_error(f"Visualization agent failed: {str(e)}")
            return state
    
    async def _test_llm_provider(self):
        """Test LLM provider connectivity."""
        try:
            test_messages = [{"role": "user", "content": "Hello, this is a connectivity test. Please respond with 'OK'."}]
            response = await self.llm_provider.generate(test_messages)
            self.logger.info("llm_provider_test_successful", response=response[:50])
        except Exception as e:
            self.logger.error("llm_provider_test_failed", error=str(e))
            raise
    
    # Public API methods
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            **self.stats,
            "initialized": self._initialized,
            "cached_schemas": list(self._schema_cache.keys()),
            "available_agents": {
                "router": self.router is not None,
                "sql": self.sql_agent is not None,
                "analysis": self.analysis_agent is not None,
                "visualization": self.visualization_agent is not None
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Simple health check."""
        health = {
            "status": "healthy",
            "initialized": self._initialized,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Test LLM provider
            await self._test_llm_provider()
            health["llm_provider"] = "healthy"
        except Exception as e:
            health["llm_provider"] = f"unhealthy: {str(e)}"
            health["status"] = "degraded"
        
        # Check agents
        health["agents"] = {
            "router": "available" if self.router else "not_available",
            "sql": "available" if self.sql_agent else "not_available",
            "analysis": "available" if self.analysis_agent else "not_available",
            "visualization": "available" if self.visualization_agent else "not_available"
        }
        
        return health
    
    def clear_cache(self) -> None:
        """Clear schema cache."""
        self._schema_cache.clear()
        self.logger.info("schema_cache_cleared")
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self.logger.info("orchestrator_cleanup_start")
            
            # Clear caches
            self.clear_cache()
            
            # Cleanup agents if they have cleanup methods
            for agent_name, agent in [("router", self.router), ("sql", self.sql_agent)]:
                if agent and hasattr(agent, 'cleanup') and callable(agent.cleanup):
                    try:
                        await agent.cleanup()
                    except Exception as e:
                        self.logger.warning(f"{agent_name}_cleanup_failed", error=str(e))
            
            self.logger.info("orchestrator_cleanup_complete")
            
        except Exception as e:
            self.logger.error("orchestrator_cleanup_failed", error=str(e))