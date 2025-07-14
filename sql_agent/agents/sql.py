"""Enhanced SQL Agent with schema-aware generation and table pre-selection."""

import re
import time
from typing import Dict, List, Any, Optional, Tuple
from langchain.schema import HumanMessage, SystemMessage

from .base import BaseAgent
from ..core.state import AgentState, QueryResult, SchemaContext
from ..core.database import db_manager
from ..utils.logging import log_query_execution


class SQLAgent(BaseAgent):
    """Enhanced SQL agent with schema-aware SQL generation."""
    
    def __init__(self, llm_provider):
        super().__init__("sql", llm_provider)
        
        # Execution mode: 'execute', 'validate', 'generate'
        self.execution_mode = 'execute'
        
        # SQL safety patterns
        self.dangerous_patterns = [
            r'\bDROP\b', r'\bDELETE\b', r'\bTRUNCATE\b', r'\bINSERT\b', 
            r'\bUPDATE\b', r'\bALTER\b', r'\bCREATE\b', r'\bGRANT\b', 
            r'\bREVOKE\b', r'\bEXEC\b', r'\bEXECUTE\b'
        ]
    
    async def process(self, state: AgentState) -> AgentState:
        """Process the query with schema-aware SQL generation."""
        start_time = time.time()
        
        self.logger.info("sql_agent_processing", 
                        query=state.query[:100],
                        session_id=state.session_id,
                        execution_mode=self.execution_mode,
                        selected_tables=state.metadata.get("selected_tables", []))
        
        try:
            # Use schema context from router (Phase 2 enhancement)
            schema_context = self._get_schema_context_from_state(state)
            
            # Generate SQL using schema-aware prompting (Phase 2)
            generated_sql = await self._generate_sql_with_schema_context(
                state.query, schema_context, state.metadata
            )
            
            if not generated_sql:
                state.add_error("Failed to generate SQL query")
                return state
            
            # Safety and validation
            if not self._is_sql_safe(generated_sql):
                state.add_error("Generated SQL contains potentially dangerous operations")
                return state
            
            is_valid, validation_error = await self._validate_sql(generated_sql)
            if not is_valid:
                state.add_error(f"SQL validation failed: {validation_error}")
                return state
            
            # Update state with generated SQL
            state.generated_sql = generated_sql
            
            # Execute based on mode
            if self.execution_mode == 'execute':
                query_result = await self._execute_query(
                    generated_sql, 
                    state.metadata.get('max_results', 1000)
                )
                state.query_result = query_result
                
                log_query_execution(
                    self.logger,
                    sql=generated_sql,
                    execution_time=query_result.execution_time,
                    row_count=query_result.row_count,
                    error=query_result.error
                )
            
            elif self.execution_mode == 'validate':
                state.query_result = QueryResult(
                    sql_query=generated_sql,
                    data=[],
                    columns=[],
                    row_count=0,
                    execution_time=0.0,
                    is_validated=True
                )
            
            processing_time = time.time() - start_time
            
            self.logger.info("sql_agent_complete",
                           session_id=state.session_id,
                           sql_length=len(generated_sql),
                           execution_mode=self.execution_mode,
                           row_count=state.query_result.row_count if state.query_result else 0,
                           processing_time=processing_time,
                           selected_tables_count=len(state.metadata.get("selected_tables", [])))
            
        except Exception as e:
            self.logger.error("sql_agent_error", 
                            error=str(e), 
                            session_id=state.session_id,
                            exc_info=True)
            state.add_error(f"SQL agent failed: {e}")
        
        return state
    
    def _get_schema_context_from_state(self, state: AgentState) -> Dict[str, Any]:
        """Extract schema context from router's enhanced state (Phase 2)."""
        try:
            # Get pre-selected tables and enriched context from router
            selected_tables = state.metadata.get("selected_tables", [])
            enriched_context = state.metadata.get("enriched_context", {})
            business_domains = state.metadata.get("routing", {}).get("business_domains", [])
            
            schema_context = {
                "selected_tables": selected_tables,
                "enriched_context": enriched_context,
                "business_domains": business_domains,
                "schema_contexts": state.schema_context or [],
                "database_name": state.database_name
            }
            
            self.logger.info("schema_context_extracted",
                           selected_tables=selected_tables,
                           business_domains=business_domains,
                           context_available=bool(enriched_context))
            
            return schema_context
            
        except Exception as e:
            self.logger.warning("schema_context_extraction_failed", error=str(e))
            return {"selected_tables": [], "enriched_context": {}, "business_domains": []}
    
    async def _generate_sql_with_schema_context(
        self, 
        query: str, 
        schema_context: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> str:
        """Generate SQL using schema-aware LLM prompting (Phase 2)."""
        try:
            # Use enhanced LLM with schema context (Phase 2)
            if hasattr(self.llm, 'generate_with_schema_context'):
                return await self.llm.generate_with_schema_context(
                    query=query,
                    schema_context=schema_context,
                    task_type="sql_generation"
                )
            else:
                # Fallback to manual schema-aware prompting
                return await self._generate_sql_manual_context(query, schema_context)
                
        except Exception as e:
            self.logger.error("schema_aware_sql_generation_failed", error=str(e))
            # Ultimate fallback to basic generation
            return await self._fallback_sql_generation(query, schema_context)
    
    async def _generate_sql_manual_context(self, query: str, schema_context: Dict[str, Any]) -> str:
        """Manual schema-aware SQL generation (fallback)."""
        selected_tables = schema_context.get("selected_tables", [])
        enriched_context = schema_context.get("enriched_context", {})
        business_domains = schema_context.get("business_domains", [])
        
        # Build schema information string
        schema_info = self._build_schema_info_string(selected_tables, enriched_context)
        
        # Build business context
        domain_context = f"Business context: {', '.join(business_domains)}" if business_domains else ""
        
        system_prompt = f"""You are an expert SQL generator. Convert natural language to SQL using the provided schema.

Available Tables and Columns:
{schema_info}

{domain_context}

SQL Generation Rules:
1. ONLY use tables and columns from the schema above
2. Use proper PostgreSQL syntax
3. ONLY generate SELECT queries (read-only)
4. Use appropriate JOINs when combining tables
5. Include WHERE clauses for filtering
6. Use LIMIT for top/first queries
7. Use aggregate functions (COUNT, SUM, AVG) appropriately
8. Return clean SQL without explanations

Format: Return only the executable SQL query."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Generate SQL for: {query}")
        ]
        
        response = await self.llm.generate(messages)
        return self._clean_sql_response(response)
    
    def _build_schema_info_string(self, selected_tables: List[str], enriched_context: Dict[str, Any]) -> str:
        """Build concise schema information for prompts."""
        if not selected_tables:
            return "No specific tables selected - use best judgment"
        
        schema_parts = []
        column_contexts = enriched_context.get("column_contexts", {})
        relationships = enriched_context.get("relationships", {})
        
        # Add table and column information
        for table_name in selected_tables:
            table_info = [f"Table: {table_name}"]
            
            # Add column information if available
            if table_name in column_contexts:
                columns = column_contexts[table_name]
                column_names = [col.get("column_name", "") for col in columns[:10]]  # Limit to 10
                if column_names:
                    table_info.append(f"  Columns: {', '.join(filter(None, column_names))}")
            
            schema_parts.append("\n".join(table_info))
        
        # Add relationship information for JOINs
        if relationships.get("relationships"):
            rel_info = []
            for rel in relationships["relationships"][:3]:  # Limit to 3 relationships
                source = rel.get("source_table", "")
                targets = rel.get("target_tables", [])
                if source in selected_tables and any(t in selected_tables for t in targets):
                    rel_info.append(f"{source} links to {', '.join(targets)}")
            
            if rel_info:
                schema_parts.append(f"\nTable Relationships:\n{chr(10).join(rel_info)}")
        
        return "\n\n".join(schema_parts)
    
    async def _fallback_sql_generation(self, query: str, schema_context: Dict[str, Any]) -> str:
        """Fallback SQL generation using simple patterns."""
        selected_tables = schema_context.get("selected_tables", [])
        query_lower = query.lower()
        
        # Use first selected table or default
        table = selected_tables[0] if selected_tables else "customers"
        
        # Simple pattern matching
        if any(word in query_lower for word in ["count", "how many", "number"]):
            return f"SELECT COUNT(*) FROM {table};"
        
        elif any(word in query_lower for word in ["top", "first", "limit"]):
            numbers = re.findall(r'\b(\d+)\b', query)
            limit = numbers[0] if numbers else "10"
            return f"SELECT * FROM {table} LIMIT {limit};"
        
        elif any(word in query_lower for word in ["average", "avg"]):
            return f"SELECT AVG(amount) FROM {table};"
        
        elif any(word in query_lower for word in ["sum", "total"]):
            return f"SELECT SUM(amount) FROM {table};"
        
        else:
            # Default select with limit for safety
            return f"SELECT * FROM {table} LIMIT 100;"
    
    def _clean_sql_response(self, response: str) -> str:
        """Clean up SQL response from LLM."""
        if not response:
            return ""
        
        # Remove markdown code blocks
        sql = re.sub(r'```sql\s*', '', response, flags=re.IGNORECASE)
        sql = re.sub(r'```\s*', '', sql)
        
        # Extract SQL statement
        lines = sql.split('\n')
        sql_lines = []
        found_sql = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line looks like SQL
            if (line.upper().startswith(('SELECT', 'WITH')) or found_sql):
                sql_lines.append(line)
                found_sql = True
            elif found_sql and line.endswith(';'):
                sql_lines.append(line)
                break
        
        if not sql_lines:
            # Try to extract any SELECT statement
            select_match = re.search(r'(SELECT\s+.*?;)', response, re.IGNORECASE | re.DOTALL)
            if select_match:
                sql = select_match.group(1)
            else:
                return ""
        else:
            sql = ' '.join(sql_lines)
        
        # Normalize whitespace and ensure semicolon
        sql = re.sub(r'\s+', ' ', sql.strip())
        if not sql.endswith(';'):
            sql += ';'
        
        return sql
    
    def _is_sql_safe(self, sql: str) -> bool:
        """Check if SQL query is safe (read-only)."""
        sql_upper = sql.upper()
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, sql_upper):
                return False
        
        # Must start with SELECT or WITH
        sql_trimmed = sql_upper.strip()
        return sql_trimmed.startswith(('SELECT', 'WITH'))
    
    async def _validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Validate the generated SQL against actual schema."""
        try:
            return await db_manager.validate_query(sql)
        except Exception as e:
            return False, str(e)
    
    async def _execute_query(self, sql: str, max_results: int = 1000) -> QueryResult:
        """Execute SQL query with safety limits."""
        try:
            # Add LIMIT if not present and not an aggregation query
            if ("LIMIT" not in sql.upper() and 
                not any(agg in sql.upper() for agg in ["COUNT", "SUM", "AVG", "MIN", "MAX", "GROUP BY"])):
                sql = sql.rstrip(';') + f" LIMIT {max_results};"
            
            return await db_manager.execute_query(sql, timeout=60)
            
        except Exception as e:
            self.logger.error("sql_execution_failed", sql=sql, error=str(e))
            return QueryResult(
                sql_query=sql,
                data=[],
                columns=[],
                row_count=0,
                execution_time=0.0,
                error=str(e)
            )
    
    async def explain_sql(self, sql: str) -> str:
        """Explain SQL query in simple terms."""
        system_prompt = """Explain this SQL query in simple, non-technical language.

Focus on:
1. What data is being retrieved
2. Any filtering or conditions  
3. Any sorting or grouping
4. Any calculations

Keep it concise and user-friendly."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Explain: {sql}")
        ]
        
        try:
            response = await self.llm.generate(messages)
            return response.strip()
        except Exception as e:
            self.logger.error("sql_explanation_failed", error=str(e))
            return "This query retrieves data from the database with specific conditions."
    
    def set_execution_mode(self, mode: str) -> None:
        """Set execution mode: 'execute', 'validate', or 'generate'."""
        valid_modes = ['execute', 'validate', 'generate']
        if mode in valid_modes:
            self.execution_mode = mode
            self.logger.info("sql_agent_mode_changed", mode=mode)
        else:
            raise ValueError(f"Invalid mode: {mode}. Valid modes: {valid_modes}")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information and status."""
        info = super().get_agent_info()
        info.update({
            "execution_mode": self.execution_mode,
            "supported_operations": ["SELECT", "WITH"],
            "safety_patterns": len(self.dangerous_patterns),
            "schema_aware": True,
            "phase": "2.0"
        })
        return info