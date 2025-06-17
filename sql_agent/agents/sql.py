"""SQL Agent for SQL Agent."""

import re
from typing import Dict, List, Any, Optional, Tuple
from langchain.schema import HumanMessage, SystemMessage
from .base import BaseAgent
from ..core.state import AgentState, QueryResult, SchemaContext
from ..core.database import db_manager
from ..utils.logging import log_query_execution


class SQLAgent(BaseAgent):
    """SQL agent that converts natural language to SQL queries."""
    
    def __init__(self, llm_provider):
        super().__init__("sql", llm_provider)
        
        # SQL generation templates
        self.sql_templates = {
            "select": "SELECT {columns} FROM {table} {where} {group_by} {order_by} {limit}",
            "count": "SELECT COUNT(*) FROM {table} {where}",
            "aggregate": "SELECT {aggregate_function}({column}) FROM {table} {where} {group_by}"
        }
    
    async def process(self, state: AgentState) -> AgentState:
        """Process the query and generate SQL."""
        self.logger.info("sql_agent_processing", query=state.query)
        
        try:
            # Get schema context if not already available
            if not state.schema_context:
                state.schema_context = await self._get_schema_context(state.query)
            
            # Generate SQL from natural language
            generated_sql = await self._generate_sql(state.query, state.schema_context)
            
            # Validate the generated SQL
            is_valid, validation_error = await self._validate_sql(generated_sql)
            
            if not is_valid:
                state.add_error(f"SQL validation failed: {validation_error}")
                return state
            
            # Execute the query
            query_result = await self._execute_query(generated_sql)
            
            # Update state
            state.generated_sql = generated_sql
            state.query_result = query_result
            
            # Log the execution
            log_query_execution(
                self.logger,
                sql=generated_sql,
                execution_time=query_result.execution_time,
                row_count=query_result.row_count,
                error=query_result.error
            )
            
            self.logger.info(
                "sql_generation_complete",
                sql=generated_sql,
                row_count=query_result.row_count,
                execution_time=query_result.execution_time
            )
            
        except Exception as e:
            self.logger.error("sql_agent_error", error=str(e), exc_info=True)
            state.add_error(f"SQL generation failed: {e}")
        
        return state
    
    async def _get_schema_context(self, query: str) -> List[SchemaContext]:
        """Get relevant schema context for the query."""
        # This is a simplified implementation
        # In Phase 3, this will use RAG with vector database
        try:
            schema_info = await db_manager.get_schema_info()
            
            # Simple keyword matching for now
            query_lower = query.lower()
            relevant_tables = []
            
            for table_name, table_info in schema_info.items():
                # Check if table name or column names appear in query
                if table_name.lower() in query_lower:
                    relevant_tables.append(table_name)
                else:
                    # Check column names
                    for column in table_info.get("columns", []):
                        if column.get("column_name", "").lower() in query_lower:
                            relevant_tables.append(table_name)
                            break
            
            # Create schema context objects
            schema_context = []
            for table_name in relevant_tables[:3]:  # Limit to 3 most relevant tables
                table_info = schema_info.get(table_name, {})
                context = SchemaContext(
                    table_name=table_name,
                    description=f"Table {table_name} with columns: {', '.join([col['column_name'] for col in table_info.get('columns', [])])}"
                )
                schema_context.append(context)
            
            return schema_context
            
        except Exception as e:
            self.logger.error("schema_context_failed", error=str(e))
            return []
    
    async def _generate_sql(self, query: str, schema_context: List[SchemaContext]) -> str:
        """Generate SQL from natural language query."""
        # Build schema context string
        schema_context_str = self._build_schema_context_string(schema_context)
        
        system_prompt = f"""You are an expert SQL generator. Convert natural language queries to SQL.

Database Schema:
{schema_context_str}

Rules:
1. Only generate SELECT queries (no INSERT, UPDATE, DELETE, DROP, etc.)
2. Use proper SQL syntax
3. Include appropriate WHERE clauses when filtering is mentioned
4. Use LIMIT when "top", "first", or specific numbers are mentioned
5. Use ORDER BY when sorting is mentioned
6. Use aggregate functions (COUNT, SUM, AVG, etc.) when appropriate
7. Use JOINs when multiple tables are involved
8. Always use table aliases for clarity

Generate only the SQL query, no explanations."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Convert this to SQL: {query}")
        ]
        
        try:
            response = await self.llm.generate(messages)
            
            # Clean up the response
            sql = self._clean_sql_response(response)
            
            self.logger.info("sql_generated", original_query=query, generated_sql=sql)
            
            return sql
            
        except Exception as e:
            self.logger.error("sql_generation_failed", error=str(e))
            # Fallback to template-based generation
            return self._fallback_sql_generation(query, schema_context)
    
    def _build_schema_context_string(self, schema_context: List[SchemaContext]) -> str:
        """Build a string representation of schema context."""
        if not schema_context:
            return "No schema context available"
        
        context_parts = []
        for context in schema_context:
            context_parts.append(f"Table: {context.table_name}")
            if context.description:
                context_parts.append(f"  Description: {context.description}")
            if context.relationships:
                context_parts.append(f"  Relationships: {', '.join(context.relationships)}")
        
        return "\n".join(context_parts)
    
    def _clean_sql_response(self, response: str) -> str:
        """Clean up the SQL response from LLM."""
        # Remove markdown code blocks
        sql = re.sub(r'```sql\s*', '', response)
        sql = re.sub(r'```\s*', '', sql)
        
        # Remove extra whitespace and newlines
        sql = re.sub(r'\s+', ' ', sql.strip())
        
        # Ensure it ends with semicolon
        if not sql.endswith(';'):
            sql += ';'
        
        return sql
    
    def _fallback_sql_generation(self, query: str, schema_context: List[SchemaContext]) -> str:
        """Fallback SQL generation using templates and pattern matching."""
        query_lower = query.lower()
        
        # Simple pattern matching
        if "count" in query_lower or "how many" in query_lower:
            table = self._extract_table_name(query, schema_context)
            return f"SELECT COUNT(*) FROM {table};"
        
        elif "top" in query_lower or "first" in query_lower:
            # Extract number
            numbers = re.findall(r'\d+', query)
            limit = numbers[0] if numbers else "10"
            table = self._extract_table_name(query, schema_context)
            return f"SELECT * FROM {table} LIMIT {limit};"
        
        else:
            # Default select all
            table = self._extract_table_name(query, schema_context)
            return f"SELECT * FROM {table};"
    
    def _extract_table_name(self, query: str, schema_context: List[SchemaContext]) -> str:
        """Extract table name from query or schema context."""
        # Check if any table names are mentioned in the query
        query_lower = query.lower()
        for context in schema_context:
            if context.table_name.lower() in query_lower:
                return context.table_name
        
        # Return first available table or default
        if schema_context:
            return schema_context[0].table_name
        
        return "customers"  # Default fallback
    
    async def _validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Validate the generated SQL."""
        return await db_manager.validate_query(sql)
    
    async def _execute_query(self, sql: str) -> QueryResult:
        """Execute the SQL query."""
        return await db_manager.execute_query(
            sql, 
            timeout=30  # Use configuration timeout
        )
    
    def get_sql_templates(self) -> Dict[str, str]:
        """Get available SQL templates."""
        return self.sql_templates.copy()
    
    async def explain_sql(self, sql: str) -> str:
        """Explain what a SQL query does."""
        system_prompt = """You are an SQL expert. Explain what the given SQL query does in simple terms.

Focus on:
1. What data is being retrieved
2. Any filtering or conditions
3. Any sorting or grouping
4. Any aggregations or calculations

Keep the explanation clear and concise."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Explain this SQL query: {sql}")
        ]
        
        try:
            response = await self.llm.generate(messages)
            return response.strip()
        except Exception as e:
            self.logger.error("sql_explanation_failed", error=str(e))
            return "Unable to explain the SQL query." 