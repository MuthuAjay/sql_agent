"""Enhanced SQL Agent for SQL Agent."""

import re
import time
from typing import Dict, List, Any, Optional, Tuple
from langchain.schema import HumanMessage, SystemMessage

from .base import BaseAgent
from ..core.state import AgentState, QueryResult, SchemaContext
from ..core.database import db_manager
from ..rag import context_manager
from ..utils.logging import log_query_execution


class SQLAgent(BaseAgent):
    """Enhanced SQL agent that converts natural language to SQL queries."""
    
    def __init__(self, llm_provider):
        super().__init__("sql", llm_provider)
        
        # Execution mode: 'execute', 'validate', 'generate'
        self.execution_mode = 'execute'
        
        # SQL generation templates for fallback
        self.sql_templates = {
            "select": "SELECT {columns} FROM {table} {where} {group_by} {order_by} {limit}",
            "count": "SELECT COUNT(*) FROM {table} {where}",
            "aggregate": "SELECT {aggregate_function}({column}) FROM {table} {where} {group_by}",
            "top_n": "SELECT * FROM {table} {where} {order_by} LIMIT {limit}",
            "join": "SELECT {columns} FROM {table1} t1 JOIN {table2} t2 ON {join_condition} {where}"
        }
        
        # SQL safety patterns
        self.dangerous_patterns = [
            r'\bDROP\b', r'\bDELETE\b', r'\bTRUNCATE\b', r'\bINSERT\b', 
            r'\bUPDATE\b', r'\bALTER\b', r'\bCREATE\b', r'\bGRANT\b', 
            r'\bREVOKE\b', r'\bEXEC\b', r'\bEXECUTE\b'
        ]
    
    async def process(self, state: AgentState) -> AgentState:
        """Process the query and generate/execute SQL."""
        start_time = time.time()
        
        self.logger.info("sql_agent_processing", 
                        query=state.query[:100],
                        session_id=state.session_id,
                        execution_mode=self.execution_mode)
        
        try:
            # Get schema context using RAG if not already available
            if not state.schema_context:
                state.schema_context = await self._get_schema_context_with_rag(
                    state.query, 
                    state.database_name
                )
            
            # Generate SQL from natural language
            generated_sql = await self._generate_sql(state.query, state.schema_context)
            
            if not generated_sql:
                state.add_error("Failed to generate SQL query")
                return state
            
            # Safety check
            if not self._is_sql_safe(generated_sql):
                state.add_error("Generated SQL contains potentially dangerous operations")
                return state
            
            # Validate the generated SQL
            is_valid, validation_error = await self._validate_sql(generated_sql)
            
            if not is_valid:
                state.add_error(f"SQL validation failed: {validation_error}")
                return state
            
            # Update state with generated SQL
            state.generated_sql = generated_sql
            
            # Execute query based on mode
            if self.execution_mode == 'execute':
                query_result = await self._execute_query(
                    generated_sql, 
                    state.metadata.get('max_results', 1000)
                )
                state.query_result = query_result
                
                # Log the execution
                log_query_execution(
                    self.logger,
                    sql=generated_sql,
                    execution_time=query_result.execution_time,
                    row_count=query_result.row_count,
                    error=query_result.error
                )
            
            elif self.execution_mode == 'validate':
                # Create a mock result for validation mode
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
                           rag_context_count=len(state.schema_context))
            
        except Exception as e:
            self.logger.error("sql_agent_error", 
                            error=str(e), 
                            session_id=state.session_id,
                            exc_info=True)
            state.add_error(f"SQL agent failed: {e}")
        
        return state
    
    async def _get_schema_context_with_rag(self, query: str, database_name: Optional[str] = None) -> List[SchemaContext]:
        """Get relevant schema context for the query using RAG."""
        try:
            # Use RAG context manager to retrieve relevant schema context
            contexts = await context_manager.retrieve_schema_context(
                query=query,
                database_name=database_name,
                limit=5,  # Get top 5 most relevant contexts
                min_similarity=0.6  # Minimum similarity threshold
            )
            
            self.logger.info("rag_context_retrieved",
                           query=query[:100],
                           database_name=database_name,
                           context_count=len(contexts),
                           contexts=[f"{ctx.table_name}.{ctx.column_name or 'table'}" for ctx in contexts])
            
            return contexts
            
        except Exception as e:
            self.logger.error("rag_context_failed", error=str(e))
            # Fallback to database schema lookup
            return await self._get_schema_context_fallback(query, database_name)
    
    async def _get_schema_context_fallback(self, query: str, database_name: Optional[str] = None) -> List[SchemaContext]:
        """Fallback schema context retrieval using database schema."""
        try:
            schema_info = await db_manager.get_schema_info(database_name)
            
            # Simple keyword matching
            query_lower = query.lower()
            relevant_tables = []
            
            # Score tables based on relevance
            table_scores = {}
            for table_name, table_info in schema_info.items():
                score = 0
                
                # Check if table name appears in query
                if table_name.lower() in query_lower:
                    score += 10
                
                # Check column names
                for column in table_info.get("columns", []):
                    column_name = column.get("column_name", "").lower()
                    if column_name in query_lower:
                        score += 5
                
                # Check table description if available
                description = table_info.get("description", "").lower()
                if description:
                    query_words = query_lower.split()
                    desc_words = description.split()
                    common_words = set(query_words) & set(desc_words)
                    score += len(common_words)
                
                if score > 0:
                    table_scores[table_name] = score
            
            # Sort by relevance and take top 3
            relevant_tables = sorted(table_scores.keys(), 
                                   key=lambda x: table_scores[x], 
                                   reverse=True)[:3]
            
            # Create schema context objects
            schema_context = []
            for table_name in relevant_tables:
                table_info = schema_info.get(table_name, {})
                columns = [col['column_name'] for col in table_info.get('columns', [])]
                
                context = SchemaContext(
                    table_name=table_name,
                    description=f"Table {table_name} with columns: {', '.join(columns[:10])}",  # Limit to 10 columns
                    relationships=table_info.get('relationships', [])
                )
                schema_context.append(context)
            
            return schema_context
            
        except Exception as e:
            self.logger.error("fallback_schema_context_failed", error=str(e))
            return []
    
    async def _generate_sql(self, query: str, schema_context: List[SchemaContext]) -> str:
        """Generate SQL from natural language query."""
        schema_context_str = self._build_schema_context_string(schema_context)
        
        system_prompt = f"""You are an expert SQL generator. Convert natural language queries to SQL.

Available Schema:
{schema_context_str}

SQL Generation Rules:
1. ONLY generate SELECT queries (no INSERT, UPDATE, DELETE, DROP, etc.)
2. Use proper SQL syntax and standard functions
3. Include appropriate WHERE clauses for filtering
4. Use LIMIT when "top", "first", or specific numbers are mentioned
5. Use ORDER BY when sorting is mentioned (ASC/DESC)
6. Use aggregate functions (COUNT, SUM, AVG, MIN, MAX) appropriately
7. Use JOINs when multiple tables are involved
8. Always use table aliases for multi-table queries
9. Use LIKE for text pattern matching
10. Use proper date functions for date comparisons
11. Return only the SQL query, no explanations

Examples:
- "top 10 customers" → SELECT * FROM customers LIMIT 10;
- "average sales by region" → SELECT region, AVG(sales) FROM sales GROUP BY region;
- "customers with high revenue" → SELECT * FROM customers WHERE revenue > 10000;"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Convert to SQL: {query}")
        ]
        
        try:
            response = await self.llm.generate(messages)
            sql = self._clean_sql_response(response)
            
            # Fallback if LLM fails to generate proper SQL
            if not sql or len(sql.strip()) < 10:
                self.logger.warning("llm_generated_sql_too_short", response=response)
                sql = self._fallback_sql_generation(query, schema_context)
            
            self.logger.info("sql_generated", 
                           original_query=query[:100], 
                           generated_sql=sql[:200])
            
            return sql
            
        except Exception as e:
            self.logger.error("sql_generation_failed", error=str(e))
            return self._fallback_sql_generation(query, schema_context)
    
    def _build_schema_context_string(self, schema_context: List[SchemaContext]) -> str:
        """Build a concise string representation of schema context."""
        if not schema_context:
            return "No schema information available."
        
        context_parts = []
        for context in schema_context[:3]:  # Limit to 3 tables to avoid token limit
            parts = [f"Table: {context.table_name}"]
            
            if context.description:
                # Truncate long descriptions
                desc = context.description[:200] + "..." if len(context.description) > 200 else context.description
                parts.append(f"  Description: {desc}")
            
            if context.relationships:
                rel_str = ", ".join(context.relationships[:3])  # Limit relationships
                parts.append(f"  Related to: {rel_str}")
            
            context_parts.append("\n".join(parts))
        
        return "\n\n".join(context_parts)
    
    def _clean_sql_response(self, response: str) -> str:
        """Clean up the SQL response from LLM."""
        if not response:
            return ""
        
        # Remove markdown code blocks
        sql = re.sub(r'```sql\s*', '', response, flags=re.IGNORECASE)
        sql = re.sub(r'```\s*', '', sql)
        
        # Remove explanatory text before/after SQL
        lines = sql.split('\n')
        sql_lines = []
        found_sql = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line looks like SQL
            if (line.upper().startswith(('SELECT', 'WITH', 'FROM', 'WHERE', 'GROUP', 'ORDER', 'HAVING', 'LIMIT')) or 
                found_sql):
                sql_lines.append(line)
                found_sql = True
            elif found_sql and line.endswith(';'):
                sql_lines.append(line)
                break
        
        if not sql_lines:
            # Fallback: try to extract any SELECT statement
            select_match = re.search(r'(SELECT\s+.*?;)', response, re.IGNORECASE | re.DOTALL)
            if select_match:
                sql = select_match.group(1)
            else:
                return ""
        else:
            sql = ' '.join(sql_lines)
        
        # Normalize whitespace
        sql = re.sub(r'\s+', ' ', sql.strip())
        
        # Ensure it ends with semicolon
        if not sql.endswith(';'):
            sql += ';'
        
        return sql
    
    def _fallback_sql_generation(self, query: str, schema_context: List[SchemaContext]) -> str:
        """Fallback SQL generation using templates and pattern matching."""
        query_lower = query.lower()
        table = self._extract_table_name(query, schema_context)
        
        if not table:
            return "SELECT 1; -- No suitable table found"
        
        # Pattern matching for common query types
        if any(word in query_lower for word in ["count", "how many", "number of"]):
            where_clause = self._extract_where_clause(query, schema_context)
            return f"SELECT COUNT(*) FROM {table}{where_clause};"
        
        elif any(word in query_lower for word in ["top", "first", "limit"]):
            numbers = re.findall(r'\b(\d+)\b', query)
            limit = numbers[0] if numbers else "10"
            order_clause = self._extract_order_clause(query)
            return f"SELECT * FROM {table}{order_clause} LIMIT {limit};"
        
        elif any(word in query_lower for word in ["average", "avg", "mean"]):
            column = self._extract_numeric_column(query, schema_context)
            group_clause = self._extract_group_clause(query, schema_context)
            return f"SELECT AVG({column}) FROM {table}{group_clause};"
        
        elif any(word in query_lower for word in ["sum", "total"]):
            column = self._extract_numeric_column(query, schema_context)
            group_clause = self._extract_group_clause(query, schema_context)
            return f"SELECT SUM({column}) FROM {table}{group_clause};"
        
        else:
            # Default select with potential WHERE clause
            where_clause = self._extract_where_clause(query, schema_context)
            limit_clause = " LIMIT 100"  # Default limit for safety
            return f"SELECT * FROM {table}{where_clause}{limit_clause};"
    
    def _extract_table_name(self, query: str, schema_context: List[SchemaContext]) -> str:
        """Extract most relevant table name."""
        query_lower = query.lower()
        
        # Score tables based on query relevance
        table_scores = {}
        for context in schema_context:
            score = 0
            table_name = context.table_name.lower()
            
            if table_name in query_lower:
                score += 10
            
            # Check for partial matches
            if any(word in query_lower for word in table_name.split('_')):
                score += 5
            
            table_scores[context.table_name] = score
        
        if table_scores:
            return max(table_scores.keys(), key=lambda x: table_scores[x])
        
        return schema_context[0].table_name if schema_context else "table"
    
    def _extract_where_clause(self, query: str, schema_context: List[SchemaContext]) -> str:
        """Extract basic WHERE clause patterns."""
        # This is a simplified implementation
        # In practice, you might want more sophisticated NLP
        
        where_patterns = []
        query_lower = query.lower()
        
        # Look for simple comparisons
        if "where" in query_lower:
            where_part = query_lower.split("where", 1)[1]
            return f" WHERE {where_part.strip()}"
        
        # Look for common filter words
        if any(word in query_lower for word in ["high", "large", "big"]):
            return " WHERE amount > 1000"  # Example condition
        
        if any(word in query_lower for word in ["recent", "latest", "new"]):
            return " WHERE created_date >= CURRENT_DATE - INTERVAL '30 days'"
        
        return ""
    
    def _extract_order_clause(self, query: str) -> str:
        """Extract ORDER BY clause."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["highest", "largest", "top", "max"]):
            return " ORDER BY amount DESC"
        elif any(word in query_lower for word in ["lowest", "smallest", "min"]):
            return " ORDER BY amount ASC"
        elif "recent" in query_lower or "latest" in query_lower:
            return " ORDER BY created_date DESC"
        
        return ""
    
    def _extract_group_clause(self, query: str, schema_context: List[SchemaContext]) -> str:
        """Extract GROUP BY clause."""
        query_lower = query.lower()
        
        # Look for grouping keywords
        group_words = ["by", "per", "each", "every"]
        for word in group_words:
            if f" {word} " in query_lower:
                # Simple heuristic: group by common dimension columns
                common_dims = ["region", "category", "type", "status", "department"]
                for dim in common_dims:
                    if dim in query_lower:
                        return f" GROUP BY {dim}"
        
        return ""
    
    def _extract_numeric_column(self, query: str, schema_context: List[SchemaContext]) -> str:
        """Extract likely numeric column for aggregation."""
        query_lower = query.lower()
        
        # Common numeric column names
        numeric_cols = ["amount", "price", "revenue", "sales", "quantity", "count", "total", "value"]
        
        for col in numeric_cols:
            if col in query_lower:
                return col
        
        return "amount"  # Default fallback
    
    def _is_sql_safe(self, sql: str) -> bool:
        """Check if SQL query is safe (read-only)."""
        sql_upper = sql.upper()
        
        for pattern in self.dangerous_patterns:
            if re.search(pattern, sql_upper):
                return False
        
        # Must start with SELECT or WITH
        sql_trimmed = sql_upper.strip()
        return sql_trimmed.startswith(('SELECT', 'WITH'))
    
    async def _validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Validate the generated SQL."""
        try:
            return await db_manager.validate_query(sql)
        except Exception as e:
            return False, str(e)
    
    async def _execute_query(self, sql: str, max_results: int = 1000) -> QueryResult:
        """Execute the SQL query with limits."""
        try:
            # Add LIMIT if not present and query doesn't have aggregation
            if "LIMIT" not in sql.upper() and not any(agg in sql.upper() for agg in ["COUNT", "SUM", "AVG", "MIN", "MAX"]):
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
        """Explain what a SQL query does in simple terms."""
        system_prompt = """Explain the SQL query in simple, non-technical language.

Focus on:
1. What data is being retrieved
2. Any filtering conditions
3. Any sorting or grouping
4. Any calculations being performed

Keep it concise and user-friendly."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Explain this SQL: {sql}")
        ]
        
        try:
            response = await self.llm.generate(messages)
            return response.strip()
        except Exception as e:
            self.logger.error("sql_explanation_failed", error=str(e))
            return "This query retrieves data from the database with specific conditions."
    
    async def optimize_sql(self, sql: str) -> List[str]:
        """Suggest optimizations for SQL query."""
        suggestions = []
        sql_upper = sql.upper()
        
        # Basic optimization suggestions
        if "SELECT *" in sql_upper:
            suggestions.append("Consider selecting only needed columns instead of SELECT *")
        
        if "LIMIT" not in sql_upper and "COUNT" not in sql_upper:
            suggestions.append("Consider adding LIMIT clause to prevent large result sets")
        
        if "WHERE" not in sql_upper and "GROUP BY" not in sql_upper:
            suggestions.append("Consider adding WHERE clause to filter results")
        
        if sql.count("JOIN") > 2:
            suggestions.append("Consider if all JOINs are necessary - complex joins can be slow")
        
        return suggestions
    
    def get_sql_templates(self) -> Dict[str, str]:
        """Get available SQL templates."""
        return self.sql_templates.copy()
    
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
            "template_count": len(self.sql_templates),
            "safety_patterns": len(self.dangerous_patterns)
        })
        return info