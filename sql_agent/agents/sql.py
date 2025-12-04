"""Enhanced SQL Agent with schema-aware generation and table pre-selection.

FIXED: Added robust error handling, improved LLM response parsing, and comprehensive logging.
"""

import re
import time
from typing import Dict, List, Any, Optional, Tuple
from langchain.schema import HumanMessage, SystemMessage

from .base import BaseAgent
from ..core.state import AgentState, QueryResult, SchemaContext
from ..core.database import db_manager
from ..utils.logging import log_query_execution


class SQLAgent(BaseAgent):
    """Enhanced SQL agent with schema-aware SQL generation and robust error handling."""
    
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

        # Fraud detection SQL patterns
        self.fraud_detection_patterns = {
            "duplicate_detection": "COUNT(*) > 1 GROUP BY",
            "outlier_detection": "STDDEV|AVG.*WHERE.*>|<",
            "temporal_anomaly": "DATE_TRUNC|EXTRACT.*COUNT",
            "statistical_check": "PERCENTILE|STDDEV|VARIANCE"
        }
    
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
            
            # FIXED: Enhanced SQL generation with better error handling
            generated_sql = await self._generate_sql_with_enhanced_error_handling(
                state.query, schema_context, state.metadata
            )
            
            if not generated_sql:
                error_msg = "Failed to generate SQL query - LLM response parsing failed"
                self.logger.error("sql_generation_completely_failed", 
                               query=state.query[:100],
                               session_id=state.session_id)
                state.add_error(error_msg)
                return state
            
            self.logger.info("sql_generated_successfully",
                           sql=generated_sql[:200],
                           sql_length=len(generated_sql),
                           session_id=state.session_id)
            
            # Around line 60-65, add this:
            # self.logger.error("DEBUG_SQL_BEFORE_SAFETY_CHECK", sql=generated_sql, sql_length=len(generated_sql))
            
            # Safety and validation
            if not self._is_sql_safe(generated_sql):
                error_msg = "Generated SQL contains potentially dangerous operations"
                self.logger.warning("sql_safety_check_failed", 
                                  sql=generated_sql,
                                  session_id=state.session_id)
                state.add_error(error_msg)
                return state
            
            is_valid, validation_error = await self._validate_sql(generated_sql)
            if not is_valid:
                error_msg = f"SQL validation failed: {validation_error}"
                self.logger.error("sql_validation_failed",
                                sql=generated_sql,
                                validation_error=validation_error,
                                session_id=state.session_id)
                state.add_error(error_msg)
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
                           selected_tables_count=len(state.metadata.get("selected_tables", [])),
                           success=True)
            
        except Exception as e:
            self.logger.error("sql_agent_error", 
                            error=str(e), 
                            session_id=state.session_id,
                            query=state.query[:100],
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
    
    async def _generate_sql_with_enhanced_error_handling(
        self, 
        query: str, 
        schema_context: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> str:
        """FIXED: Generate SQL with comprehensive error handling and logging."""
        try:
            self.logger.info("starting_sql_generation",
                           query=query[:100],
                           selected_tables=schema_context.get("selected_tables", []),
                           business_domains=schema_context.get("business_domains", []))
            
            # Use enhanced LLM with schema context (Phase 2)
            if hasattr(self.llm, 'generate_with_schema_context'):
                self.logger.debug("using_enhanced_llm_generation")
                response =  await self.llm.generate_with_schema_context(
                    query=query,
                    schema_context=schema_context,
                    task_type="sql_generation"
                )
                
                return self._clean_sql_response(response)
                
            else:
                # Fallback to manual schema-aware prompting
                self.logger.debug("using_manual_schema_context_generation")
                return await self._generate_sql_manual_context_fixed(query, schema_context)
                
        except Exception as e:
            self.logger.error("schema_aware_sql_generation_failed", 
                            error=str(e),
                            query=query[:100],
                            exc_info=True)
            # Ultimate fallback to basic generation
            self.logger.info("attempting_fallback_sql_generation")
            return await self._fallback_sql_generation(query, schema_context)
    
    async def _generate_sql_manual_context_fixed(self, query: str, schema_context: Dict[str, Any]) -> str:
        """FIXED: Manual schema-aware SQL generation with robust error handling."""
        try:
            selected_tables = schema_context.get("selected_tables", [])
            enriched_context = schema_context.get("enriched_context", {})
            business_domains = schema_context.get("business_domains", [])
            
            self.logger.debug("building_schema_prompt",
                            selected_tables=selected_tables,
                            has_enriched_context=bool(enriched_context))
            
            # Build schema information string
            schema_info = self._build_schema_info_string(selected_tables, enriched_context)

            # Log the schema info being sent to LLM for debugging
            self.logger.info("schema_info_for_llm",
                           schema_info_length=len(schema_info),
                           selected_tables=selected_tables,
                           schema_info=schema_info)  # Full schema info for debugging

            # Build business context
            domain_context = f"Business context: {', '.join(business_domains)}" if business_domains else ""

            system_prompt = f"""You are an expert SQL generator. Convert natural language to SQL using the provided schema.

Available Tables and Columns:
{schema_info}

{domain_context}

SQL Generation Rules:
1. CRITICAL: Use EXACT column names from the schema above - do NOT modify or guess column names
2. Use proper PostgreSQL syntax
3. ONLY generate SELECT queries (read-only)
4. Use appropriate JOINs when combining tables
5. Include WHERE clauses for filtering
6. Use LIMIT for top/first queries
7. Use aggregate functions (COUNT, SUM, AVG) appropriately
8. Return ONLY the SQL query, no explanations or markdown
9. If a column name has no underscore (like "isfraud"), use it exactly as shown, do NOT add underscores

IMPORTANT: Return ONLY the executable SQL query without any formatting, explanations, or markdown.
CRITICAL: Use EXACT column names from the schema - do not convert snake_case, do not add/remove underscores."""

            human_prompt = f"Generate SQL for: {query}"

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            self.logger.debug("sending_prompt_to_llm",
                            system_prompt_length=len(system_prompt),
                            human_prompt=human_prompt)
            
            # Call LLM with error handling
            try:
                response = await self.llm.generate(messages)
                self.logger.info("llm_response_received",
                               response_length=len(response) if response else 0,
                               response_preview=response[:200] if response else "Empty response")
                
                if not response:
                    self.logger.error("llm_returned_empty_response")
                    return ""
                
                # FIXED: Enhanced SQL response cleaning with logging
                cleaned_sql = self._clean_sql_response(response)
                
                if cleaned_sql:
                    self.logger.info("sql_cleaning_successful", 
                                   cleaned_sql=cleaned_sql[:200],
                                   original_length=len(response),
                                   cleaned_length=len(cleaned_sql))
                else:
                    self.logger.error("sql_cleaning_failed",
                                    raw_response=response[:500])
                
                return cleaned_sql
                
            except Exception as llm_error:
                self.logger.error("llm_generation_failed",
                                error=str(llm_error),
                                exc_info=True)
                return ""
                
        except Exception as e:
            self.logger.error("manual_sql_generation_failed",
                            error=str(e),
                            exc_info=True)
            return ""
    
    def _clean_sql_response(self, response: str) -> str:
        """FIXED: Enhanced SQL response cleaning with comprehensive logging and error handling."""
        if not response:
            self.logger.warning("clean_sql_empty_response")
            return ""
        
        try:
            self.logger.debug("cleaning_sql_response", 
                            raw_response_preview=response[:300],
                            raw_response_length=len(response))
            
            original_response = response
            
            # Remove markdown code blocks
            sql = re.sub(r'```sql\s*', '', response, flags=re.IGNORECASE)
            sql = re.sub(r'```\s*', '', sql)
            
            # Remove common LLM prefixes
            sql = re.sub(r'^(Here\'s the SQL query|The SQL query is|Query:)', '', sql, flags=re.IGNORECASE)
            sql = sql.strip()
            
            self.logger.debug("after_markdown_removal", cleaned_preview=sql[:200])
            
            # Try multiple SQL extraction strategies
            cleaned_sql = ""
            
            # Strategy 1: Look for complete SELECT statements
            select_patterns = [
                r'(SELECT\s+.*?;)',  # SELECT with semicolon
                r'(SELECT\s+.*?)(?=\n\n|\n[A-Z]|$)',  # SELECT until double newline or EOF
                r'(SELECT\s+.+)',  # Any SELECT statement
            ]
            
            for pattern in select_patterns:
                match = re.search(pattern, sql, re.IGNORECASE | re.DOTALL)
                if match:
                    cleaned_sql = match.group(1).strip()
                    self.logger.debug("sql_extracted_with_pattern", 
                                    pattern=pattern,
                                    extracted_sql=cleaned_sql[:100])
                    break
            
            # Strategy 2: Line-by-line parsing (fallback)
            if not cleaned_sql:
                self.logger.debug("attempting_line_by_line_parsing")
                lines = sql.split('\n')
                sql_lines = []
                found_sql = False
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Skip explanatory text
                    if any(skip_word in line.lower() for skip_word in 
                          ['here', 'this query', 'the sql', 'explanation', '**']):
                        continue
                        
                    # Check if line looks like SQL
                    if (line.upper().startswith(('SELECT', 'WITH')) or found_sql):
                        sql_lines.append(line)
                        found_sql = True
                    elif found_sql and (line.endswith(';') or not line):
                        if line.endswith(';'):
                            sql_lines.append(line)
                        break
                
                if sql_lines:
                    cleaned_sql = ' '.join(sql_lines)
                    self.logger.debug("sql_extracted_line_by_line", 
                                    extracted_sql=cleaned_sql[:100])
            
            # Strategy 3: Extract anything that looks like a SELECT (last resort)
            if not cleaned_sql:
                self.logger.debug("attempting_loose_select_extraction")
                loose_match = re.search(r'SELECT\s+[^;]+', sql, re.IGNORECASE | re.DOTALL)
                if loose_match:
                    cleaned_sql = loose_match.group(0).strip()
                    self.logger.debug("sql_extracted_loosely", 
                                    extracted_sql=cleaned_sql[:100])
            
            # Final cleaning and validation
            if cleaned_sql:
                # Normalize whitespace
                cleaned_sql = re.sub(r'\s+', ' ', cleaned_sql.strip())
                
                # Ensure semicolon
                if not cleaned_sql.endswith(';'):
                    cleaned_sql += ';'
                
                # Validate it starts with SELECT
                if not cleaned_sql.upper().strip().startswith(('SELECT', 'WITH')):
                    self.logger.warning("extracted_sql_doesnt_start_with_select",
                                      extracted_sql=cleaned_sql[:100])
                    return ""
                
                self.logger.info("sql_cleaning_successful",
                               original_length=len(original_response),
                               cleaned_length=len(cleaned_sql),
                               cleaned_sql=cleaned_sql[:200])
                
                return cleaned_sql
            else:
                self.logger.error("sql_cleaning_failed_all_strategies",
                                raw_response=original_response[:500])
                return ""
                
        except Exception as e:
            self.logger.error("sql_cleaning_exception",
                            error=str(e),
                            raw_response=response[:500],
                            exc_info=True)
            return ""
    
    def _build_schema_info_string(self, selected_tables: List[str], enriched_context: Dict[str, Any]) -> str:
        """Build comprehensive schema information including data types and sample data for better SQL generation."""
        if not selected_tables:
            return "No specific tables selected - use best judgment"

        schema_parts = []
        column_contexts = enriched_context.get("column_contexts", {})
        relationships = enriched_context.get("relationships", {})

        # Add detailed table and column information with data types and samples
        for table_name in selected_tables:
            table_info = [f"Table: {table_name}"]

            # Add column information with data types if available
            if table_name in column_contexts:
                columns = column_contexts[table_name]
                column_details = []

                for col in columns[:15]:  # Limit to 15 columns for context window
                    col_name = col.get("column_name", "")
                    col_type = col.get("data_type", "unknown")
                    is_nullable = col.get("nullable", True)
                    is_pk = col.get("primary_key", False)
                    is_fk = col.get("foreign_key", False)
                    sample_values = col.get("sample_values", [])

                    if not col_name:
                        continue

                    # Build column description
                    col_desc = f"{col_name} ({col_type})"

                    # Add constraints
                    constraints = []
                    if is_pk:
                        constraints.append("PK")
                    if is_fk:
                        fk_ref = col.get("foreign_key_ref", "")
                        if fk_ref:
                            constraints.append(f"FK→{fk_ref}")
                        else:
                            constraints.append("FK")
                    if not is_nullable:
                        constraints.append("NOT NULL")

                    if constraints:
                        col_desc += f" [{', '.join(constraints)}]"

                    # Add sample values if available (very helpful for LLM understanding)
                    if sample_values:
                        # Format sample values nicely
                        sample_str = ", ".join([str(v) for v in sample_values[:3]])
                        col_desc += f" — examples: {sample_str}"

                    column_details.append(col_desc)

                if column_details:
                    table_info.append(f"  Columns:")
                    for detail in column_details:
                        table_info.append(f"    - {detail}")

            schema_parts.append("\n".join(table_info))

        # Add relationship information for JOINs
        if relationships.get("relationships"):
            rel_info = []
            for rel in relationships["relationships"][:5]:  # Increased to 5 for better context
                source = rel.get("source_table", "")
                source_col = rel.get("source_column", "")
                targets = rel.get("target_tables", [])
                target_cols = rel.get("target_columns", [])

                if source in selected_tables and any(t in selected_tables for t in targets):
                    # Provide detailed JOIN information
                    if source_col and target_cols:
                        for i, target in enumerate(targets):
                            if i < len(target_cols):
                                rel_info.append(f"{source}.{source_col} = {target}.{target_cols[i]}")
                    else:
                        rel_info.append(f"{source} links to {', '.join(targets)}")

            if rel_info:
                schema_parts.append(f"\nTable Relationships (for JOINs):")
                for info in rel_info:
                    schema_parts.append(f"  - {info}")

        return "\n\n".join(schema_parts)
    
    async def _fallback_sql_generation(self, query: str, schema_context: Dict[str, Any]) -> str:
        """FIXED: Enhanced fallback SQL generation with better error handling."""
        try:
            self.logger.info("using_fallback_sql_generation", query=query[:100])
            
            selected_tables = schema_context.get("selected_tables", [])
            query_lower = query.lower()
            
            # Use first selected table or default
            table = selected_tables[0] if selected_tables else "customers"
            
            self.logger.debug("fallback_generation_config",
                            primary_table=table,
                            selected_tables=selected_tables)
            
            # Simple pattern matching with enhanced logic
            if any(word in query_lower for word in ["count", "how many", "number"]):
                sql = f"SELECT COUNT(*) FROM {table};"
            elif any(word in query_lower for word in ["top", "first", "limit"]):
                numbers = re.findall(r'\b(\d+)\b', query)
                limit = numbers[0] if numbers else "10"
                sql = f"SELECT * FROM {table} LIMIT {limit};"
            elif any(word in query_lower for word in ["average", "avg"]):
                # Try to find a numeric column
                amount_columns = ["amount", "price", "total", "value", "cost"]
                column = next((col for col in amount_columns if col in query_lower), "amount")
                sql = f"SELECT AVG({column}) FROM {table};"
            elif any(word in query_lower for word in ["sum", "total"]):
                amount_columns = ["amount", "price", "total", "value", "cost"]
                column = next((col for col in amount_columns if col in query_lower), "amount")
                sql = f"SELECT SUM({column}) FROM {table};"
            else:
                # Default select with limit for safety
                sql = f"SELECT * FROM {table} LIMIT 100;"
            
            self.logger.info("fallback_sql_generated", sql=sql)
            return sql
            
        except Exception as e:
            self.logger.error("fallback_sql_generation_failed", 
                            error=str(e),
                            exc_info=True)
            # Ultimate fallback
            return "SELECT * FROM customers LIMIT 10;"
    
    def _is_sql_safe(self, sql: str) -> bool:
        """Check if SQL query is safe (read-only)."""
        try:
            sql_upper = sql.upper()
            
            # Check for dangerous patterns
            for pattern in self.dangerous_patterns:
                if re.search(pattern, sql_upper):
                    self.logger.warning("sql_safety_violation", 
                                      sql=sql[:100],
                                      pattern=pattern)
                    return False
            
            # Must start with SELECT or WITH
            sql_trimmed = sql_upper.strip()
            is_safe = sql_trimmed.startswith(('SELECT', 'WITH'))
            
            if not is_safe:
                self.logger.warning("sql_does_not_start_with_select", sql=sql[:100])
            
            return is_safe
            
        except Exception as e:
            self.logger.error("sql_safety_check_error", 
                            error=str(e),
                            sql=sql[:100])
            return False
    
    async def _validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Validate the generated SQL against actual schema."""
        try:
            self.logger.debug("validating_sql", sql=sql[:200])
            is_valid, error = await db_manager.validate_query(sql)
            
            if not is_valid:
                self.logger.warning("sql_validation_failed", 
                                  sql=sql[:200],
                                  validation_error=error)
            else:
                self.logger.debug("sql_validation_passed")
            
            return is_valid, error
            
        except Exception as e:
            self.logger.error("sql_validation_error", 
                            error=str(e),
                            sql=sql[:200])
            return False, str(e)
    
    async def _execute_query(self, sql: str, max_results: int = 1000) -> QueryResult:
        """Execute SQL query with safety limits."""
        try:
            self.logger.info("executing_sql_query", 
                           sql=sql[:200],
                           max_results=max_results)
            
            # Add LIMIT if not present and not an aggregation query
            if ("LIMIT" not in sql.upper() and 
                not any(agg in sql.upper() for agg in ["COUNT", "SUM", "AVG", "MIN", "MAX", "GROUP BY"])):
                sql = sql.rstrip(';') + f" LIMIT {max_results};"
                self.logger.debug("added_limit_to_sql", modified_sql=sql[:200])
            
            result = await db_manager.execute_query(sql, timeout=60)
            
            self.logger.info("sql_execution_completed",
                           row_count=result.row_count,
                           execution_time=result.execution_time,
                           has_error=bool(result.error))
            
            return result
            
        except Exception as e:
            self.logger.error("sql_execution_failed", 
                            sql=sql[:200], 
                            error=str(e),
                            exc_info=True)
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
            "phase": "2.0_fixed",
            "enhancements": [
                "robust_error_handling",
                "comprehensive_logging", 
                "enhanced_sql_parsing",
                "multiple_extraction_strategies",
                "fallback_generation"
            ]
        })
        return info