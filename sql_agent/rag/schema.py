"""Schema processing for RAG functionality."""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core.database import db_manager
from ..core.state import SchemaContext
from ..rag.embeddings import embedding_service
from ..utils.logging import get_logger


class SchemaProcessor:
    """Process and extract schema information for RAG context."""
    
    def __init__(self):
        self.logger = get_logger("rag.schema_processor")
        self._schema_cache: Dict[str, Dict[str, Any]] = {}
        self._last_update: Optional[datetime] = None
    
    async def extract_schema_contexts(self, database_name: Optional[str] = None) -> List[SchemaContext]:
        """Extract schema contexts from the database."""
        try:
            self.logger.info("extracting_schema_contexts", database_name=database_name)
            
            # Get schema information from database
            schema_info = await db_manager.get_schema_info()
            
            if not schema_info:
                error_msg = "No schema information available: database may be empty, inaccessible, or misconfigured."
                self.logger.error("no_schema_info_available", error=error_msg)
                raise RuntimeError(error_msg)
            
            contexts = []
            
            # Process each table
            for table_name, table_info in schema_info.items():
                try:
                    # Create table-level context
                    table_context = await self._create_table_context(table_name, table_info)
                    contexts.append(table_context)
                except Exception as e:
                    self.logger.error("create_table_context_failed", table_name=table_name, error=repr(e), exc_info=True)
                    raise
                
                # Create column-level contexts
                columns = table_info.get("columns", [])
                for column_info in columns:
                    try:
                        column_context = await self._create_column_context(
                            table_name, column_info, table_info
                        )
                        contexts.append(column_context)
                    except Exception as e:
                        self.logger.error("create_column_context_failed", table_name=table_name, column_name=column_info.get("column_name", ""), error=repr(e), exc_info=True)
                        raise
            
            self.logger.info("schema_contexts_extracted", 
                           table_count=len(schema_info),
                           total_contexts=len(contexts))
            
            return contexts
            
        except Exception as e:
            self.logger.error("extract_schema_contexts_failed", error=repr(e), exc_info=True)
            raise
    
    async def _create_table_context(self, table_name: str, table_info: Dict[str, Any]) -> SchemaContext:
        """Create a schema context for a table."""
        try:
            # Extract table information
            columns = table_info.get("columns", [])
            column_count = len(columns)
            
            # Get sample data for table description
            sample_data_result = await db_manager.get_sample_data(table_name, limit=3)
            sample_data = []
            if sample_data_result and sample_data_result.get("columns") and sample_data_result.get("rows"):
                columns = sample_data_result["columns"]
                for row in sample_data_result["rows"]:
                    sample_data.append(dict(zip(columns, row)))
            
            # Create description
            description = f"Table {table_name} with {column_count} columns"
            if sample_data:
                description += f". Contains data like: {self._format_sample_data(sample_data)}"
            
            # Identify relationships
            relationships = self._identify_table_relationships(table_name, columns)
            
            # Create context
            context = SchemaContext(
                table_name=table_name,
                description=description,
                sample_values=self._extract_sample_values(sample_data),
                relationships=relationships
            )
            
            # Generate embedding
            context.embedding = await embedding_service.embed_schema_context(
                self._create_context_text(context)
            )
            
            return context
            
        except Exception as e:
            self.logger.error("create_table_context_failed", 
                            table_name=table_name, error=repr(e), exc_info=True)
            raise
    
    async def _create_column_context(
        self, 
        table_name: str, 
        column_info: Dict[str, Any], 
        table_info: Dict[str, Any]
    ) -> SchemaContext:
        """Create a schema context for a column."""
        try:
            # Extract column information
            column_name = column_info.get("column_name", "")
            data_type = column_info.get("data_type", "")
            is_nullable = column_info.get("is_nullable", "")
            column_default = column_info.get("column_default", "")
            
            # Get sample values for this column
            sample_values = await self._get_column_sample_values(table_name, column_name)
            
            # Create description
            description_parts = [f"Column {column_name} in table {table_name}"]
            description_parts.append(f"Data type: {data_type}")
            
            if is_nullable:
                description_parts.append(f"Nullable: {is_nullable}")
            
            if column_default:
                description_parts.append(f"Default: {column_default}")
            
            if sample_values:
                sample_str = ", ".join(sample_values[:3])
                description_parts.append(f"Sample values: {sample_str}")
            
            description = ". ".join(description_parts)
            
            # Identify relationships
            relationships = self._identify_column_relationships(table_name, column_name, column_info)
            
            # Create context
            context = SchemaContext(
                table_name=table_name,
                column_name=column_name,
                data_type=data_type,
                description=description,
                sample_values=sample_values,
                relationships=relationships
            )
            
            # Generate embedding
            context.embedding = await embedding_service.embed_schema_context(
                self._create_context_text(context)
            )
            
            return context
            
        except Exception as e:
            self.logger.error("create_column_context_failed", 
                            table_name=table_name, 
                            column_name=column_info.get("column_name", ""),
                            error=repr(e), exc_info=True)
            raise
    
    async def _get_column_sample_values(self, table_name: str, column_name: str) -> List[str]:
        """Get sample values for a specific column."""
        try:
            # Query sample data for the column
            sql = f"SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL LIMIT 10"
            
            result = await db_manager.execute_query(sql, timeout=10)
            
            if result.error or not result.data:
                return []
            
            # Extract values and convert to strings
            values = []
            for row in result.data:
                value = row.get(column_name)
                if value is not None:
                    values.append(str(value)[:50])  # Limit length
            
            return values
            
        except Exception as e:
            self.logger.warning("get_column_sample_values_failed", 
                              table_name=table_name, 
                              column_name=column_name,
                              error=repr(e), exc_info=True)
            return []
    
    def _identify_table_relationships(self, table_name: str, columns: List[Any]) -> List[str]:
        """Identify relationships for a table."""
        relationships = []
        
        # Look for foreign key patterns
        for column in columns:
            if isinstance(column, dict):
                column_name = column.get("column_name", "")
            elif isinstance(column, str):
                column_name = column
            else:
                continue
            # Common foreign key patterns
            if column_name.endswith("_id"):
                referenced_table = column_name[:-3]  # Remove "_id" suffix
                relationships.append(f"References {referenced_table} table via {column_name}")
            elif "foreign" in column_name.lower():
                relationships.append(f"Foreign key column: {column_name}")
        return relationships
    
    def _identify_column_relationships(
        self, 
        table_name: str, 
        column_name: str, 
        column_info: Dict[str, Any]
    ) -> List[str]:
        """Identify relationships for a column."""
        relationships = []
        
        # Check if it's a foreign key
        if column_name.endswith("_id"):
            referenced_table = column_name[:-3]
            relationships.append(f"Foreign key to {referenced_table} table")
        
        # Check data type for relationships
        data_type = column_info.get("data_type", "").lower()
        if "int" in data_type and column_name.endswith("_id"):
            relationships.append("Integer foreign key")
        
        return relationships
    
    def _format_sample_data(self, sample_data: List[Dict[str, Any]]) -> str:
        """Format sample data for description."""
        if not sample_data:
            return ""
        
        # Get first row and format key-value pairs
        first_row = sample_data[0]
        formatted_pairs = []
        
        for key, value in first_row.items():
            if value is not None:
                formatted_pairs.append(f"{key}={str(value)[:20]}")
        
        return ", ".join(formatted_pairs[:3])  # Limit to 3 pairs
    
    def _extract_sample_values(self, sample_data: List[Dict[str, Any]]) -> List[str]:
        """Extract sample values from sample data."""
        values = []
        
        for row in sample_data:
            for value in row.values():
                if value is not None:
                    values.append(str(value)[:30])  # Limit length
        
        return values[:5]  # Limit to 5 values
    
    def _create_context_text(self, context: SchemaContext) -> str:
        """Create text representation of context for embedding."""
        parts = []
        
        if context.column_name:
            parts.append(f"Column {context.column_name} in table {context.table_name}")
            if context.data_type:
                parts.append(f"Type: {context.data_type}")
        else:
            parts.append(f"Table {context.table_name}")
        
        if context.description:
            parts.append(context.description)
        
        if context.sample_values:
            sample_str = ", ".join(context.sample_values[:3])
            parts.append(f"Examples: {sample_str}")
        
        if context.relationships:
            rel_str = "; ".join(context.relationships)
            parts.append(f"Relationships: {rel_str}")
        
        return " | ".join(parts)
    
    async def get_schema_summary(self) -> Dict[str, Any]:
        """Get a summary of the current schema."""
        try:
            schema_info = await db_manager.get_schema_info()
            
            if not schema_info:
                return {"error": "No schema information available"}
            
            summary = {
                "total_tables": len(schema_info),
                "tables": {},
                "total_columns": 0
            }
            
            for table_name, table_info in schema_info.items():
                columns = table_info.get("columns", [])
                summary["tables"][table_name] = {
                    "column_count": len(columns),
                    "columns": [col.get("column_name") for col in columns]
                }
                summary["total_columns"] += len(columns)
            
            self.logger.info("schema_summary_generated", 
                           total_tables=summary["total_tables"],
                           total_columns=summary["total_columns"])
            
            return summary
            
        except Exception as e:
            self.logger.error("get_schema_summary_failed", error=repr(e), exc_info=True)
            raise
    
    async def refresh_schema_cache(self) -> None:
        """Refresh the schema cache."""
        try:
            self.logger.info("refreshing_schema_cache")
            
            # Clear existing cache
            self._schema_cache.clear()
            
            # Extract fresh schema information
            contexts = await self.extract_schema_contexts()
            
            # Update cache
            for context in contexts:
                if context.table_name not in self._schema_cache:
                    self._schema_cache[context.table_name] = {
                        "contexts": [],
                        "last_updated": datetime.utcnow()
                    }
                
                self._schema_cache[context.table_name]["contexts"].append(context)
            
            self._last_update = datetime.utcnow()
            
            self.logger.info("schema_cache_refreshed", 
                           table_count=len(self._schema_cache),
                           total_contexts=len(contexts))
            
        except Exception as e:
            self.logger.error("refresh_schema_cache_failed", error=repr(e), exc_info=True)
            raise
    
    def get_cached_schema(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Get cached schema information."""
        if table_name:
            return self._schema_cache.get(table_name, {})
        else:
            return self._schema_cache.copy()
    
    def is_cache_stale(self, max_age_minutes: int = 30) -> bool:
        """Check if the schema cache is stale."""
        if not self._last_update:
            return True
        
        age_minutes = (datetime.utcnow() - self._last_update).total_seconds() / 60
        return age_minutes > max_age_minutes


# Global schema processor instance
schema_processor = SchemaProcessor() 