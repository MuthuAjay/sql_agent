"""
Enhanced schema processing for RAG functionality with dynamic database introspection.

This module provides dynamic schema extraction, vectorization, and context management
for intelligent table selection in the SQL Agent system.

FIXED: Updated to use SQLAlchemy async pattern instead of cursor-based connections.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json

from sqlalchemy import text
from ..core.database import db_manager
from ..core.state import SchemaContext
from ..utils.logging import get_logger


class SchemaProcessor:
    """Enhanced schema processor with dynamic database introspection and vectorization."""
    
    def __init__(self):
        self.logger = get_logger("rag.schema_processor")
        self._schema_cache: Dict[str, Dict[str, Any]] = {}
        self._vector_cache: Dict[str, List[float]] = {}
        self._last_update: Optional[datetime] = None
        
        # Enhanced configuration
        self.config = {
            "cache_ttl_minutes": 30,
            "sample_data_limit": 5,
            "max_sample_value_length": 50,
            "max_description_length": 500,
            "include_statistics": True,
            "extract_business_concepts": True
        }
    
    async def extract_database_schema(self, database_name: str) -> Dict[str, Any]:
        """
        Dynamically extract complete database schema with metadata.
        
        This is the main method that orchestrator will call to get schema information.
        """
        try:
            self.logger.info("extracting_database_schema", database=database_name)
            
            # Check cache first
            cache_key = f"schema_{database_name}"
            if cache_key in self._schema_cache and not self._is_cache_stale(cache_key):
                self.logger.debug("schema_cache_hit", database=database_name)
                return self._schema_cache[cache_key]
            
            # FIXED: Use main database manager instead of separate introspection
            try:
                schema_data = await db_manager.get_database_schema(database_name)
                self.logger.info("schema_from_main_db_manager", database=database_name, 
                               table_count=len(schema_data.get("tables", [])))
            except Exception as e:
                self.logger.warning("main_db_manager_failed", error=str(e))
                # Fallback to introspection if main DB manager fails
                schema_data = await self._introspect_database_schema_fixed(database_name)
            
            # Enhance with business context
            enhanced_schema = await self._enhance_schema_with_context(schema_data)
            
            # Cache the result
            self._schema_cache[cache_key] = enhanced_schema
            
            self.logger.info(
                "database_schema_extracted",
                database=database_name,
                table_count=len(enhanced_schema.get("tables", [])),
                extraction_method="main_db_manager_with_enhancement"
            )
            
            return enhanced_schema
            
        except Exception as e:
            self.logger.error("extract_database_schema_failed", database=database_name, error=str(e), exc_info=True)
            # Return empty but valid schema structure
            return {
                "database_name": database_name,
                "tables": [],
                "extraction_timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    async def _introspect_database_schema_fixed(self, database_name: str) -> Dict[str, Any]:
        """
        FIXED: Perform dynamic database introspection using SQLAlchemy async pattern.
        This replaces the broken cursor-based approach.
        """
        try:
            self.logger.info("using_fixed_database_introspection", database=database_name)
            
            # FIXED: Use SQLAlchemy async pattern instead of cursor
            async with db_manager._async_engine.begin() as conn:
                
                # Tables query
                tables_query = text("""
                    SELECT 
                        t.table_name,
                        COALESCE(obj_description(c.oid), '') as table_comment,
                        t.table_type
                    FROM information_schema.tables t
                    LEFT JOIN pg_class c ON c.relname = t.table_name AND c.relkind = 'r'
                    WHERE t.table_schema = 'public' 
                    AND t.table_type = 'BASE TABLE'
                    ORDER BY t.table_name
                """)
                
                # Columns query
                columns_query = text("""
                    SELECT 
                        cols.table_name,
                        cols.column_name,
                        cols.data_type,
                        cols.is_nullable,
                        cols.column_default,
                        cols.character_maximum_length,
                        cols.numeric_precision,
                        cols.numeric_scale,
                        cols.ordinal_position,
                        COALESCE(col_description(pgc.oid, cols.ordinal_position), '') as column_comment
                    FROM information_schema.columns cols
                    LEFT JOIN pg_class pgc ON pgc.relname = cols.table_name
                    WHERE cols.table_schema = 'public'
                    ORDER BY cols.table_name, cols.ordinal_position
                """)
                
                # Foreign keys query
                foreign_keys_query = text("""
                    SELECT
                        tc.table_name,
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name,
                        tc.constraint_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                        ON ccu.constraint_name = tc.constraint_name
                        AND ccu.table_schema = tc.table_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                    AND tc.table_schema = 'public'
                """)
                
                # Primary keys query
                primary_keys_query = text("""
                    SELECT
                        tc.table_name,
                        kcu.column_name,
                        tc.constraint_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                    WHERE tc.constraint_type = 'PRIMARY KEY'
                    AND tc.table_schema = 'public'
                """)
                
                # FIXED: Execute queries using SQLAlchemy async pattern
                # Get tables
                result = await conn.execute(tables_query)
                table_rows = result.fetchall()
                
                # Get columns  
                result = await conn.execute(columns_query)
                column_rows = result.fetchall()
                
                # Get foreign keys
                result = await conn.execute(foreign_keys_query)
                fk_rows = result.fetchall()
                
                # Get primary keys
                result = await conn.execute(primary_keys_query)
                pk_rows = result.fetchall()
                
                # Get table statistics (optional)
                stats_rows = []
                try:
                    table_stats_query = text("""
                        SELECT 
                            schemaname,
                            tablename,
                            n_tup_ins as inserts,
                            n_tup_upd as updates,
                            n_tup_del as deletes,
                            n_live_tup as live_tuples,
                            n_dead_tup as dead_tuples,
                            last_vacuum,
                            last_autovacuum,
                            last_analyze,
                            last_autoanalyze
                        FROM pg_stat_user_tables 
                        WHERE schemaname = 'public'
                    """)
                    result = await conn.execute(table_stats_query)
                    stats_rows = result.fetchall()
                except Exception as e:
                    self.logger.warning("table_stats_query_failed", error=str(e))
                    stats_rows = []
            
            # Process results into structured format
            schema_data = await self._process_introspection_results(
                database_name, table_rows, column_rows, fk_rows, pk_rows, stats_rows
            )
            
            self.logger.info("database_introspection_completed", 
                           database=database_name, 
                           table_count=len(schema_data.get("tables", [])))
            
            return schema_data
            
        except Exception as e:
            self.logger.error("database_introspection_failed", database=database_name, error=str(e), exc_info=True)
            raise
    
    async def _process_introspection_results(
        self, 
        database_name: str,
        table_rows: List[Tuple],
        column_rows: List[Tuple], 
        fk_rows: List[Tuple],
        pk_rows: List[Tuple],
        stats_rows: List[Tuple]
    ) -> Dict[str, Any]:
        """Process raw introspection results into structured schema."""
        
        tables_info = {}
        
        # Process tables
        for row in table_rows:
            table_name = row[0]
            table_comment = row[1] if len(row) > 1 else ""
            table_type = row[2] if len(row) > 2 else "BASE TABLE"
            
            tables_info[table_name] = {
                "name": table_name,
                "description": table_comment or self._generate_table_description(table_name),
                "type": table_type,
                "columns": [],
                "column_details": {},
                "primary_keys": [],
                "foreign_keys": [],
                "statistics": {},
                "business_concepts": self._extract_business_concepts(table_name),
                "sample_data": None
            }
        
        # Process columns
        for row in column_rows:
            table_name = row[0]
            column_name = row[1]
            data_type = row[2]
            is_nullable = row[3]
            column_default = row[4]
            max_length = row[5]
            numeric_precision = row[6]
            numeric_scale = row[7]
            ordinal_position = row[8]
            column_comment = row[9] if len(row) > 9 else ""
            
            if table_name in tables_info:
                tables_info[table_name]["columns"].append(column_name)
                
                column_detail = {
                    "name": column_name,
                    "type": data_type,
                    "nullable": is_nullable == "YES",
                    "default": column_default,
                    "max_length": max_length,
                    "precision": numeric_precision,
                    "scale": numeric_scale,
                    "position": ordinal_position,
                    "comment": column_comment or self._generate_column_description(column_name, data_type),
                    "business_concept": self._extract_column_business_concept(column_name)
                }
                
                tables_info[table_name]["column_details"][column_name] = column_detail
        
        # Process foreign keys
        for row in fk_rows:
            table_name = row[0]
            column_name = row[1]
            foreign_table = row[2]
            foreign_column = row[3]
            constraint_name = row[4]
            
            if table_name in tables_info:
                fk_info = {
                    "column": column_name,
                    "references_table": foreign_table,
                    "references_column": foreign_column,
                    "constraint_name": constraint_name
                }
                tables_info[table_name]["foreign_keys"].append(fk_info)
        
        # Process primary keys
        for row in pk_rows:
            table_name = row[0]
            column_name = row[1]
            constraint_name = row[2]
            
            if table_name in tables_info:
                tables_info[table_name]["primary_keys"].append({
                    "column": column_name,
                    "constraint_name": constraint_name
                })
        
        # Process statistics
        for row in stats_rows:
            schema_name = row[0]
            table_name = row[1]
            
            if table_name in tables_info:
                tables_info[table_name]["statistics"] = {
                    "inserts": row[2],
                    "updates": row[3], 
                    "deletes": row[4],
                    "live_tuples": row[5],
                    "dead_tuples": row[6],
                    "last_vacuum": row[7],
                    "last_autovacuum": row[8],
                    "last_analyze": row[9],
                    "last_autoanalyze": row[10]
                }
        
        # Add sample data for each table (using main database manager)
        for table_name in tables_info.keys():
            try:
                sample_data = await self._get_table_sample_data_fixed(table_name)
                tables_info[table_name]["sample_data"] = sample_data
            except Exception as e:
                self.logger.warning("get_sample_data_failed", table=table_name, error=str(e))
                tables_info[table_name]["sample_data"] = {}
        
        return {
            "database_name": database_name,
            "tables": list(tables_info.values()),
            "extraction_method": "fixed_introspection",
            "extraction_timestamp": datetime.utcnow().isoformat(),
            "table_count": len(tables_info),
            "total_columns": sum(len(t["columns"]) for t in tables_info.values())
        }
    
    async def _enhance_schema_with_context(self, schema_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance schema with business context and semantic information."""
        try:
            enhanced_tables = []
            
            for table in schema_data.get("tables", []):
                enhanced_table = table.copy()
                
                # Enhance table description with context
                enhanced_table["enhanced_description"] = await self._create_enhanced_table_description(enhanced_table)
                
                # Add semantic tags
                enhanced_table["semantic_tags"] = self._generate_semantic_tags(enhanced_table)
                
                # Add relationship insights
                enhanced_table["relationship_insights"] = self._analyze_table_relationships(enhanced_table)
                
                # Add data quality insights
                enhanced_table["data_quality"] = await self._assess_data_quality(enhanced_table)
                
                # ADDED: Create string versions for vector storage compatibility
                business_concepts = enhanced_table.get("business_concepts", [])
                if business_concepts:
                    enhanced_table["business_concepts_str"] = ",".join(business_concepts)
                
                semantic_tags = enhanced_table.get("semantic_tags", [])
                if semantic_tags:
                    enhanced_table["semantic_tags_str"] = ",".join(semantic_tags)
                
                columns = enhanced_table.get("columns", [])
                if columns:
                    enhanced_table["columns_str"] = ",".join(columns)
                
                enhanced_tables.append(enhanced_table)
            
            enhanced_schema = schema_data.copy()
            enhanced_schema["tables"] = enhanced_tables
            enhanced_schema["enhancement_timestamp"] = datetime.utcnow().isoformat()
            
            return enhanced_schema
            
        except Exception as e:
            self.logger.warning("schema_enhancement_failed", error=str(e))
            return schema_data  # Return original if enhancement fails
    
    async def _get_table_sample_data_fixed(self, table_name: str, limit: int = 3) -> Dict[str, Any]:
        """FIXED: Get sample data from a table using main database manager."""
        try:
            # Use main database manager's method
            sample_data = await db_manager.get_sample_data(table_name, limit)
            
            if not sample_data or not sample_data.get("rows"):
                return {"rows": [], "sample_count": 0}
            
            return {
                "rows": sample_data.get("rows", [])[:limit],
                "sample_count": len(sample_data.get("rows", [])),
                "columns": sample_data.get("columns", [])
            }
            
        except Exception as e:
            self.logger.warning("get_table_sample_data_failed", table=table_name, error=str(e))
            return {"rows": [], "sample_count": 0}
    
    # ADDED: Extract schema contexts method for context manager
    async def extract_schema_contexts(self, database_name: str = "default") -> List[SchemaContext]:
        """Extract schema contexts for RAG vector storage."""
        try:
            self.logger.info("extracting_schema_contexts", database=database_name)
            
            # Get database schema
            schema_data = await self.extract_database_schema(database_name)
            tables = schema_data.get("tables", [])
            
            contexts = []
            
            for table in tables:
                table_name = table.get("name", "")
                description = table.get("enhanced_description", table.get("description", ""))
                
                # Create table-level context
                table_context = SchemaContext(
                    table_name=table_name,
                    column_name=None,
                    data_type=None,
                    description=description,
                    sample_values=[],
                    relationships=[],
                    embedding=None
                )
                contexts.append(table_context)
                
                # Create column-level contexts
                column_details = table.get("column_details", {})
                for column_name, column_detail in column_details.items():
                    column_context = SchemaContext(
                        table_name=table_name,
                        column_name=column_name,
                        data_type=column_detail.get("type"),
                        description=column_detail.get("comment", ""),
                        sample_values=[],
                        relationships=[],
                        embedding=None
                    )
                    contexts.append(column_context)
            
            self.logger.info("schema_contexts_extracted", 
                           database=database_name, 
                           context_count=len(contexts))
            
            return contexts
            
        except Exception as e:
            self.logger.error("extract_schema_contexts_failed", database=database_name, error=str(e))
            return []
    
    # ADDED: Get schema summary method for context manager
    async def get_schema_summary(self, database_name: str = "default") -> Dict[str, Any]:
        """Get schema summary for context manager statistics."""
        try:
            schema_data = await self.extract_database_schema(database_name)
            tables = schema_data.get("tables", [])
            
            return {
                "database_name": database_name,
                "table_count": len(tables),
                "total_columns": sum(len(t.get("columns", [])) for t in tables),
                "tables_with_relationships": len([t for t in tables if t.get("foreign_keys")]),
                "extraction_timestamp": schema_data.get("extraction_timestamp")
            }
            
        except Exception as e:
            self.logger.error("get_schema_summary_failed", database=database_name, error=str(e))
            return {"database_name": database_name, "table_count": 0, "total_columns": 0}
    
    def _generate_table_description(self, table_name: str) -> str:
        """Generate intelligent description for table based on name."""
        name_lower = table_name.lower()
        
        # Common business entity patterns
        if "customer" in name_lower or "client" in name_lower:
            return "Customer information and account data"
        elif "product" in name_lower or "item" in name_lower:
            return "Product catalog and inventory information"
        elif "order" in name_lower and "item" not in name_lower:
            return "Customer orders and transaction data"
        elif "order" in name_lower and "item" in name_lower:
            return "Individual line items within customer orders"
        elif "employee" in name_lower or "staff" in name_lower:
            return "Employee information and HR data"
        elif "user" in name_lower:
            return "User account and profile information"
        elif "payment" in name_lower or "transaction" in name_lower:
            return "Payment and financial transaction data"
        elif "address" in name_lower:
            return "Address and location information"
        elif "category" in name_lower:
            return "Category and classification data"
        elif "audit" in name_lower or "log" in name_lower:
            return "Audit trail and system logging data"
        else:
            return f"Data table: {table_name}"
    
    def _generate_column_description(self, column_name: str, data_type: str) -> str:
        """Generate intelligent description for column based on name and type."""
        name_lower = column_name.lower()
        
        # ID patterns
        if name_lower.endswith("_id") or name_lower == "id":
            return f"Unique identifier ({data_type})"
        
        # Name patterns
        elif "name" in name_lower:
            return f"Name or title field ({data_type})"
        
        # Email patterns
        elif "email" in name_lower:
            return f"Email address ({data_type})"
        
        # Date/time patterns
        elif any(word in name_lower for word in ["date", "time", "created", "updated", "modified"]):
            return f"Date/time field ({data_type})"
        
        # Amount/price patterns
        elif any(word in name_lower for word in ["amount", "price", "cost", "total", "balance"]):
            return f"Monetary value ({data_type})"
        
        # Status patterns
        elif "status" in name_lower or "state" in name_lower:
            return f"Status or state indicator ({data_type})"
        
        # Count/quantity patterns
        elif any(word in name_lower for word in ["count", "quantity", "qty", "number"]):
            return f"Numeric count or quantity ({data_type})"
        
        else:
            return f"Data field: {column_name} ({data_type})"
    
    def _extract_business_concepts(self, table_name: str) -> List[str]:
        """Extract business concepts from table name."""
        concepts = []
        name_lower = table_name.lower()
        
        # Business domain concepts
        business_concepts = {
            "customer_management": ["customer", "client", "account"],
            "product_catalog": ["product", "item", "inventory", "catalog"],
            "order_processing": ["order", "purchase", "transaction"],
            "financial": ["payment", "invoice", "billing", "finance"],
            "hr_management": ["employee", "staff", "hr", "personnel"],
            "logistics": ["shipping", "delivery", "logistics", "warehouse"],
            "marketing": ["campaign", "promotion", "marketing"],
            "support": ["support", "ticket", "issue", "help"]
        }
        
        for concept_category, keywords in business_concepts.items():
            if any(keyword in name_lower for keyword in keywords):
                concepts.append(concept_category)
        
        return concepts
    
    def _extract_column_business_concept(self, column_name: str) -> Optional[str]:
        """Extract business concept from column name."""
        name_lower = column_name.lower()
        
        concept_mapping = {
            "identifier": ["id", "_id", "key"],
            "personal_info": ["name", "first_name", "last_name", "email", "phone"],
            "geographic": ["address", "city", "state", "country", "zip", "postal"],
            "temporal": ["date", "time", "created", "updated", "modified"],
            "financial": ["price", "amount", "cost", "total", "balance", "salary"],
            "status": ["status", "state", "active", "enabled", "flag"],
            "metrics": ["count", "quantity", "score", "rating", "performance"]
        }
        
        for concept, keywords in concept_mapping.items():
            if any(keyword in name_lower for keyword in keywords):
                return concept
        
        return None
    
    async def _create_enhanced_table_description(self, table: Dict[str, Any]) -> str:
        """Create enhanced description with context."""
        base_description = table.get("description", "")
        table_name = table.get("name", "")
        columns = table.get("columns", [])
        sample_data = table.get("sample_data", {})
        
        parts = [base_description] if base_description else [self._generate_table_description(table_name)]
        
        # Add column count
        parts.append(f"Contains {len(columns)} columns")
        
        # Add key columns info
        key_columns = [col for col in columns if any(keyword in col.lower() for keyword in ["id", "key", "name", "email"])]
        if key_columns:
            parts.append(f"Key columns: {', '.join(key_columns[:5])}")
        
        # Add sample data context
        if sample_data.get("sample_count", 0) > 0:
            parts.append(f"Contains {sample_data['sample_count']} sample records")
        
        
        return ". ".join(parts)
    
    def _generate_semantic_tags(self, table: Dict[str, Any]) -> List[str]:
        """Generate semantic tags for table."""
        tags = []
        table_name = table.get("name", "").lower()
        columns = [col.lower() for col in table.get("columns", [])]
        
        # Entity type tags
        if any(word in table_name for word in ["customer", "client", "user"]):
            tags.append("entity:person")
        if any(word in table_name for word in ["product", "item"]):
            tags.append("entity:product")
        if any(word in table_name for word in ["order", "transaction"]):
            tags.append("entity:transaction")
        
        # Data type tags
        if any("date" in col or "time" in col for col in columns):
            tags.append("temporal_data")
        if any("amount" in col or "price" in col for col in columns):
            tags.append("financial_data")
        if any("address" in col or "location" in col for col in columns):
            tags.append("geographic_data")
        
        # Relationship tags
        foreign_keys = table.get("foreign_keys", [])
        if foreign_keys:
            tags.append("has_relationships")
        
        return tags
    
    def _analyze_table_relationships(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze table relationships."""
        foreign_keys = table.get("foreign_keys", [])
        primary_keys = table.get("primary_keys", [])
        
        return {
            "outgoing_relationships": len(foreign_keys),
            "primary_key_count": len(primary_keys),
            "referenced_tables": [fk.get("references_table") for fk in foreign_keys],
            "relationship_strength": "high" if len(foreign_keys) > 2 else "medium" if len(foreign_keys) > 0 else "low"
        }
    
    async def _assess_data_quality(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data quality of table."""
        stats = table.get("statistics", {})
        columns = table.get("columns", [])
        
        quality_assessment = {
            "has_statistics": bool(stats),
            "column_count": len(columns),
            "has_primary_key": len(table.get("primary_keys", [])) > 0,
            "has_foreign_keys": len(table.get("foreign_keys", [])) > 0,
            "data_freshness": "unknown"
        }
        
        # Assess data freshness based on statistics
        if stats.get("last_analyze"):
            try:
                last_analyze = datetime.fromisoformat(str(stats["last_analyze"]))
                days_since_analyze = (datetime.utcnow() - last_analyze).days
                
                if days_since_analyze < 7:
                    quality_assessment["data_freshness"] = "fresh"
                elif days_since_analyze < 30:
                    quality_assessment["data_freshness"] = "moderate"
                else:
                    quality_assessment["data_freshness"] = "stale"
            except:
                pass
        
        return quality_assessment
    
    def _is_cache_stale(self, cache_key: str, ttl_minutes: Optional[int] = None) -> bool:
        """Check if cache entry is stale."""
        if cache_key not in self._schema_cache:
            return True
        
        ttl = ttl_minutes or self.config["cache_ttl_minutes"]
        cache_entry = self._schema_cache[cache_key]
        
        if "extraction_timestamp" in cache_entry:
            try:
                extraction_time = datetime.fromisoformat(cache_entry["extraction_timestamp"])
                age_minutes = (datetime.utcnow() - extraction_time).total_seconds() / 60
                return age_minutes > ttl
            except:
                return True
        
        return True
    
    # Public API methods for orchestrator integration
    
    async def get_database_schema(self, database_name: str) -> Dict[str, Any]:
        """Main method for orchestrator to get database schema."""
        return await self.extract_database_schema(database_name)
    
    async def search_relevant_tables(self, query: str, database_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for tables relevant to a natural language query."""
        try:
            # Get full schema
            schema = await self.extract_database_schema(database_name)
            tables = schema.get("tables", [])
            
            if not tables:
                return []
            
            # Score tables based on query relevance
            scored_tables = []
            query_lower = query.lower()
            
            for table in tables:
                score = self._calculate_query_table_relevance(query_lower, table)
                if score > 0:
                    scored_tables.append((table, score))
            
            # Sort by relevance and return top results
            scored_tables.sort(key=lambda x: x[1], reverse=True)
            return [table for table, score in scored_tables[:limit]]
            
        except Exception as e:
            self.logger.error("search_relevant_tables_failed", query=query[:100], error=str(e))
            return []
    
    def _calculate_query_table_relevance(self, query_lower: str, table: Dict[str, Any]) -> float:
        """Calculate relevance score between query and table."""
        score = 0.0
        
        table_name = table.get("name", "").lower()
        description = table.get("enhanced_description", table.get("description", "")).lower()
        columns = [col.lower() for col in table.get("columns", [])]
        semantic_tags = table.get("semantic_tags", [])
        
        # Table name relevance (highest weight)
        for word in query_lower.split():
            if word in table_name:
                score += 0.5
        
        # Description relevance
        for word in query_lower.split():
            if word in description:
                score += 0.3
        
        # Column name relevance
        for word in query_lower.split():
            matching_columns = [col for col in columns if word in col]
            score += 0.2 * len(matching_columns)
        
        # Semantic tag relevance
        for tag in semantic_tags:
            if any(word in tag for word in query_lower.split()):
                score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    async def refresh_schema_cache(self, database_name: Optional[str] = None) -> None:
        """Refresh schema cache for database."""
        try:
            if database_name:
                cache_key = f"schema_{database_name}"
                if cache_key in self._schema_cache:
                    del self._schema_cache[cache_key]
                await self.extract_database_schema(database_name)
            else:
                # Refresh all cached schemas
                self._schema_cache.clear()
                self._vector_cache.clear()
            
            self._last_update = datetime.utcnow()
            self.logger.info("schema_cache_refreshed", database=database_name)
            
        except Exception as e:
            self.logger.error("refresh_schema_cache_failed", database=database_name, error=str(e))
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get status of schema cache."""
        return {
            "cached_databases": list(self._schema_cache.keys()),
            "cache_size": len(self._schema_cache),
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "vector_cache_size": len(self._vector_cache)
        }
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self._schema_cache.clear()
            self._vector_cache.clear()
            self.logger.info("schema_processor_cleanup_complete")
        except Exception as e:
            self.logger.error("schema_processor_cleanup_failed", error=str(e))


# Global schema processor instance
schema_processor = SchemaProcessor()