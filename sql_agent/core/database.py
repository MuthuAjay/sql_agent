"""Advanced Database management for SQL Agent with enterprise-grade schema extraction."""

import time
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib
import json

from sqlalchemy import create_engine, text, MetaData, inspect
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from .config import settings
from .state import QueryResult
from sql_agent.core.models import Table


@dataclass
class TableStatistics:
    """Advanced table statistics for optimization."""
    row_count: int
    size_bytes: int
    last_vacuum: Optional[datetime]
    last_analyze: Optional[datetime]
    index_usage: Dict[str, float]
    column_cardinality: Dict[str, int]
    null_percentages: Dict[str, float]


@dataclass
class RelationshipInfo:
    """Enhanced relationship information."""
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    relationship_type: str  # 'foreign_key', 'implicit', 'inferred'
    cardinality: str  # '1:1', '1:many', 'many:many'
    join_selectivity: Optional[float]
    confidence: float


@dataclass
class SchemaChangeInfo:
    """Schema change tracking information."""
    table_name: str
    change_type: str  # 'added', 'modified', 'deleted'
    change_details: Dict[str, Any]
    timestamp: datetime
    checksum_before: Optional[str]
    checksum_after: Optional[str]


class DatabaseManager:
    """Advanced database manager with enterprise-grade schema extraction."""
    
    def __init__(self):
        self._engine: Optional[Engine] = None
        self._async_engine = None
        self._session_factory = None
        self._metadata: Optional[MetaData] = None
        
        # Advanced features
        self._schema_cache: Dict[str, Dict[str, Any]] = {}
        self._schema_checksums: Dict[str, str] = {}
        self._last_schema_update: Optional[datetime] = None
        self._connection_pool = None
        
        # Performance optimization
        self.config = {
            "max_concurrent_queries": 10,
            "schema_cache_ttl": 1800,  # 30 minutes
            "batch_size": 50,
            "enable_parallel_extraction": True,
            "enable_statistics_collection": True,
            "enable_relationship_inference": True
        }
    
    async def initialize(self) -> None:
        """Initialize database connection with advanced features."""
        start_time = time.time()
        try:
            # Create async engine with optimized settings
            self._async_engine = create_async_engine(
                settings.database_url,
                pool_size=settings.database_pool_size or 20,
                max_overflow=settings.database_max_overflow or 30,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=settings.debug,
            )
            
            # Create session factory
            self._session_factory = async_sessionmaker(
                self._async_engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            
            # Test connection and initialize features
            await self._test_and_setup_connection()
            
        except SQLAlchemyError as e:
            return QueryResult(
                data=[],
                columns=[],
                row_count=0,
                execution_time=time.time() - start_time,
                sql_query="Initialization failed",
                error=str(e),
            )
    
    async def get_schema_info(self) -> Dict[str, Any]:
        """Legacy method - redirects to advanced schema extraction."""
        return await self.get_database_schema()
    
    async def validate_query(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Validate a SQL query without executing it."""
        try:
            sql_lower = sql.lower().strip()
            
            # Check for dangerous operations
            dangerous_keywords = [
                "drop", "delete", "truncate", "alter", "create", "insert", "update"
            ]
            
            for keyword in dangerous_keywords:
                if keyword in sql_lower:
                    return False, f"Query contains potentially dangerous keyword: {keyword}"
            
            # Check for basic SQL syntax
            if not sql_lower.startswith("select"):
                return False, "Only SELECT queries are allowed"
            
            # Advanced validation could be added here
            # - Parse SQL AST for deeper validation
            # - Check table/column existence against schema
            # - Validate permissions
            
            return True, None
            
        except Exception as e:
            return False, f"Query validation failed: {e}"
    
    async def get_connection(self, database_name: Optional[str] = None) -> Any:
        """Get database connection for direct usage."""
        if not self._async_engine:
            raise RuntimeError("Database not initialized")
        return self._async_engine.begin()
    
    async def get_tables(self) -> List[dict]:
        """Get simplified table list for backward compatibility."""
        schema_data = await self.get_database_schema()
        tables = []
        
        for table_data in schema_data.get("tables", []):
            tables.append({
                "name": table_data["name"],
                "type": table_data.get("type", "table").lower(),
                "schema": "public",
                "rowCount": table_data.get("statistics", {}).get("live_tuples"),
                "size": table_data.get("statistics", {}).get("total_size_bytes"),
                "description": table_data.get("description"),
                "lastDescriptionUpdate": None
            })
        
        return tables
    
    async def get_table_schema(self, table_name: str) -> dict:
        """Get detailed schema for a specific table."""
        schema_data = await self.get_database_schema()
        
        # Find the specific table
        table_data = None
        for table in schema_data.get("tables", []):
            if table["name"] == table_name:
                table_data = table
                break
        
        if not table_data:
            raise ValueError(f"Table {table_name} not found")
        
        # Convert to legacy format
        columns = []
        for column_name in table_data.get("columns", []):
            column_detail = table_data.get("column_details", {}).get(column_name, {})
            
            # Check if it's a primary key
            is_primary_key = any(
                pk.get("column") == column_name 
                for pk in table_data.get("primary_keys", [])
            )
            
            # Check if it's a foreign key
            foreign_key_info = None
            for fk in table_data.get("foreign_keys", []):
                if fk.get("column") == column_name:
                    foreign_key_info = {
                        "referencedTable": fk.get("references_table"),
                        "referencedColumn": fk.get("references_column")
                    }
                    break
            
            columns.append({
                "name": column_name,
                "type": column_detail.get("type", "unknown"),
                "nullable": column_detail.get("nullable", True),
                "primaryKey": is_primary_key,
                "foreignKey": foreign_key_info,
                "defaultValue": column_detail.get("default"),
                "constraints": []
            })
        
        # Convert indexes
        indexes = []
        for index_data in table_data.get("indexes", []):
            indexes.append({
                "name": index_data["name"],
                "columns": index_data["columns"],
                "unique": index_data.get("unique", False)
            })
        
        # Convert foreign keys
        foreign_keys = []
        for fk in table_data.get("foreign_keys", []):
            foreign_keys.append({
                "columnName": fk["column"],
                "referencedTable": fk["references_table"],
                "referencedColumn": fk["references_column"]
            })
        
        return {
            "tableName": table_name,
            "columns": columns,
            "indexes": indexes,
            "foreignKeys": foreign_keys
        }
    
    async def get_sample_data(self, table_name: str, limit: int = 5) -> dict:
        """Get sample data from a table."""
        if not self._async_engine:
            raise RuntimeError("Async engine is not initialized.")
        
        async with self._async_engine.begin() as conn:
            sql = f"SELECT * FROM {table_name} LIMIT {limit}"
            result = await conn.execute(text(sql))
            rows = result.fetchall()
            columns = result.keys()
            
            return {
                "columns": list(columns),
                "rows": [list(row) for row in rows]
            }
    
    async def list_tables(self, database_id: str) -> List[Table]:
        """List tables for backward compatibility."""
        schema_data = await self.get_database_schema()
        tables = []
        
        for table_data in schema_data.get("tables", []):
            tables.append(Table(
                name=table_data["name"], 
                type=table_data.get("type", "table")
            ))
        
        return tables
    
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._async_engine is not None
    
    async def test_connection(self) -> bool:
        """Test database connection."""
        try:
            if not self._async_engine:
                return False
            async with self._async_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
    
    # Advanced methods for Phase 3
    
    async def get_performance_insights(self, database_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance insights for database optimization."""
        schema_data = await self.get_database_schema(database_name)
        return schema_data.get("performance_insights", {})
    
    async def get_business_domains(self, database_name: Optional[str] = None) -> Dict[str, List[str]]:
        """Get business domain classification of tables."""
        schema_data = await self.get_database_schema(database_name)
        return schema_data.get("business_domains", {})
    
    async def get_quality_metrics(self, database_name: Optional[str] = None) -> Dict[str, Any]:
        """Get schema quality metrics."""
        schema_data = await self.get_database_schema(database_name)
        return schema_data.get("quality_metrics", {})
    
    async def optimize_schema_extraction(self, enable_parallel: bool = True) -> None:
        """Optimize schema extraction settings."""
        self.config["enable_parallel_extraction"] = enable_parallel
        
        if enable_parallel:
            self.config["max_concurrent_queries"] = min(20, self.config["max_concurrent_queries"])
        else:
            self.config["max_concurrent_queries"] = 1
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about schema extraction performance."""
        return {
            "cache_size": len(self._schema_cache),
            "cache_ttl": self.config["schema_cache_ttl"],
            "last_update": self._last_schema_update.isoformat() if self._last_schema_update else None,
            "config": self.config,
            "parallel_extraction": self.config["enable_parallel_extraction"],
            "statistics_collection": self.config["enable_statistics_collection"]
        }
    
    async def refresh_schema_cache(self, database_name: Optional[str] = None) -> None:
        """Force refresh of schema cache."""
        cache_key = database_name or "default"
        
        if cache_key in self._schema_cache:
            del self._schema_cache[cache_key]
        
        if cache_key in self._schema_checksums:
            del self._schema_checksums[cache_key]
        
        # Trigger fresh extraction
        await self.get_database_schema(database_name)
    
    async def export_schema_metadata(self, database_name: Optional[str] = None) -> Dict[str, Any]:
        """Export complete schema metadata for backup or analysis."""
        schema_data = await self.get_database_schema(database_name)
        
        return {
            "export_timestamp": datetime.utcnow().isoformat(),
            "database_name": database_name or "default",
            "schema_data": schema_data,
            "extraction_config": self.config,
            "checksum": self._schema_checksums.get(database_name or "default")
        }

    
    async def _test_and_setup_connection(self) -> None:
        """Test connection and setup advanced features."""
        async with self._async_engine.begin() as conn:
            # Test basic connectivity
            await conn.execute(text("SELECT 1"))
            
            # Setup performance monitoring if supported
            try:
                await conn.execute(text("SELECT * FROM pg_stat_activity LIMIT 1"))
                self.config["supports_pg_stats"] = True
            except:
                self.config["supports_pg_stats"] = False
            
            # Check for advanced PostgreSQL features
            try:
                await conn.execute(text("SELECT * FROM pg_stat_user_tables LIMIT 1"))
                self.config["supports_table_stats"] = True
            except:
                self.config["supports_table_stats"] = False
    
    async def close(self) -> None:
        """Close database connections and cleanup."""
        if self._async_engine:
            await self._async_engine.dispose()
        self._schema_cache.clear()
        self._schema_checksums.clear()
    
    async def get_database_schema(self, database_name: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive database schema with advanced metadata."""
        try:
            # Check cache first
            cache_key = database_name or "default"
            if self._is_schema_cache_valid(cache_key):
                return self._schema_cache[cache_key]
            
            # Extract schema with parallel processing
            if self.config["enable_parallel_extraction"]:
                schema_data = await self._extract_schema_parallel()
            else:
                schema_data = await self._extract_schema_sequential()
            
            # Enhance with advanced metadata
            enhanced_schema = await self._enhance_schema_with_metadata(schema_data)
            
            # Cache the result
            self._cache_schema(cache_key, enhanced_schema)
            
            return enhanced_schema
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract database schema: {e}")
    
    async def _extract_schema_parallel(self) -> Dict[str, Any]:
        """Extract schema using parallel processing for performance."""
        async with self._async_engine.begin() as conn:
            # Get table list first
            tables_query = """
                SELECT 
                    t.table_name,
                    t.table_type,
                    COALESCE(obj_description(c.oid), '') as table_comment,
                    COALESCE(n_tup_ins, 0) as inserts,
                    COALESCE(n_tup_upd, 0) as updates,
                    COALESCE(n_tup_del, 0) as deletes,
                    COALESCE(n_live_tup, 0) as live_tuples
                FROM information_schema.tables t
                LEFT JOIN pg_class c ON c.relname = t.table_name AND c.relkind = 'r'
                LEFT JOIN pg_stat_user_tables s ON s.relname = t.table_name
                WHERE t.table_schema = 'public' 
                AND t.table_type = 'BASE TABLE'
                ORDER BY t.table_name
            """
            
            result = await conn.execute(text(tables_query))
            table_rows = result.fetchall()
            
            # Process tables in batches for parallel processing
            table_batches = [
                table_rows[i:i + self.config["batch_size"]] 
                for i in range(0, len(table_rows), self.config["batch_size"])
            ]
            
            # Create semaphore to limit concurrent operations
            semaphore = asyncio.Semaphore(self.config["max_concurrent_queries"])
            
            # Process batches concurrently
            tasks = [
                self._process_table_batch(batch, semaphore) 
                for batch in table_batches
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            all_tables = {}
            for batch_result in batch_results:
                if isinstance(batch_result, dict):
                    all_tables.update(batch_result)

            print("**************\n all table data \n", all_tables)

            return {
                "database_name": "sql_agent_db",
                "tables": list(all_tables.values()),
                "extraction_method": "parallel",
                "extraction_timestamp": datetime.utcnow().isoformat(),
                "table_count": len(all_tables),
                "total_columns": sum(len(t.get("columns", [])) for t in all_tables.values())
            }
    
    async def _process_table_batch(self, table_batch: List, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Process a batch of tables concurrently."""
        async with semaphore:
            tables_data = {}
            
            # Create new connection for this batch
            async with self._async_engine.begin() as conn:
                for table_row in table_batch:
                    try:
                        table_name = table_row[0]
                        table_data = await self._extract_single_table_data(conn, table_row)
                        tables_data[table_name] = table_data
                    except Exception as e:
                        # Log error but continue processing other tables
                        print(f"Failed to process table {table_row[0]}: {e}")
                        continue

            print("**************\n table_data - _process_table_batch \n", tables_data)

            return tables_data
    
    async def _extract_single_table_data(self, conn, table_row) -> Dict[str, Any]:
        """Extract comprehensive data for a single table."""
        table_name = table_row[0]
        table_type = table_row[1]
        table_comment = table_row[2]
        
        # Base table information
        table_data = {
            "name": table_name,
            "type": table_type,
            "description": table_comment or self._generate_table_description(table_name),
            "columns": [],
            "column_details": {},
            "primary_keys": [],
            "foreign_keys": [],
            "indexes": [],
            "statistics": {},
            "business_concepts": self._extract_business_concepts(table_name),
            "relationships": []
        }
        
        # Extract columns with enhanced metadata
        columns_data = await self._extract_table_columns(conn, table_name)
        table_data.update(columns_data)
        
        # Extract constraints and relationships
        constraints_data = await self._extract_table_constraints(conn, table_name)
        table_data.update(constraints_data)
        
        # Extract indexes
        indexes_data = await self._extract_table_indexes(conn, table_name)
        table_data["indexes"] = indexes_data
        
        # Extract statistics if enabled
        if self.config["enable_statistics_collection"]:
            stats_data = await self._extract_table_statistics(conn, table_name)
            table_data["statistics"] = stats_data
            
        print("**************\n table_data \n", table_data)
        
        return table_data
    
    async def _extract_table_columns(self, conn, table_name: str) -> Dict[str, Any]:
        """Extract detailed column information."""
        columns_query = """
            SELECT 
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
            WHERE cols.table_name = :table_name
            AND cols.table_schema = 'public'
            ORDER BY cols.ordinal_position
        """
        print("Extracting columns for table:", table_name)
        result = await conn.execute(text(columns_query), {"table_name": table_name})
        column_rows = result.fetchall()
        
        columns = []
        column_details = {}
        
        for row in column_rows:
            column_name = row[0]
            data_type = row[1]
            
            columns.append(column_name)
            
            column_details[column_name] = {
                "name": column_name,
                "type": data_type,
                "nullable": row[2] == "YES",
                "default": row[3],
                "max_length": row[4],
                "precision": row[5],
                "scale": row[6],
                "position": row[7],
                "comment": row[8] or self._generate_column_description(column_name, data_type),
                "business_concept": self._extract_column_business_concept(column_name)
            }
        
        return {
            "columns": columns,
            "column_details": column_details
        }
    
    async def _extract_table_constraints(self, conn, table_name: str) -> Dict[str, Any]:
        """Extract primary keys and foreign keys."""
        # Primary keys
        pk_query = """
            SELECT kcu.column_name, tc.constraint_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu 
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
            AND tc.table_name = :table_name
            AND tc.table_schema = 'public'
        """
        
        result = await conn.execute(text(pk_query), {"table_name": table_name})
        pk_rows = result.fetchall()
        
        primary_keys = [
            {"column": row[0], "constraint_name": row[1]}
            for row in pk_rows
        ]
        
        # Foreign keys
        fk_query = """
            SELECT
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
            AND tc.table_name = :table_name
            AND tc.table_schema = 'public'
        """
        
        result = await conn.execute(text(fk_query), {"table_name": table_name})
        fk_rows = result.fetchall()
        
        foreign_keys = [
            {
                "column": row[0],
                "references_table": row[1],
                "references_column": row[2],
                "constraint_name": row[3]
            }
            for row in fk_rows
        ]
        
        return {
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys
        }
    
    async def _extract_table_indexes(self, conn, table_name: str) -> List[Dict[str, Any]]:
        """Extract index information."""
        indexes_query = """
            SELECT
                indexname,
                indexdef,
                indisunique,
                indisprimary
            FROM pg_indexes pi
            JOIN pg_class pc ON pc.relname = pi.indexname
            JOIN pg_index pgi ON pgi.indexrelid = pc.oid
            WHERE pi.tablename = :table_name
            AND pi.schemaname = 'public'
        """
        
        result = await conn.execute(text(indexes_query), {"table_name": table_name})
        index_rows = result.fetchall()
        
        indexes = []
        for row in index_rows:
            index_name = row[0]
            index_def = row[1]
            is_unique = row[2]
            is_primary = row[3]
            
            # Parse columns from index definition
            columns = self._parse_index_columns(index_def)
            
            indexes.append({
                "name": index_name,
                "columns": columns,
                "unique": is_unique,
                "primary": is_primary,
                "definition": index_def
            })
        
        return indexes
    
    async def _extract_table_statistics(self, conn, table_name: str) -> Dict[str, Any]:
        """Extract table statistics for optimization."""
        if not self.config.get("supports_table_stats", False):
            return {}
        
        stats_query = """
            SELECT 
                n_tup_ins,
                n_tup_upd,
                n_tup_del,
                n_live_tup,
                n_dead_tup,
                last_vacuum,
                last_autovacuum,
                last_analyze,
                last_autoanalyze,
                pg_total_relation_size(quote_ident($1)::regclass) as total_size
            FROM pg_stat_user_tables 
            WHERE relname = :table_name
        """
        
        try:
            result = await conn.execute(text(stats_query), {"table_name": table_name})
            row = result.fetchone()
            
            if row:
                return {
                    "inserts": row[0] or 0,
                    "updates": row[1] or 0,
                    "deletes": row[2] or 0,
                    "live_tuples": row[3] or 0,
                    "dead_tuples": row[4] or 0,
                    "last_vacuum": row[5],
                    "last_autovacuum": row[6],
                    "last_analyze": row[7],
                    "last_autoanalyze": row[8],
                    "total_size_bytes": row[9] or 0
                }
        except Exception as e:
            print(f"Failed to get statistics for {table_name}: {e}")
        
        return {}
    
    async def _enhance_schema_with_metadata(self, schema_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance schema with advanced metadata."""
        enhanced_schema = schema_data.copy()
        
        # Add relationship inference
        if self.config["enable_relationship_inference"]:
            inferred_relationships = await self._infer_relationships(schema_data["tables"])
            enhanced_schema["inferred_relationships"] = inferred_relationships
        
        # Add business domain classification
        enhanced_schema["business_domains"] = self._classify_business_domains(schema_data["tables"])
        
        # Add schema quality metrics
        enhanced_schema["quality_metrics"] = self._calculate_schema_quality(schema_data["tables"])
        
        # Add performance insights
        enhanced_schema["performance_insights"] = self._analyze_performance_characteristics(schema_data["tables"])
        
        return enhanced_schema
    
    async def _infer_relationships(self, tables: List[Dict[str, Any]]) -> List[RelationshipInfo]:
        """Infer implicit relationships beyond foreign keys."""
        relationships = []
        
        # Create lookup for quick access
        table_lookup = {t["name"]: t for t in tables}
        
        for table in tables:
            table_name = table["name"]
            columns = table.get("columns", [])
            
            for column in columns:
                # Look for ID columns that might reference other tables
                if column.endswith("_id") and column != "id":
                    potential_table = column[:-3]  # Remove "_id"
                    
                    # Check if potential table exists (with variations)
                    candidates = [
                        potential_table,
                        potential_table + "s",  # plural
                        potential_table.rstrip("s"),  # singular
                    ]
                    
                    for candidate in candidates:
                        if candidate in table_lookup:
                            # Found potential relationship
                            relationships.append(RelationshipInfo(
                                source_table=table_name,
                                source_column=column,
                                target_table=candidate,
                                target_column="id",  # Assume primary key is 'id'
                                relationship_type="inferred",
                                cardinality="many:1",  # Most common pattern
                                join_selectivity=None,
                                confidence=0.8
                            ))
                            break
        
        return relationships
    
    def _classify_business_domains(self, tables: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Classify tables into business domains."""
        domain_keywords = {
            "customer_management": ["customer", "client", "user", "account", "subscriber"],
            "product_catalog": ["product", "item", "inventory", "catalog", "merchandise"],
            "order_processing": ["order", "transaction", "purchase", "sale", "payment"],
            "financial": ["revenue", "profit", "cost", "budget", "finance", "accounting"],
            "hr_management": ["employee", "staff", "hr", "personnel", "payroll"],
            "marketing": ["campaign", "promotion", "lead", "conversion", "marketing"],
            "operations": ["logistics", "warehouse", "shipping", "supply", "operations"]
        }
        
        domain_classification = {}
        
        for domain, keywords in domain_keywords.items():
            domain_tables = []
            
            for table in tables:
                table_name = table["name"].lower()
                
                if any(keyword in table_name for keyword in keywords):
                    domain_tables.append(table["name"])
            
            if domain_tables:
                domain_classification[domain] = domain_tables
        
        return domain_classification
    
    def _calculate_schema_quality(self, tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate schema quality metrics."""
        total_tables = len(tables)
        tables_with_pk = sum(1 for t in tables if t.get("primary_keys"))
        tables_with_fk = sum(1 for t in tables if t.get("foreign_keys"))
        tables_with_descriptions = sum(1 for t in tables if t.get("description", "").strip())
        total_columns = sum(len(t.get("columns", [])) for t in tables)
        
        return {
            "total_tables": total_tables,
            "pk_coverage": tables_with_pk / total_tables if total_tables > 0 else 0,
            "fk_coverage": tables_with_fk / total_tables if total_tables > 0 else 0,
            "documentation_coverage": tables_with_descriptions / total_tables if total_tables > 0 else 0,
            "avg_columns_per_table": total_columns / total_tables if total_tables > 0 else 0,
            "relationship_density": sum(len(t.get("foreign_keys", [])) for t in tables) / total_tables if total_tables > 0 else 0
        }
    
    def _analyze_performance_characteristics(self, tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance characteristics of the schema."""
        large_tables = []
        tables_without_indexes = []
        tables_with_many_columns = []
        
        for table in tables:
            table_name = table["name"]
            stats = table.get("statistics", {})
            indexes = table.get("indexes", [])
            columns = table.get("columns", [])
            
            # Identify large tables
            live_tuples = stats.get("live_tuples", 0)
            if live_tuples > 100000:  # > 100K rows
                large_tables.append({
                    "table": table_name,
                    "row_count": live_tuples,
                    "size_bytes": stats.get("total_size_bytes", 0)
                })
            
            # Tables without proper indexing
            non_pk_indexes = [idx for idx in indexes if not idx.get("primary", False)]
            if not non_pk_indexes and live_tuples > 1000:
                tables_without_indexes.append(table_name)
            
            # Tables with many columns (potential normalization issues)
            if len(columns) > 20:
                tables_with_many_columns.append({
                    "table": table_name,
                    "column_count": len(columns)
                })
        
        return {
            "large_tables": large_tables,
            "tables_without_indexes": tables_without_indexes,
            "tables_with_many_columns": tables_with_many_columns,
            "performance_warnings": len(tables_without_indexes) + len(tables_with_many_columns)
        }
    
    # Utility methods
    
    def _generate_table_description(self, table_name: str) -> str:
        """Generate intelligent description for table based on name."""
        name_lower = table_name.lower()
        
        # Business entity patterns
        if "customer" in name_lower or "client" in name_lower:
            return "Customer information and account data"
        elif "product" in name_lower or "item" in name_lower:
            return "Product catalog and inventory information"
        elif "order" in name_lower:
            return "Customer orders and transaction data"
        elif "employee" in name_lower or "staff" in name_lower:
            return "Employee information and HR data"
        elif "user" in name_lower:
            return "User account and profile information"
        elif "payment" in name_lower or "transaction" in name_lower:
            return "Payment and financial transaction data"
        else:
            return f"Data table: {table_name}"
    
    def _generate_column_description(self, column_name: str, data_type: str) -> str:
        """Generate intelligent description for column."""
        name_lower = column_name.lower()
        
        if name_lower.endswith("_id") or name_lower == "id":
            return f"Unique identifier ({data_type})"
        elif "name" in name_lower:
            return f"Name or title field ({data_type})"
        elif "email" in name_lower:
            return f"Email address ({data_type})"
        elif any(word in name_lower for word in ["date", "time", "created", "updated"]):
            return f"Date/time field ({data_type})"
        elif any(word in name_lower for word in ["amount", "price", "cost", "total"]):
            return f"Monetary value ({data_type})"
        else:
            return f"Data field: {column_name} ({data_type})"
    
    def _extract_business_concepts(self, table_name: str) -> List[str]:
        """Extract business concepts from table name."""
        concepts = []
        name_lower = table_name.lower()
        
        business_concepts = {
            "customer_management": ["customer", "client", "account"],
            "product_catalog": ["product", "item", "inventory"],
            "order_processing": ["order", "purchase", "transaction"],
            "financial": ["payment", "invoice", "billing"],
            "hr_management": ["employee", "staff", "hr"],
            "logistics": ["shipping", "delivery", "warehouse"]
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
            "personal_info": ["name", "first_name", "last_name", "email"],
            "geographic": ["address", "city", "state", "country"],
            "temporal": ["date", "time", "created", "updated"],
            "financial": ["price", "amount", "cost", "total"],
            "status": ["status", "state", "active", "enabled"]
        }
        
        for concept, keywords in concept_mapping.items():
            if any(keyword in name_lower for keyword in keywords):
                return concept
        
        return None
    
    def _parse_index_columns(self, index_def: str) -> List[str]:
        """Parse column names from index definition."""
        try:
            # Extract content between parentheses
            start = index_def.find("(")
            end = index_def.rfind(")")
            
            if start != -1 and end != -1:
                columns_str = index_def[start+1:end]
                # Remove quotes and split by comma
                columns = [col.strip().strip('"') for col in columns_str.split(",")]
                return columns
        except Exception:
            pass
        
        return []
    
    def _is_schema_cache_valid(self, cache_key: str) -> bool:
        """Check if schema cache is still valid."""
        if cache_key not in self._schema_cache:
            return False
        
        if not self._last_schema_update:
            return False
        
        # Check TTL
        cache_age = (datetime.utcnow() - self._last_schema_update).total_seconds()
        return cache_age < self.config["schema_cache_ttl"]
    
    def _cache_schema(self, cache_key: str, schema_data: Dict[str, Any]) -> None:
        """Cache schema data with checksum."""
        self._schema_cache[cache_key] = schema_data
        self._last_schema_update = datetime.utcnow()
        
        # Calculate checksum for change detection
        schema_str = json.dumps(schema_data, sort_keys=True, default=str)
        checksum = hashlib.md5(schema_str.encode()).hexdigest()
        self._schema_checksums[cache_key] = checksum
    
    async def detect_schema_changes(self, database_name: Optional[str] = None) -> List[SchemaChangeInfo]:
        """Detect changes in database schema."""
        cache_key = database_name or "default"
        
        # Get current schema
        current_schema = await self.get_database_schema(database_name)
        current_checksum = self._schema_checksums.get(cache_key)
        
        # Calculate new checksum
        schema_str = json.dumps(current_schema, sort_keys=True, default=str)
        new_checksum = hashlib.md5(schema_str.encode()).hexdigest()
        
        changes = []
        
        if current_checksum and current_checksum != new_checksum:
            # Schema has changed - detailed analysis would go here
            changes.append(SchemaChangeInfo(
                table_name="SCHEMA",
                change_type="modified",
                change_details={"checksum_changed": True},
                timestamp=datetime.utcnow(),
                checksum_before=current_checksum,
                checksum_after=new_checksum
            ))
        
        return changes
    
    # Legacy methods for backward compatibility
    
    async def execute_query(
        self, 
        sql: str, 
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> QueryResult:
        """Execute a SQL query and return results."""
        start_time = time.time()
        
        try:
            if not self._session_factory:
                raise RuntimeError("Session factory is not initialized.")
            
            async with self._session_factory() as session:
                if timeout:
                    await session.execute(text(f"SET statement_timeout = {timeout * 1000}"))
                
                result = await session.execute(text(sql), parameters or {})
                rows = result.mappings().all()
                columns = list(result.keys()) if result.keys() else []
                data = [dict(row) for row in rows]
                
                return QueryResult(
                    data=data,
                    columns=columns,
                    row_count=len(data),
                    execution_time=time.time() - start_time,
                    sql_query=sql,
                )
        
        except SQLAlchemyError as e:
            raise RuntimeError(f"SQL execution failed: {e}")
        
        
# Global database manager instance
db_manager = DatabaseManager()