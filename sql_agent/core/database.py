"""Advanced Database management for SQL Agent with LLM-powered intelligence."""

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
    """Advanced database manager with LLM-powered intelligence."""
    
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
        
        # LLM Intelligence integration
        self._schema_analyzer = None
        self._llm_analysis_enabled = False
        
        # Performance optimization
        self.config = {
            "max_concurrent_queries": 10,
            "schema_cache_ttl": 1800,  # 30 minutes
            "batch_size": 50,
            "enable_parallel_extraction": True,
            "enable_statistics_collection": True,
            "enable_relationship_inference": True,
            "enable_llm_intelligence": True,  # NEW: LLM intelligence toggle
            "llm_fallback_enabled": True      # NEW: Graceful degradation
        }

        # Fraud detection query tracking
        self._fraud_query_log: List[Dict[str, Any]] = []
        self._fraud_query_stats = {
            "total_fraud_queries": 0,
            "successful_detections": 0,
            "failed_detections": 0,
            "avg_detection_time": 0.0
        }
    
    async def initialize(self) -> None:
        """Initialize database connection with LLM intelligence."""
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
            
            # Initialize LLM intelligence
            await self._initialize_llm_intelligence()
            
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
    
    async def _initialize_llm_intelligence(self) -> None:
        """Initialize LLM intelligence capabilities."""
        try:
            if self.config["enable_llm_intelligence"] and settings.enable_business_intelligence:
                # Import schema analyzer (lazy loading to avoid circular imports)
                from ..rag.schema_analyzer import schema_analyzer
                
                self._schema_analyzer = schema_analyzer
                
                # Test LLM availability
                health_check = await self._schema_analyzer.health_check()
                self._llm_analysis_enabled = health_check.get("llm_available", False)
                
                print(f"LLM Intelligence initialized: {self._llm_analysis_enabled}")
                print(f"LLM Provider: {health_check.get('llm_provider', 'None')}")
                
            else:
                print("LLM Intelligence disabled by configuration")
                
        except Exception as e:
            print(f"Failed to initialize LLM intelligence: {e}")
            self._llm_analysis_enabled = False
            
            if not self.config["llm_fallback_enabled"]:
                raise RuntimeError(f"LLM intelligence required but unavailable: {e}")
    
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
                "data": [list(row) for row in rows]
            }

    async def get_column_statistics(self, table_name: str) -> Dict[str, Any]:
        """Get comprehensive statistics for all columns in a table including min, max, count, and distinct values."""
        if not self._async_engine:
            raise RuntimeError("Async engine is not initialized.")

        try:
            async with self._async_engine.begin() as conn:
                # Get column metadata first
                columns_query = text("""
                    SELECT
                        column_name,
                        data_type,
                        is_nullable,
                        character_maximum_length,
                        numeric_precision,
                        numeric_scale
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = :table_name
                    ORDER BY ordinal_position
                """)

                columns_result = await conn.execute(columns_query, {"table_name": table_name})
                columns = columns_result.fetchall()

                column_stats = {}

                # For each column, get statistics based on data type
                for col in columns:
                    col_name = col[0]
                    col_type = col[1].lower()
                    is_nullable = col[2] == 'YES'

                    try:
                        # Base statistics for all columns
                        stats_query = text(f"""
                            SELECT
                                COUNT(*) as total_count,
                                COUNT(DISTINCT "{col_name}") as distinct_count,
                                COUNT("{col_name}") as non_null_count
                            FROM {table_name}
                        """)

                        stats_result = await conn.execute(stats_query)
                        stats = stats_result.fetchone()

                        col_stats = {
                            "column_name": col_name,
                            "data_type": col[1],
                            "is_nullable": is_nullable,
                            "total_count": stats[0],
                            "distinct_count": stats[1],
                            "non_null_count": stats[2],
                            "null_count": stats[0] - stats[2]
                        }

                        # Add min/max for numeric and date columns
                        if any(t in col_type for t in ['int', 'numeric', 'decimal', 'float', 'double', 'real', 'money']):
                            minmax_query = text(f"""
                                SELECT
                                    MIN("{col_name}")::text as min_value,
                                    MAX("{col_name}")::text as max_value,
                                    AVG("{col_name}")::numeric as avg_value
                                FROM {table_name}
                                WHERE "{col_name}" IS NOT NULL
                            """)
                            minmax_result = await conn.execute(minmax_query)
                            minmax = minmax_result.fetchone()

                            if minmax:
                                col_stats["min_value"] = minmax[0]
                                col_stats["max_value"] = minmax[1]
                                col_stats["avg_value"] = float(minmax[2]) if minmax[2] is not None else None

                        elif any(t in col_type for t in ['date', 'timestamp', 'time']):
                            minmax_query = text(f"""
                                SELECT
                                    MIN("{col_name}")::text as min_value,
                                    MAX("{col_name}")::text as max_value
                                FROM {table_name}
                                WHERE "{col_name}" IS NOT NULL
                            """)
                            minmax_result = await conn.execute(minmax_query)
                            minmax = minmax_result.fetchone()

                            if minmax:
                                col_stats["min_value"] = minmax[0]
                                col_stats["max_value"] = minmax[1]

                        # Add sample values for all column types
                        sample_query = text(f"""
                            SELECT DISTINCT "{col_name}"
                            FROM {table_name}
                            WHERE "{col_name}" IS NOT NULL
                            LIMIT 5
                        """)
                        sample_result = await conn.execute(sample_query)
                        samples = sample_result.fetchall()
                        col_stats["sample_values"] = [str(s[0]) for s in samples]

                        column_stats[col_name] = col_stats

                    except Exception as col_error:
                        # If individual column fails, log and continue with basic info
                        column_stats[col_name] = {
                            "column_name": col_name,
                            "data_type": col[1],
                            "is_nullable": is_nullable,
                            "error": str(col_error)
                        }

                return column_stats

        except Exception as e:
            return {"error": f"Failed to get column statistics: {str(e)}"}
    
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
    
    # ==================== ENHANCED LLM-POWERED METHODS ====================
    
    async def get_performance_insights(self, database_name: Optional[str] = None) -> Dict[str, Any]:
        """Get LLM-enhanced performance insights for database optimization."""
        try:
            schema_data = await self.get_database_schema(database_name)
            
            # Use LLM intelligence if available
            if self._llm_analysis_enabled and self._schema_analyzer:
                # Get enhanced schema with LLM analysis
                enhanced_schema = schema_data.get("business_intelligence", {})
                llm_performance_insights = enhanced_schema.get("optimization_priorities", [])
                
                # Combine with traditional performance analysis
                traditional_insights = self._analyze_performance_characteristics(schema_data.get("tables", []))
                
                return {
                    "llm_insights": llm_performance_insights,
                    "traditional_analysis": traditional_insights,
                    "recommendations": self._generate_performance_recommendations(
                        traditional_insights, llm_performance_insights
                    ),
                    "intelligence_source": "llm_enhanced"
                }
            else:
                # Fallback to traditional analysis
                insights = self._analyze_performance_characteristics(schema_data.get("tables", []))
                return {
                    "traditional_analysis": insights,
                    "intelligence_source": "rule_based_fallback"
                }
                
        except Exception as e:
            print(f"Failed to get performance insights: {e}")
            return {"error": str(e), "intelligence_source": "error"}
    
    async def get_business_domains(self, database_name: Optional[str] = None) -> Dict[str, Any]:
        """Get LLM-discovered business domain classification."""
        try:
            schema_data = await self.get_database_schema(database_name)
            
            # Use LLM intelligence if available
            if self._llm_analysis_enabled and self._schema_analyzer:
                # Get LLM-discovered domains
                discovered_domains = await self._schema_analyzer.discover_business_domains(schema_data)
                
                # Format for API response
                domains_response = {}
                for domain in discovered_domains:
                    domain_tables = []
                    
                    # Find tables that belong to this domain
                    for table in schema_data.get("tables", []):
                        table_business_context = table.get("business_context", {})
                        table_processes = table_business_context.get("business_processes", [])
                        
                        # Check if table processes match domain processes
                        if any(process in domain.related_processes for process in table_processes):
                            domain_tables.append(table["name"])
                    
                    domains_response[domain.name] = {
                        "tables": domain_tables,
                        "description": domain.description,
                        "confidence": domain.confidence,
                        "evidence": domain.evidence,
                        "criticality": domain.criticality
                    }
                
                return {
                    "discovered_domains": domains_response,
                    "domain_count": len(discovered_domains),
                    "intelligence_source": "llm_discovery",
                    "business_purpose": schema_data.get("business_intelligence", {}).get("business_purpose"),
                    "industry_domain": schema_data.get("business_intelligence", {}).get("industry_domain")
                }
            else:
                # Fallback to rule-based classification
                traditional_domains = self._classify_business_domains_fallback(schema_data.get("tables", []))
                return {
                    "discovered_domains": traditional_domains,
                    "domain_count": len(traditional_domains),
                    "intelligence_source": "rule_based_fallback"
                }
                
        except Exception as e:
            print(f"Failed to get business domains: {e}")
            return {"error": str(e), "intelligence_source": "error"}
    
    async def get_quality_metrics(self, database_name: Optional[str] = None) -> Dict[str, Any]:
        """Get LLM-enhanced schema quality metrics."""
        try:
            schema_data = await self.get_database_schema(database_name)
            
            # Traditional quality metrics
            traditional_quality = self._calculate_schema_quality(schema_data.get("tables", []))
            
            # Use LLM intelligence if available
            if self._llm_analysis_enabled and self._schema_analyzer:
                # Get LLM confidence metrics
                business_intelligence = schema_data.get("business_intelligence", {})
                llm_confidence = business_intelligence.get("confidence_metrics", {})
                
                return {
                    "traditional_metrics": traditional_quality,
                    "llm_confidence_metrics": llm_confidence,
                    "overall_intelligence_score": llm_confidence.get("overall_confidence", 0.5),
                    "data_completeness": traditional_quality.get("documentation_coverage", 0),
                    "business_understanding": llm_confidence.get("database_confidence", 0.5),
                    "relationship_clarity": llm_confidence.get("relationship_confidence", 0.5),
                    "intelligence_source": "llm_enhanced"
                }
            else:
                # Enhanced traditional metrics
                return {
                    "traditional_metrics": traditional_quality,
                    "overall_quality_score": traditional_quality.get("documentation_coverage", 0.5),
                    "intelligence_source": "rule_based_fallback"
                }
                
        except Exception as e:
            print(f"Failed to get quality metrics: {e}")
            return {"error": str(e), "intelligence_source": "error"}
    
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
            "statistics_collection": self.config["enable_statistics_collection"],
            "llm_intelligence_enabled": self._llm_analysis_enabled,
            "llm_provider": getattr(self._schema_analyzer, "_llm_provider", None) is not None if self._schema_analyzer else False
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
            "checksum": self._schema_checksums.get(database_name or "default"),
            "llm_intelligence_enabled": self._llm_analysis_enabled
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
        """Get comprehensive database schema with LLM intelligence."""
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
            
            # Enhance with LLM intelligence or fallback to traditional
            enhanced_schema = await self._enhance_schema_with_intelligence(schema_data)
            
            # Cache the result
            self._cache_schema(cache_key, enhanced_schema)
            
            return enhanced_schema
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract database schema: {e}")
    
    async def _enhance_schema_with_intelligence(self, schema_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance schema with LLM intelligence or traditional metadata."""
        try:
            # Try LLM intelligence first
            if self._llm_analysis_enabled and self._schema_analyzer:
                print("Enhancing schema with LLM intelligence...")
                
                # Use schema analyzer for intelligent enhancement
                enhanced_schema = await self._schema_analyzer.enhance_existing_schema(schema_data)
                
                print(f"LLM enhancement complete. Discovered {len(enhanced_schema.get('business_intelligence', {}).get('discovered_domains', []))} domains")
                
                return enhanced_schema
            else:
                print("LLM unavailable, using traditional enhancement...")
                
                # Fallback to traditional enhancement
                enhanced_schema = await self._enhance_schema_with_metadata_fallback(schema_data)
                
                return enhanced_schema
                
        except Exception as e:
            print(f"LLM enhancement failed, falling back to traditional: {e}")
            
            # Always fallback to traditional enhancement
            return await self._enhance_schema_with_metadata_fallback(schema_data)
    
    async def _enhance_schema_with_metadata_fallback(self, schema_data: Dict[str, Any]) -> Dict[str, Any]:
        """Traditional metadata enhancement when LLM unavailable."""
        enhanced_schema = schema_data.copy()
        
        # Add relationship inference
        if self.config["enable_relationship_inference"]:
            inferred_relationships = await self._infer_relationships(schema_data["tables"])
            enhanced_schema["inferred_relationships"] = inferred_relationships
        
        # Add traditional business domain classification
        enhanced_schema["business_domains"] = self._classify_business_domains_fallback(schema_data["tables"])
        
        # Add schema quality metrics
        enhanced_schema["quality_metrics"] = self._calculate_schema_quality(schema_data["tables"])
        
        # Add performance insights
        enhanced_schema["performance_insights"] = self._analyze_performance_characteristics(schema_data["tables"])
        
        # Add traditional business concepts to tables
        for table in enhanced_schema["tables"]:
            table["business_concepts"] = self._extract_business_concepts_fallback(table["name"])
            table["rich_description"] = self._generate_table_description_fallback(table["name"])
            table["business_keywords_str"] = ",".join(table["business_concepts"])
        
        # Mark as traditional enhancement
        enhanced_schema["intelligence_metadata"] = {
            "enhancement_type": "traditional",
            "llm_available": False,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
        return enhanced_schema
    
    # ==================== ORIGINAL EXTRACTION METHODS (UNCHANGED) ====================
    
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

            return {
                "database_name": "sql_agent_db",
                "tables": list(all_tables.values()),
                "extraction_method": "parallel",
                "extraction_timestamp": datetime.utcnow().isoformat(),
                "table_count": len(all_tables),
                "total_columns": sum(len(t.get("columns", [])) for t in all_tables.values())
            }
    
    async def _extract_schema_sequential(self) -> Dict[str, Any]:
        """Sequential schema extraction fallback."""
        # Implementation would be similar to parallel but without concurrency
        # For brevity, redirect to parallel method
        return await self._extract_schema_parallel()
    
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
            "description": table_comment or None,  # Will be enhanced by LLM or fallback
            "columns": [],
            "column_details": {},
            "primary_keys": [],
            "foreign_keys": [],
            "indexes": [],
            "statistics": {},
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
            
        # Add sample data for LLM analysis
        try:
            sample_data = await self.get_sample_data(table_name, 3)
            table_data["sample_data"] = sample_data
        except Exception as e:
            print(f"Failed to get sample data for {table_name}: {e}")
            table_data["sample_data"] = {"columns": [], "rows": []}
        
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
                "comment": row[8] or None  # Will be enhanced by LLM or fallback
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
    
    # ==================== FALLBACK METHODS (NO HARDCODED DOMAINS) ====================
    
    def _classify_business_domains_fallback(self, tables: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Fallback: Return empty domains when LLM unavailable - NO hardcoded domains."""
        return {
            "note": "Business domain discovery requires LLM analysis",
            "fallback_message": "Enable LLM intelligence for dynamic domain discovery"
        }
    
    def _extract_business_concepts_fallback(self, table_name: str) -> List[str]:
        """Fallback: Return simple table name analysis - NO hardcoded concepts."""
        # Simple extraction based only on table name patterns
        concepts = []
        name_parts = table_name.lower().replace('_', ' ').split()
        
        # Just return meaningful words from table name
        meaningful_words = [word for word in name_parts if len(word) > 2 and word not in ['table', 'data', 'info']]
        
        return meaningful_words[:3] if meaningful_words else [table_name.lower()]
    
    def _generate_table_description_fallback(self, table_name: str) -> str:
        """Fallback: Simple description without business assumptions."""
        # Generic description based only on table name structure
        name_clean = table_name.replace('_', ' ').title()
        return f"Data table: {name_clean}"
    
    def _generate_performance_recommendations(
        self, 
        traditional_insights: Dict[str, Any], 
        llm_insights: List[str]
    ) -> List[str]:
        """Generate performance recommendations - LLM first, basic technical fallback."""
        recommendations = []
        
        # Prioritize LLM insights if available
        if llm_insights:
            recommendations.extend(llm_insights)
        
        # Add only technical recommendations (no business assumptions)
        if traditional_insights.get("tables_without_indexes"):
            recommendations.append("Consider adding indexes to large tables for better query performance")
        
        if traditional_insights.get("tables_with_many_columns"):
            recommendations.append("Review table structure for potential optimization opportunities")
        
        if traditional_insights.get("large_tables"):
            recommendations.append("Monitor query performance on large tables")
        
        return recommendations
    
    # ==================== ORIGINAL UTILITY METHODS (UNCHANGED) ====================
    
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
    
    # ==================== LEGACY METHODS FOR BACKWARD COMPATIBILITY ====================
    
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

    def log_fraud_query(
        self,
        table_name: str,
        detector_type: str,
        scenarios_found: int,
        execution_time: float,
        success: bool = True
    ) -> None:
        """Log fraud detection query for audit trail and analytics."""
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "table_name": table_name,
                "detector_type": detector_type,
                "scenarios_found": scenarios_found,
                "execution_time": execution_time,
                "success": success
            }

            self._fraud_query_log.append(log_entry)

            # Update statistics
            self._fraud_query_stats["total_fraud_queries"] += 1
            if success:
                self._fraud_query_stats["successful_detections"] += 1
            else:
                self._fraud_query_stats["failed_detections"] += 1

            # Update average detection time
            total = self._fraud_query_stats["total_fraud_queries"]
            current_avg = self._fraud_query_stats["avg_detection_time"]
            self._fraud_query_stats["avg_detection_time"] = (
                (current_avg * (total - 1) + execution_time) / total
            )

            # Keep only last 1000 entries
            if len(self._fraud_query_log) > 1000:
                self._fraud_query_log.pop(0)

        except Exception as e:
            # Don't fail on logging errors
            pass

    def get_fraud_query_stats(self) -> Dict[str, Any]:
        """Get fraud detection query statistics."""
        return {
            **self._fraud_query_stats,
            "recent_queries": self._fraud_query_log[-10:] if self._fraud_query_log else []
        }


# Global database manager instance
db_manager = DatabaseManager()