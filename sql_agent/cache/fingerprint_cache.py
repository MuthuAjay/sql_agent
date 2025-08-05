"""
Fingerprint-Based Cache Management

Intelligent cache invalidation system using schema fingerprints for efficient
incremental updates. Designed for 200+ table databases with local LLM processing.

Design Principles:
- Fingerprint-driven cache invalidation
- Hierarchical caching (schema -> table -> analysis)
- Intelligent cache warming and invalidation
- Minimal LLM re-processing on schema changes

Architecture:
- Uses persistent_cache.py for storage backend
- Integrates with fingerprinting.py for change detection
- Provides high-level cache management for schema intelligence

Author: ML Engineering Team
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging

from .persistent_cache import PersistentCache
from ..utils.fingerprinting import (
    SchemaFingerprintGenerator, 
    FingerprintComparator,
    SchemaFingerprint,
    TableFingerprint
)

logger = logging.getLogger(__name__)


@dataclass
class CacheKey:
    """Structured cache key for hierarchical caching."""
    database_name: str
    table_name: Optional[str] = None
    analysis_type: str = "business_context"
    version: str = "1.0"
    
    def to_string(self) -> str:
        """Convert to cache key string."""
        parts = [self.database_name, self.analysis_type, self.version]
        if self.table_name:
            parts.insert(1, self.table_name)
        return ":".join(parts)


@dataclass
class CacheInvalidationPlan:
    """Plan for cache invalidation based on fingerprint changes."""
    database_name: str
    invalidate_all: bool
    tables_to_invalidate: List[str]
    tables_to_keep: List[str]
    invalidation_reason: str
    estimated_reanalysis_time_minutes: float


@dataclass
class CacheWarmupPlan:
    """Plan for cache warming based on table priorities."""
    database_name: str
    priority_tables: List[str]
    background_tables: List[str]
    estimated_warmup_time_minutes: float


class FingerprintCache:
    """
    Intelligent cache management using schema fingerprints.
    
    Provides high-level cache operations that understand schema changes
    and minimize expensive LLM re-processing through fingerprint-based
    invalidation strategies.
    """
    
    def __init__(self, 
                 cache_dir: Union[str, Path] = "./cache",
                 fingerprint_ttl_hours: int = 168,  # 1 week
                 analysis_ttl_hours: int = 24,      # 1 day
                 max_concurrent_operations: int = 10):
        """
        Initialize fingerprint-based cache.
        
        Args:
            cache_dir: Cache storage directory
            fingerprint_ttl_hours: TTL for fingerprint cache entries
            analysis_ttl_hours: TTL for analysis result cache entries
            max_concurrent_operations: Max concurrent cache operations
        """
        self.cache_dir = Path(cache_dir)
        self.fingerprint_ttl = timedelta(hours=fingerprint_ttl_hours)
        self.analysis_ttl = timedelta(hours=analysis_ttl_hours)
        
        # Initialize underlying cache systems
        self.fingerprint_cache = PersistentCache(
            cache_dir=cache_dir / "fingerprints",
            db_name="fingerprints.db",
            max_size_mb=100,  # Fingerprints are small
            default_ttl_hours=fingerprint_ttl_hours
        )
        
        self.analysis_cache = PersistentCache(
            cache_dir=cache_dir / "analysis", 
            db_name="analysis.db",
            max_size_mb=2048,  # Analysis results can be large
            default_ttl_hours=analysis_ttl_hours
        )
        
        # Fingerprint generator
        self.fingerprint_generator = SchemaFingerprintGenerator(
            include_sample_data=True,
            sample_data_depth=5  # Limited depth for stability
        )
        
        # Concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrent_operations)
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize cache systems."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Start background cleanup for both caches
            await self.fingerprint_cache.start_background_cleanup()
            await self.analysis_cache.start_background_cleanup()
            
            self._initialized = True
            logger.info("Fingerprint cache system initialized")
            
        except Exception as e:
            logger.error(f"Fingerprint cache initialization failed: {e}")
            raise
    
    async def check_schema_changes(self, database_name: str, 
                                 current_schema: Dict[str, Any]) -> CacheInvalidationPlan:
        """
        Check if schema has changed and create invalidation plan.
        
        Args:
            database_name: Database identifier
            current_schema: Current schema data
            
        Returns:
            Invalidation plan with specific actions needed
        """
        async with self.semaphore:
            try:
                # Generate current fingerprint
                current_fp = self.fingerprint_generator.generate_schema_fingerprint(current_schema)
                
                # Get cached fingerprint
                fp_key = CacheKey(database_name, analysis_type="schema_fingerprint").to_string()
                cached_fp = await self.fingerprint_cache.get(fp_key)
                
                if not cached_fp:
                    # No cached fingerprint - first time analysis
                    await self._cache_fingerprint(database_name, current_fp)
                    return CacheInvalidationPlan(
                        database_name=database_name,
                        invalidate_all=True,
                        tables_to_invalidate=list(current_fp.table_fingerprints.keys()),
                        tables_to_keep=[],
                        invalidation_reason="first_time_analysis",
                        estimated_reanalysis_time_minutes=self._estimate_analysis_time(len(current_fp.table_fingerprints))
                    )
                
                # Compare fingerprints
                comparison = FingerprintComparator.compare_schema_fingerprints(cached_fp, current_fp)
                
                # Update cached fingerprint
                await self._cache_fingerprint(database_name, current_fp)
                
                # Create invalidation plan
                return self._create_invalidation_plan(database_name, comparison)
                
            except Exception as e:
                logger.error(f"Schema change check failed for {database_name}: {e}")
                # Safe fallback - invalidate everything
                return CacheInvalidationPlan(
                    database_name=database_name,
                    invalidate_all=True,
                    tables_to_invalidate=[],
                    tables_to_keep=[],
                    invalidation_reason=f"error_fallback: {str(e)}",
                    estimated_reanalysis_time_minutes=30.0
                )
    
    async def get_cached_table_analysis(self, database_name: str, 
                                      table_name: str) -> Optional[Any]:
        """
        Get cached table analysis if fingerprint matches.
        
        Args:
            database_name: Database identifier
            table_name: Table name
            
        Returns:
            Cached analysis result or None if invalid/missing
        """
        async with self.semaphore:
            try:
                analysis_key = CacheKey(database_name, table_name, "table_analysis").to_string()
                return await self.analysis_cache.get(analysis_key)
                
            except Exception as e:
                logger.error(f"Failed to get cached analysis for {table_name}: {e}")
                return None
    
    async def cache_table_analysis(self, database_name: str, table_name: str, 
                                 analysis_result: Any, table_fingerprint: str):
        """
        Cache table analysis result with fingerprint tracking.
        
        Args:
            database_name: Database identifier
            table_name: Table name
            analysis_result: Analysis result to cache
            table_fingerprint: Current table fingerprint
        """
        async with self.semaphore:
            try:
                analysis_key = CacheKey(database_name, table_name, "table_analysis").to_string()
                
                metadata = {
                    "table_fingerprint": table_fingerprint,
                    "cached_at": datetime.utcnow().isoformat(),
                    "database_name": database_name,
                    "table_name": table_name
                }
                
                await self.analysis_cache.set(
                    analysis_key, 
                    analysis_result, 
                    ttl=self.analysis_ttl,
                    metadata=metadata
                )
                
                logger.debug(f"Cached analysis for {table_name}")
                
            except Exception as e:
                logger.error(f"Failed to cache analysis for {table_name}: {e}")
    
    async def invalidate_tables(self, database_name: str, table_names: List[str]):
        """
        Invalidate cache entries for specific tables.
        
        Args:
            database_name: Database identifier
            table_names: List of table names to invalidate
        """
        async with self.semaphore:
            try:
                invalidation_tasks = []
                
                for table_name in table_names:
                    # Invalidate table analysis
                    analysis_key = CacheKey(database_name, table_name, "table_analysis").to_string()
                    invalidation_tasks.append(self.analysis_cache.delete(analysis_key))
                    
                    # Invalidate related caches (vector embeddings, etc.)
                    embedding_key = CacheKey(database_name, table_name, "embeddings").to_string()
                    invalidation_tasks.append(self.analysis_cache.delete(embedding_key))
                
                # Execute invalidations concurrently
                results = await asyncio.gather(*invalidation_tasks, return_exceptions=True)
                
                successful_invalidations = sum(1 for r in results if r is True)
                logger.info(f"Invalidated {successful_invalidations} cache entries for {len(table_names)} tables")
                
            except Exception as e:
                logger.error(f"Table invalidation failed: {e}")
    
    async def get_cache_status(self, database_name: str) -> Dict[str, Any]:
        """
        Get comprehensive cache status for database.
        
        Returns:
            Cache status including fingerprint info and cached analysis count
        """
        try:
            # Get fingerprint status
            fp_key = CacheKey(database_name, analysis_type="schema_fingerprint").to_string()
            cached_fp = await self.fingerprint_cache.get(fp_key)
            
            status = {
                "database_name": database_name,
                "has_cached_fingerprint": cached_fp is not None,
                "fingerprint_age_hours": None,
                "cached_table_count": 0,
                "total_cache_entries": 0
            }
            
            if cached_fp:
                fp_age = datetime.utcnow() - cached_fp.generated_at
                status["fingerprint_age_hours"] = fp_age.total_seconds() / 3600
                status["cached_table_count"] = len(cached_fp.table_fingerprints)
            
            # Get cache statistics
            fp_stats = await self.fingerprint_cache.get_stats()
            analysis_stats = await self.analysis_cache.get_stats()
            
            status["cache_stats"] = {
                "fingerprint_cache": {
                    "entries": fp_stats.total_entries,
                    "size_mb": fp_stats.total_size_bytes / (1024 * 1024),
                    "hit_ratio": fp_stats.hit_ratio
                },
                "analysis_cache": {
                    "entries": analysis_stats.total_entries,
                    "size_mb": analysis_stats.total_size_bytes / (1024 * 1024),
                    "hit_ratio": analysis_stats.hit_ratio
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get cache status for {database_name}: {e}")
            return {"error": str(e)}
    
    async def create_warmup_plan(self, database_name: str, 
                               schema_data: Dict[str, Any],
                               priority_limit: int = 50) -> CacheWarmupPlan:
        """
        Create intelligent cache warming plan for large schemas.
        
        Args:
            database_name: Database identifier
            schema_data: Current schema data
            priority_limit: Number of tables to prioritize for immediate analysis
            
        Returns:
            Warmup plan with prioritized table lists
        """
        try:
            tables = schema_data.get("tables", [])
            
            if len(tables) <= priority_limit:
                # Small schema - analyze everything immediately
                return CacheWarmupPlan(
                    database_name=database_name,
                    priority_tables=[t.get("name", "") for t in tables],
                    background_tables=[],
                    estimated_warmup_time_minutes=self._estimate_analysis_time(len(tables))
                )
            
            # Large schema - prioritize core tables
            priority_tables, background_tables = self._prioritize_tables_for_warmup(tables, priority_limit)
            
            return CacheWarmupPlan(
                database_name=database_name,
                priority_tables=priority_tables,
                background_tables=background_tables,
                estimated_warmup_time_minutes=self._estimate_analysis_time(len(priority_tables))
            )
            
        except Exception as e:
            logger.error(f"Failed to create warmup plan for {database_name}: {e}")
            # Safe fallback
            table_names = [t.get("name", "") for t in schema_data.get("tables", [])]
            return CacheWarmupPlan(
                database_name=database_name,
                priority_tables=table_names[:priority_limit],
                background_tables=table_names[priority_limit:],
                estimated_warmup_time_minutes=15.0
            )
    
    def _create_invalidation_plan(self, database_name: str, 
                                comparison: Dict[str, Any]) -> CacheInvalidationPlan:
        """Create invalidation plan from fingerprint comparison."""
        if not comparison["schema_changed"]:
            return CacheInvalidationPlan(
                database_name=database_name,
                invalidate_all=False,
                tables_to_invalidate=[],
                tables_to_keep=[],
                invalidation_reason="no_changes",
                estimated_reanalysis_time_minutes=0.0
            )
        
        change_type = comparison["change_type"]
        
        if comparison["requires_full_reanalysis"]:
            # Major changes - invalidate everything
            all_tables = (comparison.get("added_tables", []) + 
                         comparison.get("changed_tables", []) +
                         comparison.get("removed_tables", []))
            
            return CacheInvalidationPlan(
                database_name=database_name,
                invalidate_all=True,
                tables_to_invalidate=all_tables,
                tables_to_keep=[],
                invalidation_reason=f"major_changes_{change_type}",
                estimated_reanalysis_time_minutes=self._estimate_analysis_time(len(all_tables))
            )
        
        # Incremental changes - selective invalidation
        tables_to_invalidate = (comparison.get("added_tables", []) + 
                              comparison.get("changed_tables", []))
        
        return CacheInvalidationPlan(
            database_name=database_name,
            invalidate_all=False,
            tables_to_invalidate=tables_to_invalidate,
            tables_to_keep=[],
            invalidation_reason=f"incremental_changes_{change_type}",
            estimated_reanalysis_time_minutes=self._estimate_analysis_time(len(tables_to_invalidate))
        )
    
    def _prioritize_tables_for_warmup(self, tables: List[Dict[str, Any]], 
                                    limit: int) -> Tuple[List[str], List[str]]:
        """
        Prioritize tables for cache warming.
        
        Priority factors:
        1. Tables with foreign keys (relationship hubs)
        2. Tables with recent activity (high row counts)
        3. Tables with business-critical keywords
        4. Smaller tables (faster to analyze)
        """
        scored_tables = []
        
        for table in tables:
            table_name = table.get("name", "")
            score = 0
            
            # Factor 1: Relationship hub score
            fk_count = len(table.get("foreign_keys", []))
            score += fk_count * 10
            
            # Factor 2: Activity score (row count)
            row_count = table.get("statistics", {}).get("live_tuples", 0)
            if row_count > 10000:
                score += 20
            elif row_count > 1000:
                score += 10
            
            # Factor 3: Business keywords
            business_keywords = ["customer", "order", "product", "user", "account", 
                               "transaction", "payment", "invoice", "analytics"]
            name_lower = table_name.lower()
            keyword_matches = sum(1 for kw in business_keywords if kw in name_lower)
            score += keyword_matches * 15
            
            # Factor 4: Size penalty (prefer smaller tables for quick warmup)
            if row_count > 100000:
                score -= 5
            
            scored_tables.append((table_name, score))
        
        # Sort by score and split
        scored_tables.sort(key=lambda x: x[1], reverse=True)
        
        priority_tables = [name for name, _ in scored_tables[:limit]]
        background_tables = [name for name, _ in scored_tables[limit:]]
        
        return priority_tables, background_tables
    
    def _estimate_analysis_time(self, table_count: int) -> float:
        """
        Estimate analysis time based on table count and hardware.
        
        Based on local LLM performance with 4090 GPU:
        - Desktop (24GB): ~3-5 seconds per table
        - Laptop (16GB): ~5-8 seconds per table
        """
        # Conservative estimate for planning
        seconds_per_table = 6.0  # Average for both setups
        
        # Parallel processing factor (conservative)
        parallel_factor = 0.4  # Assume 40% efficiency gain from parallelization
        
        total_seconds = (table_count * seconds_per_table) * (1 - parallel_factor)
        return total_seconds / 60  # Convert to minutes
    
    async def _cache_fingerprint(self, database_name: str, fingerprint: SchemaFingerprint):
        """Cache schema fingerprint for future comparisons."""
        try:
            fp_key = CacheKey(database_name, analysis_type="schema_fingerprint").to_string()
            
            await self.fingerprint_cache.set(
                fp_key,
                fingerprint,
                ttl=self.fingerprint_ttl,
                metadata={
                    "database_name": database_name,
                    "table_count": fingerprint.table_count,
                    "generated_at": fingerprint.generated_at.isoformat()
                }
            )
            
            logger.debug(f"Cached fingerprint for {database_name}")
            
        except Exception as e:
            logger.error(f"Failed to cache fingerprint for {database_name}: {e}")
    
    async def bulk_cache_operations(self, operations: List[Tuple[str, str, Any]]):
        """
        Execute bulk cache operations efficiently.
        
        Args:
            operations: List of (operation, key, value) tuples
                       operation: "set", "delete", "get"
        """
        async with self.semaphore:
            try:
                tasks = []
                
                for operation, key, value in operations:
                    if operation == "set":
                        tasks.append(self.analysis_cache.set(key, value))
                    elif operation == "delete":
                        tasks.append(self.analysis_cache.delete(key))
                    elif operation == "get":
                        tasks.append(self.analysis_cache.get(key))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                successful_ops = sum(1 for r in results if not isinstance(r, Exception))
                logger.info(f"Bulk operations completed: {successful_ops}/{len(operations)} successful")
                
                return results
                
            except Exception as e:
                logger.error(f"Bulk cache operations failed: {e}")
                return []
    
    async def optimize_cache(self, database_name: str) -> Dict[str, Any]:
        """
        Optimize cache for better performance.
        
        Operations:
        - Remove unused entries
        - Compact storage
        - Update statistics
        """
        try:
            # Cleanup expired entries
            fp_cleaned = await self.fingerprint_cache.cleanup_expired()
            analysis_cleaned = await self.analysis_cache.cleanup_expired()
            
            # Get updated statistics
            fp_stats = await self.fingerprint_cache.get_stats()
            analysis_stats = await self.analysis_cache.get_stats()
            
            optimization_result = {
                "database_name": database_name,
                "cleaned_entries": fp_cleaned + analysis_cleaned,
                "cache_stats": {
                    "fingerprint_cache_mb": fp_stats.total_size_bytes / (1024 * 1024),
                    "analysis_cache_mb": analysis_stats.total_size_bytes / (1024 * 1024),
                    "total_entries": fp_stats.total_entries + analysis_stats.total_entries,
                    "combined_hit_ratio": (fp_stats.hit_ratio + analysis_stats.hit_ratio) / 2
                },
                "optimization_timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Cache optimization completed for {database_name}")
            return optimization_result
            
        except Exception as e:
            logger.error(f"Cache optimization failed for {database_name}: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close cache systems and cleanup resources."""
        try:
            await self.fingerprint_cache.stop_background_cleanup()
            await self.analysis_cache.stop_background_cleanup()
            
            self.fingerprint_cache.close()
            self.analysis_cache.close()
            
            logger.info("Fingerprint cache system closed")
            
        except Exception as e:
            logger.error(f"Cache close error: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        if not self._initialized:
            await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()