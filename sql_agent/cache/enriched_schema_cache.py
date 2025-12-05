"""
Enriched Schema Cache

Persistent caching layer for fully enriched database schemas including:
- Column statistics (count, distinct, min/max/avg, sample values)
- Business domains and contexts
- Table relationships
- Sample data
- LLM-generated intelligence

This cache is populated on database selection and used for fast query routing
without re-querying database statistics on every query.

Author: ML Engineering Team
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging

from .persistent_cache import PersistentCache

logger = logging.getLogger(__name__)


@dataclass
class EnrichedColumn:
    """Enriched column metadata with statistics and samples."""
    column_name: str
    data_type: str
    nullable: bool
    primary_key: bool
    foreign_key: bool
    business_concept: str
    # Statistics
    total_count: Optional[int] = None
    distinct_count: Optional[int] = None
    null_count: Optional[int] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    avg_value: Optional[float] = None
    sample_values: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class EnrichedTable:
    """Enriched table metadata."""
    table_name: str
    columns: List[EnrichedColumn]
    row_count: Optional[int] = None
    business_purpose: Optional[str] = None
    business_role: Optional[str] = None
    business_domains: Optional[List[str]] = None
    sample_data: Optional[Dict[str, Any]] = None
    relationships: Optional[List[Dict[str, Any]]] = None
    criticality: Optional[str] = None
    confidence_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "table_name": self.table_name,
            "columns": [col.to_dict() for col in self.columns],
            "row_count": self.row_count,
            "business_purpose": self.business_purpose,
            "business_role": self.business_role,
            "business_domains": self.business_domains,
            "sample_data": self.sample_data,
            "relationships": self.relationships,
            "criticality": self.criticality,
            "confidence_score": self.confidence_score
        }


@dataclass
class EnrichedSchema:
    """Complete enriched database schema."""
    database_name: str
    tables: List[EnrichedTable]
    business_purpose: Optional[str] = None
    industry_domain: Optional[str] = None
    discovered_domains: Optional[List[Dict[str, Any]]] = None
    relationships: Optional[List[Dict[str, Any]]] = None
    enriched_at: Optional[datetime] = None
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "database_name": self.database_name,
            "tables": [table.to_dict() for table in self.tables],
            "business_purpose": self.business_purpose,
            "industry_domain": self.industry_domain,
            "discovered_domains": self.discovered_domains,
            "relationships": self.relationships,
            "enriched_at": self.enriched_at.isoformat() if self.enriched_at else None,
            "version": self.version
        }


class EnrichedSchemaCache:
    """
    High-level cache for fully enriched database schemas.

    This cache stores the complete enriched schema including all statistics,
    sample values, and business intelligence. It's populated when a database
    is selected and used for fast query routing.
    """

    def __init__(
        self,
        cache_dir: str = "./cache",
        default_ttl_days: int = 7,  # 7 days default
        max_size_mb: int = 2048,  # 2GB for enriched schemas
    ):
        """
        Initialize enriched schema cache.

        Args:
            cache_dir: Cache storage directory
            default_ttl_days: Default TTL for enriched schemas
            max_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.default_ttl = timedelta(days=default_ttl_days)

        # Initialize persistent cache
        self.cache = PersistentCache(
            cache_dir=cache_dir,
            db_name="enriched_schemas.db",
            max_size_mb=max_size_mb,
            default_ttl_hours=default_ttl_days * 24,
            serialization_format="pickle",
            enable_compression=True
        )

        self._initialized = False
        logger.info("EnrichedSchemaCache initialized")

    async def initialize(self):
        """Initialize the cache system."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            await self.cache.start_background_cleanup()
            self._initialized = True
            logger.info("Enriched schema cache system started")
        except Exception as e:
            logger.error(f"Failed to initialize enriched schema cache: {e}")
            raise

    def _make_schema_key(self, database_name: str) -> str:
        """Generate cache key for database schema."""
        return f"enriched_schema:{database_name}"

    def _make_table_key(self, database_name: str, table_name: str) -> str:
        """Generate cache key for individual table."""
        return f"enriched_table:{database_name}:{table_name}"

    async def get_enriched_schema(self, database_name: str) -> Optional[EnrichedSchema]:
        """
        Get complete enriched schema from cache.

        Args:
            database_name: Database identifier

        Returns:
            EnrichedSchema if cached, None otherwise
        """
        try:
            key = self._make_schema_key(database_name)
            cached_data = await self.cache.get(key)

            if cached_data:
                logger.info(f"✓ Cache HIT for enriched schema: {database_name}")
                return cached_data
            else:
                logger.info(f"✗ Cache MISS for enriched schema: {database_name}")
                return None

        except Exception as e:
            logger.error(f"Failed to get enriched schema for {database_name}: {e}")
            return None

    async def set_enriched_schema(
        self,
        enriched_schema: EnrichedSchema,
        ttl: Optional[timedelta] = None
    ) -> bool:
        """
        Cache complete enriched schema.

        Args:
            enriched_schema: Complete enriched schema to cache
            ttl: Time to live (optional, uses default if not specified)

        Returns:
            True if successful, False otherwise
        """
        try:
            key = self._make_schema_key(enriched_schema.database_name)

            # Set enriched_at timestamp
            enriched_schema.enriched_at = datetime.utcnow()

            success = await self.cache.set(
                key,
                enriched_schema,
                ttl=ttl or self.default_ttl,
                metadata={
                    "database_name": enriched_schema.database_name,
                    "table_count": len(enriched_schema.tables),
                    "cached_at": datetime.utcnow().isoformat()
                }
            )

            if success:
                logger.info(f"✓ Cached enriched schema for {enriched_schema.database_name} "
                          f"with {len(enriched_schema.tables)} tables")
            else:
                logger.warning(f"✗ Failed to cache enriched schema for {enriched_schema.database_name}")

            return success

        except Exception as e:
            logger.error(f"Failed to cache enriched schema: {e}")
            return False

    async def get_enriched_table(
        self,
        database_name: str,
        table_name: str
    ) -> Optional[EnrichedTable]:
        """
        Get enriched table metadata from cache.

        Args:
            database_name: Database identifier
            table_name: Table name

        Returns:
            EnrichedTable if cached, None otherwise
        """
        try:
            key = self._make_table_key(database_name, table_name)
            cached_data = await self.cache.get(key)

            if cached_data:
                logger.debug(f"Cache HIT for table: {database_name}.{table_name}")
                return cached_data
            else:
                logger.debug(f"Cache MISS for table: {database_name}.{table_name}")
                return None

        except Exception as e:
            logger.error(f"Failed to get enriched table {table_name}: {e}")
            return None

    async def set_enriched_table(
        self,
        database_name: str,
        enriched_table: EnrichedTable,
        ttl: Optional[timedelta] = None
    ) -> bool:
        """
        Cache individual enriched table.

        Args:
            database_name: Database identifier
            enriched_table: Enriched table to cache
            ttl: Time to live (optional)

        Returns:
            True if successful, False otherwise
        """
        try:
            key = self._make_table_key(database_name, enriched_table.table_name)

            success = await self.cache.set(
                key,
                enriched_table,
                ttl=ttl or self.default_ttl,
                metadata={
                    "database_name": database_name,
                    "table_name": enriched_table.table_name,
                    "cached_at": datetime.utcnow().isoformat()
                }
            )

            if success:
                logger.debug(f"Cached enriched table: {database_name}.{enriched_table.table_name}")

            return success

        except Exception as e:
            logger.error(f"Failed to cache enriched table: {e}")
            return False

    async def invalidate_schema(self, database_name: str) -> bool:
        """
        Invalidate cached schema for a database.

        Args:
            database_name: Database identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            key = self._make_schema_key(database_name)
            success = await self.cache.delete(key)

            if success:
                logger.info(f"✓ Invalidated enriched schema cache for {database_name}")
            else:
                logger.warning(f"✗ Failed to invalidate enriched schema for {database_name}")

            return success

        except Exception as e:
            logger.error(f"Failed to invalidate schema: {e}")
            return False

    async def invalidate_table(self, database_name: str, table_name: str) -> bool:
        """
        Invalidate cached enriched table.

        Args:
            database_name: Database identifier
            table_name: Table name

        Returns:
            True if successful, False otherwise
        """
        try:
            key = self._make_table_key(database_name, table_name)
            success = await self.cache.delete(key)

            if success:
                logger.info(f"Invalidated enriched table: {database_name}.{table_name}")

            return success

        except Exception as e:
            logger.error(f"Failed to invalidate table: {e}")
            return False

    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Cache statistics dictionary
        """
        try:
            stats = await self.cache.get_stats()

            return {
                "total_entries": stats.total_entries,
                "total_size_mb": stats.total_size_bytes / (1024 * 1024),
                "hit_count": stats.hit_count,
                "miss_count": stats.miss_count,
                "hit_ratio": stats.hit_ratio,
                "average_entry_size_mb": stats.average_entry_size / (1024 * 1024),
                "oldest_entry": stats.oldest_entry.isoformat() if stats.oldest_entry else None,
                "newest_entry": stats.newest_entry.isoformat() if stats.newest_entry else None,
            }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}

    async def get_schema_info(self, database_name: str) -> Optional[Dict[str, Any]]:
        """
        Get basic information about cached schema without loading full data.

        Args:
            database_name: Database identifier

        Returns:
            Schema info dictionary or None
        """
        try:
            enriched_schema = await self.get_enriched_schema(database_name)

            if not enriched_schema:
                return None

            return {
                "database_name": enriched_schema.database_name,
                "table_count": len(enriched_schema.tables),
                "tables": [table.table_name for table in enriched_schema.tables],
                "business_purpose": enriched_schema.business_purpose,
                "industry_domain": enriched_schema.industry_domain,
                "enriched_at": enriched_schema.enriched_at.isoformat() if enriched_schema.enriched_at else None,
                "version": enriched_schema.version
            }

        except Exception as e:
            logger.error(f"Failed to get schema info for {database_name}: {e}")
            return None

    async def warm_cache(
        self,
        database_name: str,
        schema_enricher_func,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Warm cache by enriching and caching database schema.

        Args:
            database_name: Database to warm cache for
            schema_enricher_func: Async function that returns EnrichedSchema
            force: Force re-enrichment even if cached

        Returns:
            Warmup result with timing and status
        """
        start_time = datetime.utcnow()

        try:
            # Check if already cached
            if not force:
                existing = await self.get_enriched_schema(database_name)
                if existing:
                    logger.info(f"Schema already cached for {database_name}, skipping warmup")
                    return {
                        "status": "already_cached",
                        "database_name": database_name,
                        "table_count": len(existing.tables),
                        "cached_at": existing.enriched_at.isoformat() if existing.enriched_at else None
                    }

            # Enrich schema
            logger.info(f"Starting cache warmup for {database_name}...")
            enriched_schema = await schema_enricher_func(database_name)

            if not enriched_schema:
                return {
                    "status": "error",
                    "error": "Schema enricher returned None"
                }

            # Cache the enriched schema
            success = await self.set_enriched_schema(enriched_schema)

            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            return {
                "status": "success" if success else "error",
                "database_name": database_name,
                "table_count": len(enriched_schema.tables),
                "duration_seconds": duration,
                "enriched_at": enriched_schema.enriched_at.isoformat() if enriched_schema.enriched_at else None
            }

        except Exception as e:
            logger.error(f"Cache warmup failed for {database_name}: {e}")
            return {
                "status": "error",
                "database_name": database_name,
                "error": str(e)
            }

    async def close(self):
        """Close the cache and cleanup resources."""
        try:
            await self.cache.stop_background_cleanup()
            self.cache.close()
            logger.info("Enriched schema cache closed")
        except Exception as e:
            logger.error(f"Error closing enriched schema cache: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        if not self._initialized:
            await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
