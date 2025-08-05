"""
Persistent Cache System

High-performance persistent caching system designed for schema analysis results,
LLM outputs, and vector embeddings. Optimized for local LLM workflows with
200+ table databases.

Design Principles:
- Disk-based persistence across application restarts
- Atomic operations for concurrent access
- Efficient serialization for complex data structures
- Configurable TTL and size-based eviction
- Hardware-optimized for SSD storage

Author: ML Engineering Team
"""

import asyncio
import json
import pickle
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import logging
import hashlib
import gzip
import os

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    expires_at: Optional[datetime]
    size_bytes: int
    access_count: int
    metadata: Dict[str, Any]


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_entries: int
    total_size_bytes: int
    hit_count: int
    miss_count: int
    eviction_count: int
    hit_ratio: float
    average_entry_size: float
    oldest_entry: Optional[datetime]
    newest_entry: Optional[datetime]


class CacheSerializer:
    """
    High-performance serialization for cache entries.
    
    Supports multiple formats optimized for different data types:
    - JSON: Human-readable, good for configuration
    - Pickle: Python objects, fastest for complex data
    - Compressed: For large data structures
    """
    
    @staticmethod
    def serialize(data: Any, format: str = "pickle", compress: bool = False) -> bytes:
        """Serialize data to bytes."""
        try:
            if format == "json":
                serialized = json.dumps(data, default=str, ensure_ascii=False).encode('utf-8')
            elif format == "pickle":
                serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            if compress:
                serialized = gzip.compress(serialized)
            
            return serialized
            
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            raise
    
    @staticmethod
    def deserialize(data: bytes, format: str = "pickle", compressed: bool = False) -> Any:
        """Deserialize bytes to data."""
        try:
            if compressed:
                data = gzip.decompress(data)
            
            if format == "json":
                return json.loads(data.decode('utf-8'))
            elif format == "pickle":
                return pickle.loads(data)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            raise


class PersistentCache:
    """
    Enterprise-grade persistent cache with SQLite backend.
    
    Optimized for:
    - Schema analysis results (large complex objects)
    - LLM responses (text with metadata)
    - Vector embeddings (numeric arrays)
    - High-frequency reads with periodic writes
    """
    
    def __init__(self, 
                 cache_dir: Union[str, Path] = "./cache",
                 db_name: str = "persistent_cache.db",
                 max_size_mb: int = 1024,  # 1GB default
                 default_ttl_hours: int = 24,
                 cleanup_interval_minutes: int = 60,
                 serialization_format: str = "pickle",
                 enable_compression: bool = True):
        """
        Initialize persistent cache.
        
        Args:
            cache_dir: Directory for cache files
            db_name: SQLite database filename
            max_size_mb: Maximum cache size in MB
            default_ttl_hours: Default TTL for entries
            cleanup_interval_minutes: Background cleanup frequency
            serialization_format: Default serialization format
            enable_compression: Enable compression for large entries
        """
        self.cache_dir = Path(cache_dir)
        self.db_path = self.cache_dir / db_name
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self.cleanup_interval = timedelta(minutes=cleanup_interval_minutes)
        self.serialization_format = serialization_format
        self.enable_compression = enable_compression
        
        # Thread-safe database connection
        self._local = threading.local()
        self._lock = threading.RLock()
        
        # Performance tracking
        self._stats = CacheStats(
            total_entries=0, total_size_bytes=0, hit_count=0, 
            miss_count=0, eviction_count=0, hit_ratio=0.0,
            average_entry_size=0.0, oldest_entry=None, newest_entry=None
        )
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Initialize cache
        self._initialize()
    
    def _initialize(self):
        """Initialize cache directory and database."""
        try:
            # Create cache directory
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize database schema
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        value BLOB NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        accessed_at TIMESTAMP NOT NULL,
                        expires_at TIMESTAMP,
                        size_bytes INTEGER NOT NULL,
                        access_count INTEGER DEFAULT 1,
                        format TEXT DEFAULT 'pickle',
                        compressed BOOLEAN DEFAULT FALSE,
                        metadata TEXT DEFAULT '{}'
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache_entries(accessed_at)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_size_bytes ON cache_entries(size_bytes)
                """)
                
                conn.commit()
            
            # Load initial stats
            self._refresh_stats()
            
            logger.info(f"Persistent cache initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Cache initialization failed: {e}")
            raise
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
            
            # Optimize for SSD storage
            self._local.connection.execute("PRAGMA journal_mode = WAL")
            self._local.connection.execute("PRAGMA synchronous = NORMAL")
            self._local.connection.execute("PRAGMA cache_size = 10000")
            self._local.connection.execute("PRAGMA temp_store = MEMORY")
            
        return self._local.connection
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._get_sync, key, default)
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            return default
    
    def _get_sync(self, key: str, default: Any = None) -> Any:
        """Synchronous get implementation."""
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute("""
                    SELECT value, expires_at, access_count, format, compressed
                    FROM cache_entries 
                    WHERE key = ?
                """, (key,))
                
                row = cursor.fetchone()
                if not row:
                    self._stats.miss_count += 1
                    return default
                
                # Check expiration
                if row['expires_at']:
                    expires_at = datetime.fromisoformat(row['expires_at'])
                    if datetime.utcnow() > expires_at:
                        self._delete_sync(key)
                        self._stats.miss_count += 1
                        return default
                
                # Update access statistics
                conn.execute("""
                    UPDATE cache_entries 
                    SET accessed_at = ?, access_count = access_count + 1
                    WHERE key = ?
                """, (datetime.utcnow().isoformat(), key))
                conn.commit()
                
                # Deserialize value
                value = CacheSerializer.deserialize(
                    row['value'], 
                    format=row['format'],
                    compressed=row['compressed']
                )
                
                self._stats.hit_count += 1
                return value
                
            except Exception as e:
                logger.error(f"Sync get failed for key {key}: {e}")
                self._stats.miss_count += 1
                return default
    
    async def set(self, key: str, value: Any, ttl: Optional[timedelta] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            True if successful
        """
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._set_sync, key, value, ttl, metadata)
        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            return False
    
    def _set_sync(self, key: str, value: Any, ttl: Optional[timedelta] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Synchronous set implementation."""
        with self._lock:
            try:
                # Serialize value
                serialized_value = CacheSerializer.serialize(
                    value, 
                    format=self.serialization_format,
                    compress=self.enable_compression
                )
                
                size_bytes = len(serialized_value)
                now = datetime.utcnow()
                expires_at = now + (ttl or self.default_ttl)
                
                # Check if we need to make space
                if self._stats.total_size_bytes + size_bytes > self.max_size_bytes:
                    self._evict_entries(size_bytes)
                
                conn = self._get_connection()
                
                # Insert or replace entry
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries
                    (key, value, created_at, accessed_at, expires_at, size_bytes, 
                     access_count, format, compressed, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                """, (
                    key, serialized_value, now.isoformat(), now.isoformat(),
                    expires_at.isoformat(), size_bytes, self.serialization_format,
                    self.enable_compression, json.dumps(metadata or {})
                ))
                
                conn.commit()
                self._refresh_stats()
                
                return True
                
            except Exception as e:
                logger.error(f"Sync set failed for key {key}: {e}")
                return False
    
    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._delete_sync, key)
        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {e}")
            return False
    
    def _delete_sync(self, key: str) -> bool:
        """Synchronous delete implementation."""
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                conn.commit()
                
                deleted = cursor.rowcount > 0
                if deleted:
                    self._refresh_stats()
                
                return deleted
                
            except Exception as e:
                logger.error(f"Sync delete failed for key {key}: {e}")
                return False
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._clear_sync)
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return False
    
    def _clear_sync(self) -> bool:
        """Synchronous clear implementation."""
        with self._lock:
            try:
                conn = self._get_connection()
                conn.execute("DELETE FROM cache_entries")
                conn.commit()
                self._refresh_stats()
                return True
            except Exception as e:
                logger.error(f"Sync clear failed: {e}")
                return False
    
    def _evict_entries(self, needed_bytes: int):
        """Evict entries to make space using LRU strategy."""
        try:
            conn = self._get_connection()
            
            # Calculate how much to evict (25% overhead)
            target_free_bytes = needed_bytes * 1.25
            
            # Find LRU entries to evict
            cursor = conn.execute("""
                SELECT key, size_bytes FROM cache_entries
                ORDER BY accessed_at ASC
            """)
            
            entries_to_evict = []
            freed_bytes = 0
            
            for row in cursor:
                entries_to_evict.append(row['key'])
                freed_bytes += row['size_bytes']
                
                if freed_bytes >= target_free_bytes:
                    break
            
            # Evict entries
            if entries_to_evict:
                placeholders = ','.join(['?'] * len(entries_to_evict))
                conn.execute(f"DELETE FROM cache_entries WHERE key IN ({placeholders})", 
                           entries_to_evict)
                conn.commit()
                
                self._stats.eviction_count += len(entries_to_evict)
                logger.info(f"Evicted {len(entries_to_evict)} entries, freed {freed_bytes} bytes")
                
        except Exception as e:
            logger.error(f"Entry eviction failed: {e}")
    
    def _refresh_stats(self):
        """Refresh cache statistics."""
        try:
            conn = self._get_connection()
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_entries,
                    COALESCE(SUM(size_bytes), 0) as total_size_bytes,
                    COALESCE(AVG(size_bytes), 0) as average_entry_size,
                    MIN(created_at) as oldest_entry,
                    MAX(created_at) as newest_entry
                FROM cache_entries
            """)
            
            row = cursor.fetchone()
            if row:
                self._stats.total_entries = row['total_entries']
                self._stats.total_size_bytes = row['total_size_bytes']
                self._stats.average_entry_size = row['average_entry_size']
                
                if row['oldest_entry']:
                    self._stats.oldest_entry = datetime.fromisoformat(row['oldest_entry'])
                if row['newest_entry']:
                    self._stats.newest_entry = datetime.fromisoformat(row['newest_entry'])
                
                # Calculate hit ratio
                total_requests = self._stats.hit_count + self._stats.miss_count
                if total_requests > 0:
                    self._stats.hit_ratio = self._stats.hit_count / total_requests
                    
        except Exception as e:
            logger.error(f"Stats refresh failed: {e}")
    
    async def cleanup_expired(self) -> int:
        """Remove expired entries."""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._cleanup_expired_sync)
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 0
    
    def _cleanup_expired_sync(self) -> int:
        """Synchronous cleanup implementation."""
        with self._lock:
            try:
                conn = self._get_connection()
                now = datetime.utcnow().isoformat()
                
                cursor = conn.execute("""
                    DELETE FROM cache_entries 
                    WHERE expires_at IS NOT NULL AND expires_at < ?
                """, (now,))
                
                conn.commit()
                removed_count = cursor.rowcount
                
                if removed_count > 0:
                    self._refresh_stats()
                    logger.info(f"Cleaned up {removed_count} expired entries")
                
                return removed_count
                
            except Exception as e:
                logger.error(f"Sync cleanup failed: {e}")
                return 0
    
    async def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        self._refresh_stats()
        return self._stats
    
    async def start_background_cleanup(self):
        """Start background cleanup task."""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._background_cleanup_loop())
        logger.info("Background cleanup started")
    
    async def stop_background_cleanup(self):
        """Stop background cleanup task."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Background cleanup stopped")
    
    async def _background_cleanup_loop(self):
        """Background cleanup loop."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval.total_seconds())
                if self._running:
                    await self.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
    
    def close(self):
        """Close cache and cleanup resources."""
        try:
            if hasattr(self._local, 'connection'):
                self._local.connection.close()
            logger.info("Persistent cache closed")
        except Exception as e:
            logger.error(f"Cache close error: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_background_cleanup()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_background_cleanup()
        self.close()