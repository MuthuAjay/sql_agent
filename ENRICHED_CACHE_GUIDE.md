# Enriched Schema Cache Guide

## Overview

The Enriched Schema Cache is a persistent caching system that stores complete database schema enrichments including:

- **Column Statistics**: count, distinct values, min/max/avg, sample values
- **Business Intelligence**: LLM-generated business domains, purposes, and contexts
- **Table Relationships**: Foreign keys and inferred relationships
- **Sample Data**: Representative data samples from each table

This cache dramatically improves query performance by eliminating the need to:
1. Re-query database statistics on every query
2. Re-run LLM analysis on every query
3. Re-fetch sample data repeatedly

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Database Selection                        │
│                           ↓                                  │
│                  POST /api/schema/enrich_cache              │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│              SchemaEnrichmentService                         │
│  ┌───────────────────────────────────────────────────┐     │
│  │ 1. Get base schema (tables, columns)              │     │
│  │ 2. Get column statistics (min/max/avg/samples)    │     │
│  │ 3. Get business intelligence (LLM analysis)       │     │
│  │ 4. Get sample data (5 rows per table)             │     │
│  │ 5. Build EnrichedSchema object                    │     │
│  └───────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│              EnrichedSchemaCache                             │
│  ┌───────────────────────────────────────────────────┐     │
│  │ Store in PersistentCache (SQLite)                 │     │
│  │ - TTL: 7 days (configurable)                      │     │
│  │ - Compression: Enabled                            │     │
│  │ - Format: Pickle (efficient for Python objects)   │     │
│  └───────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                   Query Processing                           │
│  ┌───────────────────────────────────────────────────┐     │
│  │ 1. Router agent retrieves enriched schema         │     │
│  │ 2. No DB queries for statistics                   │     │
│  │ 3. No LLM calls for business context              │     │
│  │ 4. Fast query routing and SQL generation          │     │
│  └───────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## API Endpoints

### 1. Warm the Cache (Enrich Schema)

**Endpoint**: `POST /api/schema/enrich_cache`

**Description**: Enrich and cache complete database schema with all statistics, samples, and business intelligence.

**Parameters**:
- `database_name` (optional): Database to enrich (defaults to "default")
- `force` (optional): Force re-enrichment even if cached (default: false)

**Example**:
```bash
# First time enrichment
curl -X POST "http://localhost:8000/api/schema/enrich_cache"

# Force re-enrichment
curl -X POST "http://localhost:8000/api/schema/enrich_cache?force=true"
```

**Response**:
```json
{
  "status": "success",
  "database_name": "default",
  "table_count": 1,
  "duration_seconds": 12.45,
  "enriched_at": "2025-12-05T10:30:00.000000Z",
  "business_purpose": "This database system tracks financial transactions...",
  "industry_domain": "Financial Services",
  "message": "Successfully enriched and cached 1 tables"
}
```

### 2. Check Cache Status

**Endpoint**: `GET /api/schema/enriched_cache/status`

**Description**: Get status of enriched schema cache.

**Example**:
```bash
curl "http://localhost:8000/api/schema/enriched_cache/status"
```

**Response**:
```json
{
  "database_name": "default",
  "is_cached": true,
  "schema_info": {
    "database_name": "default",
    "table_count": 1,
    "tables": ["transactions"],
    "business_purpose": "...",
    "industry_domain": "Financial Services",
    "enriched_at": "2025-12-05T10:30:00.000000Z",
    "version": "1.0"
  },
  "cache_stats": {
    "total_entries": 5,
    "total_size_mb": 15.2,
    "hit_count": 42,
    "miss_count": 3,
    "hit_ratio": 0.93
  }
}
```

### 3. Get Cached Enriched Schema

**Endpoint**: `GET /api/schema/enriched_cache/schema`

**Description**: Get the full enriched schema from cache.

**Example**:
```bash
curl "http://localhost:8000/api/schema/enriched_cache/schema"
```

**Response**:
```json
{
  "database_name": "default",
  "tables": [
    {
      "table_name": "transactions",
      "columns": [
        {
          "column_name": "step",
          "data_type": "integer",
          "nullable": true,
          "primary_key": false,
          "foreign_key": false,
          "business_concept": "",
          "total_count": 6362620,
          "distinct_count": 743,
          "null_count": 0,
          "min_value": "1",
          "max_value": "743",
          "avg_value": 243.39,
          "sample_values": ["1", "2", "3", "4", "5"]
        },
        ...
      ],
      "row_count": 6362620,
      "business_purpose": "Fraud detection and transaction monitoring",
      "business_role": "Core data source for identifying fraudulent transactions",
      "business_domains": ["fraud_detection", "transaction_management"],
      "criticality": "Critical",
      "confidence_score": 0.95
    }
  ],
  "business_purpose": "This database system tracks financial transactions...",
  "industry_domain": "Financial Services",
  "discovered_domains": [...],
  "relationships": [...],
  "enriched_at": "2025-12-05T10:30:00.000000Z"
}
```

### 4. Invalidate Cache

**Endpoint**: `DELETE /api/schema/enriched_cache`

**Description**: Invalidate (delete) the enriched schema cache. Use when schema changes.

**Example**:
```bash
curl -X DELETE "http://localhost:8000/api/schema/enriched_cache"
```

**Response**:
```json
{
  "status": "success",
  "database_name": "default",
  "message": "Enriched cache invalidated for database 'default'"
}
```

## Usage Workflow

### Initial Setup (One-Time Per Database)

1. **Start your application**:
   ```bash
   uvicorn sql_agent.api.main:app --reload
   ```

2. **Warm the cache** (run once after selecting database):
   ```bash
   curl -X POST "http://localhost:8000/api/schema/enrich_cache"
   ```

   This will:
   - Analyze your database schema
   - Collect column statistics
   - Generate business intelligence with LLM
   - Cache everything for 7 days

3. **Verify cache status**:
   ```bash
   curl "http://localhost:8000/api/schema/enriched_cache/status"
   ```

### Normal Query Operations

Once cached, all queries will automatically use the enriched schema:

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "How many fraudulent transactions are there?"}'
```

**Benefits**:
- ✅ No database statistics queries
- ✅ No LLM analysis calls
- ✅ Instant access to column metadata
- ✅ Fast query routing

### Schema Changes

When your database schema changes (new tables, columns, etc.):

```bash
# Invalidate old cache
curl -X DELETE "http://localhost:8000/api/schema/enriched_cache"

# Re-enrich with new schema
curl -X POST "http://localhost:8000/api/schema/enrich_cache?force=true"
```

## Performance Comparison

### Without Enriched Cache

```
Query: "Count fraudulent transactions"
├─ Schema introspection: 250ms
├─ Column statistics (11 columns): 850ms
├─ LLM business analysis: 3200ms
├─ Router processing: 150ms
├─ SQL generation: 120ms
└─ TOTAL: ~4.6 seconds
```

### With Enriched Cache (After Initial Warmup)

```
Query: "Count fraudulent transactions"
├─ Cache lookup: 15ms ✓
├─ Router processing: 150ms
├─ SQL generation: 120ms
└─ TOTAL: ~0.3 seconds
```

**Speed Improvement**: 15x faster! 🚀

## Configuration

### Cache Settings

Edit settings in [main.py:107-121](main.py#L107-L121):

```python
cache = EnrichedSchemaCache(
    cache_dir="./cache",           # Cache storage location
    default_ttl_days=7,             # How long to keep cache (7 days)
    max_size_mb=2048                # Max cache size (2GB)
)
```

### Enrichment Settings

Edit settings in [main.py:124-129](main.py#L124-L129):

```python
service = SchemaEnrichmentService(
    db_manager=database_manager,
    schema_processor=sp,
    schema_analyzer=None            # Set to SchemaAnalyzer for LLM analysis
)
```

## Cache Structure

### On Disk

```
./cache/
├── enriched_schemas.db          # SQLite database with enriched schemas
├── enriched_schemas.db-wal      # Write-ahead log for concurrency
└── enriched_schemas.db-shm      # Shared memory for WAL mode
```

### In Memory (EnrichedSchema Object)

```python
EnrichedSchema(
    database_name="default",
    tables=[
        EnrichedTable(
            table_name="transactions",
            columns=[
                EnrichedColumn(
                    column_name="step",
                    data_type="integer",
                    total_count=6362620,
                    distinct_count=743,
                    sample_values=["1", "2", "3"]
                ),
                ...
            ],
            business_purpose="...",
            sample_data={...}
        )
    ],
    business_purpose="...",
    industry_domain="Financial Services"
)
```

## Best Practices

### 1. **Warm Cache on Startup** (Recommended)

Add automatic cache warming to your startup script:

```bash
#!/bin/bash
# Start server
uvicorn sql_agent.api.main:app &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Warm cache
curl -X POST "http://localhost:8000/api/schema/enrich_cache"

# Keep server running
wait $SERVER_PID
```

### 2. **Schedule Cache Refresh**

Use cron to refresh cache weekly:

```cron
# Refresh cache every Sunday at 2 AM
0 2 * * 0 curl -X POST "http://localhost:8000/api/schema/enrich_cache?force=true"
```

### 3. **Monitor Cache Health**

Add monitoring to track cache performance:

```bash
# Check cache status
curl "http://localhost:8000/api/schema/enriched_cache/status" | jq '.cache_stats.hit_ratio'
```

If hit ratio < 0.8, consider:
- Increasing TTL (longer cache lifetime)
- Increasing max_size_mb (more cache space)

### 4. **Handle Schema Migrations**

In your migration scripts:

```python
# After running database migration
import requests

# Invalidate old cache
requests.delete("http://localhost:8000/api/schema/enriched_cache")

# Warm new cache
requests.post("http://localhost:8000/api/schema/enrich_cache?force=true")
```

## Troubleshooting

### Cache Not Found Error

**Error**: `No enriched schema cached for database 'default'`

**Solution**:
```bash
curl -X POST "http://localhost:8000/api/schema/enrich_cache"
```

### Cache Initialization Failed

**Error**: `Enriched schema cache not available`

**Cause**: Missing dependencies or permissions

**Solution**:
1. Check cache directory permissions: `chmod 755 ./cache`
2. Verify dependencies are installed: `pip install -r requirements.txt`
3. Check logs for detailed error messages

### Stale Cache Data

**Symptom**: Query results don't reflect recent schema changes

**Solution**:
```bash
# Force refresh cache
curl -X POST "http://localhost:8000/api/schema/enrich_cache?force=true"
```

### High Memory Usage

**Cause**: Cache size too large for available RAM

**Solution**: Reduce `max_size_mb` in configuration:
```python
cache = EnrichedSchemaCache(
    max_size_mb=1024  # Reduce from 2048 to 1024
)
```

## Advanced Usage

### Custom Enrichment Logic

Extend `SchemaEnrichmentService` to add custom enrichment:

```python
from sql_agent.services.schema_enrichment import SchemaEnrichmentService

class CustomEnrichmentService(SchemaEnrichmentService):
    async def _enrich_table(self, database_name, table_data, include_sample_data):
        enriched = await super()._enrich_table(database_name, table_data, include_sample_data)

        # Add custom enrichment
        enriched.custom_metadata = await self.get_custom_metadata(table_data)

        return enriched
```

### Selective Table Caching

Cache only specific tables:

```python
from sql_agent.api.dependencies import get_enrichment_service, get_enriched_cache

enrichment_service = get_enrichment_service()
enriched_cache = get_enriched_cache()

# Enrich only high-priority tables
priority_tables = ["transactions", "customers", "orders"]
enriched_tables = await enrichment_service.enrich_specific_tables(
    database_name="default",
    table_names=priority_tables,
    include_sample_data=True
)

# Cache individually
for table in enriched_tables:
    await enriched_cache.set_enriched_table("default", table)
```

## FAQ

**Q: How much disk space does the cache use?**
A: Depends on schema size. For a 1-table database (transactions), ~5-10MB. For 100 tables, ~500MB-1GB.

**Q: Does the cache persist across server restarts?**
A: Yes! The cache is stored in SQLite and persists across restarts.

**Q: What happens if the cache expires?**
A: The system automatically falls back to querying the database directly. You'll see slightly slower performance until cache is warmed again.

**Q: Can I use this with multiple databases?**
A: Yes! Each database has its own cache entry. Use the `database_name` parameter.

**Q: How do I disable the cache?**
A: Simply don't call the `/enrich_cache` endpoint. The system will work without cache (with slower performance).

## Related Documentation

- [PersistentCache Implementation](sql_agent/cache/persistent_cache.py)
- [EnrichedSchemaCache Implementation](sql_agent/cache/enriched_schema_cache.py)
- [SchemaEnrichmentService Implementation](sql_agent/services/schema_enrichment.py)
- [API Routes](sql_agent/api/routes/schema.py)

## Support

For issues or questions:
1. Check the application logs for detailed error messages
2. Verify cache status: `GET /api/schema/enriched_cache/status`
3. Try invalidating and re-warming the cache
4. Check GitHub issues for similar problems
