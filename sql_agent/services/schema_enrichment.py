"""
Schema Enrichment Service

Service that orchestrates complete schema enrichment including:
- Column statistics and sample values
- Business intelligence from LLM
- Table relationships
- Sample data

This service is used to build EnrichedSchema objects for caching.

Author: ML Engineering Team
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..cache.enriched_schema_cache import (
    EnrichedSchema,
    EnrichedTable,
    EnrichedColumn
)

logger = logging.getLogger(__name__)


class SchemaEnrichmentService:
    """
    Service for building fully enriched schemas.

    Orchestrates data collection from:
    - DatabaseManager (introspection, statistics, samples)
    - SchemaAnalyzer (LLM intelligence)
    - SchemaProcessor (context enhancement)
    """

    def __init__(
        self,
        db_manager,
        schema_processor,
        schema_analyzer=None
    ):
        """
        Initialize schema enrichment service.

        Args:
            db_manager: DatabaseManager instance
            schema_processor: SchemaProcessor instance
            schema_analyzer: SchemaAnalyzer instance (optional)
        """
        self.db_manager = db_manager
        self.schema_processor = schema_processor
        self.schema_analyzer = schema_analyzer

    async def enrich_full_schema(
        self,
        database_name: str,
        include_sample_data: bool = True,
        max_concurrent_tables: int = 5
    ) -> Optional[EnrichedSchema]:
        """
        Build a complete enriched schema with all metadata.

        Args:
            database_name: Database to enrich
            include_sample_data: Whether to include sample data
            max_concurrent_tables: Max tables to process concurrently

        Returns:
            EnrichedSchema object with all enrichment data
        """
        try:
            logger.info(f"Starting full schema enrichment for {database_name}...")
            start_time = datetime.utcnow()

            # Step 1: Get base schema from database
            base_schema = await self.db_manager.get_database_schema(database_name)
            if not base_schema:
                logger.error(f"Failed to get base schema for {database_name}")
                return None

            tables_data = base_schema.get("tables", [])
            logger.info(f"Found {len(tables_data)} tables in {database_name}")

            # Step 2: Get business intelligence from LLM (if available)
            business_intelligence = None
            if self.schema_analyzer:
                try:
                    logger.info("Analyzing database business intelligence...")
                    business_intelligence = await self.schema_analyzer.analyze_database_intelligence(
                        base_schema
                    )
                except Exception as e:
                    logger.warning(f"Business intelligence analysis failed: {e}")

            # Step 3: Enrich each table with statistics and samples
            enriched_tables = []

            # Process tables in batches for concurrency control
            semaphore = asyncio.Semaphore(max_concurrent_tables)

            async def enrich_single_table(table_data: Dict[str, Any]) -> Optional[EnrichedTable]:
                async with semaphore:
                    return await self._enrich_table(
                        database_name,
                        table_data,
                        include_sample_data
                    )

            # Enrich all tables concurrently (with limit)
            logger.info(f"Enriching {len(tables_data)} tables (max {max_concurrent_tables} concurrent)...")
            enrichment_tasks = [enrich_single_table(table) for table in tables_data]
            enriched_results = await asyncio.gather(*enrichment_tasks, return_exceptions=True)

            # Collect successful enrichments
            for result in enriched_results:
                if isinstance(result, EnrichedTable):
                    enriched_tables.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Table enrichment failed: {result}")

            logger.info(f"Successfully enriched {len(enriched_tables)}/{len(tables_data)} tables")

            # Step 4: Build EnrichedSchema object
            enriched_schema = EnrichedSchema(
                database_name=database_name,
                tables=enriched_tables,
                business_purpose=business_intelligence.get("business_purpose") if business_intelligence else None,
                industry_domain=business_intelligence.get("industry_domain") if business_intelligence else None,
                discovered_domains=business_intelligence.get("discovered_domains") if business_intelligence else None,
                relationships=base_schema.get("relationships", []),
                enriched_at=datetime.utcnow(),
                version="1.0"
            )

            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"✓ Schema enrichment completed in {duration:.2f}s for {database_name}")

            return enriched_schema

        except Exception as e:
            logger.error(f"Full schema enrichment failed for {database_name}: {e}", exc_info=True)
            return None

    async def _enrich_table(
        self,
        database_name: str,
        table_data: Dict[str, Any],
        include_sample_data: bool = True
    ) -> Optional[EnrichedTable]:
        """
        Enrich a single table with all metadata.

        Args:
            database_name: Database name
            table_data: Base table data from introspection
            include_sample_data: Whether to include sample data

        Returns:
            EnrichedTable with all enrichment
        """
        try:
            table_name = table_data.get("name", "")
            logger.debug(f"Enriching table: {table_name}")

            # Get column statistics
            column_stats = await self.db_manager.get_column_statistics(table_name)
            if not column_stats:
                logger.warning(f"No column statistics for {table_name}")
                column_stats = {}

            # Get sample data (if requested)
            sample_data = None
            if include_sample_data:
                try:
                    sample_data = await self.db_manager.get_sample_data(table_name, limit=5)
                except Exception as e:
                    logger.debug(f"Failed to get sample data for {table_name}: {e}")

            # Build enriched columns
            enriched_columns = []
            columns_data = table_data.get("columns", [])

            for col_data in columns_data:
                col_name = col_data.get("name", "")
                col_stats = column_stats.get(col_name, {})

                enriched_col = EnrichedColumn(
                    column_name=col_name,
                    data_type=col_data.get("data_type", "unknown"),
                    nullable=col_data.get("nullable", True),
                    primary_key=col_data.get("primary_key", False),
                    foreign_key=col_data.get("foreign_key", False),
                    business_concept=col_data.get("business_concept", ""),
                    total_count=col_stats.get("total_count"),
                    distinct_count=col_stats.get("distinct_count"),
                    null_count=col_stats.get("null_count"),
                    min_value=col_stats.get("min_value"),
                    max_value=col_stats.get("max_value"),
                    avg_value=col_stats.get("avg_value"),
                    sample_values=col_stats.get("sample_values", [])
                )
                enriched_columns.append(enriched_col)

            # Get table business context (if available from analyzer)
            business_context = None
            if self.schema_analyzer:
                try:
                    # Try to get table-specific business context
                    table_context = await self.schema_analyzer.analyze_table_business_context(
                        table_data
                    )
                    business_context = table_context
                except Exception as e:
                    logger.debug(f"No business context for {table_name}: {e}")

            # Build EnrichedTable
            enriched_table = EnrichedTable(
                table_name=table_name,
                columns=enriched_columns,
                row_count=table_data.get("statistics", {}).get("live_tuples"),
                business_purpose=business_context.get("business_purpose") if business_context else None,
                business_role=business_context.get("business_role") if business_context else None,
                business_domains=business_context.get("business_domains") if business_context else None,
                sample_data=sample_data,
                relationships=table_data.get("foreign_keys", []),
                criticality=business_context.get("criticality") if business_context else None,
                confidence_score=business_context.get("confidence_score") if business_context else None
            )

            return enriched_table

        except Exception as e:
            logger.error(f"Failed to enrich table {table_data.get('name', 'unknown')}: {e}")
            return None

    async def enrich_specific_tables(
        self,
        database_name: str,
        table_names: List[str],
        include_sample_data: bool = True
    ) -> List[EnrichedTable]:
        """
        Enrich only specific tables.

        Args:
            database_name: Database name
            table_names: List of table names to enrich
            include_sample_data: Whether to include sample data

        Returns:
            List of EnrichedTable objects
        """
        try:
            logger.info(f"Enriching {len(table_names)} specific tables in {database_name}")

            # Get base schema
            base_schema = await self.db_manager.get_database_schema(database_name)
            if not base_schema:
                return []

            # Filter to requested tables
            all_tables = base_schema.get("tables", [])
            tables_to_enrich = [
                table for table in all_tables
                if table.get("name") in table_names
            ]

            # Enrich each table
            enriched_tables = []
            for table_data in tables_to_enrich:
                enriched = await self._enrich_table(
                    database_name,
                    table_data,
                    include_sample_data
                )
                if enriched:
                    enriched_tables.append(enriched)

            logger.info(f"Successfully enriched {len(enriched_tables)}/{len(table_names)} tables")
            return enriched_tables

        except Exception as e:
            logger.error(f"Failed to enrich specific tables: {e}")
            return []

    async def get_enriched_context_for_query(
        self,
        database_name: str,
        selected_tables: List[str]
    ) -> Dict[str, Any]:
        """
        Get enriched context for specific tables (optimized for query routing).

        This is what the Router agent needs - column contexts with statistics.

        Args:
            database_name: Database name
            selected_tables: Tables relevant to the query

        Returns:
            Enriched context dictionary in Router agent format
        """
        try:
            # Enrich the selected tables
            enriched_tables = await self.enrich_specific_tables(
                database_name,
                selected_tables,
                include_sample_data=False  # Query routing doesn't need sample data
            )

            # Convert to Router agent format
            column_contexts = {}
            for enriched_table in enriched_tables:
                column_contexts[enriched_table.table_name] = [
                    col.to_dict() for col in enriched_table.columns
                ]

            return {
                "selected_tables": selected_tables,
                "table_count": len(selected_tables),
                "database_name": database_name,
                "column_contexts": column_contexts
            }

        except Exception as e:
            logger.error(f"Failed to get enriched context for query: {e}")
            return {
                "selected_tables": selected_tables,
                "table_count": len(selected_tables),
                "database_name": database_name,
                "column_contexts": {}
            }
