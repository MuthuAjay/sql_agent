"""
Context management for RAG functionality.

FIXED: Proper initialization handling and graceful degradation when vector store is unavailable.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core.state import SchemaContext
from ..rag.schema import schema_processor
from ..utils.logging import get_logger


class VectorSearchResult:
    """Simple result class for vector search results when vector store is unavailable."""
    def __init__(self, content: str, metadata: Dict[str, Any], distance: float = 0.0, embedding: Optional[List[float]] = None):
        self.content = content
        self.metadata = metadata
        self.distance = distance
        self.embedding = embedding


class ContextManager:
    """Manage RAG context retrieval and storage with graceful degradation."""
    
    def __init__(self):
        self.logger = get_logger("rag.context_manager")
        self._initialized = False
        self._last_schema_update: Optional[datetime] = None
        self._vector_store_available = False
        self._embedding_service_available = False
        
        # FIXED: Initialize these to None and handle gracefully
        self.vector_store = None
        self.embedding_service = None
    
    async def initialize(self) -> None:
        """
        FIXED: Initialize the context manager with graceful degradation.
        If vector store or embedding services are unavailable, still allow basic functionality.
        """
        try:
            self.logger.info("initializing_context_manager")
            
            # Try to initialize vector store (optional)
            try:
                from ..rag.vector_store import vector_store
                await vector_store.initialize()
                self.vector_store = vector_store
                self._vector_store_available = True
                self.logger.info("vector_store_initialized")
            except Exception as e:
                self.logger.warning("vector_store_initialization_failed", error=str(e))
                self._vector_store_available = False
            
            # Try to initialize embedding service (optional)
            try:
                from ..rag.embeddings import embedding_service
                await embedding_service.get_embedding_dimension()
                self.embedding_service = embedding_service
                self._embedding_service_available = True
                self.logger.info("embedding_service_initialized")
            except Exception as e:
                self.logger.warning("embedding_service_initialization_failed", error=str(e))
                self._embedding_service_available = False
            
            # FIXED: Always try to initialize basic schema contexts
            try:
                await self._initialize_schema_contexts()
                self.logger.info("basic_schema_contexts_initialized")
            except Exception as e:
                self.logger.warning("schema_context_initialization_failed", error=str(e))
                # Don't fail completely - we can still provide basic functionality
            
            # FIXED: Mark as initialized even if some components failed
            self._initialized = True
            
            self.logger.info("context_manager_initialized", 
                           vector_store_available=self._vector_store_available,
                           embedding_service_available=self._embedding_service_available)
            
        except Exception as e:
            self.logger.error("context_manager_initialization_failed", error=str(e), exc_info=True)
            # FIXED: Still mark as initialized for basic functionality
            self._initialized = True
            self.logger.warning("context_manager_initialized_with_degraded_functionality")
    
    def _ensure_initialized(self) -> None:
        """Ensure the context manager is initialized."""
        if not self._initialized:
            raise RuntimeError("Context manager not initialized. Call initialize() first.")
    
    async def _initialize_schema_contexts(self) -> None:
        """FIXED: Initialize schema contexts with graceful handling."""
        try:
            self.logger.info("initializing_schema_contexts")
            
            # Extract schema contexts using the fixed schema processor
            contexts = await schema_processor.extract_schema_contexts()
            
            if not contexts:
                self.logger.warning("no_schema_contexts_extracted")
                return
            
            # If vector store is available, add contexts to it
            if self._vector_store_available and self.vector_store:
                try:
                    context_ids = await self.vector_store.add_schema_contexts(contexts)
                    self.logger.info("contexts_added_to_vector_store", context_count=len(context_ids))
                except Exception as e:
                    self.logger.warning("add_contexts_to_vector_store_failed", error=str(e))
            
            self._last_schema_update = datetime.utcnow()
            
            self.logger.info("schema_contexts_initialized", 
                           context_count=len(contexts))
            
        except Exception as e:
            self.logger.error("initialize_schema_contexts_failed", error=str(e))
            # Don't raise - allow degraded functionality
    
    async def retrieve_schema_context(
        self, 
        query: str, 
        limit: int = 5,
        min_similarity: float = 0.7
    ) -> List[SchemaContext]:
        """
        FIXED: Retrieve relevant schema context with fallback to simple matching.
        """
        self._ensure_initialized()
        
        try:
            self.logger.info("retrieving_schema_context", 
                           query=query[:100], 
                           limit=limit)
            
            # If vector store is available, use it
            if self._vector_store_available and self.vector_store and self._embedding_service_available and self.embedding_service:
                return await self._retrieve_schema_context_vector(query, limit, min_similarity)
            else:
                # Fallback to simple keyword-based retrieval
                return await self._retrieve_schema_context_fallback(query, limit)
            
        except Exception as e:
            self.logger.error("retrieve_schema_context_failed", 
                            query=query[:100], error=str(e))
            # Return empty list instead of raising
            return []
    
    async def _retrieve_schema_context_vector(self, query: str, limit: int, min_similarity: float) -> List[SchemaContext]:
        """Retrieve schema context using vector search."""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_query(query)
            
            # Search for similar contexts
            search_results = await self.vector_store.similarity_search(
                query_embedding, 
                limit=limit * 2  # Get more results to filter by similarity
            )
            
            # Filter by similarity threshold
            relevant_results = [
                result for result in search_results 
                if result.distance >= min_similarity
            ]
            
            # Convert to SchemaContext objects
            contexts = []
            for result in relevant_results[:limit]:
                context = self._create_schema_context_from_result(result)
                contexts.append(context)
            
            self.logger.info("schema_context_retrieved_vector", 
                           query=query[:100],
                           results_count=len(contexts))
            
            return contexts
            
        except Exception as e:
            self.logger.warning("vector_retrieval_failed", error=str(e))
            # Fallback to simple retrieval
            return await self._retrieve_schema_context_fallback(query, limit)
    
    async def _retrieve_schema_context_fallback(self, query: str, limit: int) -> List[SchemaContext]:
        """FIXED: Simple fallback retrieval using schema processor."""
        try:
            self.logger.info("using_fallback_schema_retrieval", query=query[:100])
            
            # Get database schema
            schema_data = await schema_processor.extract_database_schema("default")
            tables = schema_data.get("tables", [])
            
            if not tables:
                return []
            
            # Simple keyword matching
            query_lower = query.lower()
            contexts = []
            
            for table in tables:
                table_name = table.get("name", "")
                description = table.get("enhanced_description", table.get("description", ""))
                
                # Check if table is relevant
                table_relevant = False
                if any(word in table_name.lower() for word in query_lower.split()):
                    table_relevant = True
                elif any(word in description.lower() for word in query_lower.split()):
                    table_relevant = True
                
                if table_relevant:
                    # Create table context
                    context = SchemaContext(
                        table_name=table_name,
                        column_name=None,
                        data_type=None,
                        description=description,
                        sample_values=[],
                        relationships=[],
                        embedding=None
                    )
                    contexts.append(context)
                    
                    # Add relevant column contexts
                    column_details = table.get("column_details", {})
                    for column_name, column_detail in column_details.items():
                        if any(word in column_name.lower() for word in query_lower.split()):
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
                
                if len(contexts) >= limit:
                    break
            
            self.logger.info("schema_context_retrieved_fallback", 
                           query=query[:100],
                           results_count=len(contexts))
            
            return contexts[:limit]
            
        except Exception as e:
            self.logger.error("fallback_retrieval_failed", error=str(e))
            return []
    
    async def retrieve_context_by_tables(
        self, 
        table_names: List[str]
    ) -> List[SchemaContext]:
        """FIXED: Retrieve schema context for specific tables."""
        self._ensure_initialized()
        
        try:
            self.logger.info("retrieving_context_by_tables", table_names=table_names)
            
            # If vector store is available, use it
            if self._vector_store_available and self.vector_store:
                return await self._retrieve_context_by_tables_vector(table_names)
            else:
                # Fallback to direct schema lookup
                return await self._retrieve_context_by_tables_fallback(table_names)
            
        except Exception as e:
            self.logger.error("retrieve_context_by_tables_failed", 
                            table_names=table_names, error=str(e))
            return []
    
    async def _retrieve_context_by_tables_vector(self, table_names: List[str]) -> List[SchemaContext]:
        """Retrieve context by tables using vector store."""
        try:
            contexts = []
            
            for table_name in table_names:
                # Get contexts for this table
                table_results = await self.vector_store.get_by_table_name(table_name)
                
                # Convert to SchemaContext objects
                for result in table_results:
                    context = self._create_schema_context_from_result(result)
                    contexts.append(context)
            
            return contexts
            
        except Exception as e:
            self.logger.warning("vector_table_retrieval_failed", error=str(e))
            return await self._retrieve_context_by_tables_fallback(table_names)
    
    async def _retrieve_context_by_tables_fallback(self, table_names: List[str]) -> List[SchemaContext]:
        """FIXED: Fallback retrieval for specific tables."""
        try:
            # Get database schema
            schema_data = await schema_processor.extract_database_schema("default")
            tables = schema_data.get("tables", [])
            
            contexts = []
            
            for table in tables:
                table_name = table.get("name", "")
                if table_name in table_names:
                    # Create table context
                    description = table.get("enhanced_description", table.get("description", ""))
                    context = SchemaContext(
                        table_name=table_name,
                        column_name=None,
                        data_type=None,
                        description=description,
                        sample_values=[],
                        relationships=[],
                        embedding=None
                    )
                    contexts.append(context)
                    
                    # Add column contexts
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
            
            self.logger.info("context_by_tables_retrieved_fallback", 
                           table_names=table_names,
                           context_count=len(contexts))
            
            return contexts
            
        except Exception as e:
            self.logger.error("fallback_table_retrieval_failed", error=str(e))
            return []
    
    async def search_schema_by_keywords(
        self, 
        keywords: List[str], 
        limit: int = 10
    ) -> List[SchemaContext]:
        """FIXED: Search schema by keywords with fallback."""
        self._ensure_initialized()
        
        try:
            self.logger.info("searching_schema_by_keywords", keywords=keywords)
            
            # Use fallback approach (keyword matching)
            contexts = []
            
            for keyword in keywords:
                keyword_contexts = await self._retrieve_schema_context_fallback(keyword, limit)
                contexts.extend(keyword_contexts)
            
            # Remove duplicates
            unique_contexts = self._deduplicate_contexts(contexts)
            
            self.logger.info("schema_search_by_keywords_completed", 
                           keywords=keywords,
                           unique_contexts=len(unique_contexts))
            
            return unique_contexts[:limit]
            
        except Exception as e:
            self.logger.error("search_schema_by_keywords_failed", 
                            keywords=keywords, error=str(e))
            return []
    
    def _create_schema_context_from_result(self, result: VectorSearchResult) -> SchemaContext:
        """Create a SchemaContext from a VectorSearchResult."""
        metadata = result.metadata
        
        context = SchemaContext(
            table_name=metadata.get("table_name", ""),
            column_name=metadata.get("column_name") or None,
            data_type=metadata.get("data_type") or None,
            description=result.content,
            embedding=result.embedding
        )
        
        return context
    
    def _deduplicate_contexts(self, contexts: List[SchemaContext]) -> List[SchemaContext]:
        """Remove duplicate contexts based on table_name and column_name."""
        seen = set()
        unique_contexts = []
        
        for context in contexts:
            key = (context.table_name, context.column_name)
            if key not in seen:
                seen.add(key)
                unique_contexts.append(context)
        
        return unique_contexts
    
    async def refresh_schema_contexts(self) -> None:
        """FIXED: Refresh schema contexts with graceful handling."""
        self._ensure_initialized()
        
        try:
            self.logger.info("refreshing_schema_contexts")
            
            # Clear existing contexts if vector store is available
            if self._vector_store_available and self.vector_store:
                try:
                    await self.vector_store.clear_collection()
                except Exception as e:
                    self.logger.warning("clear_vector_store_failed", error=str(e))
            
            # Extract fresh schema contexts
            contexts = await schema_processor.extract_schema_contexts()
            
            if not contexts:
                self.logger.warning("no_schema_contexts_extracted_during_refresh")
                return
            
            # Add new contexts to vector store if available
            if self._vector_store_available and self.vector_store:
                try:
                    context_ids = await self.vector_store.add_schema_contexts(contexts)
                    self.logger.info("contexts_refreshed_in_vector_store", context_count=len(context_ids))
                except Exception as e:
                    self.logger.warning("add_refreshed_contexts_failed", error=str(e))
            
            self._last_schema_update = datetime.utcnow()
            
            self.logger.info("schema_contexts_refreshed", 
                           context_count=len(contexts))
            
        except Exception as e:
            self.logger.error("refresh_schema_contexts_failed", error=str(e))
            # Don't raise - allow continued operation
    
    async def get_context_statistics(self) -> Dict[str, Any]:
        """FIXED: Get statistics with graceful handling."""
        self._ensure_initialized()
        
        try:
            stats = {
                "vector_store_available": self._vector_store_available,
                "embedding_service_available": self._embedding_service_available,
                "last_update": self._last_schema_update.isoformat() if self._last_schema_update else None,
            }
            
            # Get vector store statistics if available
            if self._vector_store_available and self.vector_store:
                try:
                    vector_stats = await self.vector_store.get_collection_stats()
                    stats["vector_store"] = vector_stats
                except Exception as e:
                    self.logger.warning("get_vector_stats_failed", error=str(e))
                    stats["vector_store"] = {"error": str(e)}
            
            # Get schema summary
            try:
                schema_summary = await schema_processor.get_schema_summary()
                stats["schema"] = schema_summary
            except Exception as e:
                self.logger.warning("get_schema_summary_failed", error=str(e))
                stats["schema"] = {"error": str(e)}
            
            # Get embedding dimension if available
            if self._embedding_service_available and self.embedding_service:
                try:
                    embedding_dim = await self.embedding_service.get_embedding_dimension()
                    stats["embedding_dimension"] = embedding_dim
                except Exception as e:
                    self.logger.warning("get_embedding_dimension_failed", error=str(e))
            
            self.logger.info("context_statistics_retrieved", available_services={
                "vector_store": self._vector_store_available,
                "embedding_service": self._embedding_service_available
            })
            
            return stats
            
        except Exception as e:
            self.logger.error("get_context_statistics_failed", error=str(e))
            return {"error": str(e)}
    
    async def update_table_context(self, table_name: str) -> None:
        """FIXED: Update context for specific table with graceful handling."""
        self._ensure_initialized()
        
        try:
            self.logger.info("updating_table_context", table_name=table_name)
            
            # Delete existing contexts if vector store is available
            deleted_count = 0
            if self._vector_store_available and self.vector_store:
                try:
                    deleted_count = await self.vector_store.delete_by_table_name(table_name)
                except Exception as e:
                    self.logger.warning("delete_table_contexts_failed", error=str(e))
            
            # Get fresh schema info for this table
            schema_info = await schema_processor.extract_schema_contexts()
            
            # Filter contexts for this table
            table_contexts = [
                context for context in schema_info 
                if context.table_name == table_name
            ]
            
            if table_contexts:
                # Add new contexts if vector store is available
                if self._vector_store_available and self.vector_store:
                    try:
                        context_ids = await self.vector_store.add_schema_contexts(table_contexts)
                        self.logger.info("table_context_updated", 
                                       table_name=table_name,
                                       deleted_count=deleted_count,
                                       new_contexts=len(context_ids))
                    except Exception as e:
                        self.logger.warning("add_table_contexts_failed", error=str(e))
                else:
                    self.logger.info("table_context_prepared", 
                                   table_name=table_name,
                                   new_contexts=len(table_contexts))
            else:
                self.logger.warning("no_contexts_found_for_table", table_name=table_name)
            
        except Exception as e:
            self.logger.error("update_table_context_failed", 
                            table_name=table_name, error=str(e))
    
    async def get_relevant_tables(self, query: str, limit: int = 5) -> List[str]:
        """FIXED: Get relevant table names with fallback."""
        self._ensure_initialized()
        
        try:
            # Retrieve schema context
            contexts = await self.retrieve_schema_context(query, limit=limit * 2)
            
            # Extract unique table names
            table_names = list(set(context.table_name for context in contexts if context.table_name))
            
            return table_names[:limit]
            
        except Exception as e:
            self.logger.error("get_relevant_tables_failed", 
                            query=query[:100], error=str(e))
            return []
    
    def is_schema_stale(self, max_age_minutes: int = 60) -> bool:
        """Check if the schema context is stale."""
        if not self._last_schema_update:
            return True
        
        age_minutes = (datetime.utcnow() - self._last_schema_update).total_seconds() / 60
        return age_minutes > max_age_minutes
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of RAG services."""
        return {
            "initialized": self._initialized,
            "vector_store_available": self._vector_store_available,
            "embedding_service_available": self._embedding_service_available,
            "last_schema_update": self._last_schema_update.isoformat() if self._last_schema_update else None
        }


# Global context manager instance
context_manager = ContextManager()