"""Context management for RAG functionality."""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core.state import SchemaContext
from ..rag.vector_store import vector_store, VectorSearchResult
from ..rag.embeddings import embedding_service
from ..rag.schema import schema_processor
from ..utils.logging import get_logger


class ContextManager:
    """Manage RAG context retrieval and storage."""
    
    def __init__(self):
        self.logger = get_logger("rag.context_manager")
        self._initialized = False
        self._last_schema_update: Optional[datetime] = None
    
    async def initialize(self) -> None:
        """Initialize the context manager and vector store."""
        try:
            self.logger.info("initializing_context_manager")
            
            # Initialize vector store
            await vector_store.initialize()
            
            # Initialize embedding service
            await embedding_service.get_embedding_dimension()
            
            # Extract and store initial schema contexts
            try:
                await self._initialize_schema_contexts()
            except Exception as e:
                self.logger.error("schema_context_initialization_failed", error=repr(e), exc_info=True)
                raise RuntimeError(f"Failed to initialize schema contexts: {e}")
            
            self._initialized = True
            self.logger.info("context_manager_initialized")
            
        except Exception as e:
            self.logger.error("context_manager_initialization_failed", error=repr(e), exc_info=True)
            raise
    
    def _ensure_initialized(self) -> None:
        """Ensure the context manager is initialized."""
        if not self._initialized:
            raise RuntimeError("Context manager not initialized. Call initialize() first.")
    
    async def _initialize_schema_contexts(self) -> None:
        """Initialize schema contexts in the vector store."""
        try:
            self.logger.info("initializing_schema_contexts")
            
            # Extract schema contexts
            contexts = await schema_processor.extract_schema_contexts()
            
            if not contexts:
                self.logger.warning("no_schema_contexts_extracted")
                return
            
            # Add contexts to vector store
            context_ids = await vector_store.add_schema_contexts(contexts)
            
            self._last_schema_update = datetime.utcnow()
            
            self.logger.info("schema_contexts_initialized", 
                           context_count=len(contexts),
                           context_ids=len(context_ids))
            
        except Exception as e:
            self.logger.error("initialize_schema_contexts_failed", error=str(e))
            raise
    
    async def retrieve_schema_context(
        self, 
        query: str, 
        limit: int = 5,
        min_similarity: float = 0.7
    ) -> List[SchemaContext]:
        """Retrieve relevant schema context for a query."""
        self._ensure_initialized()
        
        try:
            self.logger.info("retrieving_schema_context", 
                           query=query[:100], 
                           limit=limit)
            
            # Generate query embedding
            query_embedding = await embedding_service.embed_query(query)
            
            # Search for similar contexts
            search_results = await vector_store.similarity_search(
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
            
            self.logger.info("schema_context_retrieved", 
                           query=query[:100],
                           results_count=len(contexts),
                           total_searched=len(search_results))
            
            return contexts
            
        except Exception as e:
            self.logger.error("retrieve_schema_context_failed", 
                            query=query[:100], error=str(e))
            raise
    
    async def retrieve_context_by_tables(
        self, 
        table_names: List[str]
    ) -> List[SchemaContext]:
        """Retrieve schema context for specific tables."""
        self._ensure_initialized()
        
        try:
            self.logger.info("retrieving_context_by_tables", table_names=table_names)
            
            contexts = []
            
            for table_name in table_names:
                # Get contexts for this table
                table_results = await vector_store.get_by_table_name(table_name)
                
                # Convert to SchemaContext objects
                for result in table_results:
                    context = self._create_schema_context_from_result(result)
                    contexts.append(context)
            
            self.logger.info("context_by_tables_retrieved", 
                           table_names=table_names,
                           context_count=len(contexts))
            
            return contexts
            
        except Exception as e:
            self.logger.error("retrieve_context_by_tables_failed", 
                            table_names=table_names, error=str(e))
            raise
    
    async def search_schema_by_keywords(
        self, 
        keywords: List[str], 
        limit: int = 10
    ) -> List[SchemaContext]:
        """Search schema by keywords."""
        self._ensure_initialized()
        
        try:
            self.logger.info("searching_schema_by_keywords", keywords=keywords)
            
            contexts = []
            
            for keyword in keywords:
                # Search by text
                search_results = await vector_store.search_by_text(keyword, limit=limit)
                
                # Convert to SchemaContext objects
                for result in search_results:
                    context = self._create_schema_context_from_result(result)
                    contexts.append(context)
            
            # Remove duplicates based on table_name and column_name
            unique_contexts = self._deduplicate_contexts(contexts)
            
            self.logger.info("schema_search_by_keywords_completed", 
                           keywords=keywords,
                           unique_contexts=len(unique_contexts))
            
            return unique_contexts[:limit]
            
        except Exception as e:
            self.logger.error("search_schema_by_keywords_failed", 
                            keywords=keywords, error=str(e))
            raise
    
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
        """Refresh schema contexts from the database."""
        self._ensure_initialized()
        
        try:
            self.logger.info("refreshing_schema_contexts")
            
            # Clear existing contexts
            await vector_store.clear_collection()
            
            # Extract fresh schema contexts
            contexts = await schema_processor.extract_schema_contexts()
            
            if not contexts:
                self.logger.warning("no_schema_contexts_extracted_during_refresh")
                return
            
            # Add new contexts to vector store
            context_ids = await vector_store.add_schema_contexts(contexts)
            
            self._last_schema_update = datetime.utcnow()
            
            self.logger.info("schema_contexts_refreshed", 
                           context_count=len(contexts),
                           context_ids=len(context_ids))
            
        except Exception as e:
            self.logger.error("refresh_schema_contexts_failed", error=str(e))
            raise
    
    async def get_context_statistics(self) -> Dict[str, Any]:
        """Get statistics about the context store."""
        self._ensure_initialized()
        
        try:
            # Get vector store statistics
            vector_stats = await vector_store.get_collection_stats()
            
            # Get schema summary
            schema_summary = await schema_processor.get_schema_summary()
            
            stats = {
                "vector_store": vector_stats,
                "schema": schema_summary,
                "last_update": self._last_schema_update.isoformat() if self._last_schema_update else None,
                "embedding_dimension": await embedding_service.get_embedding_dimension()
            }
            
            self.logger.info("context_statistics_retrieved", stats=stats)
            return stats
            
        except Exception as e:
            self.logger.error("get_context_statistics_failed", error=str(e))
            raise
    
    async def update_table_context(self, table_name: str) -> None:
        """Update context for a specific table."""
        self._ensure_initialized()
        
        try:
            self.logger.info("updating_table_context", table_name=table_name)
            
            # Delete existing contexts for this table
            deleted_count = await vector_store.delete_by_table_name(table_name)
            
            # Get fresh schema info for this table
            schema_info = await schema_processor.extract_schema_contexts()
            
            # Filter contexts for this table
            table_contexts = [
                context for context in schema_info 
                if context.table_name == table_name
            ]
            
            if table_contexts:
                # Add new contexts
                context_ids = await vector_store.add_schema_contexts(table_contexts)
                
                self.logger.info("table_context_updated", 
                               table_name=table_name,
                               deleted_count=deleted_count,
                               new_contexts=len(table_contexts))
            else:
                self.logger.warning("no_contexts_found_for_table", table_name=table_name)
            
        except Exception as e:
            self.logger.error("update_table_context_failed", 
                            table_name=table_name, error=str(e))
            raise
    
    async def get_relevant_tables(self, query: str, limit: int = 5) -> List[str]:
        """Get relevant table names for a query."""
        self._ensure_initialized()
        
        try:
            # Retrieve schema context
            contexts = await self.retrieve_schema_context(query, limit=limit * 2)
            
            # Extract unique table names
            table_names = list(set(context.table_name for context in contexts))
            
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


# Global context manager instance
context_manager = ContextManager() 