"""Vector store for RAG functionality using ChromaDB."""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import uuid

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.api.models.Collection import Collection

from ..core.config import settings
from ..core.state import SchemaContext
from ..utils.logging import get_logger


@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""
    
    id: str
    content: str
    metadata: Dict[str, Any]
    distance: float
    embedding: Optional[List[float]] = None


class VectorStore:
    """Vector store using ChromaDB for schema context storage and retrieval."""
    
    def __init__(self, collection_name: str = "sql_agent_schema"):
        self.logger = get_logger("rag.vector_store")
        self.collection_name = collection_name
        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[Collection] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the vector store and create collection."""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=settings.chroma_db_path or "./chroma_db",
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "SQL Agent schema context embeddings"}
            )
            
            self._initialized = True
            self.logger.info("vector_store_initialized", collection_name=self.collection_name)
            
        except Exception as e:
            self.logger.error("vector_store_initialization_failed", error=str(e))
            raise
    
    def _ensure_initialized(self) -> None:
        """Ensure the vector store is initialized."""
        if not self._initialized:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
    
    async def add_schema_contexts(self, contexts: List[SchemaContext]) -> List[str]:
        """Add schema contexts to the vector store."""
        self._ensure_initialized()
        
        try:
            if not contexts:
                return []
            
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            embeddings = []
            
            for context in contexts:
                # Generate unique ID
                context_id = str(uuid.uuid4())
                ids.append(context_id)
                
                # Create document content
                content = self._create_context_content(context)
                documents.append(content)
                
                # Create metadata
                metadata = {
                    "table_name": context.table_name,
                    "column_name": context.column_name or "",
                    "data_type": context.data_type or "",
                    "type": "schema_context",
                    "context_id": context_id
                }
                metadatas.append(metadata)
                
                # Use existing embedding or generate new one
                if context.embedding:
                    embeddings.append(context.embedding)
                else:
                    # This will be handled by ChromaDB's embedding function
                    embeddings.append(None)
            
            # Add to collection
            if embeddings[0] is not None:
                # Use provided embeddings
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
            else:
                # Let ChromaDB generate embeddings
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
            
            self.logger.info("schema_contexts_added", count=len(contexts))
            return ids
            
        except Exception as e:
            self.logger.error("add_schema_contexts_failed", error=str(e))
            raise
    
    async def similarity_search(
        self, 
        query_embedding: List[float], 
        limit: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar schema contexts."""
        self._ensure_initialized()
        
        try:
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=filter_dict
            )
            
            # Convert to VectorSearchResult objects
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    result = VectorSearchResult(
                        id=results['ids'][0][i],
                        content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i],
                        distance=results['distances'][0][i] if results['distances'] else 0.0,
                        embedding=results['embeddings'][0][i] if results['embeddings'] else None
                    )
                    search_results.append(result)
            
            self.logger.info("similarity_search_completed", 
                           query_length=len(query_embedding), 
                           results_count=len(search_results))
            
            return search_results
            
        except Exception as e:
            self.logger.error("similarity_search_failed", error=str(e))
            raise
    
    async def search_by_text(
        self, 
        query: str, 
        limit: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar contexts using text query."""
        self._ensure_initialized()
        
        try:
            # Perform text-based search
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where=filter_dict
            )
            
            # Convert to VectorSearchResult objects
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    result = VectorSearchResult(
                        id=results['ids'][0][i],
                        content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i],
                        distance=results['distances'][0][i] if results['distances'] else 0.0,
                        embedding=results['embeddings'][0][i] if results['embeddings'] else None
                    )
                    search_results.append(result)
            
            self.logger.info("text_search_completed", 
                           query=query[:100], 
                           results_count=len(search_results))
            
            return search_results
            
        except Exception as e:
            self.logger.error("text_search_failed", error=str(e))
            raise
    
    async def get_by_table_name(self, table_name: str) -> List[VectorSearchResult]:
        """Get all contexts for a specific table."""
        self._ensure_initialized()
        
        try:
            results = self.collection.get(
                where={"table_name": table_name}
            )
            
            search_results = []
            if results['ids']:
                for i in range(len(results['ids'])):
                    result = VectorSearchResult(
                        id=results['ids'][i],
                        content=results['documents'][i],
                        metadata=results['metadatas'][i],
                        distance=0.0,  # No distance for direct lookup
                        embedding=results['embeddings'][i] if results['embeddings'] else None
                    )
                    search_results.append(result)
            
            self.logger.info("table_contexts_retrieved", 
                           table_name=table_name, 
                           count=len(search_results))
            
            return search_results
            
        except Exception as e:
            self.logger.error("get_by_table_name_failed", error=str(e), table_name=table_name)
            raise
    
    async def delete_by_table_name(self, table_name: str) -> int:
        """Delete all contexts for a specific table."""
        self._ensure_initialized()
        
        try:
            # Get IDs to delete
            results = self.collection.get(
                where={"table_name": table_name}
            )
            
            if not results['ids']:
                return 0
            
            # Delete the contexts
            self.collection.delete(ids=results['ids'])
            
            deleted_count = len(results['ids'])
            self.logger.info("table_contexts_deleted", 
                           table_name=table_name, 
                           count=deleted_count)
            
            return deleted_count
            
        except Exception as e:
            self.logger.error("delete_by_table_name_failed", error=str(e), table_name=table_name)
            raise
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        self._ensure_initialized()
        
        try:
            count = self.collection.count()
            
            # Get sample metadata to understand structure
            sample_results = self.collection.get(limit=1)
            table_names = set()
            
            if sample_results['metadatas']:
                all_results = self.collection.get()
                for metadata in all_results['metadatas']:
                    if metadata and 'table_name' in metadata:
                        table_names.add(metadata['table_name'])
            
            stats = {
                "total_contexts": count,
                "unique_tables": len(table_names),
                "table_names": list(table_names),
                "collection_name": self.collection_name
            }
            
            self.logger.info("collection_stats_retrieved", stats=stats)
            return stats
            
        except Exception as e:
            self.logger.error("get_collection_stats_failed", error=str(e))
            raise
    
    def _create_context_content(self, context: SchemaContext) -> str:
        """Create a text representation of schema context for embedding."""
        content_parts = []
        
        # Basic table/column info
        if context.column_name:
            content_parts.append(f"Column {context.column_name} in table {context.table_name}")
            if context.data_type:
                content_parts.append(f"Data type: {context.data_type}")
        else:
            content_parts.append(f"Table {context.table_name}")
        
        # Description
        if context.description:
            content_parts.append(f"Description: {context.description}")
        
        # Sample values
        if context.sample_values:
            sample_str = ", ".join(context.sample_values[:5])  # Limit to 5 samples
            content_parts.append(f"Sample values: {sample_str}")
        
        # Relationships
        if context.relationships:
            rel_str = ", ".join(context.relationships)
            content_parts.append(f"Relationships: {rel_str}")
        
        return " | ".join(content_parts)
    
    async def clear_collection(self) -> None:
        """Clear all data from the collection."""
        self._ensure_initialized()
        
        try:
            self.collection.delete(where={})
            self.logger.info("collection_cleared", collection_name=self.collection_name)
            
        except Exception as e:
            self.logger.error("clear_collection_failed", error=str(e))
            raise


# Global vector store instance
vector_store = VectorStore() 