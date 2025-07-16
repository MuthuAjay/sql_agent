"""
Enhanced Vector store for RAG functionality using ChromaDB.

This module provides intelligent vector storage and retrieval for database schema contexts,
optimized for dynamic table selection in natural language to SQL conversion.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import uuid
import json
from datetime import datetime, timedelta

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.api.models.Collection import Collection

from ..core.config import settings
from ..core.state import SchemaContext
from ..utils.logging import get_logger


@dataclass
class VectorSearchResult:
    """Enhanced result from vector similarity search."""
    
    id: str
    content: str
    metadata: Dict[str, Any]
    distance: float
    relevance_score: float  # Normalized relevance score (0-1)
    table_name: str
    column_name: Optional[str] = None
    data_type: Optional[str] = None
    business_concepts: List[str] = None
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        if self.business_concepts is None:
            self.business_concepts = []


@dataclass
class SchemaSearchContext:
    """Context for schema search operations."""
    
    query: str
    database_name: str
    search_type: str  # 'table_selection', 'column_discovery', 'relationship_mapping'
    filters: Optional[Dict[str, Any]] = None
    limit: int = 10
    include_relationships: bool = True
    include_sample_data: bool = True


class VectorStore:
    """Enhanced vector store for intelligent schema context management."""
    
    def __init__(self, collection_name: str = "sql_agent_schema_v2"):
        self.logger = get_logger("rag.enhanced_vector_store")
        self.collection_name = collection_name
        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[Collection] = None
        self._initialized = False
        
        # Enhanced configuration
        self.config = {
            "embedding_dimension": 384,  # Default for sentence-transformers
            "distance_threshold": 0.8,   # Maximum distance for relevant results
            "relevance_boost_factors": {
                "exact_table_match": 0.3,
                "business_concept_match": 0.2,
                "data_type_match": 0.1,
                "relationship_match": 0.15
            },
            "batch_size": 100,
            "cache_ttl_minutes": 30
        }
        
        # Search result cache
        self._search_cache: Dict[str, Tuple[List[VectorSearchResult], datetime]] = {}
    
    async def initialize(self) -> None:
        """Initialize the enhanced vector store."""
        try:
            self.logger.info("initializing_enhanced_vector_store")
            
            # Initialize ChromaDB client with enhanced settings
            chroma_settings = ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
            
            self.client = chromadb.PersistentClient(
                path=settings.chroma_db_path or "./chroma_db",
                settings=chroma_settings
            )
            
            # Get or create collection with metadata
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Enhanced SQL Agent schema context embeddings",
                    "version": "2.0",
                    "embedding_function": "sentence-transformers",
                    "created_at": datetime.utcnow().isoformat()
                }
            )
            
            self._initialized = True
            
            # Log collection statistics
            try:
                stats = await self.get_collection_stats()
                self.logger.info(
                    "enhanced_vector_store_initialized", 
                    collection_name=self.collection_name,
                    total_contexts=stats.get("total_contexts", 0),
                    unique_tables=stats.get("unique_tables", 0)
                )
            except Exception as e:
                self.logger.warning("stats_retrieval_failed_during_init", error=str(e))
            
        except Exception as e:
            self.logger.error("enhanced_vector_store_initialization_failed", error=str(e), exc_info=True)
            raise
    
    def _ensure_initialized(self) -> None:
        """Ensure the vector store is initialized."""
        if not self._initialized:
            raise RuntimeError("Enhanced vector store not initialized. Call initialize() first.")
    
    async def ingest_database_schema(self, schema_data: Dict[str, Any]) -> Dict[str, int]:
        """
        Ingest complete database schema into vector store.
        
        This is the main method called by schema processor to store schema contexts.
        """
        self._ensure_initialized()
        
        try:
            database_name = schema_data.get("database_name", "unknown")
            tables = schema_data.get("tables", [])
            
            self.logger.info(
                "ingesting_database_schema",
                database=database_name,
                table_count=len(tables)
            )
            
            # Clear existing schema for this database
            await self._clear_database_schema(database_name)
            
            # Process tables in batches
            total_contexts = 0
            table_contexts = 0
            column_contexts = 0
            
            for table in tables:
                contexts = self._create_table_contexts(table, database_name)
                
                if contexts:
                    await self._add_contexts_batch(contexts)
                    
                    # Count different types
                    for context in contexts:
                        if context.get("metadata", {}).get("context_type") == "table":
                            table_contexts += 1
                        else:
                            column_contexts += 1
                    
                    total_contexts += len(contexts)
            
            stats = {
                "total_contexts": total_contexts,
                "table_contexts": table_contexts,
                "column_contexts": column_contexts,
                "database_name": database_name
            }
            
            self.logger.info("database_schema_ingested", **stats)
            return stats
            
        except Exception as e:
            self.logger.error("ingest_database_schema_failed", database=database_name, error=str(e), exc_info=True)
            raise
    
    def _create_table_contexts(self, table_data: Dict[str, Any], database_name: str) -> List[Dict[str, Any]]:
        """Create vector contexts from table data."""
        contexts = []
        table_name = table_data.get("name", "")
        
        if not table_name:
            return contexts
        
        try:
            # Create table-level context
            table_context = self._create_table_level_context(table_data, database_name)
            contexts.append(table_context)
            
            # Create column-level contexts
            columns = table_data.get("columns", [])
            column_details = table_data.get("column_details", {})
            
            for column_name in columns:
                column_detail = column_details.get(column_name, {})
                column_context = self._create_column_level_context(
                    table_data, column_name, column_detail, database_name
                )
                contexts.append(column_context)
            
            # Create relationship contexts if significant relationships exist
            foreign_keys = table_data.get("foreign_keys", [])
            if len(foreign_keys) > 0:
                relationship_context = self._create_relationship_context(table_data, database_name)
                contexts.append(relationship_context)
            
            return contexts
            
        except Exception as e:
            self.logger.error("create_table_contexts_failed", table=table_name, error=str(e))
            return []
    
    def _create_table_level_context(self, table_data: Dict[str, Any], database_name: str) -> Dict[str, Any]:
        """Create table-level context for vectorization."""
        table_name = table_data.get("name", "")
        
        # Create rich content for embedding
        content_parts = []
        
        # Basic table information
        content_parts.append(f"Table: {table_name}")
        
        # Enhanced description
        description = table_data.get("enhanced_description") or table_data.get("description", "")
        if description:
            content_parts.append(f"Purpose: {description}")
        
        # Business concepts
        business_concepts = table_data.get("business_concepts", [])
        if business_concepts:
            content_parts.append(f"Business domains: {', '.join(business_concepts)}")
        
        # Column information summary
        columns = table_data.get("columns", [])
        if columns:
            key_columns = [col for col in columns if any(keyword in col.lower() for keyword in ["id", "key", "name", "email"])]
            if key_columns:
                content_parts.append(f"Key columns: {', '.join(key_columns[:5])}")
        
        # Semantic tags
        semantic_tags = table_data.get("semantic_tags", [])
        if semantic_tags:
            content_parts.append(f"Data types: {', '.join(semantic_tags)}")
        
        # Sample data context
        sample_data = table_data.get("sample_data", {})
        if sample_data.get("sample_count", 0) > 0:
            content_parts.append(f"Contains {sample_data['sample_count']} example records")
        
        content = " | ".join(content_parts)
        
        # Create comprehensive metadata
        metadata = {
            "context_type": "table",
            "table_name": table_name,
            "database_name": database_name,
            "column_count": len(columns),
            "business_concepts": table_data.get("business_concepts_str", ",".join(business_concepts) if business_concepts else ""),
            "semantic_tags": table_data.get("semantic_tags_str", ",".join(semantic_tags) if semantic_tags else ""),
            "has_relationships": len(table_data.get("foreign_keys", [])) > 0,
            "data_quality_score": table_data.get("data_quality", {}).get("has_primary_key", False),
            "created_at": datetime.utcnow().isoformat(),
            "search_keywords": ",".join(self._extract_search_keywords(table_data))
        }
        
        return {
            "id": f"table_{database_name}_{table_name}_{uuid.uuid4().hex[:8]}",
            "content": content,
            "metadata": metadata
        }
    
    def _create_column_level_context(
        self, 
        table_data: Dict[str, Any], 
        column_name: str, 
        column_detail: Dict[str, Any], 
        database_name: str
    ) -> Dict[str, Any]:
        """Create column-level context for vectorization."""
        table_name = table_data.get("name", "")
        
        # Create rich content for embedding
        content_parts = []
        
        # Basic column information
        content_parts.append(f"Column: {column_name} in table {table_name}")
        
        # Data type information
        data_type = column_detail.get("type", "")
        if data_type:
            content_parts.append(f"Data type: {data_type}")
        
        # Column description
        column_comment = column_detail.get("comment", "")
        if column_comment:
            content_parts.append(f"Description: {column_comment}")
        
        # Business concept
        business_concept = column_detail.get("business_concept")
        if business_concept:
            content_parts.append(f"Business concept: {business_concept}")
        
        # Constraints and properties
        properties = []
        if not column_detail.get("nullable", True):
            properties.append("required")
        if column_detail.get("default"):
            properties.append("has_default")
        if column_name.endswith("_id"):
            properties.append("identifier")
        
        if properties:
            content_parts.append(f"Properties: {', '.join(properties)}")
        
        content = " | ".join(content_parts)
        
        # Create metadata
        metadata = {
            "context_type": "column",
            "table_name": table_name,
            "column_name": column_name,
            "database_name": database_name,
            "data_type": data_type,
            "business_concept": business_concept or "",
            "is_nullable": column_detail.get("nullable", True),
            "is_primary_key": column_name in [pk.get("column", "") for pk in table_data.get("primary_keys", [])],
            "is_foreign_key": column_name in [fk.get("column", "") for fk in table_data.get("foreign_keys", [])],
            "created_at": datetime.utcnow().isoformat(),
            "search_keywords": ",".join([column_name, data_type, business_concept] if business_concept else [column_name, data_type])
        }
        
        return {
            "id": f"column_{database_name}_{table_name}_{column_name}_{uuid.uuid4().hex[:8]}",
            "content": content,
            "metadata": metadata
        }
    
    def _create_relationship_context(self, table_data: Dict[str, Any], database_name: str) -> Dict[str, Any]:
        """Create relationship context for vectorization."""
        table_name = table_data.get("name", "")
        foreign_keys = table_data.get("foreign_keys", [])
        
        # Create content describing relationships
        content_parts = []
        content_parts.append(f"Table relationships for {table_name}")
        
        referenced_tables = []
        for fk in foreign_keys:
            ref_table = fk.get("references_table", "")
            ref_column = fk.get("references_column", "")
            fk_column = fk.get("column", "")
            
            if ref_table:
                referenced_tables.append(ref_table)
                content_parts.append(f"Links to {ref_table} via {fk_column} -> {ref_column}")
        
        content = " | ".join(content_parts)
        
        # Create metadata
        metadata = {
            "context_type": "relationship",
            "table_name": table_name,
            "database_name": database_name,
            "referenced_tables": ",".join(referenced_tables),
            "relationship_count": len(foreign_keys),
            "relationship_strength": table_data.get("relationship_insights", {}).get("relationship_strength", "unknown"),
            "created_at": datetime.utcnow().isoformat(),
            "search_keywords": ",".join([table_name] + referenced_tables)
        }
        
        return {
            "id": f"relationship_{database_name}_{table_name}_{uuid.uuid4().hex[:8]}",
            "content": content,
            "metadata": metadata
        }
    
    def _extract_search_keywords(self, table_data: Dict[str, Any]) -> List[str]:
        """Extract search keywords from table data."""
        keywords = []
        
        # Table name
        table_name = table_data.get("name", "")
        if table_name:
            keywords.append(table_name)
            # Add word variations
            keywords.extend(table_name.lower().split("_"))
        
        # Business concepts
        business_concepts = table_data.get("business_concepts", [])
        keywords.extend(business_concepts)
        
        # Semantic tags
        semantic_tags = table_data.get("semantic_tags", [])
        keywords.extend(semantic_tags)
        
        # Key column names
        columns = table_data.get("columns", [])
        key_columns = [col for col in columns if any(keyword in col.lower() for keyword in ["name", "email", "type", "status"])]
        keywords.extend(key_columns[:5])  # Limit to 5 key columns
        
        return list(set(keywords))  # Remove duplicates
    
    async def _add_contexts_batch(self, contexts: List[Dict[str, Any]]) -> None:
        """Add contexts to vector store in batch."""
        try:
            if not contexts:
                return
            
            ids = [ctx["id"] for ctx in contexts]
            documents = [ctx["content"] for ctx in contexts]
            metadatas = [ctx["metadata"] for ctx in contexts]
            
            # Add to collection (ChromaDB will generate embeddings)
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            self.logger.debug("contexts_batch_added", count=len(contexts))
            
        except Exception as e:
            self.logger.error("add_contexts_batch_failed", error=str(e), count=len(contexts))
            raise
    
    async def search_for_table_selection(self, search_context: SchemaSearchContext) -> List[VectorSearchResult]:
        """
        Main method for table selection - searches for relevant schema contexts.
        
        This is called by the orchestrator for intelligent table selection.
        """
        self._ensure_initialized()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(search_context)
            cached_results = self._get_cached_results(cache_key)
            if cached_results:
                self.logger.debug("table_selection_cache_hit", query=search_context.query[:50])
                return cached_results
            
            # Perform vector search
            raw_results = await self._perform_enhanced_search(search_context)
            
            # Post-process and rank results
            ranked_results = self._rank_and_filter_results(raw_results, search_context)
            
            # Cache results
            self._cache_results(cache_key, ranked_results)
            
            self.logger.info(
                "table_selection_search_completed",
                query=search_context.query[:50],
                database=search_context.database_name,
                raw_results=len(raw_results),
                final_results=len(ranked_results)
            )
            
            return ranked_results
            
        except Exception as e:
            self.logger.error(
                "search_for_table_selection_failed",
                query=search_context.query[:50],
                error=str(e),
                exc_info=True
            )
            return []
    
    async def _perform_enhanced_search(self, search_context: SchemaSearchContext) -> List[VectorSearchResult]:
        """Perform enhanced vector search with multiple strategies."""
        all_results = []
        
        # Strategy 1: Direct text similarity search
        text_results = await self._search_by_text_enhanced(search_context)
        all_results.extend(text_results)
        
        # Strategy 2: Business concept search
        concept_results = await self._search_by_business_concepts(search_context)
        all_results.extend(concept_results)
        
        # Strategy 3: Keyword-based search
        keyword_results = await self._search_by_keywords(search_context)
        all_results.extend(keyword_results)
        
        # Remove duplicates while preserving order
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                unique_results.append(result)
        
        return unique_results
    
    async def _search_by_text_enhanced(self, search_context: SchemaSearchContext) -> List[VectorSearchResult]:
        """Enhanced text-based vector search."""
        try:
            # Build filters
            filters = {"database_name": search_context.database_name}
            if search_context.filters:
                filters.update(search_context.filters)
            
            # Perform search
            chroma_results = self.collection.query(
                query_texts=[search_context.query],
                n_results=search_context.limit * 2,  # Get more for filtering
                where=filters
            )
            
            return self._convert_chroma_results(chroma_results, "text_similarity")
            
        except Exception as e:
            self.logger.warning("text_search_failed", error=str(e))
            return []
    
    async def _search_by_business_concepts(self, search_context: SchemaSearchContext) -> List[VectorSearchResult]:
        """Search by extracted business concepts."""
        try:
            # Extract business concepts from query
            concepts = self._extract_business_concepts_from_query(search_context.query)
            
            if not concepts:
                return []
            
            concept_results = []
            for concept in concepts:
                filters = {
                    "database_name": search_context.database_name,
                    "business_concepts": {"$contains": concept}
                }
                
                try:
                    chroma_results = self.collection.query(
                        query_texts=[concept],
                        n_results=5,
                        where=filters
                    )
                    
                    results = self._convert_chroma_results(chroma_results, "business_concept")
                    concept_results.extend(results)
                    
                except Exception as e:
                    self.logger.debug("business_concept_search_failed", concept=concept, error=str(e))
                    continue
            
            return concept_results
            
        except Exception as e:
            self.logger.warning("business_concept_search_failed", error=str(e))
            return []
    
    async def _search_by_keywords(self, search_context: SchemaSearchContext) -> List[VectorSearchResult]:
        """Search by extracted keywords."""
        try:
            # Extract keywords from query
            keywords = self._extract_keywords_from_query(search_context.query)
            
            if not keywords:
                return []
            
            keyword_results = []
            for keyword in keywords[:3]:  # Limit to top 3 keywords
                filters = {
                    "database_name": search_context.database_name,
                    "search_keywords": {"$contains": keyword}
                }
                
                try:
                    chroma_results = self.collection.query(
                        query_texts=[keyword],
                        n_results=3,
                        where=filters
                    )
                    
                    results = self._convert_chroma_results(chroma_results, "keyword_match")
                    keyword_results.extend(results)
                    
                except Exception as e:
                    self.logger.debug("keyword_search_failed", keyword=keyword, error=str(e))
                    continue
            
            return keyword_results
            
        except Exception as e:
            self.logger.warning("keyword_search_failed", error=str(e))
            return []
    
    def _convert_chroma_results(self, chroma_results: Dict, search_type: str) -> List[VectorSearchResult]:
        """Convert ChromaDB results to VectorSearchResult objects."""
        results = []
        
        if not chroma_results.get('ids') or not chroma_results['ids'][0]:
            return results
        
        for i in range(len(chroma_results['ids'][0])):
            metadata = chroma_results['metadatas'][0][i]
            distance = chroma_results['distances'][0][i] if chroma_results.get('distances') else 0.0
            
            # Calculate relevance score (distance -> similarity)
            relevance_score = max(0.0, 1.0 - distance)
            
            result = VectorSearchResult(
                id=chroma_results['ids'][0][i],
                content=chroma_results['documents'][0][i],
                metadata=metadata,
                distance=distance,
                relevance_score=relevance_score,
                table_name=metadata.get("table_name", ""),
                column_name=metadata.get("column_name"),
                data_type=metadata.get("data_type"),
                business_concepts=metadata.get("business_concepts", []),
                embedding=chroma_results['embeddings'][0][i] if chroma_results.get('embeddings') else None
            )
            
            results.append(result)
        
        return results
    
    def _rank_and_filter_results(self, results: List[VectorSearchResult], search_context: SchemaSearchContext) -> List[VectorSearchResult]:
        """Rank and filter search results based on relevance."""
        if not results:
            return []
        
        # Apply relevance boosting
        boosted_results = []
        query_lower = search_context.query.lower()
        
        for result in results:
            boosted_score = result.relevance_score
            
            # Boost for exact table name matches
            if result.table_name.lower() in query_lower:
                boosted_score += self.config["relevance_boost_factors"]["exact_table_match"]
            
            # Boost for business concept matches
            for concept in result.business_concepts:
                if concept.lower() in query_lower:
                    boosted_score += self.config["relevance_boost_factors"]["business_concept_match"]
            
            # Boost for data type relevance
            if result.data_type and any(word in query_lower for word in ["number", "text", "date", "time"]):
                boosted_score += self.config["relevance_boost_factors"]["data_type_match"]
            
            # Create enhanced result
            enhanced_result = result
            enhanced_result.relevance_score = min(1.0, boosted_score)  # Cap at 1.0
            boosted_results.append(enhanced_result)
        
        # Sort by boosted relevance score
        boosted_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Filter by distance threshold
        filtered_results = [
            r for r in boosted_results 
            if r.distance <= self.config["distance_threshold"]
        ]
        
        # Return top results
        return filtered_results[:search_context.limit]
    
    def _extract_business_concepts_from_query(self, query: str) -> List[str]:
        """Extract business concepts from natural language query."""
        query_lower = query.lower()
        concepts = []
        
        concept_keywords = {
            "customer_management": ["customer", "client", "user", "account", "buyer"],
            "product_catalog": ["product", "item", "goods", "merchandise", "inventory"],
            "order_processing": ["order", "purchase", "transaction", "sale", "booking"],
            "financial": ["payment", "revenue", "profit", "cost", "amount", "price"],
            "hr_management": ["employee", "staff", "worker", "personnel"],
            "logistics": ["shipping", "delivery", "warehouse", "stock"]
        }
        
        for concept, keywords in concept_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                concepts.append(concept)
        
        return concepts
    
    def _extract_keywords_from_query(self, query: str) -> List[str]:
        """Extract relevant keywords from query."""
        # Simple keyword extraction (can be enhanced with NLP)
        query_lower = query.lower()
        
        # Remove common stop words
        stop_words = {"show", "me", "the", "all", "get", "find", "list", "what", "which", "how", "many", "from", "with", "and", "or", "of", "in", "on", "at", "to", "for", "by"}
        
        words = query_lower.split()
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords[:5]  # Return top 5 keywords
    
    def _generate_cache_key(self, search_context: SchemaSearchContext) -> str:
        """Generate cache key for search context."""
        import hashlib
        
        key_data = f"{search_context.query}:{search_context.database_name}:{search_context.search_type}:{search_context.limit}"
        if search_context.filters:
            key_data += f":{json.dumps(search_context.filters, sort_keys=True)}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_results(self, cache_key: str) -> Optional[List[VectorSearchResult]]:
        """Get cached search results."""
        if cache_key in self._search_cache:
            results, timestamp = self._search_cache[cache_key]
            
            # Check if cache is still valid
            if datetime.utcnow() - timestamp < timedelta(minutes=self.config["cache_ttl_minutes"]):
                return results
            else:
                # Remove expired cache
                del self._search_cache[cache_key]
        
        return None
    
    def _cache_results(self, cache_key: str, results: List[VectorSearchResult]) -> None:
        """Cache search results."""
        self._search_cache[cache_key] = (results, datetime.utcnow())
        
        # Clean old cache entries (keep only last 100)
        if len(self._search_cache) > 100:
            oldest_key = min(self._search_cache.keys(), key=lambda k: self._search_cache[k][1])
            del self._search_cache[oldest_key]
    
    async def _clear_database_schema(self, database_name: str) -> None:
        """Clear existing schema contexts for a database."""
        try:
            # Get all contexts for this database
            results = self.collection.get(where={"database_name": database_name})
            
            if results['ids']:
                # Delete in batches
                batch_size = self.config["batch_size"]
                ids = results['ids']
                
                for i in range(0, len(ids), batch_size):
                    batch_ids = ids[i:i + batch_size]
                    self.collection.delete(ids=batch_ids)
                
                self.logger.info("database_schema_cleared", database=database_name, count=len(ids))
            
        except Exception as e:
            self.logger.warning("clear_database_schema_failed", database=database_name, error=str(e))
    
    # Public API methods
    
    async def get_tables_for_query(self, query: str, database_name: str, limit: int = 5) -> List[str]:
        """Get table names most relevant to a query. Main method for orchestrator."""
        search_context = SchemaSearchContext(
            query=query,
            database_name=database_name,
            search_type="table_selection",
            limit=limit * 2  # Get more results for better filtering
        )
        
        results = await self.search_for_table_selection(search_context)
        
        # Extract unique table names, prioritizing table-level contexts
        table_scores = {}
        for result in results:
            table_name = result.table_name
            if table_name:
                # Give higher weight to table-level contexts
                weight = 1.0 if result.metadata.get("context_type") == "table" else 0.7
                current_score = table_scores.get(table_name, 0.0)
                table_scores[table_name] = max(current_score, result.relevance_score * weight)
        
        # Sort by score and return top table names
        sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Extract just the table names, limited to requested count
        selected_tables = [table_name for table_name, score in sorted_tables[:limit]]
        
        self.logger.info(
            "tables_selected_for_query",
            query=query[:50],
            database=database_name,
            selected_tables=selected_tables,
            total_candidates=len(table_scores)
        )
        
        return selected_tables
    
    async def get_table_context(self, table_name: str, database_name: str) -> Dict[str, Any]:
        """Get comprehensive context for a specific table."""
        try:
            filters = {
                "database_name": database_name,
                "table_name": table_name
            }
            
            results = self.collection.get(where=filters)
            
            if not results['ids']:
                return {}
            
            # Organize contexts by type
            table_context = {}
            column_contexts = []
            relationship_contexts = []
            
            for i in range(len(results['ids'])):
                metadata = results['metadatas'][i]
                content = results['documents'][i]
                context_type = metadata.get("context_type", "unknown")
                
                if context_type == "table":
                    table_context = {
                        "content": content,
                        "metadata": metadata,
                        "business_concepts": metadata.get("business_concepts", []),
                        "semantic_tags": metadata.get("semantic_tags", [])
                    }
                elif context_type == "column":
                    column_contexts.append({
                        "column_name": metadata.get("column_name", ""),
                        "content": content,
                        "metadata": metadata
                    })
                elif context_type == "relationship":
                    relationship_contexts.append({
                        "content": content,
                        "metadata": metadata
                    })
            
            return {
                "table_name": table_name,
                "table_context": table_context,
                "column_contexts": column_contexts,
                "relationship_contexts": relationship_contexts,
                "total_contexts": len(results['ids'])
            }
            
        except Exception as e:
            self.logger.error("get_table_context_failed", table=table_name, error=str(e))
            return {}
    
    async def search_columns_for_query(self, query: str, table_names: List[str], database_name: str) -> List[Dict[str, Any]]:
        """Search for relevant columns within specific tables."""
        try:
            search_context = SchemaSearchContext(
                query=query,
                database_name=database_name,
                search_type="column_discovery",
                filters={"context_type": "column"},
                limit=20
            )
            
            results = await self.search_for_table_selection(search_context)
            
            # Filter to only include specified tables
            filtered_results = [
                result for result in results 
                if result.table_name in table_names
            ]
            
            # Group by table
            columns_by_table = {}
            for result in filtered_results:
                table_name = result.table_name
                if table_name not in columns_by_table:
                    columns_by_table[table_name] = []
                
                columns_by_table[table_name].append({
                    "column_name": result.column_name,
                    "data_type": result.data_type,
                    "relevance_score": result.relevance_score,
                    "business_concept": result.metadata.get("business_concept"),
                    "content": result.content
                })
            
            return columns_by_table
            
        except Exception as e:
            self.logger.error("search_columns_for_query_failed", error=str(e))
            return {}
    
    async def get_relationship_insights(self, table_names: List[str], database_name: str) -> Dict[str, Any]:
        """Get relationship insights for a set of tables."""
        try:
            filters = {
                "database_name": database_name,
                "context_type": "relationship"
            }
            
            results = self.collection.get(where=filters)
            
            relationships = []
            relationship_map = {}
            
            for i in range(len(results['ids'])):
                metadata = results['metadatas'][i]
                content = results['documents'][i]
                table_name = metadata.get("table_name", "")
                
                if table_name in table_names:
                    referenced_tables = metadata.get("referenced_tables", [])
                    
                    # Check if any referenced tables are in our table list
                    relevant_refs = [ref for ref in referenced_tables if ref in table_names]
                    
                    if relevant_refs:
                        relationship_info = {
                            "source_table": table_name,
                            "target_tables": relevant_refs,
                            "relationship_strength": metadata.get("relationship_strength", "unknown"),
                            "content": content
                        }
                        relationships.append(relationship_info)
                        
                        # Build relationship map
                        if table_name not in relationship_map:
                            relationship_map[table_name] = []
                        relationship_map[table_name].extend(relevant_refs)
            
            return {
                "relationships": relationships,
                "relationship_map": relationship_map,
                "connected_tables": len(relationship_map),
                "total_relationships": len(relationships)
            }
            
        except Exception as e:
            self.logger.error("get_relationship_insights_failed", error=str(e))
            return {}
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the collection."""
        self._ensure_initialized()
        
        try:
            # Get basic count
            total_count = self.collection.count()
            
            if total_count == 0:
                return {
                    "total_contexts": 0,
                    "unique_tables": 0,
                    "unique_databases": 0,
                    "context_types": {},
                    "collection_name": self.collection_name
                }
            
            # Get all metadata for analysis
            all_results = self.collection.get()
            
            # Analyze metadata
            databases = set()
            tables = set()
            context_types = {}
            business_concepts = set()
            
            for metadata in all_results['metadatas']:
                if metadata:
                    # Databases
                    db_name = metadata.get("database_name")
                    if db_name:
                        databases.add(db_name)
                    
                    # Tables
                    table_name = metadata.get("table_name")
                    if table_name:
                        tables.add(table_name)
                    
                    # Context types
                    context_type = metadata.get("context_type", "unknown")
                    context_types[context_type] = context_types.get(context_type, 0) + 1
                    
                    # Business concepts
                    concepts = metadata.get("business_concepts", [])
                    if isinstance(concepts, list):
                        business_concepts.update(concepts)
            
            stats = {
                "total_contexts": total_count,
                "unique_tables": len(tables),
                "unique_databases": len(databases),
                "context_types": context_types,
                "business_concepts": list(business_concepts),
                "collection_name": self.collection_name,
                "cache_size": len(self._search_cache),
                "config": self.config
            }
            
            self.logger.info("collection_stats_generated", **{k: v for k, v in stats.items() if k != "config"})
            return stats
            
        except Exception as e:
            self.logger.error("get_collection_stats_failed", error=str(e), exc_info=True)
            return {"error": str(e), "total_contexts": 0}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on vector store."""
        health_status = {
            "status": "healthy",
            "initialized": self._initialized,
            "collection_name": self.collection_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            if self._initialized:
                # Test basic operations
                stats = await self.get_collection_stats()
                health_status.update({
                    "total_contexts": stats.get("total_contexts", 0),
                    "unique_tables": stats.get("unique_tables", 0),
                    "cache_size": len(self._search_cache)
                })
                
                # Test search functionality with a simple query
                test_results = await self.get_tables_for_query("test query", "test_db", limit=1)
                health_status["search_functional"] = True
                
            else:
                health_status["status"] = "not_initialized"
                
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
            self.logger.error("vector_store_health_check_failed", error=str(e))
        
        return health_status
    
    async def clear_cache(self) -> None:
        """Clear search result cache."""
        self._search_cache.clear()
        self.logger.info("vector_store_cache_cleared")
    
    async def rebuild_collection(self, database_name: Optional[str] = None) -> Dict[str, int]:
        """Rebuild collection by clearing and re-ingesting schema data."""
        try:
            if database_name:
                await self._clear_database_schema(database_name)
                self.logger.info("collection_partially_rebuilt", database=database_name)
                return {"cleared_database": database_name}
            else:
                # Clear entire collection
                self.collection.delete(where={})
                self.logger.info("collection_fully_rebuilt")
                return {"cleared_all": True}
                
        except Exception as e:
            self.logger.error("rebuild_collection_failed", error=str(e))
            raise
    
    async def export_collection_data(self) -> Dict[str, Any]:
        """Export collection data for backup or analysis."""
        try:
            all_results = self.collection.get()
            
            export_data = {
                "collection_name": self.collection_name,
                "export_timestamp": datetime.utcnow().isoformat(),
                "total_contexts": len(all_results['ids']),
                "contexts": []
            }
            
            for i in range(len(all_results['ids'])):
                context_data = {
                    "id": all_results['ids'][i],
                    "content": all_results['documents'][i],
                    "metadata": all_results['metadatas'][i]
                }
                export_data["contexts"].append(context_data)
            
            self.logger.info("collection_data_exported", total_contexts=export_data["total_contexts"])
            return export_data
            
        except Exception as e:
            self.logger.error("export_collection_data_failed", error=str(e))
            raise
    
    async def cleanup(self) -> None:
        """Cleanup resources and close connections."""
        try:
            self.logger.info("vector_store_cleanup_start")
            
            # Clear caches
            self._search_cache.clear()
            
            # Reset state
            self._initialized = False
            self.collection = None
            self.client = None
            
            self.logger.info("vector_store_cleanup_complete")
            
        except Exception as e:
            self.logger.error("vector_store_cleanup_failed", error=str(e))


# Global enhanced vector store instance
vector_store = VectorStore()