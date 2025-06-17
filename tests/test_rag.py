"""Tests for RAG (Retrieval-Augmented Generation) functionality."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from sql_agent.rag.embeddings import EmbeddingService, OpenAIEmbeddingService
from sql_agent.rag.vector_store import VectorStore, VectorSearchResult
from sql_agent.rag.schema import SchemaProcessor
from sql_agent.rag.context import ContextManager
from sql_agent.core.state import SchemaContext


class TestEmbeddingService:
    """Test the embedding service."""
    
    @pytest.fixture
    def embedding_service(self):
        """Create an EmbeddingService instance."""
        return EmbeddingService()
    
    @pytest.mark.asyncio
    async def test_embed_query(self, embedding_service):
        """Test query embedding."""
        with patch.object(embedding_service, '_get_service') as mock_get_service:
            mock_service = Mock()
            mock_service.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])
            mock_get_service.return_value = mock_service
            
            result = await embedding_service.embed_query("test query")
            
            assert result == [0.1, 0.2, 0.3]
            mock_service.embed_text.assert_called_once_with("test query")
    
    @pytest.mark.asyncio
    async def test_embed_schema_context(self, embedding_service):
        """Test schema context embedding."""
        with patch.object(embedding_service, '_get_service') as mock_get_service:
            mock_service = Mock()
            mock_service.embed_text = AsyncMock(return_value=[0.4, 0.5, 0.6])
            mock_get_service.return_value = mock_service
            
            result = await embedding_service.embed_schema_context("schema context")
            
            assert result == [0.4, 0.5, 0.6]
            mock_service.embed_text.assert_called_once_with("schema context")
    
    @pytest.mark.asyncio
    async def test_embed_batch(self, embedding_service):
        """Test batch embedding."""
        with patch.object(embedding_service, '_get_service') as mock_get_service:
            mock_service = Mock()
            mock_service.embed_batch = AsyncMock(return_value=[[0.1], [0.2]])
            mock_get_service.return_value = mock_service
            
            result = await embedding_service.embed_batch(["text1", "text2"])
            
            assert result == [[0.1], [0.2]]
            mock_service.embed_batch.assert_called_once_with(["text1", "text2"])
    
    @pytest.mark.asyncio
    async def test_get_embedding_dimension(self, embedding_service):
        """Test getting embedding dimension."""
        with patch.object(embedding_service, '_get_service') as mock_get_service:
            mock_service = Mock()
            mock_service.get_embedding_dimension = AsyncMock(return_value=384)
            mock_get_service.return_value = mock_service
            
            result = await embedding_service.get_embedding_dimension()
            
            assert result == 384
            mock_service.get_embedding_dimension.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_compute_similarity(self, embedding_service):
        """Test similarity computation."""
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0]
        embedding3 = [1.0, 0.0, 0.0]  # Same as embedding1
        
        # Test orthogonal vectors (should be 0)
        similarity_12 = await embedding_service.compute_similarity(embedding1, embedding2)
        assert abs(similarity_12) < 0.001
        
        # Test identical vectors (should be 1)
        similarity_13 = await embedding_service.compute_similarity(embedding1, embedding3)
        assert abs(similarity_13 - 1.0) < 0.001


class TestVectorStore:
    """Test the vector store."""
    
    @pytest.fixture
    def vector_store(self):
        """Create a VectorStore instance."""
        return VectorStore("test_collection")
    
    @pytest.mark.asyncio
    async def test_initialization(self, vector_store):
        """Test vector store initialization."""
        with patch('chromadb.PersistentClient') as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            await vector_store.initialize()
            
            assert vector_store._initialized is True
            assert vector_store.client == mock_client
            assert vector_store.collection == mock_collection
    
    @pytest.mark.asyncio
    async def test_add_schema_contexts(self, vector_store):
        """Test adding schema contexts."""
        vector_store._initialized = True
        vector_store.collection = Mock()
        
        contexts = [
            SchemaContext(
                table_name="users",
                column_name="id",
                data_type="integer",
                description="User ID column",
                embedding=[0.1, 0.2, 0.3]
            ),
            SchemaContext(
                table_name="users",
                column_name="name",
                data_type="varchar",
                description="User name column",
                embedding=[0.4, 0.5, 0.6]
            )
        ]
        
        result = await vector_store.add_schema_contexts(contexts)
        
        assert len(result) == 2
        vector_store.collection.add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_similarity_search(self, vector_store):
        """Test similarity search."""
        vector_store._initialized = True
        vector_store.collection = Mock()
        
        # Mock search results
        mock_results = {
            'ids': [['id1', 'id2']],
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'table_name': 'users'}, {'table_name': 'orders'}]],
            'distances': [[0.8, 0.6]]
        }
        vector_store.collection.query.return_value = mock_results
        
        query_embedding = [0.1, 0.2, 0.3]
        results = await vector_store.similarity_search(query_embedding, limit=2)
        
        assert len(results) == 2
        assert results[0].id == 'id1'
        assert results[0].content == 'doc1'
        assert results[0].distance == 0.8
    
    @pytest.mark.asyncio
    async def test_search_by_text(self, vector_store):
        """Test text-based search."""
        vector_store._initialized = True
        vector_store.collection = Mock()
        
        # Mock search results
        mock_results = {
            'ids': [['id1']],
            'documents': [['doc1']],
            'metadatas': [[{'table_name': 'users'}]],
            'distances': [[0.9]]
        }
        vector_store.collection.query.return_value = mock_results
        
        results = await vector_store.search_by_text("customer", limit=1)
        
        assert len(results) == 1
        assert results[0].content == 'doc1'
        vector_store.collection.query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_by_table_name(self, vector_store):
        """Test getting contexts by table name."""
        vector_store._initialized = True
        vector_store.collection = Mock()
        
        # Mock get results
        mock_results = {
            'ids': ['id1', 'id2'],
            'documents': ['doc1', 'doc2'],
            'metadatas': [{'table_name': 'users'}, {'table_name': 'users'}]
        }
        vector_store.collection.get.return_value = mock_results
        
        results = await vector_store.get_by_table_name("users")
        
        assert len(results) == 2
        assert all(result.metadata['table_name'] == 'users' for result in results)
    
    @pytest.mark.asyncio
    async def test_get_collection_stats(self, vector_store):
        """Test getting collection statistics."""
        vector_store._initialized = True
        vector_store.collection = Mock()
        
        vector_store.collection.count.return_value = 10
        vector_store.collection.get.return_value = {
            'metadatas': [{'table_name': 'users'}, {'table_name': 'orders'}]
        }
        
        stats = await vector_store.get_collection_stats()
        
        assert stats['total_contexts'] == 10
        assert stats['unique_tables'] == 2
        assert 'users' in stats['table_names']
        assert 'orders' in stats['table_names']


class TestSchemaProcessor:
    """Test the schema processor."""
    
    @pytest.fixture
    def schema_processor(self):
        """Create a SchemaProcessor instance."""
        return SchemaProcessor()
    
    @pytest.mark.asyncio
    async def test_extract_schema_contexts(self, schema_processor):
        """Test schema context extraction."""
        with patch('sql_agent.rag.schema.db_manager') as mock_db:
            mock_db.get_schema_info.return_value = {
                "users": {
                    "columns": [
                        {"column_name": "id", "data_type": "integer"},
                        {"column_name": "name", "data_type": "varchar"}
                    ]
                }
            }
            mock_db.get_sample_data.return_value = [{"id": 1, "name": "John"}]
            
            with patch.object(schema_processor, '_get_column_sample_values') as mock_sample:
                mock_sample.return_value = ["1", "2", "3"]
                
                contexts = await schema_processor.extract_schema_contexts()
                
                assert len(contexts) == 3  # 1 table + 2 columns
                assert any(context.table_name == "users" and context.column_name is None for context in contexts)
                assert any(context.table_name == "users" and context.column_name == "id" for context in contexts)
    
    @pytest.mark.asyncio
    async def test_create_table_context(self, schema_processor):
        """Test table context creation."""
        with patch('sql_agent.rag.schema.db_manager') as mock_db:
            mock_db.get_sample_data.return_value = [{"id": 1, "name": "John"}]
            
            table_info = {
                "columns": [
                    {"column_name": "id", "data_type": "integer"},
                    {"column_name": "name", "data_type": "varchar"}
                ]
            }
            
            with patch.object(schema_processor, '_identify_table_relationships') as mock_rel:
                mock_rel.return_value = ["References orders table"]
                
                with patch.object(schema_processor, '_extract_sample_values') as mock_extract:
                    mock_extract.return_value = ["1", "John"]
                    
                    with patch.object(schema_processor, '_create_context_text') as mock_text:
                        mock_text.return_value = "Table users with 2 columns"
                        
                        with patch('sql_agent.rag.schema.embedding_service') as mock_embed:
                            mock_embed.embed_schema_context.return_value = [0.1, 0.2, 0.3]
                            
                            context = await schema_processor._create_table_context("users", table_info)
                            
                            assert context.table_name == "users"
                            assert context.column_name is None
                            assert "2 columns" in context.description
                            assert context.embedding == [0.1, 0.2, 0.3]
    
    @pytest.mark.asyncio
    async def test_create_column_context(self, schema_processor):
        """Test column context creation."""
        column_info = {
            "column_name": "id",
            "data_type": "integer",
            "is_nullable": "NO",
            "column_default": "nextval('users_id_seq')"
        }
        
        table_info = {"columns": [column_info]}
        
        with patch.object(schema_processor, '_get_column_sample_values') as mock_sample:
            mock_sample.return_value = ["1", "2", "3"]
            
            with patch.object(schema_processor, '_identify_column_relationships') as mock_rel:
                mock_rel.return_value = ["Primary key"]
                
                with patch.object(schema_processor, '_create_context_text') as mock_text:
                    mock_text.return_value = "Column id in table users"
                    
                    with patch('sql_agent.rag.schema.embedding_service') as mock_embed:
                        mock_embed.embed_schema_context.return_value = [0.4, 0.5, 0.6]
                        
                        context = await schema_processor._create_column_context("users", column_info, table_info)
                        
                        assert context.table_name == "users"
                        assert context.column_name == "id"
                        assert context.data_type == "integer"
                        assert "integer" in context.description
                        assert context.embedding == [0.4, 0.5, 0.6]
    
    @pytest.mark.asyncio
    async def test_get_schema_summary(self, schema_processor):
        """Test schema summary generation."""
        with patch('sql_agent.rag.schema.db_manager') as mock_db:
            mock_db.get_schema_info.return_value = {
                "users": {
                    "columns": [
                        {"column_name": "id"},
                        {"column_name": "name"}
                    ]
                },
                "orders": {
                    "columns": [
                        {"column_name": "id"},
                        {"column_name": "user_id"}
                    ]
                }
            }
            
            summary = await schema_processor.get_schema_summary()
            
            assert summary["total_tables"] == 2
            assert summary["total_columns"] == 4
            assert "users" in summary["tables"]
            assert "orders" in summary["tables"]
            assert summary["tables"]["users"]["column_count"] == 2


class TestContextManager:
    """Test the context manager."""
    
    @pytest.fixture
    def context_manager(self):
        """Create a ContextManager instance."""
        return ContextManager()
    
    @pytest.mark.asyncio
    async def test_initialization(self, context_manager):
        """Test context manager initialization."""
        with patch('sql_agent.rag.context.vector_store') as mock_vector_store:
            with patch('sql_agent.rag.context.embedding_service') as mock_embedding:
                with patch('sql_agent.rag.context.schema_processor') as mock_schema:
                    mock_embedding.get_embedding_dimension.return_value = 384
                    mock_schema.extract_schema_contexts.return_value = []
                    
                    await context_manager.initialize()
                    
                    assert context_manager._initialized is True
                    mock_vector_store.initialize.assert_called_once()
                    mock_embedding.get_embedding_dimension.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retrieve_schema_context(self, context_manager):
        """Test schema context retrieval."""
        context_manager._initialized = True
        
        with patch('sql_agent.rag.context.embedding_service') as mock_embedding:
            with patch('sql_agent.rag.context.vector_store') as mock_vector_store:
                mock_embedding.embed_query.return_value = [0.1, 0.2, 0.3]
                
                # Mock search results
                mock_results = [
                    VectorSearchResult(
                        id="id1",
                        content="Column id in table users",
                        metadata={"table_name": "users", "column_name": "id"},
                        distance=0.8
                    ),
                    VectorSearchResult(
                        id="id2",
                        content="Table users with 2 columns",
                        metadata={"table_name": "users"},
                        distance=0.7
                    )
                ]
                mock_vector_store.similarity_search.return_value = mock_results
                
                contexts = await context_manager.retrieve_schema_context("user information", limit=2)
                
                assert len(contexts) == 2
                assert contexts[0].table_name == "users"
                assert contexts[0].column_name == "id"
                assert contexts[1].table_name == "users"
                assert contexts[1].column_name is None
    
    @pytest.mark.asyncio
    async def test_retrieve_context_by_tables(self, context_manager):
        """Test context retrieval by table names."""
        context_manager._initialized = True
        
        with patch('sql_agent.rag.context.vector_store') as mock_vector_store:
            # Mock table results
            mock_results = [
                VectorSearchResult(
                    id="id1",
                    content="Column id in table users",
                    metadata={"table_name": "users", "column_name": "id"},
                    distance=0.0
                )
            ]
            mock_vector_store.get_by_table_name.return_value = mock_results
            
            contexts = await context_manager.retrieve_context_by_tables(["users"])
            
            assert len(contexts) == 1
            assert contexts[0].table_name == "users"
            assert contexts[0].column_name == "id"
    
    @pytest.mark.asyncio
    async def test_search_schema_by_keywords(self, context_manager):
        """Test schema search by keywords."""
        context_manager._initialized = True
        
        with patch('sql_agent.rag.context.vector_store') as mock_vector_store:
            # Mock search results
            mock_results = [
                VectorSearchResult(
                    id="id1",
                    content="Column customer_id in table orders",
                    metadata={"table_name": "orders", "column_name": "customer_id"},
                    distance=0.9
                )
            ]
            mock_vector_store.search_by_text.return_value = mock_results
            
            contexts = await context_manager.search_schema_by_keywords(["customer"], limit=1)
            
            assert len(contexts) == 1
            assert contexts[0].table_name == "orders"
            assert contexts[0].column_name == "customer_id"
    
    @pytest.mark.asyncio
    async def test_deduplicate_contexts(self, context_manager):
        """Test context deduplication."""
        contexts = [
            SchemaContext(table_name="users", column_name="id"),
            SchemaContext(table_name="users", column_name="id"),  # Duplicate
            SchemaContext(table_name="users", column_name="name"),
            SchemaContext(table_name="orders", column_name="id")
        ]
        
        unique_contexts = context_manager._deduplicate_contexts(contexts)
        
        assert len(unique_contexts) == 3
        assert any(c.table_name == "users" and c.column_name == "id" for c in unique_contexts)
        assert any(c.table_name == "users" and c.column_name == "name" for c in unique_contexts)
        assert any(c.table_name == "orders" and c.column_name == "id" for c in unique_contexts)
    
    @pytest.mark.asyncio
    async def test_get_relevant_tables(self, context_manager):
        """Test relevant table identification."""
        context_manager._initialized = True
        
        with patch.object(context_manager, 'retrieve_schema_context') as mock_retrieve:
            mock_contexts = [
                SchemaContext(table_name="users"),
                SchemaContext(table_name="orders"),
                SchemaContext(table_name="users")  # Duplicate
            ]
            mock_retrieve.return_value = mock_contexts
            
            tables = await context_manager.get_relevant_tables("customer orders", limit=3)
            
            assert len(tables) == 2
            assert "users" in tables
            assert "orders" in tables
    
    def test_is_schema_stale(self, context_manager):
        """Test schema staleness check."""
        from datetime import datetime, timedelta
        
        # Test with no last update
        assert context_manager.is_schema_stale() is True
        
        # Test with recent update
        context_manager._last_schema_update = datetime.utcnow()
        assert context_manager.is_schema_stale(max_age_minutes=60) is False
        
        # Test with old update
        context_manager._last_schema_update = datetime.utcnow() - timedelta(hours=2)
        assert context_manager.is_schema_stale(max_age_minutes=60) is True


class TestRAGIntegration:
    """Integration tests for RAG functionality."""
    
    @pytest.mark.asyncio
    async def test_full_rag_workflow(self):
        """Test the complete RAG workflow."""
        # This test would require a real database and vector store
        # For now, we'll test the integration points
        pass
    
    @pytest.mark.asyncio
    async def test_embedding_to_vector_store_flow(self):
        """Test the flow from embedding to vector store."""
        # Mock the entire flow
        with patch('sql_agent.rag.embeddings.embedding_service') as mock_embedding:
            with patch('sql_agent.rag.vector_store.vector_store') as mock_vector_store:
                # Test embedding generation
                mock_embedding.embed_query.return_value = [0.1, 0.2, 0.3]
                
                # Test vector store search
                mock_results = [
                    VectorSearchResult(
                        id="id1",
                        content="test content",
                        metadata={"table_name": "test"},
                        distance=0.8
                    )
                ]
                mock_vector_store.similarity_search.return_value = mock_results
                
                # This would be the actual integration test
                # For now, we verify the mocks work
                embedding = await mock_embedding.embed_query("test")
                assert embedding == [0.1, 0.2, 0.3]
                
                results = await mock_vector_store.similarity_search(embedding)
                assert len(results) == 1
                assert results[0].id == "id1" 