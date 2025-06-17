"""Embedding service for RAG functionality."""

import asyncio
from typing import List, Optional, Union
from abc import ABC, abstractmethod

import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from ..core.config import settings
from ..utils.logging import get_logger


class BaseEmbeddingService(ABC):
    """Abstract base class for embedding services."""
    
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings."""
        pass
    
    @abstractmethod
    async def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        pass


class OpenAIEmbeddingService(BaseEmbeddingService):
    """OpenAI embedding service."""
    
    def __init__(self):
        self.logger = get_logger("rag.openai_embeddings")
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key
        )
        self._dimension: Optional[int] = None
    
    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        try:
            result = await asyncio.to_thread(
                self.embeddings.embed_query, text
            )
            return result
        except Exception as e:
            self.logger.error("openai_embed_text_failed", error=str(e), text=text[:100])
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings."""
        try:
            results = await asyncio.to_thread(
                self.embeddings.embed_documents, texts
            )
            return results
        except Exception as e:
            self.logger.error("openai_embed_batch_failed", error=str(e), batch_size=len(texts))
            raise
    
    async def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        if self._dimension is None:
            # Test embedding to get dimension
            test_embedding = await self.embed_text("test")
            self._dimension = len(test_embedding)
        return self._dimension


class GoogleEmbeddingService(BaseEmbeddingService):
    """Google Generative AI embedding service."""
    
    def __init__(self):
        self.logger = get_logger("rag.google_embeddings")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=settings.google_api_key
        )
        self._dimension: Optional[int] = None
    
    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        try:
            result = await asyncio.to_thread(
                self.embeddings.embed_query, text
            )
            return result
        except Exception as e:
            self.logger.error("google_embed_text_failed", error=str(e), text=text[:100])
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings."""
        try:
            results = await asyncio.to_thread(
                self.embeddings.embed_documents, texts
            )
            return results
        except Exception as e:
            self.logger.error("google_embed_batch_failed", error=str(e), batch_size=len(texts))
            raise
    
    async def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        if self._dimension is None:
            # Test embedding to get dimension
            test_embedding = await self.embed_text("test")
            self._dimension = len(test_embedding)
        return self._dimension


class HuggingFaceEmbeddingService(BaseEmbeddingService):
    """HuggingFace embedding service for local embeddings."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.logger = get_logger("rag.huggingface_embeddings")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self._dimension: Optional[int] = None
    
    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        try:
            result = await asyncio.to_thread(
                self.embeddings.embed_query, text
            )
            return result
        except Exception as e:
            self.logger.error("huggingface_embed_text_failed", error=str(e), text=text[:100])
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings."""
        try:
            results = await asyncio.to_thread(
                self.embeddings.embed_documents, texts
            )
            return results
        except Exception as e:
            self.logger.error("huggingface_embed_batch_failed", error=str(e), batch_size=len(texts))
            raise
    
    async def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        if self._dimension is None:
            # Test embedding to get dimension
            test_embedding = await self.embed_text("test")
            self._dimension = len(test_embedding)
        return self._dimension


class EmbeddingService:
    """Main embedding service that routes to appropriate provider."""
    
    def __init__(self):
        self.logger = get_logger("rag.embedding_service")
        self._service: Optional[BaseEmbeddingService] = None
    
    def _get_service(self) -> BaseEmbeddingService:
        """Get the appropriate embedding service based on configuration."""
        if self._service is None:
            if settings.llm_provider == "openai" and settings.openai_api_key:
                self._service = OpenAIEmbeddingService()
            elif settings.llm_provider == "google" and settings.google_api_key:
                self._service = GoogleEmbeddingService()
            else:
                # Fallback to HuggingFace for local embeddings
                self._service = HuggingFaceEmbeddingService()
                self.logger.info("using_huggingface_embeddings_fallback")
        
        return self._service
    
    async def embed_query(self, query: str) -> List[float]:
        """Embed a query for similarity search."""
        service = self._get_service()
        return await service.embed_text(query)
    
    async def embed_schema_context(self, context: str) -> List[float]:
        """Embed schema context for storage."""
        service = self._get_service()
        return await service.embed_text(context)
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        service = self._get_service()
        return await service.embed_batch(texts)
    
    async def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        service = self._get_service()
        return await service.get_embedding_dimension()
    
    async def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Normalize vectors
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)
            
            # Compute cosine similarity
            similarity = np.dot(vec1_norm, vec2_norm)
            return float(similarity)
        except Exception as e:
            self.logger.error("similarity_computation_failed", error=str(e))
            return 0.0


# Global embedding service instance
embedding_service = EmbeddingService() 