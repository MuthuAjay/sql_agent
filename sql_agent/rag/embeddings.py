"""
Enhanced Embedding service for RAG functionality with specialized embeddings.

This module provides task-specific embeddings for different content types,
business domain optimization, and production-grade performance.
"""

import asyncio
import time
from typing import List, Optional, Union, Dict, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib

import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from ..core.config import settings
from ..utils.logging import get_logger


@dataclass
class EmbeddingResult:
    """Result of embedding operation with metadata."""
    embedding: List[float]
    dimension: int
    model_name: str
    processing_time: float
    cache_hit: bool = False
    confidence: Optional[float] = None


@dataclass
class EmbeddingBatch:
    """Batch embedding result with performance metrics."""
    embeddings: List[List[float]]
    dimension: int
    model_name: str
    batch_size: int
    total_processing_time: float
    avg_processing_time: float
    cache_hits: int = 0


class EmbeddingType:
    """Content type classifications for specialized embeddings."""
    TABLE_NAME = "table_name"
    COLUMN_NAME = "column_name"
    SCHEMA_DESCRIPTION = "schema_description"
    QUERY_PATTERN = "query_pattern"
    BUSINESS_CONCEPT = "business_concept"
    RELATIONSHIP = "relationship"
    SAMPLE_DATA = "sample_data"
    DOCUMENTATION = "documentation"


class BusinessDomain:
    """Business domain classifications for domain-aware embeddings."""
    FINANCIAL = "financial"
    HEALTHCARE = "healthcare"
    ECOMMERCE = "ecommerce"
    HR = "hr"
    MARKETING = "marketing"
    OPERATIONS = "operations"
    LOGISTICS = "logistics"
    GENERAL = "general"


class BaseEmbeddingService(ABC):
    """Abstract base class for embedding services."""
    
    @abstractmethod
    async def embed_text(self, text: str, content_type: Optional[str] = None) -> List[float]:
        """Embed a single text string with optional content type optimization."""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str], content_type: Optional[str] = None) -> List[List[float]]:
        """Embed a batch of text strings with optional content type optimization."""
        pass
    
    @abstractmethod
    async def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        pass
    
    @abstractmethod
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities."""
        pass


class OpenAIEmbeddingService(BaseEmbeddingService):
    """Enhanced OpenAI embedding service with specialized configurations."""
    
    def __init__(self):
        self.logger = get_logger("rag.openai_embeddings")
        
        # Model selection based on use case
        self.models = {
            "primary": "text-embedding-3-small",  # Fast, good for general use
            "high_precision": "text-embedding-3-large",  # Higher accuracy for complex queries
            "legacy": "text-embedding-ada-002"  # Fallback option
        }
        
        # Initialize primary model
        self.embeddings = OpenAIEmbeddings(
            model=self.models["primary"],
            openai_api_key=settings.openai_api_key,
            chunk_size=1000,  # Optimize batch processing
            max_retries=3,
            request_timeout=30
        )
        
        # High precision model for complex cases
        self.high_precision_embeddings = OpenAIEmbeddings(
            model=self.models["high_precision"],
            openai_api_key=settings.openai_api_key,
            chunk_size=500,  # Smaller chunks for better quality
            max_retries=3,
            request_timeout=60
        )
        
        self._dimension: Optional[int] = None
        self._high_precision_dimension: Optional[int] = None
    
    async def embed_text(self, text: str, content_type: Optional[str] = None) -> List[float]:
        """Embed text with content-type optimization."""
        start_time = time.time()
        
        try:
            # Use high precision model for complex content types
            use_high_precision = content_type in [
                EmbeddingType.SCHEMA_DESCRIPTION,
                EmbeddingType.BUSINESS_CONCEPT,
                EmbeddingType.QUERY_PATTERN
            ]
            
            # Preprocess text based on content type
            processed_text = self._preprocess_text(text, content_type)
            
            if use_high_precision:
                result = await asyncio.to_thread(
                    self.high_precision_embeddings.embed_query, processed_text
                )
            else:
                result = await asyncio.to_thread(
                    self.embeddings.embed_query, processed_text
                )
            
            processing_time = time.time() - start_time
            
            self.logger.debug("openai_embed_text_success",
                            content_type=content_type,
                            text_length=len(text),
                            processing_time=processing_time,
                            high_precision=use_high_precision)
            
            return result
            
        except Exception as e:
            self.logger.error("openai_embed_text_failed", 
                            error=str(e), 
                            text=text[:100],
                            content_type=content_type)
            raise
    
    async def embed_batch(self, texts: List[str], content_type: Optional[str] = None) -> List[List[float]]:
        """Embed batch with optimized processing."""
        start_time = time.time()
        
        try:
            # Determine if we need high precision
            use_high_precision = content_type in [
                EmbeddingType.SCHEMA_DESCRIPTION,
                EmbeddingType.BUSINESS_CONCEPT,
                EmbeddingType.QUERY_PATTERN
            ]
            
            # Preprocess all texts
            processed_texts = [self._preprocess_text(text, content_type) for text in texts]
            
            # Use appropriate model
            embeddings_service = self.high_precision_embeddings if use_high_precision else self.embeddings
            
            # Process in optimized batches
            batch_size = 500 if use_high_precision else 1000
            results = []
            
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i + batch_size]
                batch_results = await asyncio.to_thread(
                    embeddings_service.embed_documents, batch
                )
                results.extend(batch_results)
            
            processing_time = time.time() - start_time
            
            self.logger.info("openai_embed_batch_success",
                           batch_size=len(texts),
                           content_type=content_type,
                           processing_time=processing_time,
                           high_precision=use_high_precision)
            
            return results
            
        except Exception as e:
            self.logger.error("openai_embed_batch_failed", 
                            error=str(e), 
                            batch_size=len(texts),
                            content_type=content_type)
            raise
    
    def _preprocess_text(self, text: str, content_type: Optional[str]) -> str:
        """Preprocess text based on content type for better embeddings."""
        if not content_type:
            return text
        
        # Content-specific preprocessing
        if content_type == EmbeddingType.TABLE_NAME:
            # Enhance table names with context
            return f"Database table: {text.replace('_', ' ')}"
        
        elif content_type == EmbeddingType.COLUMN_NAME:
            # Enhance column names with context
            return f"Database column: {text.replace('_', ' ')}"
        
        elif content_type == EmbeddingType.BUSINESS_CONCEPT:
            # Add business context
            return f"Business concept: {text}"
        
        elif content_type == EmbeddingType.QUERY_PATTERN:
            # Add SQL context
            return f"SQL query pattern: {text}"
        
        elif content_type == EmbeddingType.RELATIONSHIP:
            # Add relationship context
            return f"Database relationship: {text}"
        
        return text
    
    async def get_embedding_dimension(self) -> int:
        """Get embedding dimension for primary model."""
        if self._dimension is None:
            test_embedding = await self.embed_text("test")
            self._dimension = len(test_embedding)
        return self._dimension
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "provider": "openai",
            "primary_model": self.models["primary"],
            "high_precision_model": self.models["high_precision"],
            "dimension": await self.get_embedding_dimension(),
            "supports_batch": True,
            "supports_content_types": True,
            "max_tokens": 8191
        }


class GoogleEmbeddingService(BaseEmbeddingService):
    """Enhanced Google embedding service with domain optimization."""
    
    def __init__(self):
        self.logger = get_logger("rag.google_embeddings")
        
        # Google models for different use cases
        self.models = {
            "primary": "models/embedding-001",
            "text": "models/text-embedding-004"  # If available
        }
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.models["primary"],
            google_api_key=settings.google_api_key,
            task_type="retrieval_query"
        )
        
        # Document embeddings for schema content
        self.doc_embeddings = GoogleGenerativeAIEmbeddings(
            model=self.models["primary"],
            google_api_key=settings.google_api_key,
            task_type="retrieval_document"
        )
        
        self._dimension: Optional[int] = None
    
    async def embed_text(self, text: str, content_type: Optional[str] = None) -> List[float]:
        """Embed text with task-type optimization."""
        start_time = time.time()
        
        try:
            # Choose appropriate embeddings based on content type
            is_query_type = content_type in [
                EmbeddingType.QUERY_PATTERN,
                EmbeddingType.BUSINESS_CONCEPT
            ]
            
            embeddings_service = self.embeddings if is_query_type else self.doc_embeddings
            processed_text = self._preprocess_text(text, content_type)
            
            result = await asyncio.to_thread(
                embeddings_service.embed_query, processed_text
            )
            
            processing_time = time.time() - start_time
            
            self.logger.debug("google_embed_text_success",
                            content_type=content_type,
                            text_length=len(text),
                            processing_time=processing_time,
                            is_query_type=is_query_type)
            
            return result
            
        except Exception as e:
            self.logger.error("google_embed_text_failed", 
                            error=str(e), 
                            text=text[:100],
                            content_type=content_type)
            raise
    
    async def embed_batch(self, texts: List[str], content_type: Optional[str] = None) -> List[List[float]]:
        """Embed batch with task-type optimization."""
        start_time = time.time()
        
        try:
            # Choose appropriate service
            is_query_type = content_type in [
                EmbeddingType.QUERY_PATTERN,
                EmbeddingType.BUSINESS_CONCEPT
            ]
            
            embeddings_service = self.embeddings if is_query_type else self.doc_embeddings
            processed_texts = [self._preprocess_text(text, content_type) for text in texts]
            
            results = await asyncio.to_thread(
                embeddings_service.embed_documents, processed_texts
            )
            
            processing_time = time.time() - start_time
            
            self.logger.info("google_embed_batch_success",
                           batch_size=len(texts),
                           content_type=content_type,
                           processing_time=processing_time,
                           is_query_type=is_query_type)
            
            return results
            
        except Exception as e:
            self.logger.error("google_embed_batch_failed", 
                            error=str(e), 
                            batch_size=len(texts),
                            content_type=content_type)
            raise
    
    def _preprocess_text(self, text: str, content_type: Optional[str]) -> str:
        """Google-specific text preprocessing."""
        if not content_type:
            return text
        
        # Add prefixes for better semantic understanding
        prefixes = {
            EmbeddingType.TABLE_NAME: "Table:",
            EmbeddingType.COLUMN_NAME: "Column:",
            EmbeddingType.SCHEMA_DESCRIPTION: "Schema:",
            EmbeddingType.BUSINESS_CONCEPT: "Business:",
            EmbeddingType.QUERY_PATTERN: "SQL:",
            EmbeddingType.RELATIONSHIP: "Relation:"
        }
        
        prefix = prefixes.get(content_type, "")
        return f"{prefix} {text}" if prefix else text
    
    async def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            test_embedding = await self.embed_text("test")
            self._dimension = len(test_embedding)
        return self._dimension
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "provider": "google",
            "model": self.models["primary"],
            "dimension": await self.get_embedding_dimension(),
            "supports_batch": True,
            "supports_content_types": True,
            "supports_task_types": True
        }


class HuggingFaceEmbeddingService(BaseEmbeddingService):
    """Enhanced HuggingFace service with domain-specific models."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.logger = get_logger("rag.huggingface_embeddings")
        
        # Domain-specific models
        self.models = {
            "general": "sentence-transformers/all-MiniLM-L6-v2",
            "code": "microsoft/codebert-base",
            "financial": "ProsusAI/finbert",
            "biomedical": "dmis-lab/biobert-base-cased-v1.1",
            "legal": "nlpaueb/legal-bert-base-uncased"
        }
        
        # Use provided model or default
        selected_model = model_name or self.models["general"]
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=selected_model,
            model_kwargs={
                'device': 'cpu',
                'trust_remote_code': True
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32,
                # 'show_progress_bar': False
            }
        )
        
        self.current_model = selected_model
        self._dimension: Optional[int] = None
        
        # Domain-specific preprocessing
        self.domain_patterns = {
            BusinessDomain.FINANCIAL: ["revenue", "profit", "cost", "budget", "financial"],
            BusinessDomain.HEALTHCARE: ["patient", "treatment", "diagnosis", "medical"],
            BusinessDomain.ECOMMERCE: ["product", "order", "customer", "purchase", "cart"],
            BusinessDomain.HR: ["employee", "payroll", "performance", "department"]
        }
    
    async def embed_text(self, text: str, content_type: Optional[str] = None) -> List[float]:
        """Embed text with domain awareness."""
        start_time = time.time()
        
        try:
            # Detect domain and optimize accordingly
            domain = self._detect_domain(text)
            processed_text = self._preprocess_text(text, content_type, domain)
            
            result = await asyncio.to_thread(
                self.embeddings.embed_query, processed_text
            )
            
            processing_time = time.time() - start_time
            
            self.logger.debug("huggingface_embed_text_success",
                            content_type=content_type,
                            domain=domain,
                            text_length=len(text),
                            processing_time=processing_time)
            
            return result
            
        except Exception as e:
            self.logger.error("huggingface_embed_text_failed", 
                            error=str(e), 
                            text=text[:100],
                            content_type=content_type)
            raise
    
    async def embed_batch(self, texts: List[str], content_type: Optional[str] = None) -> List[List[float]]:
        """Embed batch with domain optimization."""
        start_time = time.time()
        
        try:
            # Process texts with domain awareness
            processed_texts = []
            for text in texts:
                domain = self._detect_domain(text)
                processed_text = self._preprocess_text(text, content_type, domain)
                processed_texts.append(processed_text)
            
            results = await asyncio.to_thread(
                self.embeddings.embed_documents, processed_texts
            )
            
            processing_time = time.time() - start_time
            
            self.logger.info("huggingface_embed_batch_success",
                           batch_size=len(texts),
                           content_type=content_type,
                           processing_time=processing_time)
            
            return results
            
        except Exception as e:
            self.logger.error("huggingface_embed_batch_failed", 
                            error=str(e), 
                            batch_size=len(texts),
                            content_type=content_type)
            raise
    
    def _detect_domain(self, text: str) -> str:
        """Detect business domain from text content."""
        text_lower = text.lower()
        
        domain_scores = {}
        for domain, keywords in self.domain_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        
        return BusinessDomain.GENERAL
    
    def _preprocess_text(self, text: str, content_type: Optional[str], domain: str) -> str:
        """Domain and content-aware preprocessing."""
        processed = text
        
        # Add domain context
        if domain != BusinessDomain.GENERAL:
            processed = f"[{domain.upper()}] {processed}"
        
        # Add content type context
        if content_type:
            type_prefixes = {
                EmbeddingType.TABLE_NAME: "[TABLE]",
                EmbeddingType.COLUMN_NAME: "[COLUMN]",
                EmbeddingType.BUSINESS_CONCEPT: "[CONCEPT]",
                EmbeddingType.QUERY_PATTERN: "[QUERY]"
            }
            
            prefix = type_prefixes.get(content_type)
            if prefix:
                processed = f"{prefix} {processed}"
        
        return processed
    
    async def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            test_embedding = await self.embed_text("test")
            self._dimension = len(test_embedding)
        return self._dimension
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "provider": "huggingface",
            "model": self.current_model,
            "dimension": await self.get_embedding_dimension(),
            "supports_batch": True,
            "supports_content_types": True,
            "supports_domains": True,
            "available_models": list(self.models.keys())
        }


class EmbeddingService:
    """Main embedding service with specialized content handling and caching."""
    
    def __init__(self):
        self.logger = get_logger("rag.embedding_service")
        self._service: Optional[BaseEmbeddingService] = None
        
        # Performance optimization
        self._embedding_cache: Dict[str, Tuple[List[float], datetime]] = {}
        self.cache_ttl = timedelta(hours=24)
        self.max_cache_size = 10000
        
        # Performance metrics
        self.metrics = {
            "total_embeddings": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0
        }
    
    def _get_service(self) -> BaseEmbeddingService:
        """Get appropriate embedding service based on configuration."""
        if self._service is None:
            if settings.llm_provider == "openai" and settings.openai_api_key:
                self._service = OpenAIEmbeddingService()
                self.logger.info("using_openai_embeddings")
            elif settings.llm_provider == "google" and settings.google_api_key:
                self._service = GoogleEmbeddingService()
                self.logger.info("using_google_embeddings")
            else:
                # Smart model selection for HuggingFace
                domain = getattr(settings, 'primary_business_domain', None)
                model_mapping = {
                    BusinessDomain.FINANCIAL: "ProsusAI/finbert",
                    BusinessDomain.HEALTHCARE: "dmis-lab/biobert-base-cased-v1.1"
                }
                
                model_name = model_mapping.get(domain)
                self._service = HuggingFaceEmbeddingService(model_name)
                self.logger.info("using_huggingface_embeddings", 
                               domain=domain, 
                               model=model_name or "general")
        
        return self._service
    
    def _get_cache_key(self, text: str, content_type: Optional[str] = None) -> str:
        """Generate cache key for text and content type."""
        key_data = f"{text}:{content_type or 'default'}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_embedding(self, cache_key: str) -> Optional[List[float]]:
        """Get embedding from cache if valid."""
        if cache_key in self._embedding_cache:
            embedding, timestamp = self._embedding_cache[cache_key]
            
            if datetime.utcnow() - timestamp < self.cache_ttl:
                self.metrics["cache_hits"] += 1
                return embedding
            else:
                # Remove expired entry
                del self._embedding_cache[cache_key]
        
        self.metrics["cache_misses"] += 1
        return None
    
    def _cache_embedding(self, cache_key: str, embedding: List[float]) -> None:
        """Cache embedding with TTL."""
        # Implement LRU eviction if cache is full
        if len(self._embedding_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = min(self._embedding_cache.keys(), 
                           key=lambda k: self._embedding_cache[k][1])
            del self._embedding_cache[oldest_key]
        
        self._embedding_cache[cache_key] = (embedding, datetime.utcnow())
    
    async def embed_query(self, query: str, content_type: Optional[str] = None) -> EmbeddingResult:
        """Embed query with caching and performance tracking."""
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(query, content_type)
        cached_embedding = self._get_cached_embedding(cache_key)
        
        if cached_embedding:
            processing_time = time.time() - start_time
            service = self._get_service()
            model_info = await service.get_model_info()
            
            return EmbeddingResult(
                embedding=cached_embedding,
                dimension=len(cached_embedding),
                model_name=model_info.get("model", "unknown"),
                processing_time=processing_time,
                cache_hit=True,
                confidence=1.0
            )
        
        # Generate new embedding
        service = self._get_service()
        embedding = await service.embed_text(query, content_type)
        
        # Cache the result
        self._cache_embedding(cache_key, embedding)
        
        # Update metrics
        processing_time = time.time() - start_time
        self.metrics["total_embeddings"] += 1
        self.metrics["total_processing_time"] += processing_time
        self.metrics["avg_processing_time"] = (
            self.metrics["total_processing_time"] / self.metrics["total_embeddings"]
        )
        
        model_info = await service.get_model_info()
        
        return EmbeddingResult(
            embedding=embedding,
            dimension=len(embedding),
            model_name=model_info.get("model", "unknown"),
            processing_time=processing_time,
            cache_hit=False,
            confidence=0.95  # High confidence for new embeddings
        )
    
    async def embed_schema_context(self, context: str, content_type: str = EmbeddingType.SCHEMA_DESCRIPTION) -> List[float]:
        """Embed schema context with specialized handling."""
        result = await self.embed_query(context, content_type)
        return result.embedding
    
    async def embed_table_name(self, table_name: str) -> List[float]:
        """Embed table name with specialized preprocessing."""
        result = await self.embed_query(table_name, EmbeddingType.TABLE_NAME)
        return result.embedding
    
    async def embed_column_name(self, column_name: str, table_context: Optional[str] = None) -> List[float]:
        """Embed column name with optional table context."""
        text = f"{table_context}.{column_name}" if table_context else column_name
        result = await self.embed_query(text, EmbeddingType.COLUMN_NAME)
        return result.embedding
    
    async def embed_business_concept(self, concept: str, domain: Optional[str] = None) -> List[float]:
        """Embed business concept with domain awareness."""
        text = f"[{domain}] {concept}" if domain else concept
        result = await self.embed_query(text, EmbeddingType.BUSINESS_CONCEPT)
        return result.embedding
    
    async def embed_query_pattern(self, pattern: str) -> List[float]:
        """Embed SQL query pattern for similarity matching."""
        result = await self.embed_query(pattern, EmbeddingType.QUERY_PATTERN)
        return result.embedding
    
    async def embed_batch(self, texts: List[str], content_type: Optional[str] = None) -> EmbeddingBatch:
        """Embed batch with caching and performance tracking."""
        start_time = time.time()
        
        # Check cache for each text
        uncached_texts = []
        uncached_indices = []
        cached_embeddings = {}
        cache_hits = 0
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text, content_type)
            cached_embedding = self._get_cached_embedding(cache_key)
            
            if cached_embedding:
                cached_embeddings[i] = cached_embedding
                cache_hits += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            service = self._get_service()
            new_embeddings = await service.embed_batch(uncached_texts, content_type)
            
            # Cache new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                cache_key = self._get_cache_key(text, content_type)
                self._cache_embedding(cache_key, embedding)
        else:
            new_embeddings = []
        
        # Combine cached and new embeddings
        all_embeddings = [None] * len(texts)
        
        # Place cached embeddings
        for i, embedding in cached_embeddings.items():
            all_embeddings[i] = embedding
        
        # Place new embeddings
        for i, embedding in zip(uncached_indices, new_embeddings):
            all_embeddings[i] = embedding
        
        # Update metrics
        processing_time = time.time() - start_time
        self.metrics["total_embeddings"] += len(texts)
        self.metrics["total_processing_time"] += processing_time
        self.metrics["avg_processing_time"] = (
            self.metrics["total_processing_time"] / self.metrics["total_embeddings"]
        )
        
        service = self._get_service()
        model_info = await service.get_model_info()
        dimension = len(all_embeddings[0]) if all_embeddings else 0
        
        return EmbeddingBatch(
            embeddings=all_embeddings,
            dimension=dimension,
            model_name=model_info.get("model", "unknown"),
            batch_size=len(texts),
            total_processing_time=processing_time,
            avg_processing_time=processing_time / len(texts) if texts else 0,
            cache_hits=cache_hits
        )
    
    async def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        service = self._get_service()
        return await service.get_embedding_dimension()
    
    async def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity with enhanced precision."""
        try:
            vec1 = np.array(embedding1, dtype=np.float64)
            vec2 = np.array(embedding2, dtype=np.float64)
            
            # Ensure vectors are normalized
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            vec1_norm = vec1 / norm1
            vec2_norm = vec2 / norm2
            
            # Compute cosine similarity
            similarity = np.dot(vec1_norm, vec2_norm)
            
            # Clamp to [-1, 1] range
            similarity = np.clip(similarity, -1.0, 1.0)
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error("embedding_similarity_failed", error=str(e))
            return 0.0
        
embedding_service = EmbeddingService()