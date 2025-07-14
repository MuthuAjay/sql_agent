"""LLM provider management for SQL Agent with schema-aware prompting."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from langchain.llms.base import LLM
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from .config import settings

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers with schema awareness."""
    
    def __init__(self, model_name: str, temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        self._llm: Optional[BaseChatModel] = None
    
    @abstractmethod
    def get_llm(self) -> BaseChatModel:
        """Get the LLM instance."""
        pass
    
    @abstractmethod
    async def generate(self, messages: List[BaseMessage]) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def generate_with_tools(
        self, 
        messages: List[BaseMessage], 
        tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a response with tool calls."""
        pass
    
    # New schema-aware methods for Phase 2
    
    async def generate_with_schema_context(
        self, 
        query: str, 
        schema_context: Dict[str, Any],
        task_type: str = "sql_generation"
    ) -> str:
        """Generate response with schema context for intelligent prompting."""
        try:
            # Build schema-aware prompt
            schema_prompt = self._build_schema_aware_prompt(query, schema_context, task_type)
            
            messages = [
                SystemMessage(content=schema_prompt["system_prompt"]),
                HumanMessage(content=schema_prompt["user_prompt"])
            ]
            
            return await self.generate(messages)
            
        except Exception as e:
            logger.error(f"Schema-aware generation failed: {e}")
            # Fallback to basic generation
            return await self.generate([HumanMessage(content=query)])
    
    def _build_schema_aware_prompt(
        self, 
        query: str, 
        schema_context: Dict[str, Any], 
        task_type: str
    ) -> Dict[str, str]:
        """Build schema-aware prompts based on task type."""
        
        # Extract schema information
        selected_tables = schema_context.get("selected_tables", [])
        enriched_context = schema_context.get("enriched_context", {})
        business_domains = schema_context.get("business_domains", [])
        relationships = enriched_context.get("relationships", {})
        
        # Build schema context string
        schema_info = self._format_schema_context(
            selected_tables, enriched_context, relationships
        )
        
        # Task-specific prompts
        if task_type == "sql_generation":
            return self._build_sql_generation_prompt(query, schema_info, business_domains)
        elif task_type == "analysis":
            return self._build_analysis_prompt(query, schema_info, business_domains)
        elif task_type == "visualization":
            return self._build_visualization_prompt(query, schema_info, business_domains)
        else:
            return self._build_generic_prompt(query, schema_info)
    
    def _format_schema_context(
        self, 
        selected_tables: List[str], 
        enriched_context: Dict[str, Any],
        relationships: Dict[str, Any]
    ) -> str:
        """Format schema context for prompts."""
        context_parts = []
        
        if selected_tables:
            context_parts.append(f"Relevant Tables: {', '.join(selected_tables)}")
        
        # Add column information if available
        column_contexts = enriched_context.get("column_contexts", {})
        for table_name, columns in column_contexts.items():
            if columns:
                column_names = [col.get("column_name", "") for col in columns[:10]]  # Limit to 10 columns
                context_parts.append(f"{table_name} columns: {', '.join(filter(None, column_names))}")
        
        # Add relationship information
        if relationships.get("relationships"):
            rel_info = []
            for rel in relationships["relationships"][:3]:  # Limit to 3 relationships
                source = rel.get("source_table", "")
                targets = rel.get("target_tables", [])
                if source and targets:
                    rel_info.append(f"{source} → {', '.join(targets)}")
            
            if rel_info:
                context_parts.append(f"Table relationships: {'; '.join(rel_info)}")
        
        return "\n".join(context_parts) if context_parts else "No schema context available"
    
    def _build_sql_generation_prompt(
        self, 
        query: str, 
        schema_info: str, 
        business_domains: List[str]
    ) -> Dict[str, str]:
        """Build SQL generation prompt with schema context."""
        
        domain_context = f"Business context: {', '.join(business_domains)}" if business_domains else ""
        
        system_prompt = f"""You are an expert SQL developer. Generate accurate SQL queries based on natural language requests.

Database Schema Context:
{schema_info}

{domain_context}

Guidelines:
1. Use ONLY the tables and columns mentioned in the schema context
2. Use proper JOIN syntax when combining tables
3. Follow PostgreSQL syntax standards
4. Include appropriate WHERE clauses for filtering
5. Use meaningful aliases for tables
6. Return clean, executable SQL without explanations

Format: Return only the SQL query, no additional text."""

        user_prompt = f"Generate SQL for: {query}"
        
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
        }
    
    def _build_analysis_prompt(
        self, 
        query: str, 
        schema_info: str, 
        business_domains: List[str]
    ) -> Dict[str, str]:
        """Build analysis prompt with schema context."""
        
        domain_context = f"Business domains: {', '.join(business_domains)}" if business_domains else ""
        
        system_prompt = f"""You are a data analyst expert. Analyze data and provide business insights.

Available Data Schema:
{schema_info}

{domain_context}

Guidelines:
1. Focus on the specific tables and columns available
2. Consider business context when providing insights
3. Suggest relevant metrics and KPIs
4. Identify patterns and trends
5. Provide actionable recommendations
6. Structure your response clearly

Format: Provide analysis insights in a structured format."""

        user_prompt = f"Analyze: {query}"
        
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
        }
    
    def _build_visualization_prompt(
        self, 
        query: str, 
        schema_info: str, 
        business_domains: List[str]
    ) -> Dict[str, str]:
        """Build visualization prompt with schema context."""
        
        domain_context = f"Business context: {', '.join(business_domains)}" if business_domains else ""
        
        system_prompt = f"""You are a data visualization expert. Design appropriate charts and visualizations.

Available Data Schema:
{schema_info}

{domain_context}

Guidelines:
1. Choose the most appropriate chart type for the data
2. Consider the business context and audience
3. Ensure the visualization clearly communicates insights
4. Suggest interactive features if relevant
5. Recommend color schemes and styling
6. Structure your response clearly

Format: Provide visualization recommendations in a structured format."""

        user_prompt = f"Create visualization for: {query}"
        
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
        }
    
    def _build_generic_prompt(self, query: str, schema_info: str) -> Dict[str, str]:
        """Build generic prompt with schema context."""
        
        system_prompt = f"""You are an intelligent database assistant.

Available Database Schema:
{schema_info}

Guidelines:
1. Use the available schema information to provide accurate responses
2. Be specific and practical in your recommendations
3. Consider the database structure in your responses

Format: Provide clear, actionable responses."""

        user_prompt = query
        
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
        }


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider with schema awareness."""
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.1):
        super().__init__(model_name, temperature)
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        try:
            from langchain_openai import ChatOpenAI
            self._chat_openai = ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai is not installed. "
                "Install it with: poetry install --with llm"
            )
    
    def get_llm(self) -> BaseChatModel:
        """Get the OpenAI LLM instance."""
        if self._llm is None:
            self._llm = self._chat_openai(
                model=self.model_name,
                temperature=self.temperature,
                api_key=settings.openai_api_key,
            )
        return self._llm
    
    async def generate(self, messages: List[BaseMessage]) -> str:
        """Generate a response from OpenAI."""
        llm = self.get_llm()
        response = await llm.ainvoke(messages)
        if isinstance(response, AIMessage):
            return response.content
        return str(response.content)
    
    async def generate_with_tools(
        self, 
        messages: List[BaseMessage], 
        tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a response with tool calls from OpenAI."""
        llm = self.get_llm()
        response = await llm.ainvoke(messages)
        content = response.content if isinstance(response, AIMessage) else str(response.content)
        return {
            "content": content,
            "tool_calls": []  # Simplified for now
        }


class GoogleProvider(LLMProvider):
    """Google LLM provider with schema awareness."""
    
    def __init__(self, model_name: str = "gemini-pro", temperature: float = 0.1):
        super().__init__(model_name, temperature)
        if not settings.google_api_key:
            raise ValueError("Google API key is required")
        
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self._chat_google = ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai is not installed. "
                "Install it with: poetry install --with llm"
            )
    
    def get_llm(self) -> BaseChatModel:
        """Get the Google LLM instance."""
        if self._llm is None:
            self._llm = self._chat_google(
                model=self.model_name,
                temperature=self.temperature,
                google_api_key=settings.google_api_key,
            )
        return self._llm
    
    async def generate(self, messages: List[BaseMessage]) -> str:
        """Generate a response from Google."""
        llm = self.get_llm()
        response = await llm.ainvoke(messages)
        if isinstance(response, AIMessage):
            return response.content
        return str(response.content)
    
    async def generate_with_tools(
        self, 
        messages: List[BaseMessage], 
        tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        
        """Generate a response with tool calls from Google."""
        llm = self.get_llm()
        response = await llm.ainvoke(messages)
        content = response.content if isinstance(response, AIMessage) else str(response.content)
        return {
            "content": content,
            "tool_calls": []  # Simplified for now
        }


class OllamaProvider(LLMProvider):
    """Ollama LLM provider for local models with schema awareness."""
    
    def __init__(self, model_name: str = "llama2", temperature: float = 0.1):
        super().__init__(model_name, temperature)
        
        try:
            from langchain_ollama import ChatOllama
            self._chat_ollama = ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-ollama is not installed. "
                "Install it with: poetry install --with llm"
            )
    
    def get_llm(self) -> BaseChatModel:
        """Get the Ollama LLM instance."""
        if self._llm is None:
            self._llm = self._chat_ollama(
                model=self.model_name,
                temperature=self.temperature,
                base_url=settings.ollama_base_url,
            )
        return self._llm
    
    async def generate(self, messages: List[BaseMessage]) -> str:
        """Generate a response from Ollama."""
        llm = self.get_llm()
        response = await llm.ainvoke(messages)
        if isinstance(response, AIMessage):
            return response.content
        return str(response.content)
    
    async def generate_with_tools(
        self, 
        messages: List[BaseMessage], 
        tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a response with tool calls from Ollama."""
        llm = self.get_llm()
        response = await llm.ainvoke(messages)
        content = response.content if isinstance(response, AIMessage) else str(response.content)
        return {
            "content": content,
            "tool_calls": []  # Simplified for now
        }


class LLMFactory:
    """Factory for creating LLM providers with schema awareness."""
    
    _providers = {
        "openai": OpenAIProvider,
        "google": GoogleProvider,
        "ollama": OllamaProvider,
    }
    
    @classmethod
    def create_provider(
        cls, 
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1
    ) -> LLMProvider:
        """Create an LLM provider instance."""
        provider_name = provider_name or settings.effective_llm_provider
        
        if provider_name not in cls._providers:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")
        
        provider_class = cls._providers[provider_name]
        
        # Set default model names based on provider
        if model_name is None:
            if provider_name == "openai":
                model_name = settings.openai_model
            elif provider_name == "google":
                model_name = settings.google_model
            elif provider_name == "ollama":
                model_name = settings.ollama_model
            else:
                model_name = "default"
        
        try:
            return provider_class(model_name=model_name, temperature=temperature)
        except (ImportError, ValueError) as e:
            logger.warning(f"Failed to create {provider_name} provider: {e}")
            # Fallback to Ollama if available
            if provider_name != "ollama":
                logger.info("Falling back to Ollama provider")
                return cls.create_provider("ollama", settings.ollama_model, temperature)
            else:
                raise
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available LLM providers."""
        available = []
        for provider_name, provider_class in cls._providers.items():
            try:
                # Test if provider can be instantiated
                if provider_name == "openai" and settings.openai_api_key:
                    available.append(provider_name)
                elif provider_name == "google" and settings.google_api_key:
                    available.append(provider_name)
                elif provider_name == "ollama":
                    # For Ollama, we assume it's available if the import works
                    try:
                        from langchain_ollama import ChatOllama
                        available.append(provider_name)
                    except ImportError:
                        pass
            except Exception:
                continue
        return available
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """Register a new LLM provider."""
        if not issubclass(provider_class, LLMProvider):
            raise ValueError("Provider class must inherit from LLMProvider")
        cls._providers[name] = provider_class