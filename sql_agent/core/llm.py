"""LLM provider management for SQL Agent."""

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
    """Abstract base class for LLM providers."""
    
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


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""
    
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
        # For OpenAI, we need to use function calling
        # This is a simplified implementation
        response = await llm.ainvoke(messages)
        content = response.content if isinstance(response, AIMessage) else str(response.content)
        return {
            "content": content,
            "tool_calls": []  # Simplified for now
        }


class GoogleProvider(LLMProvider):
    """Google LLM provider."""
    
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
        # For Google, we need to use function calling
        # This is a simplified implementation
        response = await llm.ainvoke(messages)
        content = response.content if isinstance(response, AIMessage) else str(response.content)
        return {
            "content": content,
            "tool_calls": []  # Simplified for now
        }


class OllamaProvider(LLMProvider):
    """Ollama LLM provider for local models."""
    
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
        # For Ollama, we need to use function calling
        # This is a simplified implementation
        response = await llm.ainvoke(messages)
        content = response.content if isinstance(response, AIMessage) else str(response.content)
        return {
            "content": content,
            "tool_calls": []  # Simplified for now
        }


class LLMFactory:
    """Factory for creating LLM providers."""
    
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