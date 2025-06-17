"""LLM provider management for SQL Agent."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from langchain.llms.base import LLM
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from .config import settings


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, model_name: str, temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        self._llm: Optional[LLM] = None
    
    @abstractmethod
    def get_llm(self) -> LLM:
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
    
    def get_llm(self) -> LLM:
        """Get the OpenAI LLM instance."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=settings.openai_api_key,
            )
        return self._llm
    
    async def generate(self, messages: List[BaseMessage]) -> str:
        """Generate a response from OpenAI."""
        llm = self.get_llm()
        response = await llm.agenerate([messages])
        return response.generations[0][0].text
    
    async def generate_with_tools(
        self, 
        messages: List[BaseMessage], 
        tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a response with tool calls from OpenAI."""
        llm = self.get_llm()
        # For OpenAI, we need to use function calling
        # This is a simplified implementation
        response = await llm.agenerate([messages])
        return {
            "content": response.generations[0][0].text,
            "tool_calls": []  # Simplified for now
        }


class GoogleProvider(LLMProvider):
    """Google LLM provider."""
    
    def __init__(self, model_name: str = "gemini-pro", temperature: float = 0.1):
        super().__init__(model_name, temperature)
        if not settings.google_api_key:
            raise ValueError("Google API key is required")
    
    def get_llm(self) -> LLM:
        """Get the Google LLM instance."""
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                google_api_key=settings.google_api_key,
            )
        return self._llm
    
    async def generate(self, messages: List[BaseMessage]) -> str:
        """Generate a response from Google."""
        llm = self.get_llm()
        response = await llm.agenerate([messages])
        return response.generations[0][0].text
    
    async def generate_with_tools(
        self, 
        messages: List[BaseMessage], 
        tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a response with tool calls from Google."""
        llm = self.get_llm()
        # For Google, we need to use function calling
        # This is a simplified implementation
        response = await llm.agenerate([messages])
        return {
            "content": response.generations[0][0].text,
            "tool_calls": []  # Simplified for now
        }


class LocalProvider(LLMProvider):
    """Local LLM provider (placeholder for future implementation)."""
    
    def __init__(self, model_name: str = "local-model", temperature: float = 0.1):
        super().__init__(model_name, temperature)
        # This would be implemented with local models like Ollama, etc.
        raise NotImplementedError("Local LLM provider not yet implemented")
    
    def get_llm(self) -> LLM:
        """Get the local LLM instance."""
        raise NotImplementedError("Local LLM provider not yet implemented")
    
    async def generate(self, messages: List[BaseMessage]) -> str:
        """Generate a response from local LLM."""
        raise NotImplementedError("Local LLM provider not yet implemented")
    
    async def generate_with_tools(
        self, 
        messages: List[BaseMessage], 
        tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a response with tool calls from local LLM."""
        raise NotImplementedError("Local LLM provider not yet implemented")


class LLMFactory:
    """Factory for creating LLM providers."""
    
    _providers = {
        "openai": OpenAIProvider,
        "google": GoogleProvider,
        "local": LocalProvider,
    }
    
    @classmethod
    def create_provider(
        cls, 
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1
    ) -> LLMProvider:
        """Create an LLM provider instance."""
        provider_name = provider_name or settings.llm_provider
        
        if provider_name not in cls._providers:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")
        
        provider_class = cls._providers[provider_name]
        
        # Set default model names based on provider
        if model_name is None:
            if provider_name == "openai":
                model_name = settings.openai_model
            elif provider_name == "google":
                model_name = settings.google_model
            else:
                model_name = "default"
        
        return provider_class(model_name=model_name, temperature=temperature)
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available LLM providers."""
        return list(cls._providers.keys())
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """Register a new LLM provider."""
        if not issubclass(provider_class, LLMProvider):
            raise ValueError("Provider class must inherit from LLMProvider")
        cls._providers[name] = provider_class 