"""Tests for configuration management."""

import pytest
from sql_agent.core.config import Settings


def test_settings_defaults():
    """Test that settings have correct defaults."""
    settings = Settings()
    
    assert settings.app_name == "SQL Agent"
    assert settings.app_version == "0.1.0"
    assert settings.debug is False
    assert settings.llm_provider == "openai"
    assert settings.database_type == "postgresql"
    assert settings.vector_db_type == "chromadb"


def test_settings_validation():
    """Test settings validation."""
    # Test that database_url is required
    with pytest.raises(ValueError, match="Database URL is required"):
        Settings(database_url="")


def test_settings_llm_provider_validation():
    """Test LLM provider validation."""
    # Test OpenAI without API key
    with pytest.raises(ValueError, match="OpenAI API key is required"):
        Settings(llm_provider="openai", openai_api_key=None)
    
    # Test Google without API key
    with pytest.raises(ValueError, match="Google API key is required"):
        Settings(llm_provider="google", google_api_key=None) 