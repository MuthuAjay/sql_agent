"""
SQL Agent API Module

This module provides the FastAPI REST API for the SQL Agent,
enabling natural language to SQL conversion, analysis, and visualization.
"""

from .main import app

__all__ = ["app"] 