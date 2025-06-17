"""
SQL Routes

This module contains endpoints for SQL generation and execution.
"""

import time
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse

from sql_agent.api.models import (
    SQLGenerationRequest, SQLExecutionRequest, SQLResult
)
from sql_agent.agents.sql import SQLAgent
from sql_agent.core.llm import LLMFactory
from sql_agent.core.database import get_database_manager

router = APIRouter()


def get_sql_agent() -> SQLAgent:
    """Get the SQL agent instance."""
    llm_provider = LLMFactory.create_provider()
    return SQLAgent(llm_provider)


def get_database() -> Any:
    """Get the database manager instance."""
    from sql_agent.api.main import database_manager
    if database_manager is None:
        raise HTTPException(status_code=503, detail="Database manager not initialized")
    return database_manager


@router.post("/generate", response_model=Dict[str, Any])
async def generate_sql(
    request: SQLGenerationRequest,
    req: Request,
    sql_agent: SQLAgent = Depends(get_sql_agent)
) -> Dict[str, Any]:
    """
    Generate SQL from natural language query.
    
    This endpoint uses the SQL agent to convert natural language to SQL
    without executing the query.
    """
    start_time = time.time()
    request_id = getattr(req.state, "request_id", "unknown")
    
    try:
        # Create a mock state for SQL generation only
        from sql_agent.core.state import AgentState
        state = AgentState(
            query=request.query,
            session_id=f"sql_gen_{request_id}",
            database_name=request.database_name
        )
        
        # Run SQL agent
        result_state = await sql_agent.run(state)
        
        processing_time = time.time() - start_time
        
        return {
            "query": request.query,
            "generated_sql": result_state.generated_sql,
            "explanation": result_state.generated_sql if request.include_explanation else None,
            "processing_time": processing_time,
            "has_errors": result_state.has_errors(),
            "errors": result_state.errors
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate SQL: {str(e)}"
        )


@router.post("/execute", response_model=SQLResult)
async def execute_sql(
    request: SQLExecutionRequest,
    req: Request,
    database_manager = Depends(get_database)
) -> SQLResult:
    """
    Execute a SQL query directly.
    
    This endpoint executes the provided SQL query and returns the results.
    """
    start_time = time.time()
    request_id = getattr(req.state, "request_id", "unknown")
    
    try:
        # Execute SQL query
        result = await database_manager.execute_query(
            sql=request.sql,
            database_name=request.database_name,
            max_results=request.max_results
        )
        
        execution_time = time.time() - start_time
        
        return SQLResult(
            sql=request.sql,
            data=result.get("data", []),
            row_count=result.get("row_count", 0),
            execution_time=execution_time,
            columns=result.get("columns", []),
            explanation=None
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute SQL: {str(e)}"
        )


@router.post("/validate", response_model=Dict[str, Any])
async def validate_sql(
    request: SQLExecutionRequest,
    req: Request,
    database_manager = Depends(get_database)
) -> Dict[str, Any]:
    """
    Validate SQL syntax without execution.
    
    This endpoint validates the SQL syntax and provides feedback
    without actually executing the query.
    """
    request_id = getattr(req.state, "request_id", "unknown")
    
    try:
        # Validate SQL syntax
        validation_result = await database_manager.validate_sql(
            sql=request.sql,
            database_name=request.database_name
        )
        
        return {
            "sql": request.sql,
            "is_valid": validation_result.get("is_valid", False),
            "errors": validation_result.get("errors", []),
            "warnings": validation_result.get("warnings", []),
            "suggestions": validation_result.get("suggestions", [])
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to validate SQL: {str(e)}"
        )


@router.get("/templates", response_model=Dict[str, Any])
async def get_sql_templates() -> Dict[str, Any]:
    """
    Get common SQL query templates.
    
    This endpoint returns a collection of useful SQL query templates
    that users can reference or modify.
    """
    templates = {
        "basic_select": {
            "name": "Basic SELECT",
            "template": "SELECT column1, column2 FROM table_name WHERE condition",
            "description": "Basic SELECT query with WHERE clause"
        },
        "aggregation": {
            "name": "Aggregation Query",
            "template": "SELECT column1, COUNT(*) as count, AVG(column2) as average FROM table_name GROUP BY column1",
            "description": "Query with aggregation functions"
        },
        "join": {
            "name": "JOIN Query",
            "template": "SELECT t1.column1, t2.column2 FROM table1 t1 JOIN table2 t2 ON t1.id = t2.id",
            "description": "Query with table joins"
        },
        "subquery": {
            "name": "Subquery",
            "template": "SELECT column1 FROM table1 WHERE column2 IN (SELECT column2 FROM table2 WHERE condition)",
            "description": "Query with subquery"
        },
        "window_function": {
            "name": "Window Function",
            "template": "SELECT column1, ROW_NUMBER() OVER (PARTITION BY column2 ORDER BY column3) as row_num FROM table_name",
            "description": "Query with window function"
        }
    }
    
    return {
        "templates": templates,
        "total": len(templates)
    } 