"""
Query Routes

This module contains the main query endpoints for natural language to SQL conversion,
analysis, and visualization using the multi-agent system.
"""

import time
import uuid
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
import structlog

from sql_agent.api.models import (
    QueryRequest, QueryResponse, QueryIntent, ErrorResponse
)
from sql_agent.core.state import AgentState
from sql_agent.agents.orchestrator import AgentOrchestrator

# Configure structured logging
logger = structlog.get_logger(__name__)

router = APIRouter()


def get_orchestrator() -> AgentOrchestrator:
    """Get the agent orchestrator instance."""
    from sql_agent.api.main import orchestrator
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Agent orchestrator not initialized")
    return orchestrator


@router.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    req: Request,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> QueryResponse:
    """
    Process a natural language query and return SQL results, analysis, and visualization.
    
    This endpoint coordinates all agents to:
    1. Analyze query intent
    2. Generate and execute SQL
    3. Analyze results (if requested)
    4. Create visualizations (if requested)
    """
    start_time = time.time()
    request_id = getattr(req.state, "request_id", "unknown")
    
    logger.info(
        "Processing query",
        query=request.query,
        request_id=request_id,
        include_analysis=request.include_analysis,
        include_visualization=request.include_visualization
    )
    
    try:
        # Process query through orchestrator
        final_state = await orchestrator.process_query(
            query=request.query,
            database_name=request.database_name
        )
        
        # Extract results from state
        intent = QueryIntent.UNKNOWN
        if final_state.metadata.get("routing"):
            routing = final_state.metadata["routing"]
            primary_agent = routing.get("primary_agent", "sql")
            if primary_agent == "sql":
                intent = QueryIntent.SQL_GENERATION
            elif primary_agent == "analysis":
                intent = QueryIntent.ANALYSIS
            elif primary_agent == "visualization":
                intent = QueryIntent.VISUALIZATION
        
        # Convert state results to API models
        sql_result = None
        if final_state.query_result:
            from sql_agent.api.models import SQLResult
            sql_result = SQLResult(
                sql=final_state.query_result.sql_query,
                data=final_state.query_result.data,
                row_count=final_state.query_result.row_count,
                execution_time=final_state.query_result.execution_time,
                columns=final_state.query_result.columns,
                explanation=final_state.generated_sql
            )
        
        analysis_result = None
        if final_state.analysis_result:
            from sql_agent.api.models import AnalysisResult
            analysis_result = AnalysisResult(
                summary={},  # Will be populated from analysis_result
                insights=final_state.analysis_result.insights,
                anomalies=[],  # Will be populated from analysis_result
                trends=[],  # Will be populated from analysis_result
                recommendations=final_state.analysis_result.recommendations,
                data_quality_score=final_state.analysis_result.data_quality_score or 0.0
            )
        
        visualization_result = None
        if final_state.visualization_config:
            from sql_agent.api.models import VisualizationResult, ChartType
            visualization_result = VisualizationResult(
                chart_type=ChartType(final_state.visualization_config.chart_type),
                chart_config=final_state.visualization_config.config,
                chart_data={},  # Will be populated from visualization
                title=final_state.visualization_config.title or "Generated Chart",
                description=None,
                export_formats=["json", "html"]
            )
        
        processing_time = time.time() - start_time
        
        logger.info(
            "Query processed successfully",
            request_id=request_id,
            intent=intent,
            processing_time=processing_time,
            has_sql_result=sql_result is not None,
            has_analysis=analysis_result is not None,
            has_visualization=visualization_result is not None
        )
        
        return QueryResponse(
            query=request.query,
            intent=intent,
            sql_result=sql_result,
            analysis_result=analysis_result,
            visualization_result=visualization_result,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            "Query processing failed",
            request_id=request_id,
            error=str(e),
            processing_time=processing_time,
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )


@router.post("/query/simple", response_model=Dict[str, Any])
async def simple_query(
    request: QueryRequest,
    req: Request,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Simple query endpoint that returns just the SQL and results without analysis.
    
    This is a simplified version for basic SQL generation and execution.
    """
    start_time = time.time()
    request_id = getattr(req.state, "request_id", "unknown")
    
    logger.info(
        "Processing simple query",
        query=request.query,
        request_id=request_id
    )
    
    try:
        # Process through orchestrator
        final_state = await orchestrator.process_query(
            query=request.query,
            database_name=request.database_name
        )
        
        # Extract SQL result only
        sql_result = final_state.query_result
        intent = QueryIntent.UNKNOWN
        if final_state.metadata.get("routing"):
            routing = final_state.metadata["routing"]
            primary_agent = routing.get("primary_agent", "sql")
            if primary_agent == "sql":
                intent = QueryIntent.SQL_GENERATION
        
        processing_time = time.time() - start_time
        
        logger.info(
            "Simple query processed",
            request_id=request_id,
            intent=intent,
            processing_time=processing_time
        )
        
        return {
            "query": request.query,
            "intent": intent,
            "sql": final_state.generated_sql,
            "data": sql_result.data if sql_result else [],
            "row_count": sql_result.row_count if sql_result else 0,
            "execution_time": sql_result.execution_time if sql_result else 0,
            "processing_time": processing_time,
            "explanation": final_state.generated_sql
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            "Simple query failed",
            request_id=request_id,
            error=str(e),
            processing_time=processing_time,
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process simple query: {str(e)}"
        )


@router.get("/query/history", response_model=Dict[str, Any])
async def get_query_history(
    req: Request,
    limit: int = 10,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Get recent query history for the session.
    
    This endpoint returns recent queries processed by the system.
    """
    request_id = getattr(req.state, "request_id", "unknown")
    
    logger.info(
        "Getting query history",
        request_id=request_id,
        limit=limit
    )
    
    try:
        # For now, return empty history - this would be implemented with a database
        # to store query history
        history = []
        
        return {
            "queries": history,
            "total": len(history),
            "limit": limit
        }
        
    except Exception as e:
        logger.error(
            "Failed to get query history",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get query history: {str(e)}"
        )


@router.delete("/query/history")
async def clear_query_history(
    req: Request,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, str]:
    """
    Clear query history for the session.
    """
    request_id = getattr(req.state, "request_id", "unknown")
    
    logger.info(
        "Clearing query history",
        request_id=request_id
    )
    
    try:
        # For now, just return success - this would clear from database
        return {"message": "Query history cleared successfully"}
        
    except Exception as e:
        logger.error(
            "Failed to clear query history",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear query history: {str(e)}"
        ) 