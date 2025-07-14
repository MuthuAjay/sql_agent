"""
Enhanced Query Routes - Phase 1 Fixes

This module contains enhanced query endpoints for natural language to SQL conversion,
analysis, and visualization using the multi-agent system.

Google-grade fixes:
1. Fixed route paths to match test expectations
2. Added graceful orchestrator fallback
3. Simplified response models for reliability
4. Enhanced error handling
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Union
import asyncio
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks, Query as FastAPIQuery
from fastapi.responses import JSONResponse
import structlog
from cachetools import TTLCache

from sql_agent.api.models import (
    QueryRequest, QueryResponse, QueryIntent, ErrorResponse, SQLResult,
    AnalysisResult, VisualizationResult, PaginationParams, PaginatedResponse,
    FeedbackRequest, ValidationResult
)
from sql_agent.core.state import AgentState
from sql_agent.agents.orchestrator import AgentOrchestrator
from sql_agent.api.models import ChartType

# Configure structured logging
logger = structlog.get_logger(__name__)

router = APIRouter()

# Cache for query results (TTL: 1 hour)
query_cache = TTLCache(maxsize=1000, ttl=3600)

# Query history storage (in production, use database)
query_history: List[Dict[str, Any]] = []


def get_orchestrator() -> Optional[AgentOrchestrator]:
    """Get the agent orchestrator instance with graceful fallback."""
    try:
        from sql_agent.api.main import orchestrator
        return orchestrator
    except ImportError as e:
        logger.warning("Failed to import orchestrator", error=str(e))
        return None
    except Exception as e:
        logger.warning("Orchestrator not available", error=str(e))
        return None


def get_orchestrator_required() -> AgentOrchestrator:
    """Get orchestrator instance with required dependency (raises 503 if not available)."""
    orchestrator = get_orchestrator()
    if orchestrator is None:
        raise HTTPException(
            status_code=503, 
            detail={
                "error": "Agent orchestrator not initialized",
                "user_message": "The AI query processing service is temporarily unavailable",
                "suggestions": [
                    "Try using the direct SQL execution endpoints instead",
                    "Contact support if this issue persists"
                ],
                "type": "service_unavailable"
            }
        )
    return orchestrator


async def cache_query_result(cache_key: str, result: Dict[str, Any], ttl: int = 3600):
    """Cache query result with TTL."""
    try:
        query_cache[cache_key] = {
            "result": result,
            "timestamp": datetime.utcnow(),
            "ttl": ttl
        }
        logger.debug("Query result cached", cache_key=cache_key, ttl=ttl)
    except Exception as e:
        logger.warning("Failed to cache query result", error=str(e), cache_key=cache_key)


async def get_cached_result(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get cached query result if available and valid."""
    try:
        cached = query_cache.get(cache_key)
        if cached:
            # Check if cache is still valid
            if datetime.utcnow() - cached["timestamp"] < timedelta(seconds=cached["ttl"]):
                logger.debug("Cache hit", cache_key=cache_key)
                return cached["result"]
            else:
                # Remove expired cache
                del query_cache[cache_key]
                logger.debug("Cache expired, removed", cache_key=cache_key)
    except Exception as e:
        logger.warning("Failed to get cached result", error=str(e), cache_key=cache_key)
    return None


def generate_cache_key(query: str, database_name: Optional[str], include_analysis: bool, include_viz: bool) -> str:
    """Generate cache key for query."""
    import hashlib
    cache_data = f"{query}:{database_name}:{include_analysis}:{include_viz}"
    return hashlib.md5(cache_data.encode()).hexdigest()


async def store_query_history(query_data: Dict[str, Any]):
    """Store query in history (background task)."""
    try:
        query_history.append({
            **query_data,
            "timestamp": datetime.utcnow().isoformat()
        })
        # Keep only last 1000 queries
        if len(query_history) > 1000:
            query_history.pop(0)
        logger.debug("Query stored in history", query_id=query_data.get("request_id"))
    except Exception as e:
        logger.warning("Failed to store query history", error=str(e))


def create_simple_sql_result(sql: str, data: List[Dict], execution_time: float = 0.0) -> SQLResult:
    """Create a simplified SQL result for fallback scenarios."""
    columns = list(data[0].keys()) if data else []
    return SQLResult(
        sql=sql,
        data=data,
        row_count=len(data),
        total_rows=len(data),
        execution_time=execution_time,
        columns=columns,
        column_types=None,
        explanation=f"Generated SQL: {sql}",
        query_plan=None,
        cache_hit=False,
        warnings=[]
    )


# FIXED ROUTE: Changed from "/query" to "/process" to match test expectations
@router.post("/process", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    req: Request,
    background_tasks: BackgroundTasks
) -> QueryResponse:
    """
    Enhanced query processing endpoint with graceful fallback.
    
    Route: POST /api/v1/query/process (matches test expectations)
    
    This endpoint coordinates all agents to:
    1. Check cache for existing results
    2. Try orchestrator for full AI processing
    3. Fallback to basic SQL generation if orchestrator unavailable
    4. Store results in history and cache
    """
    start_time = time.time()
    request_id = getattr(req.state, "request_id", request.request_id or str(uuid.uuid4()))
    
    logger.info(
        "Processing query",
        query=request.query[:100],  # Truncate for logging
        request_id=request_id,
        database_name=request.database_name,
        include_analysis=request.include_analysis,
        include_visualization=request.include_visualization,
        session_id=request.session_id
    )
    
    try:
        # Generate cache key
        cache_key = generate_cache_key(
            request.query, 
            request.database_name, 
            request.include_analysis or False,
            request.include_visualization or False
        )
        
        # Check cache first
        cached_result = await get_cached_result(cache_key)
        if cached_result:
            logger.info("Returning cached result", request_id=request_id, cache_key=cache_key)
            cached_result["cached"] = True
            cached_result["processing_time"] = time.time() - start_time
            return QueryResponse(**cached_result)
        
        # Try to get orchestrator
        orchestrator = get_orchestrator()
        
        if orchestrator is not None:
            # Full AI processing with orchestrator
            final_state = await process_with_orchestrator(
                orchestrator, request, request_id, start_time
            )
        else:
            # Graceful fallback without orchestrator
            logger.warning("Orchestrator unavailable, using fallback processing", request_id=request_id)
            final_state = await process_with_fallback(request, request_id)
        
        processing_time = time.time() - start_time
        
        # Create response data
        response_data = {
            "request_id": request_id,
            "timestamp": datetime.utcnow(),
            "processing_time": processing_time,
            "query": request.query,
            "intent": final_state.get("intent", QueryIntent.SQL_GENERATION),
            "confidence": final_state.get("confidence", 0.8),
            "sql_result": final_state.get("sql_result"),
            "analysis_result": final_state.get("analysis_result"),
            "visualization_result": final_state.get("visualization_result"),
            "suggestions": final_state.get("suggestions", []),
            "cached": False
        }
        
        # Cache the result in background
        background_tasks.add_task(cache_query_result, cache_key, response_data)
        
        # Store in history in background
        background_tasks.add_task(store_query_history, {
            "request_id": request_id,
            "query": request.query,
            "database_name": request.database_name,
            "intent": response_data["intent"].value if hasattr(response_data["intent"], 'value') else str(response_data["intent"]),
            "processing_time": processing_time,
            "success": True,
            "row_count": response_data["sql_result"].row_count if response_data["sql_result"] else 0,
            "orchestrator_used": orchestrator is not None
        })
        
        logger.info(
            "Query processed successfully",
            request_id=request_id,
            intent=response_data["intent"],
            confidence=response_data["confidence"],
            processing_time=processing_time,
            has_sql_result=response_data["sql_result"] is not None,
            has_analysis=response_data["analysis_result"] is not None,
            has_visualization=response_data["visualization_result"] is not None,
            orchestrator_used=orchestrator is not None,
            cached=False
        )
        
        return QueryResponse(**response_data)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        return await handle_query_error(e, request, request_id, start_time, background_tasks)


async def process_with_orchestrator(
    orchestrator: AgentOrchestrator, 
    request: QueryRequest, 
    request_id: str, 
    start_time: float
) -> Dict[str, Any]:
    """Process query using the full orchestrator."""
    try:
        # Create enhanced agent state
        state = AgentState(
            query=request.query,
            session_id=request.session_id or f"session_{request_id}",
            database_name=request.database_name,
            metadata={
                "request_id": request_id,
                "max_results": request.max_results,
                "include_analysis": request.include_analysis,
                "include_visualization": request.include_visualization,
                "chart_type": request.chart_type.value if request.chart_type else None,
                "analysis_type": request.analysis_type.value if request.analysis_type else None,
                "context": request.context or {}
            }
        )
        
        # Process query through orchestrator with timeout
        final_state = await asyncio.wait_for(
            orchestrator.process_query(
                query=request.query,
                database_name=request.database_name,
                context=request.context or {}
            ),
            timeout=120  # Reduced timeout for better UX
        )
        
        # Enhanced intent detection with confidence
        intent = QueryIntent.UNKNOWN
        confidence = 0.0
        
        if final_state.metadata.get("routing"):
            routing = final_state.metadata["routing"]
            primary_agent = routing.get("primary_agent", "sql")
            confidence = routing.get("confidence", 0.0)
            
            intent_mapping = {
                "sql": QueryIntent.SQL_GENERATION,
                "analysis": QueryIntent.ANALYSIS,
                "visualization": QueryIntent.VISUALIZATION,
                "schema": QueryIntent.SCHEMA_INFO
            }
            intent = intent_mapping.get(primary_agent, QueryIntent.UNKNOWN)
        
        # Convert state results to API models
        sql_result = None
        if final_state.query_result:
            sql_result = SQLResult(
                sql=final_state.query_result.sql_query or final_state.generated_sql or "",
                data=final_state.query_result.data[:request.max_results] if final_state.query_result.data else [],
                row_count=len(final_state.query_result.data[:request.max_results]) if final_state.query_result.data else 0,
                total_rows=len(final_state.query_result.data) if final_state.query_result.data else 0,
                execution_time=getattr(final_state.query_result, 'execution_time', 0.0),
                columns=getattr(final_state.query_result, 'columns', []),
                column_types=getattr(final_state.query_result, 'column_types', None),
                explanation=final_state.generated_sql or "SQL generated by AI agent",
                query_plan=getattr(final_state.query_result, 'query_plan', None),
                cache_hit=False,
                warnings=getattr(final_state.query_result, 'warnings', [])
            )
        
        # Simplified analysis result handling
        analysis_result = None
        if request.include_analysis and final_state.analysis_result:
            try:
                from sql_agent.api.models import StatisticalSummary
                
                summary = getattr(final_state.analysis_result, "summary", {})
                analysis_result = AnalysisResult(
                    summary=StatisticalSummary(
                        count=summary.get("count", 0),
                        numeric_columns=summary.get("numeric_columns", {}),
                        categorical_columns=summary.get("categorical_columns", {}),
                        missing_values=summary.get("missing_values", {}),
                        data_types=summary.get("data_types", {}),
                        correlations=summary.get("correlations", None)
                    ),
                    insights=[],  # Simplified for now
                    anomalies=[],  # Simplified for now
                    trends=[],  # Simplified for now
                    recommendations=[],  # Simplified for now
                    data_quality_score=getattr(final_state.analysis_result, "data_quality_score", 0.8),
                    confidence_score=getattr(final_state.analysis_result, 'confidence_score', 0.8),
                    processing_metadata=getattr(final_state.analysis_result, 'metadata', None)
                )
            except Exception as e:
                logger.warning("Failed to create analysis result", error=str(e))
                analysis_result = None
        
        # Simplified visualization result handling
        visualization_result = None
        if request.include_visualization and final_state.visualization_config:
            try:
                from sql_agent.api.models import ChartConfig
                
                chart_type = ChartType(final_state.visualization_config.chart_type)
                chart_config = ChartConfig(
                    type=chart_type,
                    title=getattr(final_state.visualization_config, 'title', "Generated Chart"),
                    x_axis=getattr(final_state.visualization_config, 'x_axis', ""),
                    y_axis=getattr(final_state.visualization_config, 'y_axis', ""),
                    color_by=getattr(final_state.visualization_config, 'color_by', None),
                    size_by=getattr(final_state.visualization_config, 'size_by', None),
                    aggregation=getattr(final_state.visualization_config, 'aggregation', None),
                    theme=getattr(final_state.visualization_config, 'theme', 'light'),
                    interactive=getattr(final_state.visualization_config, 'interactive', True),
                    responsive=getattr(final_state.visualization_config, 'responsive', True),
                    animations=getattr(final_state.visualization_config, 'animations', True),
                    legend=getattr(final_state.visualization_config, 'legend', True),
                    grid=getattr(final_state.visualization_config, 'grid', True)
                )
                
                visualization_result = VisualizationResult(
                    chart_type=chart_type,
                    chart_config=chart_config,
                    chart_data=getattr(final_state.visualization_config, 'config', {}),
                    title=getattr(final_state.visualization_config, 'title', "Generated Chart"),
                    description=getattr(final_state.visualization_config, 'description', None),
                    export_formats=["json", "html", "png", "svg"],
                    alternative_charts=getattr(final_state.visualization_config, 'alternatives', []),
                    data_insights=getattr(final_state.visualization_config, 'insights', [])
                )
            except Exception as e:
                logger.warning("Failed to create visualization result", error=str(e))
                visualization_result = None
        
        # Generate suggestions
        suggestions = []
        if sql_result and sql_result.data:
            suggestions.extend([
                f"Try filtering the results: Add WHERE conditions",
                f"Aggregate the data: Use GROUP BY to summarize",
                "Export results: Download the data in CSV format"
            ])
        
        if not request.include_analysis:
            suggestions.append("Get insights: Enable analysis to discover patterns")
        
        if not request.include_visualization:
            suggestions.append("Visualize data: Enable visualization to create charts")
        
        return {
            "intent": intent,
            "confidence": confidence,
            "sql_result": sql_result,
            "analysis_result": analysis_result,
            "visualization_result": visualization_result,
            "suggestions": suggestions[:5]
        }
        
    except asyncio.TimeoutError:
        logger.warning("Orchestrator processing timed out", request_id=request_id)
        raise HTTPException(
            status_code=408,
            detail={
                "error": "Query processing timed out",
                "user_message": "Your query took too long to process",
                "suggestions": [
                    "Try a simpler query",
                    "Add more specific filters",
                    "Break complex queries into smaller parts"
                ],
                "request_id": request_id,
                "type": "timeout_error"
            }
        )


async def process_with_fallback(request: QueryRequest, request_id: str) -> Dict[str, Any]:
    """Fallback processing when orchestrator is unavailable."""
    logger.info("Processing with fallback logic", request_id=request_id)
    
    # Basic SQL generation fallback
    # In a real implementation, this could use a simpler SQL generation service
    fallback_sql = f"-- Fallback SQL for: {request.query}\nSELECT 'Orchestrator unavailable' as message, 'Try direct SQL execution' as suggestion;"
    
    sql_result = create_simple_sql_result(
        sql=fallback_sql,
        data=[{
            "message": "AI processing temporarily unavailable",
            "suggestion": "Use direct SQL execution endpoints",
            "query": request.query
        }],
        execution_time=0.001
    )
    
    return {
        "intent": QueryIntent.SQL_GENERATION,
        "confidence": 0.3,  # Low confidence for fallback
        "sql_result": sql_result,
        "analysis_result": None,
        "visualization_result": None,
        "suggestions": [
            "Try using /api/v1/sql/execute for direct SQL",
            "Contact support if AI processing issues persist",
            "Check system status for service availability"
        ]
    }


async def handle_query_error(
    error: Exception, 
    request: QueryRequest, 
    request_id: str, 
    start_time: float, 
    background_tasks: BackgroundTasks
) -> QueryResponse:
    """Handle query processing errors with user-friendly messages."""
    processing_time = time.time() - start_time
    
    logger.error(
        "Query processing failed",
        request_id=request_id,
        error=str(error),
        error_type=type(error).__name__,
        processing_time=processing_time,
        exc_info=True
    )
    
    # Store failed query in history
    background_tasks.add_task(store_query_history, {
        "request_id": request_id,
        "query": request.query,
        "database_name": request.database_name,
        "processing_time": processing_time,
        "success": False,
        "error": str(error),
        "error_type": type(error).__name__
    })
    
    # Provide user-friendly error messages
    user_message = "An error occurred while processing your query."
    suggestions = [
        "Try rephrasing your question more clearly",
        "Check if the database and table names are correct",
        "Simplify your query and try again"
    ]
    error_type = "query_processing_error"
    
    if "timeout" in str(error).lower():
        user_message = "Your query took too long to process."
        suggestions = [
            "Try limiting your results with specific date ranges",
            "Use more specific filters to reduce data volume",
            "Break complex queries into smaller parts"
        ]
        error_type = "timeout_error"
    elif "connection" in str(error).lower():
        user_message = "Database connection issue."
        suggestions = [
            "Check if the database is available",
            "Try again in a few moments",
            "Contact support if the issue persists"
        ]
        error_type = "connection_error"
    elif "orchestrator" in str(error).lower():
        user_message = "AI processing service is temporarily unavailable."
        suggestions = [
            "Try using direct SQL execution instead",
            "Check system status for service updates",
            "Contact support if needed"
        ]
        error_type = "service_unavailable"
    
    raise HTTPException(
        status_code=500,
        detail={
            "error": str(error),
            "user_message": user_message,
            "suggestions": suggestions,
            "request_id": request_id,
            "type": error_type,
            "processing_time": processing_time
        }
    )


# FIXED ROUTE: Added alias for backward compatibility
@router.post("/query", response_model=QueryResponse)
async def process_query_legacy(
    request: QueryRequest,
    req: Request,
    background_tasks: BackgroundTasks
) -> QueryResponse:
    """Legacy endpoint - redirects to /process for backward compatibility."""
    return await process_query(request, req, background_tasks)


# ENHANCED: Simplified endpoint that works without orchestrator
@router.post("/simple", response_model=Dict[str, Any])
async def simple_query(
    request: QueryRequest,
    req: Request,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Simplified query endpoint for basic SQL generation and execution.
    
    This endpoint works even when the orchestrator is unavailable.
    Returns minimal response structure for lightweight clients.
    """
    start_time = time.time()
    request_id = getattr(req.state, "request_id", request.request_id or str(uuid.uuid4()))
    
    logger.info(
        "Processing simple query",
        query=request.query[:100],
        request_id=request_id,
        database_name=request.database_name
    )
    
    try:
        # Check cache for simple queries
        cache_key = generate_cache_key(request.query, request.database_name, False, False)
        cached_result = await get_cached_result(cache_key)
        
        if cached_result and "sql_result" in cached_result:
            sql_result = cached_result["sql_result"]
            return {
                "query": request.query,
                "intent": cached_result.get("intent", "sql_generation"),
                "sql": sql_result.get("sql", ""),
                "data": sql_result.get("data", [])[:request.max_results],
                "row_count": min(len(sql_result.get("data", [])), request.max_results),
                "execution_time": sql_result.get("execution_time", 0),
                "processing_time": time.time() - start_time,
                "explanation": sql_result.get("explanation", ""),
                "cached": True
            }
        
        # Try orchestrator first, fallback if unavailable
        orchestrator = get_orchestrator()
        
        if orchestrator is not None:
            try:
                # Process through orchestrator with shorter timeout
                final_state = await asyncio.wait_for(
                    orchestrator.process_query(
                        query=request.query,
                        database_name=request.database_name
                    ),
                    timeout=60  # 1 minute for simple queries
                )
                
                sql_result = final_state.query_result
                result = {
                    "query": request.query,
                    "intent": "sql_generation",
                    "sql": final_state.generated_sql or "",
                    "data": sql_result.data[:request.max_results] if sql_result and sql_result.data else [],
                    "row_count": min(len(sql_result.data), request.max_results) if sql_result and sql_result.data else 0,
                    "execution_time": sql_result.execution_time if sql_result else 0,
                    "processing_time": time.time() - start_time,
                    "explanation": final_state.generated_sql or "Generated by AI agent",
                    "cached": False
                }
                
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=408,
                    detail="Simple query processing timed out. Please try a more specific query."
                )
        else:
            # Fallback processing
            result = {
                "query": request.query,
                "intent": "fallback",
                "sql": f"-- AI processing unavailable\n-- Query: {request.query}",
                "data": [{"message": "AI processing temporarily unavailable", "query": request.query}],
                "row_count": 1,
                "execution_time": 0.001,
                "processing_time": time.time() - start_time,
                "explanation": "Fallback processing - orchestrator unavailable",
                "cached": False
            }
        
        # Cache simple result
        background_tasks.add_task(cache_query_result, cache_key, {"sql_result": result, "intent": result["intent"]})
        
        logger.info(
            "Simple query processed",
            request_id=request_id,
            intent=result["intent"],
            processing_time=result["processing_time"],
            orchestrator_used=orchestrator is not None
        )
        
        return result
        
    except HTTPException:
        raise
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
            detail={
                "error": f"Failed to process simple query: {str(e)}",
                "user_message": "Query processing failed",
                "suggestions": [
                    "Try rephrasing your query",
                    "Use more specific terms",
                    "Contact support if issue persists"
                ],
                "request_id": request_id,
                "type": "simple_query_error"
            }
        )


# Keep all other endpoints unchanged...
@router.get("/history", response_model=PaginatedResponse)
async def get_query_history(
    req: Request,
    pagination: PaginationParams = Depends(),
    database_name: Optional[str] = FastAPIQuery(None, description="Filter by database"),
    success_only: bool = FastAPIQuery(False, description="Show only successful queries"),
    search: Optional[str] = FastAPIQuery(None, description="Search in query text")
) -> PaginatedResponse:
    """Get paginated query history with filtering and search capabilities."""
    request_id = getattr(req.state, "request_id", "unknown")
    
    logger.info(
        "Getting query history",
        request_id=request_id,
        page=pagination.page,
        limit=pagination.limit,
        database_name=database_name,
        success_only=success_only,
        search=search
    )
    
    try:
        # Filter history
        filtered_history = query_history.copy()
        
        if database_name:
            filtered_history = [q for q in filtered_history if q.get("database_name") == database_name]
        
        if success_only:
            filtered_history = [q for q in filtered_history if q.get("success", True)]
        
        if search:
            search_lower = search.lower()
            filtered_history = [
                q for q in filtered_history 
                if search_lower in q.get("query", "").lower()
            ]
        
        # Sort by timestamp (newest first)
        filtered_history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Paginate
        total = len(filtered_history)
        start_idx = (pagination.page - 1) * pagination.limit
        end_idx = start_idx + pagination.limit
        page_items = filtered_history[start_idx:end_idx]
        
        total_pages = (total + pagination.limit - 1) // pagination.limit
        
        return PaginatedResponse(
            items=page_items,
            total=total,
            page=pagination.page,
            limit=pagination.limit,
            pages=total_pages,
            has_next=pagination.page < total_pages,
            has_prev=pagination.page > 1
        )
        
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


@router.delete("/history")
async def clear_query_history(
    req: Request,
    confirm: bool = FastAPIQuery(False, description="Confirmation required")
) -> Dict[str, str]:
    """Clear query history with confirmation."""
    request_id = getattr(req.state, "request_id", "unknown")
    
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required. Add ?confirm=true to clear history."
        )
    
    logger.info("Clearing query history", request_id=request_id)
    
    try:
        cleared_count = len(query_history)
        query_history.clear()
        
        return {
            "message": f"Query history cleared successfully. Removed {cleared_count} queries.",
            "cleared_count": str(cleared_count)
        }
        
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


@router.post("/validate", response_model=ValidationResult)
async def validate_query_intent(
    request: QueryRequest,
    req: Request
) -> ValidationResult:
    """
    Validate query and detect intent without execution.
    Works with or without orchestrator.
    """
    request_id = getattr(req.state, "request_id", request.request_id or str(uuid.uuid4()))
    
    logger.info(
        "Validating query intent",
        query=request.query[:100],
        request_id=request_id
    )
    
    try:
        orchestrator = get_orchestrator()
        
        if orchestrator is not None:
            # Full validation with orchestrator
            try:
                state = AgentState(
                    query=request.query,
                    session_id=f"validate_{request_id}",
                    database_name=request.database_name
                )
                
                # Use orchestrator's router to detect intent
                routing_result = await orchestrator.router.route(state)
                
                suggestions = []
                warnings = []
                errors = []
                
                # Generate suggestions based on intent
                primary_agent = routing_result.get("primary_agent", "sql")
                if primary_agent == "sql":
                    suggestions.append("This query will generate and execute SQL")
                    if not request.database_name:
                        warnings.append("No database specified, will use default")
                elif primary_agent == "analysis":
                    suggestions.append("This query will perform data analysis")
                    suggestions.append("Consider including 'analyze' or 'insights' for better results")
                elif primary_agent == "visualization":
                    suggestions.append("This query will create a visualization")
                    suggestions.append("Specify chart type (bar, line, pie) for better results")
                
                confidence = routing_result.get("confidence", 0.0)
                if confidence < 0.7:
                    warnings.append("Query intent is unclear, consider rephrasing")
                
            except Exception as e:
                logger.warning("Orchestrator validation failed, using fallback", error=str(e))
                # Fallback to basic validation
                suggestions, warnings, errors = basic_query_validation(request.query)
        else:
            # Basic validation without orchestrator
            suggestions, warnings, errors = basic_query_validation(request.query)
        
        # Basic query validation
        if len(request.query.split()) < 3:
            warnings.append("Query seems very short, consider adding more detail")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            estimated_cost=None,  # Could be implemented based on query complexity
            estimated_rows=None   # Could be estimated based on database statistics
        )
        
    except Exception as e:
        logger.error(
            "Query validation failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        return ValidationResult(
            is_valid=False,
            errors=[f"Validation failed: {str(e)}"],
            warnings=[],
            suggestions=["Try rephrasing your query", "Check for typos"],
            estimated_cost=None,
            estimated_rows=None
        )


def basic_query_validation(query: str) -> tuple[List[str], List[str], List[str]]:
    """Basic query validation without orchestrator."""
    suggestions = []
    warnings = []
    errors = []
    
    query_lower = query.lower()
    
    # Basic intent detection
    if any(word in query_lower for word in ['show', 'list', 'get', 'find', 'select']):
        suggestions.append("This appears to be a data retrieval query")
    elif any(word in query_lower for word in ['analyze', 'analysis', 'insight', 'pattern']):
        suggestions.append("This appears to be an analysis query")
    elif any(word in query_lower for word in ['chart', 'graph', 'plot', 'visualize']):
        suggestions.append("This appears to be a visualization query")
    
    # Basic warnings
    if len(query.split()) < 3:
        warnings.append("Query is very short, consider adding more detail")
    
    if not any(word in query_lower for word in ['customer', 'product', 'order', 'employee', 'sales']):
        warnings.append("Query doesn't reference common business entities")
    
    # Basic suggestions
    suggestions.extend([
        "Be specific about what data you want",
        "Include time ranges if relevant",
        "Specify filtering criteria clearly"
    ])
    
    return suggestions, warnings, errors


@router.post("/feedback")
async def submit_query_feedback(
    feedback: FeedbackRequest,
    req: Request,
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """Submit feedback for a query result."""
    request_id = getattr(req.state, "request_id", feedback.request_id or str(uuid.uuid4()))
    
    logger.info(
        "Receiving query feedback",
        query_id=feedback.query_id,
        rating=feedback.rating,
        request_id=request_id
    )
    
    try:
        # Store feedback (in production, save to database)
        feedback_data = {
            "feedback_id": str(uuid.uuid4()),
            "query_id": feedback.query_id,
            "rating": feedback.rating,
            "feedback": feedback.feedback,
            "category": feedback.category,
            "specific_issues": feedback.specific_issues,
            "suggestions": feedback.suggestions,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id
        }
        
        # In production, this would be saved to a database
        # For now, just log it
        background_tasks.add_task(
            lambda: logger.info("Feedback stored", feedback_data=feedback_data)
        )
        
        return {
            "message": "Feedback submitted successfully",
            "feedback_id": feedback_data["feedback_id"]
        }
        
    except Exception as e:
        logger.error(
            "Failed to submit feedback",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit feedback: {str(e)}"
        )


@router.get("/suggestions")
async def get_query_suggestions(
    req: Request,
    database_name: Optional[str] = FastAPIQuery(None, description="Database to get suggestions for"),
    category: Optional[str] = FastAPIQuery(None, description="Suggestion category"),
    limit: int = FastAPIQuery(10, description="Number of suggestions", ge=1, le=50)
) -> Dict[str, List[str]]:
    """Get query suggestions based on database schema and common patterns."""
    request_id = getattr(req.state, "request_id", "unknown")
    
    logger.info(
        "Getting query suggestions",
        request_id=request_id,
        database_name=database_name,
        category=category
    )
    
    try:
        # Generate contextual suggestions based on your actual database
        suggestions = {
            "customer_queries": [
                "Show me all premium customers",
                "Which customers have the highest account balance?",
                "List customers from USA",
                "Find customers who haven't logged in recently",
                "Show customer demographics by country"
            ],
            "product_queries": [
                "What are the top products by price?",
                "Show products running low on stock", 
                "List all electronics products",
                "Find products with low inventory",
                "Show product categories and their counts"
            ],
            "sales_queries": [
                "What's the total revenue from all orders?",
                "How many orders were placed this month?",
                "Show top orders by value",
                "Analyze sales by country",
                "Find orders with highest discounts"
            ],
            "employee_queries": [
                "Show employee performance by department",
                "Which employees exceeded their sales targets?",
                "List top performing sales representatives",
                "Show average performance score by department",
                "Find employees with high training hours"
            ],
            "analysis_queries": [
                "Analyze customer purchase patterns",
                "Find correlations in sales data",
                "Identify seasonal trends in orders",
                "Compare premium vs regular customers",
                "Analyze product performance by category"
            ],
            "visualization_queries": [
                "Create a bar chart of sales by month",
                "Show a pie chart of product categories",
                "Plot customer growth over time",
                "Visualize orders by status",
                "Generate a chart of employee performance"
            ]
        }
        
        if category and category in suggestions:
            return {category: suggestions[category][:limit]}
        
        # Return all categories with limited items
        result = {}
        for cat, items in suggestions.items():
            result[cat] = items[:min(limit, len(items))]
        
        return result
        
    except Exception as e:
        logger.error(
            "Failed to get suggestions",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get suggestions: {str(e)}"
        )


# Health check endpoint for query service
@router.get("/health")
async def query_service_health() -> Dict[str, Any]:
    """Health check for query processing service."""
    orchestrator = get_orchestrator()
    
    health_data = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "orchestrator_available": orchestrator is not None,
        "cache_size": len(query_cache),
        "history_size": len(query_history),
        "service_capabilities": {
            "basic_query_processing": True,
            "ai_query_processing": orchestrator is not None,
            "query_validation": True,
            "query_suggestions": True,
            "query_history": True,
            "query_caching": True
        }
    }
    
    if orchestrator is None:
        health_data["warnings"] = [
            "AI orchestrator not available - using fallback processing",
            "Advanced analysis and visualization features unavailable"
        ]
    
    return health_data


# Status endpoint for debugging
@router.get("/status")
async def query_service_status() -> Dict[str, Any]:
    """Detailed status information for debugging."""
    orchestrator = get_orchestrator()
    
    return {
        "service": "query_processing",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "orchestrator": {
            "available": orchestrator is not None,
            "type": type(orchestrator).__name__ if orchestrator else None,
            "status": "initialized" if orchestrator else "not_available"
        },
        "cache": {
            "enabled": True,
            "size": len(query_cache),
            "max_size": query_cache.maxsize,
            "ttl": query_cache.ttl
        },
        "history": {
            "enabled": True,
            "size": len(query_history),
            "max_size": 1000
        },
        "endpoints": {
            "process": "/api/v1/query/process",
            "simple": "/api/v1/query/simple", 
            "validate": "/api/v1/query/validate",
            "history": "/api/v1/query/history",
            "suggestions": "/api/v1/query/suggestions",
            "feedback": "/api/v1/query/feedback"
        }
    }