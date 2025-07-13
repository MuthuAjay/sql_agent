"""
Enhanced Query Routes

This module contains enhanced query endpoints for natural language to SQL conversion,
analysis, and visualization using the multi-agent system.
"""

import time
import uuid
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks, Query as FastAPIQuery
from fastapi.responses import JSONResponse, StreamingResponse
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


def get_orchestrator() -> AgentOrchestrator:
    """Get the agent orchestrator instance."""
    from sql_agent.api.main import orchestrator
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Agent orchestrator not initialized")
    return orchestrator


async def cache_query_result(cache_key: str, result: Dict[str, Any], ttl: int = 3600):
    """Cache query result with TTL."""
    try:
        query_cache[cache_key] = {
            "result": result,
            "timestamp": datetime.utcnow(),
            "ttl": ttl
        }
    except Exception as e:
        logger.warning("Failed to cache query result", error=str(e))


async def get_cached_result(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get cached query result if available and valid."""
    try:
        cached = query_cache.get(cache_key)
        if cached:
            # Check if cache is still valid
            if datetime.utcnow() - cached["timestamp"] < timedelta(seconds=cached["ttl"]):
                return cached["result"]
            else:
                # Remove expired cache
                del query_cache[cache_key]
    except Exception as e:
        logger.warning("Failed to get cached result", error=str(e))
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
    except Exception as e:
        logger.warning("Failed to store query history", error=str(e))


@router.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    req: Request,
    background_tasks: BackgroundTasks,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> QueryResponse:
    """
    Enhanced query processing endpoint with caching, history, and improved error handling.
    
    This endpoint coordinates all agents to:
    1. Check cache for existing results
    2. Analyze query intent with confidence scoring
    3. Generate and execute SQL with validation
    4. Analyze results with statistical insights
    5. Create visualizations with multiple options
    6. Store results in history and cache
    """
    start_time = time.time()
    request_id = getattr(req.state, "request_id", request.request_id)
    
    logger.info(
        "Processing enhanced query",
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
        try:
            final_state = await asyncio.wait_for(
                orchestrator.process_query(
                    query=request.query,
                    database_name=request.database_name,
                    context=request.context or {}
                ),
                timeout=300  # 5 minutes timeout
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=408,
                detail="Query processing timed out. Please try a simpler query or contact support."
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
        
        # Convert state results to enhanced API models
        sql_result = None
        if final_state.query_result:
            sql_result = SQLResult(
                sql=final_state.query_result.sql_query,
                data=final_state.query_result.data[:request.max_results],
                row_count=len(final_state.query_result.data[:request.max_results]),
                total_rows=len(final_state.query_result.data),
                execution_time=final_state.query_result.execution_time,
                columns=final_state.query_result.columns,
                column_types=getattr(final_state.query_result, 'column_types', None),
                explanation=final_state.generated_sql,
                query_plan=getattr(final_state.query_result, 'query_plan', None),
                cache_hit=False,
                warnings=getattr(final_state.query_result, 'warnings', [])
            )
        
        # Enhanced analysis result
        analysis_result = None
        if request.include_analysis and final_state.analysis_result:
            from sql_agent.api.models import (
                StatisticalSummary, Insight, Anomaly, Trend, Recommendation
            )
            
            # Convert analysis result with enhanced structure
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
                insights=[
                    Insight(
                        type=insight.get("type", "general"),
                        title=insight.get("title", ""),
                        description=insight.get("description", ""),
                        confidence=insight.get("confidence", 0.8),
                        impact=insight.get("impact", "medium"),
                        supporting_data=insight.get("supporting_data", None)
                    ) for insight in getattr(final_state.analysis_result, "insights", [])
                ] if isinstance(getattr(final_state.analysis_result, "insights"), list) else [],
                anomalies=[
                    Anomaly(
                        type=anomaly.get("type", "statistical"),
                        column=anomaly.get("column", None),
                        description=anomaly.get("description", ""),
                        severity=anomaly.get("severity", "medium"),
                        affected_rows=anomaly.get("affected_rows", None),
                        threshold=anomaly.get("threshold", None)
                    ) for anomaly in getattr(final_state.analysis_result, "anomalies", [])
                ] if hasattr(final_state.analysis_result, 'anomalies') else [],
                trends=[
                    Trend(
                        type=trend.get("type", "temporal"),
                        column=trend.get("column", ""),
                        direction=trend.get("direction", "stable"),
                        strength=trend.get("strength", 0.5),
                        period=trend.get("period", None),
                        description=trend.get("description", "")
                    ) for trend in getattr(final_state.analysis_result, "trends", [])
                ] if hasattr(final_state.analysis_result, 'trends') else [],
                recommendations=[
                    Recommendation(
                        type=rec.get("type", "optimization"),
                        title=rec.get("title", ""),
                        description=rec.get("description", ""),
                        priority=rec.get("priority", "medium"),
                        effort=rec.get("effort", "medium"),
                        expected_impact=rec.get("expected_impact", "medium"),
                        action_items=rec.get("action_items", [])
                    ) for rec in getattr(final_state.analysis_result, "recommendations", [])
                ] if isinstance(getattr(final_state.analysis_result, "recommendations"), list) else [],
                data_quality_score=getattr(final_state.analysis_result, "data_quality_score", 0.8),
                confidence_score=getattr(final_state.analysis_result, 'confidence_score', 0.8),
                processing_metadata=getattr(final_state.analysis_result, 'metadata', None)
            )
        
        # Enhanced visualization result
        visualization_result = None
        if request.include_visualization and final_state.visualization_config:
            from sql_agent.api.models import ChartConfig
            
            chart_type = ChartType(final_state.visualization_config.chart_type)
            chart_config = ChartConfig(
                type=chart_type,
                title=final_state.visualization_config.title or "Generated Chart",
                x_axis=final_state.visualization_config.x_axis,
                y_axis=final_state.visualization_config.y_axis,
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
                chart_data=final_state.visualization_config.config,
                title=final_state.visualization_config.title or "Generated Chart",
                description=getattr(final_state.visualization_config, 'description', None),
                export_formats=["json", "html", "png", "svg"],
                alternative_charts=getattr(final_state.visualization_config, 'alternatives', []),
                data_insights=getattr(final_state.visualization_config, 'insights', [])
            )
        
        # Generate follow-up suggestions
        suggestions = []
        if sql_result:
            suggestions.extend([
                f"Try filtering the results: Add WHERE conditions to {sql_result.sql}",
                f"Aggregate the data: Use GROUP BY to summarize {', '.join(sql_result.columns[:3])}",
                "Export results: Download the data in CSV or Excel format"
            ])
        
        if not request.include_analysis:
            suggestions.append("Get insights: Enable analysis to discover patterns and trends")
        
        if not request.include_visualization:
            suggestions.append("Visualize data: Enable visualization to create charts and graphs")
        
        processing_time = time.time() - start_time
        
        # Create response
        response_data = {
            "request_id": request_id,
            "timestamp": datetime.utcnow(),
            "processing_time": processing_time,
            "query": request.query,
            "intent": intent,
            "confidence": confidence,
            "sql_result": sql_result,
            "analysis_result": analysis_result,
            "visualization_result": visualization_result,
            "suggestions": suggestions[:5],  # Limit to 5 suggestions
            "cached": False
        }
        
        # Cache the result in background
        background_tasks.add_task(cache_query_result, cache_key, response_data)
        
        # Store in history in background
        background_tasks.add_task(store_query_history, {
            "request_id": request_id,
            "query": request.query,
            "database_name": request.database_name,
            "intent": intent.value,
            "processing_time": processing_time,
            "success": True,
            "row_count": sql_result.row_count if sql_result else 0
        })
        
        logger.info(
            "Query processed successfully",
            request_id=request_id,
            intent=intent.value,
            confidence=confidence,
            processing_time=processing_time,
            has_sql_result=sql_result is not None,
            has_analysis=analysis_result is not None,
            has_visualization=visualization_result is not None,
            cached=False
        )
        
        return QueryResponse(**response_data)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        
        logger.error(
            "Query processing failed",
            request_id=request_id,
            error=str(e),
            error_type=type(e).__name__,
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
            "error": str(e)
        })
        
        # Provide user-friendly error messages
        user_message = "An error occurred while processing your query."
        suggestions = [
            "Try rephrasing your question more clearly",
            "Check if the database and table names are correct",
            "Simplify your query and try again"
        ]
        
        if "timeout" in str(e).lower():
            user_message = "Your query took too long to process."
            suggestions = [
                "Try limiting your results with specific date ranges",
                "Use more specific filters to reduce data volume",
                "Break complex queries into smaller parts"
            ]
        elif "connection" in str(e).lower():
            user_message = "Database connection issue."
            suggestions = [
                "Check if the database is available",
                "Try again in a few moments",
                "Contact support if the issue persists"
            ]
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "user_message": user_message,
                "suggestions": suggestions,
                "request_id": request_id,
                "type": "query_processing_error"
            }
        )


@router.post("/query/simple", response_model=Dict[str, Any])
async def simple_query(
    request: QueryRequest,
    req: Request,
    background_tasks: BackgroundTasks,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Simplified query endpoint for basic SQL generation and execution.
    
    Returns minimal response structure for lightweight clients.
    """
    start_time = time.time()
    request_id = getattr(req.state, "request_id", request.request_id)
    
    logger.info(
        "Processing simple query",
        query=request.query[:100],
        request_id=request_id,
        database_name=request.database_name
    )
    
    try:
        # Check cache for simple queries too
        cache_key = generate_cache_key(request.query, request.database_name, False, False)
        cached_result = await get_cached_result(cache_key)
        
        if cached_result and "sql_result" in cached_result:
            sql_result = cached_result["sql_result"]
            return {
                "query": request.query,
                "intent": cached_result.get("intent", "sql_generation"),
                "sql": sql_result.get("sql", ""),
                "data": sql_result.get("data", [])[:request.max_results],
                "row_count": min((sql_result.row_count or 0), (request.max_results or 0)) if sql_result else 0,
                "execution_time": sql_result.get("execution_time", 0),
                "processing_time": time.time() - start_time,
                "explanation": sql_result.get("explanation", ""),
                "cached": True
            }
        
        # Process through orchestrator
        final_state = await asyncio.wait_for(
            orchestrator.process_query(
                query=request.query,
                database_name=request.database_name
            ),
            timeout=120  # 2 minutes for simple queries
        )
        
        # Extract basic results
        intent = QueryIntent.SQL_GENERATION
        if final_state.metadata.get("routing"):
            routing = final_state.metadata["routing"]
            primary_agent = routing.get("primary_agent", "sql")
            if primary_agent == "sql":
                intent = QueryIntent.SQL_GENERATION
        
        sql_result = final_state.query_result
        processing_time = time.time() - start_time
        
        result = {
            "query": request.query,
            "intent": intent.value,
            "sql": final_state.generated_sql or "",
            "data": sql_result.data[:request.max_results] if sql_result else [],
            "row_count": min((sql_result.row_count or 0), (request.max_results or 0)) if sql_result else 0,
            "execution_time": sql_result.execution_time if sql_result else 0,
            "processing_time": processing_time,
            "explanation": final_state.generated_sql or "",
            "cached": False
        }
        
        # Cache simple result
        background_tasks.add_task(cache_query_result, cache_key, {"sql_result": result, "intent": intent.value})
        
        logger.info(
            "Simple query processed",
            request_id=request_id,
            intent=intent.value,
            processing_time=processing_time
        )
        
        return result
        
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail="Query processing timed out. Please try a simpler query."
        )
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


@router.get("/query/history", response_model=PaginatedResponse)
async def get_query_history(
    req: Request,
    pagination: PaginationParams = Depends(),
    database_name: Optional[str] = FastAPIQuery(None, description="Filter by database"),
    success_only: bool = FastAPIQuery(False, description="Show only successful queries"),
    search: Optional[str] = FastAPIQuery(None, description="Search in query text")
) -> PaginatedResponse:
    """
    Get paginated query history with filtering and search capabilities.
    """
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


@router.delete("/query/history")
async def clear_query_history(
    req: Request,
    confirm: bool = FastAPIQuery(False, description="Confirmation required")
) -> Dict[str, str]:
    """
    Clear query history with confirmation.
    """
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


@router.post("/query/validate", response_model=ValidationResult)
async def validate_query_intent(
    request: QueryRequest,
    req: Request,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
) -> ValidationResult:
    """
    Validate query and detect intent without execution.
    """
    request_id = getattr(req.state, "request_id", request.request_id)
    
    logger.info(
        "Validating query intent",
        query=request.query[:100],
        request_id=request_id
    )
    
    try:
        # Quick intent detection and validation
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
        if routing_result.get("primary_agent") == "sql":
            suggestions.append("This query will generate and execute SQL")
            if not request.database_name:
                warnings.append("No database specified, will use default")
        elif routing_result.get("primary_agent") == "analysis":
            suggestions.append("This query will perform data analysis")
            suggestions.append("Consider including 'analyze' or 'insights' for better results")
        elif routing_result.get("primary_agent") == "visualization":
            suggestions.append("This query will create a visualization")
            suggestions.append("Specify chart type (bar, line, pie) for better results")
        
        # Basic query validation
        if len(request.query.split()) < 3:
            warnings.append("Query seems very short, consider adding more detail")
        
        confidence = routing_result.get("confidence", 0.0)
        if confidence < 0.7:
            warnings.append("Query intent is unclear, consider rephrasing")
        
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


@router.post("/query/feedback")
async def submit_query_feedback(
    feedback: FeedbackRequest,
    req: Request,
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """
    Submit feedback for a query result.
    """
    request_id = getattr(req.state, "request_id", feedback.request_id)
    
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


@router.get("/query/suggestions")
async def get_query_suggestions(
    req: Request,
    database_name: Optional[str] = FastAPIQuery(None, description="Database to get suggestions for"),
    category: Optional[str] = FastAPIQuery(None, description="Suggestion category"),
    limit: int = FastAPIQuery(10, description="Number of suggestions", ge=1, le=50)
) -> Dict[str, List[str]]:
    """
    Get query suggestions based on database schema and common patterns.
    """
    request_id = getattr(req.state, "request_id", "unknown")
    
    logger.info(
        "Getting query suggestions",
        request_id=request_id,
        database_name=database_name,
        category=category
    )
    
    try:
        # Generate contextual suggestions
        suggestions = {
            "basic_queries": [
                "Show me the top 10 customers by sales",
                "What are the monthly sales trends?",
                "List all products with low inventory",
                "Show customer demographics",
                "Analyze sales by region"
            ],
            "analysis_queries": [
                "Analyze customer purchase patterns",
                "Find correlations in sales data",
                "Identify seasonal trends",
                "Detect anomalies in revenue",
                "Compare year-over-year growth"
            ],
            "visualization_queries": [
                "Create a bar chart of sales by month",
                "Show a pie chart of product categories",
                "Plot customer growth over time",
                "Visualize geographic sales distribution",
                "Generate a heatmap of user activity"
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


# WebSocket endpoint for real-time query processing (optional)
@router.websocket("/query/stream")
async def stream_query_processing(websocket):
    """
    WebSocket endpoint for streaming query processing updates.
    
    This would be useful for long-running queries to provide real-time updates.
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive query from client
            data = await websocket.receive_json()
            query = data.get("query", "")
            
            if not query:
                await websocket.send_json({"error": "No query provided"})
                continue
            
            # Send processing updates
            await websocket.send_json({"status": "processing", "stage": "intent_detection"})
            await asyncio.sleep(0.5)  # Simulate processing time
            
            await websocket.send_json({"status": "processing", "stage": "sql_generation"})
            await asyncio.sleep(1)
            
            await websocket.send_json({"status": "processing", "stage": "query_execution"})
            await asyncio.sleep(2)
            
            await websocket.send_json({"status": "complete", "result": {"message": "Query completed"}})
            
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()