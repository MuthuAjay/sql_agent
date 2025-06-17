"""
Analysis Routes

This module contains endpoints for data analysis functionality.
"""

import time
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends, Request

from sql_agent.api.models import (
    AnalysisRequest, AnalysisResult
)
from sql_agent.agents.analysis import AnalysisAgent
from sql_agent.core.llm import LLMFactory

router = APIRouter()


def get_analysis_agent() -> AnalysisAgent:
    """Get the analysis agent instance."""
    llm_provider = LLMFactory.create_provider()
    return AnalysisAgent(llm_provider)


@router.post("/analyze", response_model=AnalysisResult)
async def analyze_data(
    request: AnalysisRequest,
    req: Request,
    analysis_agent: AnalysisAgent = Depends(get_analysis_agent)
) -> AnalysisResult:
    """
    Analyze data and provide insights.
    
    This endpoint uses the analysis agent to analyze the provided data
    and generate business insights, recommendations, and statistical summaries.
    """
    start_time = time.time()
    request_id = getattr(req.state, "request_id", "unknown")
    
    try:
        # Create a mock state for analysis
        from sql_agent.core.state import AgentState, QueryResult
        from sql_agent.core.state import AnalysisResult as StateAnalysisResult
        
        # Create query result with the provided data
        query_result = QueryResult(
            data=request.data,
            row_count=len(request.data),
            columns=list(request.data[0].keys()) if request.data else []
        )
        
        state = AgentState(
            query=request.query_context or "Analyze the provided data",
            session_id=f"analysis_{request_id}",
            query_result=query_result
        )
        
        # Run analysis agent
        result_state = await analysis_agent.run(state)
        
        processing_time = time.time() - start_time
        
        # Convert state analysis result to API model
        if result_state.analysis_result:
            return AnalysisResult(
                summary={},  # Will be populated from analysis_result
                insights=result_state.analysis_result.insights,
                anomalies=[],  # Will be populated from analysis_result
                trends=[],  # Will be populated from analysis_result
                recommendations=result_state.analysis_result.recommendations,
                data_quality_score=result_state.analysis_result.data_quality_score or 0.0
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Analysis agent did not return results"
            )
        
    except Exception as e:
        processing_time = time.time() - start_time
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze data: {str(e)}"
        )


@router.post("/analyze/sql", response_model=AnalysisResult)
async def analyze_sql_results(
    request: AnalysisRequest,
    req: Request,
    analysis_agent: AnalysisAgent = Depends(get_analysis_agent)
) -> AnalysisResult:
    """
    Analyze results from a SQL query.
    
    This endpoint is specifically designed to analyze data that comes from
    SQL query execution results.
    """
    start_time = time.time()
    request_id = getattr(req.state, "request_id", "unknown")
    
    try:
        # Create a mock state for SQL result analysis
        from sql_agent.core.state import AgentState, QueryResult
        
        # Create query result with the provided data
        query_result = QueryResult(
            data=request.data,
            row_count=len(request.data),
            columns=list(request.data[0].keys()) if request.data else []
        )
        
        state = AgentState(
            query=request.query_context or "Analyze the SQL query results",
            session_id=f"sql_analysis_{request_id}",
            query_result=query_result
        )
        
        # Run analysis agent
        result_state = await analysis_agent.run(state)
        
        processing_time = time.time() - start_time
        
        # Convert state analysis result to API model
        if result_state.analysis_result:
            return AnalysisResult(
                summary={},  # Will be populated from analysis_result
                insights=result_state.analysis_result.insights,
                anomalies=[],  # Will be populated from analysis_result
                trends=[],  # Will be populated from analysis_result
                recommendations=result_state.analysis_result.recommendations,
                data_quality_score=result_state.analysis_result.data_quality_score or 0.0
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Analysis agent did not return results"
            )
        
    except Exception as e:
        processing_time = time.time() - start_time
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze SQL results: {str(e)}"
        )


@router.get("/types", response_model=Dict[str, Any])
async def get_analysis_types() -> Dict[str, Any]:
    """
    Get available analysis types.
    
    This endpoint returns the different types of analysis that can be performed.
    """
    analysis_types = {
        "comprehensive": {
            "name": "Comprehensive Analysis",
            "description": "Full analysis including insights, trends, and recommendations",
            "includes": ["statistical_summary", "insights", "trends", "anomalies", "recommendations"]
        },
        "statistical": {
            "name": "Statistical Analysis",
            "description": "Focus on statistical measures and data quality",
            "includes": ["statistical_summary", "data_quality_score"]
        },
        "business": {
            "name": "Business Analysis",
            "description": "Focus on business insights and actionable recommendations",
            "includes": ["insights", "recommendations", "trends"]
        },
        "quality": {
            "name": "Data Quality Analysis",
            "description": "Focus on data quality assessment",
            "includes": ["data_quality_score", "anomalies", "statistical_summary"]
        }
    }
    
    return {
        "analysis_types": analysis_types,
        "total": len(analysis_types)
    } 