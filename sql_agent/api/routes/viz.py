"""
Visualization Routes

This module contains endpoints for data visualization functionality.
"""

import time
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends, Request

from sql_agent.api.models import (
    VisualizationRequest, VisualizationResult, ChartType
)
from sql_agent.agents.viz import VisualizationAgent
from sql_agent.core.llm import LLMFactory

router = APIRouter()


def get_visualization_agent() -> VisualizationAgent:
    """Get the visualization agent instance."""
    llm_provider = LLMFactory.create_provider()
    return VisualizationAgent(llm_provider)


@router.post("/create", response_model=VisualizationResult)
async def create_visualization(
    request: VisualizationRequest,
    req: Request,
    viz_agent: VisualizationAgent = Depends(get_visualization_agent)
) -> VisualizationResult:
    """
    Create a data visualization.
    
    This endpoint uses the visualization agent to create charts
    from the provided data.
    """
    start_time = time.time()
    request_id = getattr(req.state, "request_id", "unknown")
    
    try:
        # Create a mock state for visualization
        from sql_agent.core.state import AgentState, QueryResult, VisualizationConfig
        
        # Create query result with the provided data
        query_result = QueryResult(
            data=request.data,
            row_count=len(request.data),
            columns=list(request.data[0].keys()) if request.data else []
        )
        
        # Create visualization config
        viz_config = VisualizationConfig(
            chart_type=request.chart_type.value if request.chart_type else "bar",
            title=request.title,
            x_axis=request.x_axis,
            y_axis=request.y_axis
        )
        
        state = AgentState(
            query=request.title or "Create a visualization of the data",
            session_id=f"viz_{request_id}",
            query_result=query_result,
            visualization_config=viz_config
        )
        
        # Run visualization agent
        result_state = await viz_agent.run(state)
        
        processing_time = time.time() - start_time
        
        # Convert state visualization result to API model
        if result_state.visualization_config:
            return VisualizationResult(
                chart_type=ChartType(result_state.visualization_config.chart_type),
                chart_config=result_state.visualization_config.config,
                chart_data={},  # Will be populated from visualization
                title=result_state.visualization_config.title or "Generated Chart",
                description=None,
                export_formats=["json", "html"]
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Visualization agent did not return results"
            )
        
    except Exception as e:
        processing_time = time.time() - start_time
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create visualization: {str(e)}"
        )


@router.post("/suggest", response_model=Dict[str, Any])
async def suggest_chart_type(
    request: VisualizationRequest,
    req: Request,
    viz_agent: VisualizationAgent = Depends(get_visualization_agent)
) -> Dict[str, Any]:
    """
    Suggest appropriate chart type for the data.
    
    This endpoint analyzes the data and suggests the best chart type
    for visualization.
    """
    start_time = time.time()
    request_id = getattr(req.state, "request_id", "unknown")
    
    try:
        # Create a mock state for chart type suggestion
        from sql_agent.core.state import AgentState, QueryResult
        
        # Create query result with the provided data
        query_result = QueryResult(
            data=request.data,
            row_count=len(request.data),
            columns=list(request.data[0].keys()) if request.data else []
        )
        
        state = AgentState(
            query="Suggest the best chart type for this data",
            session_id=f"suggest_{request_id}",
            query_result=query_result
        )
        
        # Run visualization agent to get suggestions
        result_state = await viz_agent.run(state)
        
        processing_time = time.time() - start_time
        
        # Extract suggestions from agent metadata
        suggestions = result_state.metadata.get("chart_suggestions", [])
        
        return {
            "data_characteristics": {
                "row_count": len(request.data),
                "column_count": len(request.data[0]) if request.data else 0,
                "columns": list(request.data[0].keys()) if request.data else []
            },
            "suggested_charts": suggestions,
            "processing_time": processing_time
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        raise HTTPException(
            status_code=500,
            detail=f"Failed to suggest chart type: {str(e)}"
        )


@router.get("/types", response_model=Dict[str, Any])
async def get_chart_types() -> Dict[str, Any]:
    """
    Get available chart types and their descriptions.
    
    This endpoint returns all supported chart types with descriptions
    of when to use each type.
    """
    chart_types = {
        "bar": {
            "name": "Bar Chart",
            "description": "Compare quantities across categories",
            "best_for": ["categorical data", "comparisons", "rankings"],
            "data_requirements": ["categorical x-axis", "numerical y-axis"]
        },
        "line": {
            "name": "Line Chart",
            "description": "Show trends over time or continuous data",
            "best_for": ["time series", "trends", "continuous data"],
            "data_requirements": ["ordered x-axis", "numerical y-axis"]
        },
        "pie": {
            "name": "Pie Chart",
            "description": "Show proportions of a whole",
            "best_for": ["proportions", "percentages", "parts of whole"],
            "data_requirements": ["categorical data", "numerical values"]
        },
        "scatter": {
            "name": "Scatter Plot",
            "description": "Show relationship between two variables",
            "best_for": ["correlations", "distributions", "outliers"],
            "data_requirements": ["two numerical variables"]
        },
        "histogram": {
            "name": "Histogram",
            "description": "Show distribution of numerical data",
            "best_for": ["distributions", "frequency analysis"],
            "data_requirements": ["single numerical variable"]
        },
        "table": {
            "name": "Data Table",
            "description": "Display data in tabular format",
            "best_for": ["detailed data", "exact values", "sorting"],
            "data_requirements": ["any data structure"]
        }
    }
    
    return {
        "chart_types": chart_types,
        "total": len(chart_types)
    }


@router.post("/export", response_model=Dict[str, Any])
async def export_visualization(
    request: VisualizationRequest,
    req: Request,
    format: str = "json"
) -> Dict[str, Any]:
    """
    Export visualization in various formats.
    
    This endpoint exports the visualization data in the specified format.
    """
    request_id = getattr(req.state, "request_id", "unknown")
    
    try:
        # For now, return the data in the requested format
        # In a full implementation, this would generate actual chart files
        
        if format.lower() == "json":
            return {
                "format": "json",
                "data": request.data,
                "chart_config": {
                    "type": request.chart_type.value if request.chart_type else "bar",
                    "title": request.title,
                    "x_axis": request.x_axis,
                    "y_axis": request.y_axis
                }
            }
        elif format.lower() == "html":
            # Generate simple HTML chart
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{request.title or 'Data Visualization'}</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <div id="chart"></div>
                <script>
                    var data = {request.data};
                    var layout = {{
                        title: '{request.title or 'Data Visualization'}',
                        xaxis: {{ title: '{request.x_axis or 'X Axis'}' }},
                        yaxis: {{ title: '{request.y_axis or 'Y Axis'}' }}
                    }};
                    Plotly.newPlot('chart', data, layout);
                </script>
            </body>
            </html>
            """
            return {
                "format": "html",
                "content": html_content
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {format}. Supported formats: json, html"
            )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export visualization: {str(e)}"
        ) 