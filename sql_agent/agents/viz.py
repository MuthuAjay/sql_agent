"""Visualization Agent for SQL Agent."""

import json
from typing import Dict, List, Any, Optional
from langchain.schema import HumanMessage, SystemMessage
from .base import BaseAgent
from ..core.state import AgentState, VisualizationConfig, QueryResult
from ..utils.logging import log_agent_decision


class VisualizationAgent(BaseAgent):
    """Visualization agent that creates charts and visualizations."""
    
    def __init__(self, llm_provider):
        super().__init__("visualization", llm_provider)
        
        # Supported chart types
        self.chart_types = {
            "bar": {
                "description": "Bar chart for categorical data comparison",
                "requirements": ["categorical_x", "numeric_y"],
                "examples": ["customer revenue by category", "sales by region"]
            },
            "line": {
                "description": "Line chart for time series or trends",
                "requirements": ["numeric_x", "numeric_y"],
                "examples": ["revenue over time", "growth trends"]
            },
            "pie": {
                "description": "Pie chart for proportions and percentages",
                "requirements": ["categorical_data"],
                "examples": ["market share", "category distribution"]
            },
            "scatter": {
                "description": "Scatter plot for correlation analysis",
                "requirements": ["numeric_x", "numeric_y"],
                "examples": ["correlation between variables", "data distribution"]
            },
            "histogram": {
                "description": "Histogram for data distribution",
                "requirements": ["numeric_data"],
                "examples": ["data distribution", "frequency analysis"]
            }
        }
    
    async def process(self, state: AgentState) -> AgentState:
        """Process query results and generate visualization configuration."""
        self.logger.info("visualization_agent_processing", query=state.query)
        
        if not state.query_result or state.query_result.error:
            state.add_error("No valid query results to visualize")
            return state
        
        try:
            # Analyze data to determine best chart type
            chart_type = await self._determine_chart_type(state.query, state.query_result)
            
            # Generate visualization configuration
            viz_config = await self._generate_visualization_config(
                state.query, 
                state.query_result, 
                chart_type
            )
            
            # Update state
            state.visualization_config = viz_config
            
            # Log the visualization decision
            log_agent_decision(
                self.logger,
                agent=self.name,
                decision="visualization_configured",
                reasoning=f"Selected {chart_type} chart based on data characteristics",
                metadata={
                    "chart_type": chart_type,
                    "x_axis": viz_config.x_axis,
                    "y_axis": viz_config.y_axis,
                    "title": viz_config.title
                }
            )
            
            self.logger.info(
                "visualization_configured",
                chart_type=chart_type,
                x_axis=viz_config.x_axis,
                y_axis=viz_config.y_axis
            )
            
        except Exception as e:
            self.logger.error("visualization_agent_error", error=str(e), exc_info=True)
            state.add_error(f"Visualization configuration failed: {e}")
        
        return state
    
    async def _determine_chart_type(self, query: str, query_result: QueryResult) -> str:
        """Determine the best chart type for the data."""
        # Analyze data characteristics
        data_characteristics = self._analyze_data_characteristics(query_result)
        
        # Use LLM to determine chart type
        system_prompt = f"""You are a data visualization expert. Determine the best chart type for the given data and query.

Available chart types:
{json.dumps(self.chart_types, indent=2)}

Data characteristics:
{json.dumps(data_characteristics, indent=2)}

Consider:
1. The type of data (categorical, numeric, time-series)
2. The number of data points
3. The query intent
4. The best way to represent the relationships in the data

Respond with only the chart type name (e.g., "bar", "line", "pie", "scatter", "histogram")."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {query}\n\nDetermine the best chart type.")
        ]
        
        try:
            response = await self.llm.generate(messages)
            chart_type = response.strip().lower()
            
            # Validate chart type
            if chart_type in self.chart_types:
                return chart_type
            else:
                # Fallback to default based on data characteristics
                return self._fallback_chart_type(data_characteristics)
                
        except Exception as e:
            self.logger.error("chart_type_determination_failed", error=str(e))
            return self._fallback_chart_type(data_characteristics)
    
    def _analyze_data_characteristics(self, query_result: QueryResult) -> Dict[str, Any]:
        """Analyze the characteristics of the query results."""
        if not query_result.data or not query_result.columns:
            return {"error": "No data available"}
        
        characteristics = {
            "row_count": len(query_result.data),
            "column_count": len(query_result.columns),
            "columns": {},
            "data_types": {},
            "unique_values": {},
            "numeric_columns": [],
            "categorical_columns": [],
            "time_columns": []
        }
        
        for column in query_result.columns:
            values = [row[column] for row in query_result.data if row[column] is not None]
            
            if values:
                # Determine data type
                if self._is_numeric_column(values):
                    characteristics["numeric_columns"].append(column)
                    characteristics["data_types"][column] = "numeric"
                elif self._is_time_column(values):
                    characteristics["time_columns"].append(column)
                    characteristics["data_types"][column] = "time"
                else:
                    characteristics["categorical_columns"].append(column)
                    characteristics["data_types"][column] = "categorical"
                
                # Count unique values
                characteristics["unique_values"][column] = len(set(values))
                
                # Column details
                characteristics["columns"][column] = {
                    "data_type": characteristics["data_types"][column],
                    "unique_count": characteristics["unique_values"][column],
                    "sample_values": values[:5]  # First 5 values
                }
        
        return characteristics
    
    def _is_numeric_column(self, values: List[Any]) -> bool:
        """Check if a column contains numeric data."""
        try:
            numeric_count = 0
            for value in values[:10]:  # Check first 10 values
                if isinstance(value, (int, float)) or str(value).replace('.', '').replace('-', '').isdigit():
                    numeric_count += 1
            
            return numeric_count > len(values[:10]) * 0.7  # 70% are numeric
        except:
            return False
    
    def _is_time_column(self, values: List[Any]) -> bool:
        """Check if a column contains time/date data."""
        # Simple check for common date patterns
        time_patterns = ['-', '/', ':', 'T', 'Z']
        for value in values[:5]:
            if isinstance(value, str):
                if any(pattern in value for pattern in time_patterns):
                    return True
        return False
    
    def _fallback_chart_type(self, data_characteristics: Dict[str, Any]) -> str:
        """Fallback chart type selection based on data characteristics."""
        numeric_count = len(data_characteristics.get("numeric_columns", []))
        categorical_count = len(data_characteristics.get("categorical_columns", []))
        time_count = len(data_characteristics.get("time_columns", []))
        
        if time_count > 0 and numeric_count > 0:
            return "line"
        elif categorical_count > 0 and numeric_count > 0:
            return "bar"
        elif categorical_count > 0:
            return "pie"
        elif numeric_count > 1:
            return "scatter"
        else:
            return "bar"  # Default fallback
    
    async def _generate_visualization_config(
        self, 
        query: str, 
        query_result: QueryResult, 
        chart_type: str
    ) -> VisualizationConfig:
        """Generate visualization configuration."""
        # Analyze data to determine axes
        x_axis, y_axis = self._determine_axes(query_result, chart_type)
        
        # Generate title
        title = await self._generate_chart_title(query, chart_type)
        
        # Generate color scheme
        color_scheme = self._determine_color_scheme(chart_type)
        
        # Create configuration
        config = VisualizationConfig(
            chart_type=chart_type,
            x_axis=x_axis,
            y_axis=y_axis,
            title=title,
            color_scheme=color_scheme,
            config=self._get_chart_specific_config(chart_type, query_result)
        )
        
        return config
    
    def _determine_axes(self, query_result: QueryResult, chart_type: str) -> tuple[Optional[str], Optional[str]]:
        """Determine X and Y axes for the chart."""
        if not query_result.columns:
            return None, None
        
        numeric_columns = []
        categorical_columns = []
        
        for column in query_result.columns:
            values = [row[column] for row in query_result.data if row[column] is not None]
            if self._is_numeric_column(values):
                numeric_columns.append(column)
            else:
                categorical_columns.append(column)
        
        if chart_type == "bar":
            # Bar chart: categorical X, numeric Y
            x_axis = categorical_columns[0] if categorical_columns else None
            y_axis = numeric_columns[0] if numeric_columns else None
            
        elif chart_type == "line":
            # Line chart: numeric X, numeric Y
            if len(numeric_columns) >= 2:
                x_axis, y_axis = numeric_columns[0], numeric_columns[1]
            else:
                x_axis, y_axis = None, numeric_columns[0] if numeric_columns else None
                
        elif chart_type == "pie":
            # Pie chart: categorical data
            x_axis = categorical_columns[0] if categorical_columns else None
            y_axis = None
            
        elif chart_type == "scatter":
            # Scatter plot: numeric X, numeric Y
            if len(numeric_columns) >= 2:
                x_axis, y_axis = numeric_columns[0], numeric_columns[1]
            else:
                x_axis, y_axis = None, None
                
        else:  # histogram
            # Histogram: numeric data
            x_axis = numeric_columns[0] if numeric_columns else None
            y_axis = None
        
        return x_axis, y_axis
    
    async def _generate_chart_title(self, query: str, chart_type: str) -> str:
        """Generate a title for the chart."""
        system_prompt = """You are a data visualization expert. Generate a clear, concise title for a chart based on the query and chart type.

The title should:
1. Be descriptive but concise (under 60 characters)
2. Clearly indicate what the chart shows
3. Be professional and business-friendly
4. Include the chart type if relevant

Respond with only the title, no quotes or formatting."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {query}\nChart type: {chart_type}\n\nGenerate a title.")
        ]
        
        try:
            response = await self.llm.generate(messages)
            title = response.strip().strip('"').strip("'")
            
            # Fallback if title is too long or empty
            if len(title) > 60 or not title:
                return f"{chart_type.title()} Chart - {query[:40]}..."
            
            return title
            
        except Exception as e:
            self.logger.error("title_generation_failed", error=str(e))
            return f"{chart_type.title()} Chart"
    
    def _determine_color_scheme(self, chart_type: str) -> str:
        """Determine appropriate color scheme for the chart."""
        color_schemes = {
            "bar": "viridis",
            "line": "plasma", 
            "pie": "Set3",
            "scatter": "coolwarm",
            "histogram": "Blues"
        }
        
        return color_schemes.get(chart_type, "viridis")
    
    def _get_chart_specific_config(self, chart_type: str, query_result: QueryResult) -> Dict[str, Any]:
        """Get chart-specific configuration."""
        base_config: Dict[str, Any] = {
            "responsive": True,
            "displayModeBar": True,
            "displaylogo": False
        }
        
        if chart_type == "bar":
            base_config["orientation"] = "v"
            base_config["barmode"] = "group"
        elif chart_type == "line":
            base_config["line"] = {"shape": "linear"}
        elif chart_type == "pie":
            base_config["hole"] = 0.0
        elif chart_type == "scatter":
            base_config["mode"] = "markers"
        elif chart_type == "histogram":
            base_config["nbinsx"] = min(20, len(query_result.data) // 2) if query_result.data else 10
        
        return base_config
    
    def get_supported_chart_types(self) -> Dict[str, Dict[str, Any]]:
        """Get supported chart types and their requirements."""
        return self.chart_types.copy()
    
    async def generate_chart_data(self, query_result: QueryResult, viz_config: VisualizationConfig) -> Dict[str, Any]:
        """Generate chart data in a format suitable for plotting libraries."""
        if not query_result.data or not viz_config.x_axis:
            return {"error": "No data available for chart generation"}
        
        chart_data = {
            "type": viz_config.chart_type,
            "data": [],
            "layout": {
                "title": viz_config.title or "Chart",
                "xaxis": {"title": viz_config.x_axis},
                "yaxis": {"title": viz_config.y_axis} if viz_config.y_axis else {},
                "colorway": [viz_config.color_scheme] if viz_config.color_scheme else []
            }
        }
        
        # Extract data for the chart
        x_values = [row[viz_config.x_axis] for row in query_result.data if row[viz_config.x_axis] is not None]
        
        if viz_config.y_axis:
            y_values = [row[viz_config.y_axis] for row in query_result.data if row[viz_config.y_axis] is not None]
        else:
            y_values = None
        
        # Create trace data based on chart type
        if viz_config.chart_type == "bar":
            chart_data["data"] = [{
                "x": x_values,
                "y": y_values,
                "type": "bar",
                "name": viz_config.y_axis or "Count"
            }]
            
        elif viz_config.chart_type == "line":
            chart_data["data"] = [{
                "x": x_values,
                "y": y_values,
                "type": "scatter",
                "mode": "lines+markers",
                "name": viz_config.y_axis or "Value"
            }]
            
        elif viz_config.chart_type == "pie":
            # Count occurrences for pie chart
            from collections import Counter
            value_counts = Counter(x_values)
            chart_data["data"] = [{
                "labels": list(value_counts.keys()),
                "values": list(value_counts.values()),
                "type": "pie",
                "name": viz_config.x_axis
            }]
            
        elif viz_config.chart_type == "scatter":
            chart_data["data"] = [{
                "x": x_values,
                "y": y_values,
                "type": "scatter",
                "mode": "markers",
                "name": f"{viz_config.x_axis} vs {viz_config.y_axis}"
            }]
            
        elif viz_config.chart_type == "histogram":
            chart_data["data"] = [{
                "x": x_values,
                "type": "histogram",
                "name": viz_config.x_axis
            }]
        
        return chart_data 