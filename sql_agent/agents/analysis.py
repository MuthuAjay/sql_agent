"""Analysis Agent for SQL Agent."""

import statistics
from typing import Dict, List, Any, Optional
from langchain.schema import HumanMessage, SystemMessage
from .base import BaseAgent
from ..core.state import AgentState, AnalysisResult, QueryResult
from ..utils.logging import log_agent_decision


class AnalysisAgent(BaseAgent):
    """Analysis agent that provides insights from query results."""
    
    def __init__(self, llm_provider):
        super().__init__("analysis", llm_provider)
        
        # Analysis types
        self.analysis_types = {
            "statistical": ["mean", "median", "mode", "variance", "std_dev"],
            "trend": ["growth", "decline", "pattern", "seasonality"],
            "comparison": ["higher", "lower", "difference", "ratio"],
            "outlier": ["anomaly", "unusual", "extreme", "outlier"],
            "distribution": ["spread", "concentration", "distribution"]
        }
    
    async def process(self, state: AgentState) -> AgentState:
        """Process query results and generate analysis."""
        self.logger.info("analysis_agent_processing", query=state.query)
        
        if not state.query_result or state.query_result.error:
            state.add_error("No valid query results to analyze")
            return state
        
        try:
            # Perform statistical analysis
            statistical_insights = await self._perform_statistical_analysis(state.query_result)
            
            # Generate LLM-based insights
            llm_insights = await self._generate_llm_insights(state.query, state.query_result)
            
            # Calculate data quality score
            data_quality_score = self._calculate_data_quality_score(state.query_result)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(state.query_result, llm_insights)
            
            # Create analysis result
            analysis_result = AnalysisResult(
                insights=statistical_insights + llm_insights,
                recommendations=recommendations,
                data_quality_score=data_quality_score,
                performance_metrics={
                    "row_count": state.query_result.row_count,
                    "execution_time": state.query_result.execution_time,
                    "column_count": len(state.query_result.columns)
                }
            )
            
            state.analysis_result = analysis_result
            
            # Log the analysis
            log_agent_decision(
                self.logger,
                agent=self.name,
                decision="analysis_complete",
                reasoning=f"Generated {len(analysis_result.insights)} insights and {len(analysis_result.recommendations)} recommendations",
                metadata={
                    "data_quality_score": data_quality_score,
                    "insight_count": len(analysis_result.insights),
                    "recommendation_count": len(analysis_result.recommendations)
                }
            )
            
            self.logger.info(
                "analysis_complete",
                insight_count=len(analysis_result.insights),
                recommendation_count=len(analysis_result.recommendations),
                data_quality_score=data_quality_score
            )
            
        except Exception as e:
            self.logger.error("analysis_agent_error", error=str(e), exc_info=True)
            state.add_error(f"Analysis failed: {e}")
        
        return state
    
    async def _perform_statistical_analysis(self, query_result: QueryResult) -> List[str]:
        """Perform statistical analysis on the query results."""
        insights = []
        
        if not query_result.data or not query_result.columns:
            return ["No data available for statistical analysis"]
        
        try:
            # Analyze numeric columns
            numeric_columns = self._identify_numeric_columns(query_result)
            
            for column in numeric_columns:
                values = [row[column] for row in query_result.data if row[column] is not None]
                
                if len(values) > 1:
                    # Basic statistics
                    mean_val = statistics.mean(values)
                    median_val = statistics.median(values)
                    min_val = min(values)
                    max_val = max(values)
                    
                    insights.append(f"Column '{column}': Mean={mean_val:.2f}, Median={median_val:.2f}, Range=[{min_val:.2f}, {max_val:.2f}]")
                    
                    # Check for outliers
                    if len(values) > 3:
                        q1, q3 = statistics.quantiles(values, n=4)[0], statistics.quantiles(values, n=4)[2]
                        iqr = q3 - q1
                        outliers = [v for v in values if v < q1 - 1.5 * iqr or v > q3 + 1.5 * iqr]
                        
                        if outliers:
                            insights.append(f"Column '{column}' has {len(outliers)} potential outliers")
            
            # Analyze categorical columns
            categorical_columns = self._identify_categorical_columns(query_result)
            
            for column in categorical_columns:
                values = [row[column] for row in query_result.data if row[column] is not None]
                
                if values:
                    unique_values = set(values)
                    most_common = max(set(values), key=values.count)
                    
                    insights.append(f"Column '{column}': {len(unique_values)} unique values, most common: '{most_common}'")
            
            # Overall data insights
            insights.append(f"Dataset contains {len(query_result.data)} rows and {len(query_result.columns)} columns")
            
        except Exception as e:
            self.logger.error("statistical_analysis_failed", error=str(e))
            insights.append("Statistical analysis encountered an error")
        
        return insights
    
    def _identify_numeric_columns(self, query_result: QueryResult) -> List[str]:
        """Identify numeric columns in the query results."""
        numeric_columns = []
        
        for column in query_result.columns:
            try:
                # Check if column contains numeric data
                sample_values = [row[column] for row in query_result.data[:10] if row[column] is not None]
                
                if sample_values:
                    # Try to convert to float
                    float_values = [float(v) for v in sample_values if str(v).replace('.', '').replace('-', '').isdigit()]
                    
                    if len(float_values) > len(sample_values) * 0.7:  # 70% are numeric
                        numeric_columns.append(column)
            except (ValueError, TypeError):
                continue
        
        return numeric_columns
    
    def _identify_categorical_columns(self, query_result: QueryResult) -> List[str]:
        """Identify categorical columns in the query results."""
        numeric_columns = set(self._identify_numeric_columns(query_result))
        categorical_columns = [col for col in query_result.columns if col not in numeric_columns]
        return categorical_columns
    
    async def _generate_llm_insights(self, query: str, query_result: QueryResult) -> List[str]:
        """Generate insights using LLM."""
        # Prepare data summary for LLM
        data_summary = self._prepare_data_summary(query_result)
        
        system_prompt = """You are a data analyst expert. Analyze the given query results and provide business insights.

Focus on:
1. Key patterns and trends in the data
2. Business implications of the findings
3. Interesting observations about the data
4. Potential areas for further investigation

Provide 3-5 concise insights. Each insight should be actionable and business-relevant."""

        user_prompt = f"""Query: {query}

Data Summary:
{data_summary}

Generate insights about this data."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = await self.llm.generate(messages)
            
            # Parse insights from response
            insights = self._parse_insights_from_response(response)
            
            return insights
            
        except Exception as e:
            self.logger.error("llm_insights_failed", error=str(e))
            return ["Unable to generate LLM-based insights"]
    
    def _prepare_data_summary(self, query_result: QueryResult) -> str:
        """Prepare a summary of the query results for LLM analysis."""
        summary_parts = [
            f"Rows: {query_result.row_count}",
            f"Columns: {', '.join(query_result.columns)}"
        ]
        
        # Add sample data
        if query_result.data:
            summary_parts.append("Sample data:")
            for i, row in enumerate(query_result.data[:3]):  # First 3 rows
                summary_parts.append(f"  Row {i+1}: {dict(row)}")
        
        return "\n".join(summary_parts)
    
    def _parse_insights_from_response(self, response: str) -> List[str]:
        """Parse insights from LLM response."""
        # Simple parsing - split by lines and clean up
        lines = response.strip().split('\n')
        insights = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and len(line) > 10:
                # Remove bullet points and numbering
                line = line.lstrip('•-*1234567890. ')
                if line:
                    insights.append(line)
        
        return insights[:5]  # Limit to 5 insights
    
    def _calculate_data_quality_score(self, query_result: QueryResult) -> float:
        """Calculate a data quality score (0-1)."""
        if not query_result.data:
            return 0.0
        
        score = 1.0
        
        # Penalize for missing data
        total_cells = len(query_result.data) * len(query_result.columns)
        missing_cells = sum(
            1 for row in query_result.data 
            for value in row.values() 
            if value is None or value == ""
        )
        
        if total_cells > 0:
            missing_ratio = missing_cells / total_cells
            score -= missing_ratio * 0.3  # Missing data penalty
        
        # Penalize for very small datasets
        if len(query_result.data) < 5:
            score -= 0.2
        
        # Bonus for good execution time
        if query_result.execution_time < 1.0:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    async def _generate_recommendations(self, query_result: QueryResult, insights: List[str]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Data quality recommendations
        if query_result.row_count < 10:
            recommendations.append("Consider collecting more data for more reliable analysis")
        
        if len(query_result.columns) > 10:
            recommendations.append("Consider focusing on fewer columns for clearer insights")
        
        # Performance recommendations
        if query_result.execution_time > 5.0:
            recommendations.append("Query execution time is high - consider optimizing the query or adding indexes")
        
        # Business recommendations based on insights
        if insights:
            # Generate follow-up recommendations
            recommendations.append("Consider running follow-up queries to explore the identified patterns")
            recommendations.append("Share these insights with stakeholders for business decision-making")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def get_analysis_types(self) -> Dict[str, List[str]]:
        """Get available analysis types."""
        return self.analysis_types.copy()
    
    async def analyze_trends(self, query_result: QueryResult, time_column: str) -> List[str]:
        """Analyze trends in time-series data."""
        if not query_result.data or time_column not in query_result.columns:
            return ["No time-series data available for trend analysis"]
        
        # This would implement trend analysis logic
        # For now, return a placeholder
        return ["Trend analysis requires time-series data with proper date formatting"]
    
    async def detect_anomalies(self, query_result: QueryResult) -> List[str]:
        """Detect anomalies in the data."""
        if not query_result.data:
            return ["No data available for anomaly detection"]
        
        anomalies = []
        
        # Simple anomaly detection for numeric columns
        numeric_columns = self._identify_numeric_columns(query_result)
        
        for column in numeric_columns:
            values = [row[column] for row in query_result.data if row[column] is not None]
            
            if len(values) > 3:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0
                
                if std_val > 0:
                    # Detect values more than 2 standard deviations from mean
                    for i, value in enumerate(values):
                        if abs(value - mean_val) > 2 * std_val:
                            anomalies.append(f"Anomaly in row {i+1}, column '{column}': {value} (mean: {mean_val:.2f})")
        
        return anomalies if anomalies else ["No significant anomalies detected"] 