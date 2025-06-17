"""Tests for the multi-agent system."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from sql_agent.agents import (
    RouterAgent, 
    SQLAgent, 
    AnalysisAgent, 
    VisualizationAgent,
    AgentOrchestrator
)
from sql_agent.core.state import AgentState, QueryResult, AnalysisResult, VisualizationConfig
from sql_agent.core.llm import LLMProvider


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self, model_name: str = "test-model", temperature: float = 0.1):
        super().__init__(model_name, temperature)
        self.responses = {
            "intent_analysis": '{"intents": {"sql_generation": {"confidence": 0.8, "reasoning": "test"}}, "primary_intent": "sql_generation", "overall_confidence": 0.8}',
            "sql_generation": "SELECT * FROM customers LIMIT 10;",
            "insights": "Sample insight 1\nSample insight 2",
            "chart_type": "bar",
            "chart_title": "Test Chart"
        }
    
    def get_llm(self):
        return Mock()
    
    async def generate(self, messages):
        # Return appropriate response based on message content
        content = str(messages[-1].content).lower()
        if "intent" in content:
            return self.responses["intent_analysis"]
        elif "sql" in content:
            return self.responses["sql_generation"]
        elif "insight" in content:
            return self.responses["insights"]
        elif "chart type" in content:
            return self.responses["chart_type"]
        elif "title" in content:
            return self.responses["chart_title"]
        else:
            return "Test response"
    
    async def generate_with_tools(self, messages, tools):
        return {"content": "Test response", "tool_calls": []}


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def sample_query_result():
    """Create a sample query result for testing."""
    return QueryResult(
        data=[
            {"id": 1, "name": "John", "revenue": 1000.0},
            {"id": 2, "name": "Jane", "revenue": 2000.0},
            {"id": 3, "name": "Bob", "revenue": 1500.0}
        ],
        columns=["id", "name", "revenue"],
        row_count=3,
        execution_time=0.5,
        sql_query="SELECT * FROM customers LIMIT 3;"
    )


@pytest.fixture
def sample_state():
    """Create a sample agent state for testing."""
    return AgentState(
        query="Show me the top customers by revenue",
        session_id="test-session-123"
    )


class TestRouterAgent:
    """Test the Router Agent."""
    
    @pytest.mark.asyncio
    async def test_router_agent_initialization(self, mock_llm_provider):
        """Test router agent initialization."""
        agent = RouterAgent(mock_llm_provider)
        assert agent.name == "router"
        assert agent.llm == mock_llm_provider
        assert "sql_generation" in agent.intent_patterns
    
    @pytest.mark.asyncio
    async def test_router_agent_process(self, mock_llm_provider, sample_state):
        """Test router agent processing."""
        agent = RouterAgent(mock_llm_provider)
        
        result_state = await agent.process(sample_state)
        
        assert result_state.metadata["routing"] is not None
        assert result_state.metadata["intent_analysis"] is not None
        assert result_state.metadata["next_agent"] is not None
    
    @pytest.mark.asyncio
    async def test_router_agent_fallback_analysis(self, mock_llm_provider):
        """Test router agent fallback intent analysis."""
        agent = RouterAgent(mock_llm_provider)
        
        # Test with analysis keywords
        query = "Analyze the customer data trends"
        result = agent._fallback_intent_analysis(query)
        
        assert "analysis" in result["intents"]
        assert result["primary_intent"] in ["sql_generation", "analysis", "visualization"]
    
    def test_router_agent_routing_rules(self, mock_llm_provider):
        """Test router agent routing rules."""
        agent = RouterAgent(mock_llm_provider)
        rules = agent.get_routing_rules()
        
        assert "intent_patterns" in rules
        assert "agent_mapping" in rules
        assert "sql_generation" in rules["agent_mapping"]


class TestSQLAgent:
    """Test the SQL Agent."""
    
    @pytest.mark.asyncio
    async def test_sql_agent_initialization(self, mock_llm_provider):
        """Test SQL agent initialization."""
        agent = SQLAgent(mock_llm_provider)
        assert agent.name == "sql"
        assert agent.llm == mock_llm_provider
        assert "select" in agent.sql_templates
    
    @pytest.mark.asyncio
    async def test_sql_agent_process(self, mock_llm_provider, sample_state):
        """Test SQL agent processing."""
        agent = SQLAgent(mock_llm_provider)
        
        # Mock database manager
        with patch('sql_agent.agents.sql.db_manager') as mock_db:
            mock_db.get_schema_info.return_value = {
                "customers": {
                    "columns": [{"column_name": "id"}, {"column_name": "name"}, {"column_name": "revenue"}]
                }
            }
            mock_db.validate_query.return_value = (True, None)
            mock_db.execute_query.return_value = QueryResult(
                data=[{"id": 1, "name": "Test"}],
                columns=["id", "name"],
                row_count=1,
                execution_time=0.1,
                sql_query="SELECT * FROM customers;"
            )
            
            result_state = await agent.process(sample_state)
            
            assert result_state.generated_sql is not None
            assert result_state.query_result is not None
            assert result_state.query_result.error is None
    
    def test_sql_agent_clean_sql_response(self, mock_llm_provider):
        """Test SQL response cleaning."""
        agent = SQLAgent(mock_llm_provider)
        
        # Test with markdown code blocks
        response = "```sql\nSELECT * FROM customers;\n```"
        cleaned = agent._clean_sql_response(response)
        assert cleaned == "SELECT * FROM customers;"
        
        # Test without semicolon
        response = "SELECT * FROM customers"
        cleaned = agent._clean_sql_response(response)
        assert cleaned == "SELECT * FROM customers;"
    
    def test_sql_agent_fallback_generation(self, mock_llm_provider):
        """Test SQL fallback generation."""
        agent = SQLAgent(mock_llm_provider)
        
        # Test count query
        query = "How many customers do we have?"
        sql = agent._fallback_sql_generation(query, [])
        assert "COUNT" in sql.upper()
        
        # Test top query
        query = "Show me the top 5 customers"
        sql = agent._fallback_sql_generation(query, [])
        assert "LIMIT" in sql.upper()


class TestAnalysisAgent:
    """Test the Analysis Agent."""
    
    @pytest.mark.asyncio
    async def test_analysis_agent_initialization(self, mock_llm_provider):
        """Test analysis agent initialization."""
        agent = AnalysisAgent(mock_llm_provider)
        assert agent.name == "analysis"
        assert agent.llm == mock_llm_provider
        assert "statistical" in agent.analysis_types
    
    @pytest.mark.asyncio
    async def test_analysis_agent_process(self, mock_llm_provider, sample_state, sample_query_result):
        """Test analysis agent processing."""
        agent = AnalysisAgent(mock_llm_provider)
        sample_state.query_result = sample_query_result
        
        result_state = await agent.process(sample_state)
        
        assert result_state.analysis_result is not None
        assert isinstance(result_state.analysis_result, AnalysisResult)
        assert len(result_state.analysis_result.insights) > 0
    
    def test_analysis_agent_identify_numeric_columns(self, mock_llm_provider, sample_query_result):
        """Test numeric column identification."""
        agent = AnalysisAgent(mock_llm_provider)
        
        numeric_columns = agent._identify_numeric_columns(sample_query_result)
        assert "id" in numeric_columns
        assert "revenue" in numeric_columns
        assert "name" not in numeric_columns
    
    def test_analysis_agent_calculate_data_quality_score(self, mock_llm_provider, sample_query_result):
        """Test data quality score calculation."""
        agent = AnalysisAgent(mock_llm_provider)
        
        score = agent._calculate_data_quality_score(sample_query_result)
        assert 0.0 <= score <= 1.0
    
    @pytest.mark.asyncio
    async def test_analysis_agent_detect_anomalies(self, mock_llm_provider, sample_query_result):
        """Test anomaly detection."""
        agent = AnalysisAgent(mock_llm_provider)
        
        anomalies = await agent.detect_anomalies(sample_query_result)
        assert isinstance(anomalies, list)


class TestVisualizationAgent:
    """Test the Visualization Agent."""
    
    @pytest.mark.asyncio
    async def test_visualization_agent_initialization(self, mock_llm_provider):
        """Test visualization agent initialization."""
        agent = VisualizationAgent(mock_llm_provider)
        assert agent.name == "visualization"
        assert agent.llm == mock_llm_provider
        assert "bar" in agent.chart_types
    
    @pytest.mark.asyncio
    async def test_visualization_agent_process(self, mock_llm_provider, sample_state, sample_query_result):
        """Test visualization agent processing."""
        agent = VisualizationAgent(mock_llm_provider)
        sample_state.query_result = sample_query_result
        
        result_state = await agent.process(sample_state)
        
        assert result_state.visualization_config is not None
        assert isinstance(result_state.visualization_config, VisualizationConfig)
        assert result_state.visualization_config.chart_type in agent.chart_types
    
    def test_visualization_agent_analyze_data_characteristics(self, mock_llm_provider, sample_query_result):
        """Test data characteristics analysis."""
        agent = VisualizationAgent(mock_llm_provider)
        
        characteristics = agent._analyze_data_characteristics(sample_query_result)
        assert "row_count" in characteristics
        assert "numeric_columns" in characteristics
        assert "categorical_columns" in characteristics
    
    def test_visualization_agent_is_numeric_column(self, mock_llm_provider):
        """Test numeric column detection."""
        agent = VisualizationAgent(mock_llm_provider)
        
        # Test numeric values
        numeric_values = [1, 2, 3, 4, 5]
        assert agent._is_numeric_column(numeric_values) is True
        
        # Test mixed values
        mixed_values = [1, "text", 3, 4, 5]
        assert agent._is_numeric_column(mixed_values) is False
    
    def test_visualization_agent_fallback_chart_type(self, mock_llm_provider):
        """Test fallback chart type selection."""
        agent = VisualizationAgent(mock_llm_provider)
        
        # Test with time and numeric data
        characteristics = {
            "time_columns": ["date"],
            "numeric_columns": ["value"],
            "categorical_columns": []
        }
        chart_type = agent._fallback_chart_type(characteristics)
        assert chart_type == "line"
    
    @pytest.mark.asyncio
    async def test_visualization_agent_generate_chart_data(self, mock_llm_provider, sample_query_result):
        """Test chart data generation."""
        agent = VisualizationAgent(mock_llm_provider)
        
        viz_config = VisualizationConfig(
            chart_type="bar",
            x_axis="name",
            y_axis="revenue",
            title="Test Chart"
        )
        
        chart_data = await agent.generate_chart_data(sample_query_result, viz_config)
        assert "type" in chart_data
        assert "data" in chart_data
        assert "layout" in chart_data


class TestAgentOrchestrator:
    """Test the Agent Orchestrator."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, mock_llm_provider):
        """Test orchestrator initialization."""
        with patch('sql_agent.agents.orchestrator.LLMFactory.create_provider', return_value=mock_llm_provider):
            orchestrator = AgentOrchestrator()
            
            assert orchestrator.router_agent is not None
            assert orchestrator.sql_agent is not None
            assert orchestrator.analysis_agent is not None
            assert orchestrator.visualization_agent is not None
    
    @pytest.mark.asyncio
    async def test_orchestrator_process_query(self, mock_llm_provider):
        """Test orchestrator query processing."""
        with patch('sql_agent.agents.orchestrator.LLMFactory.create_provider', return_value=mock_llm_provider):
            orchestrator = AgentOrchestrator()
            
            # Mock the workflow to avoid complex LangGraph setup
            with patch.object(orchestrator, 'workflow') as mock_workflow:
                mock_workflow.ainvoke.return_value = AgentState(
                    query="test query",
                    session_id="test-session",
                    generated_sql="SELECT * FROM test;",
                    query_result=QueryResult(data=[], columns=[], row_count=0)
                )
                
                result = await orchestrator.process_query("Show me customers")
                
                assert result.session_id is not None
                assert result.query == "Show me customers"
    
    def test_orchestrator_get_agent_info(self, mock_llm_provider):
        """Test orchestrator agent info."""
        with patch('sql_agent.agents.orchestrator.LLMFactory.create_provider', return_value=mock_llm_provider):
            orchestrator = AgentOrchestrator()
            
            agent_info = orchestrator.get_agent_info()
            assert "router" in agent_info
            assert "sql" in agent_info
            assert "analysis" in agent_info
            assert "visualization" in agent_info
    
    def test_orchestrator_get_workflow_info(self, mock_llm_provider):
        """Test orchestrator workflow info."""
        with patch('sql_agent.agents.orchestrator.LLMFactory.create_provider', return_value=mock_llm_provider):
            orchestrator = AgentOrchestrator()
            
            workflow_info = orchestrator.get_workflow_info()
            assert "agents" in workflow_info
            assert "default_flow" in workflow_info
            assert "llm_provider" in workflow_info
    
    @pytest.mark.asyncio
    async def test_orchestrator_health_check(self, mock_llm_provider):
        """Test orchestrator health check."""
        with patch('sql_agent.agents.orchestrator.LLMFactory.create_provider', return_value=mock_llm_provider):
            orchestrator = AgentOrchestrator()
            
            health_status = await orchestrator.health_check()
            assert "orchestrator" in health_status
            assert "agents" in health_status
            assert "llm_provider" in health_status 