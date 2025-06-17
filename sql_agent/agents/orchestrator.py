"""Agent orchestrator for SQL Agent."""

import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from langgraph.graph import StateGraph, END
from .base import BaseAgent
from .router import RouterAgent
from .sql import SQLAgent
from .analysis import AnalysisAgent
from .viz import VisualizationAgent
from ..core.state import AgentState
from ..core.llm import LLMFactory
from ..utils.logging import get_logger


class AgentOrchestrator:
    """Orchestrates the workflow between all agents."""
    
    def __init__(self, llm_provider_name: Optional[str] = None):
        self.logger = get_logger("orchestrator")
        
        # Create LLM provider
        self.llm_provider = LLMFactory.create_provider(llm_provider_name)
        
        # Initialize agents
        self.router_agent = RouterAgent(self.llm_provider)
        self.sql_agent = SQLAgent(self.llm_provider)
        self.analysis_agent = AnalysisAgent(self.llm_provider)
        self.visualization_agent = VisualizationAgent(self.llm_provider)
        
        # Create workflow graph
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self._run_router)
        workflow.add_node("sql", self._run_sql)
        workflow.add_node("analysis", self._run_analysis)
        workflow.add_node("visualization", self._run_visualization)
        
        # Add edges
        workflow.add_edge("router", "sql")
        workflow.add_edge("sql", "analysis")
        workflow.add_edge("analysis", "visualization")
        workflow.add_edge("visualization", END)
        
        # Add conditional edges based on routing
        workflow.add_conditional_edges(
            "router",
            self._route_to_agent,
            {
                "sql": "sql",
                "analysis": "analysis", 
                "visualization": "visualization"
            }
        )
        
        return workflow.compile()
    
    async def _run_router(self, state: AgentState) -> AgentState:
        """Run the router agent."""
        self.logger.info("orchestrator_router_start", session_id=state.session_id)
        return await self.router_agent.run(state)
    
    async def _run_sql(self, state: AgentState) -> AgentState:
        """Run the SQL agent."""
        self.logger.info("orchestrator_sql_start", session_id=state.session_id)
        return await self.sql_agent.run(state)
    
    async def _run_analysis(self, state: AgentState) -> AgentState:
        """Run the analysis agent."""
        self.logger.info("orchestrator_analysis_start", session_id=state.session_id)
        return await self.analysis_agent.run(state)
    
    async def _run_visualization(self, state: AgentState) -> AgentState:
        """Run the visualization agent."""
        self.logger.info("orchestrator_visualization_start", session_id=state.session_id)
        return await self.visualization_agent.run(state)
    
    def _route_to_agent(self, state: AgentState) -> str:
        """Route to the appropriate agent based on router decision."""
        routing = state.metadata.get("routing", {})
        primary_agent = routing.get("primary_agent", "sql")
        
        self.logger.info(
            "orchestrator_routing",
            session_id=state.session_id,
            primary_agent=primary_agent,
            confidence=routing.get("confidence", 0.0)
        )
        
        return primary_agent
    
    async def process_query(self, query: str, database_name: Optional[str] = None) -> AgentState:
        """Process a natural language query through the agent workflow."""
        # Create initial state
        session_id = str(uuid.uuid4())
        state = AgentState(
            query=query,
            session_id=session_id,
            database_name=database_name,
            start_time=datetime.utcnow()
        )
        
        self.logger.info(
            "orchestrator_query_start",
            session_id=session_id,
            query=query,
            database_name=database_name
        )
        
        try:
            # Run the workflow
            final_state = await self.workflow.ainvoke(state)
            
            # Calculate processing time
            final_state.end_time = datetime.utcnow()
            if final_state.start_time:
                final_state.processing_time = (final_state.end_time - final_state.start_time).total_seconds()
            
            self.logger.info(
                "orchestrator_query_complete",
                session_id=session_id,
                processing_time=final_state.processing_time,
                has_errors=final_state.has_errors(),
                error_count=len(final_state.errors)
            )
            
            return final_state
            
        except Exception as e:
            self.logger.error(
                "orchestrator_query_failed",
                session_id=session_id,
                error=str(e),
                exc_info=True
            )
            
            # Add error to state
            state.add_error(f"Workflow execution failed: {e}")
            state.end_time = datetime.utcnow()
            if state.start_time:
                state.processing_time = (state.end_time - state.start_time).total_seconds()
            
            return state
    
    async def process_query_with_custom_flow(
        self, 
        query: str, 
        agent_sequence: List[str],
        database_name: Optional[str] = None
    ) -> AgentState:
        """Process a query with a custom agent sequence."""
        session_id = str(uuid.uuid4())
        state = AgentState(
            query=query,
            session_id=session_id,
            database_name=database_name,
            start_time=datetime.utcnow()
        )
        
        self.logger.info(
            "orchestrator_custom_flow_start",
            session_id=session_id,
            agent_sequence=agent_sequence
        )
        
        try:
            # Run agents in custom sequence
            for agent_name in agent_sequence:
                agent = self._get_agent_by_name(agent_name)
                if agent:
                    state = await agent.run(state)
                    
                    # Check for errors
                    if state.has_errors():
                        self.logger.warning(
                            "orchestrator_agent_error",
                            session_id=session_id,
                            agent=agent_name,
                            errors=state.errors
                        )
                        break
                else:
                    state.add_error(f"Unknown agent: {agent_name}")
                    break
            
            # Calculate processing time
            state.end_time = datetime.utcnow()
            if state.start_time:
                state.processing_time = (state.end_time - state.start_time).total_seconds()
            
            return state
            
        except Exception as e:
            self.logger.error(
                "orchestrator_custom_flow_failed",
                session_id=session_id,
                error=str(e),
                exc_info=True
            )
            
            state.add_error(f"Custom flow execution failed: {e}")
            state.end_time = datetime.utcnow()
            if state.start_time:
                state.processing_time = (state.end_time - state.start_time).total_seconds()
            
            return state
    
    def _get_agent_by_name(self, agent_name: str) -> Optional[BaseAgent]:
        """Get agent by name."""
        agents = {
            "router": self.router_agent,
            "sql": self.sql_agent,
            "analysis": self.analysis_agent,
            "visualization": self.visualization_agent
        }
        return agents.get(agent_name)
    
    def get_agent_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all agents."""
        return {
            "router": self.router_agent.get_agent_info(),
            "sql": self.sql_agent.get_agent_info(),
            "analysis": self.analysis_agent.get_agent_info(),
            "visualization": self.visualization_agent.get_agent_info()
        }
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the workflow."""
        return {
            "agents": list(self.get_agent_info().keys()),
            "default_flow": ["router", "sql", "analysis", "visualization"],
            "llm_provider": self.llm_provider.model_name
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on all agents."""
        health_status = {
            "orchestrator": "healthy",
            "agents": {},
            "llm_provider": "unknown"
        }
        
        # Check LLM provider
        try:
            # Simple test to check if LLM is working
            test_messages = [{"role": "user", "content": "Hello"}]
            await self.llm_provider.generate(test_messages)
            health_status["llm_provider"] = "healthy"
        except Exception as e:
            health_status["llm_provider"] = f"unhealthy: {str(e)}"
        
        # Check each agent
        for agent_name, agent in [
            ("router", self.router_agent),
            ("sql", self.sql_agent),
            ("analysis", self.analysis_agent),
            ("visualization", self.visualization_agent)
        ]:
            try:
                agent_info = agent.get_agent_info()
                health_status["agents"][agent_name] = "healthy"
            except Exception as e:
                health_status["agents"][agent_name] = f"unhealthy: {str(e)}"
        
        return health_status 