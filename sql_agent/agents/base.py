"""Base agent class for SQL Agent."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from ..core.state import AgentState
from ..core.llm import LLMProvider
from ..utils.logging import get_logger


class BaseAgent(ABC):
    """Abstract base class for all agents in the SQL Agent system."""
    
    def __init__(self, name: str, llm_provider: LLMProvider):
        self.name = name
        self.llm = llm_provider
        self.logger = get_logger(f"agent.{name}")
    
    @abstractmethod
    async def process(self, state: AgentState) -> AgentState:
        """Process the current state and return updated state."""
        pass
    
    async def pre_process(self, state: AgentState) -> AgentState:
        """Pre-processing hook for agents."""
        self.logger.info(
            "agent_started",
            agent=self.name,
            session_id=state.session_id,
            query=state.query,
        )
        state.set_current_agent(self.name)
        return state
    
    async def post_process(self, state: AgentState) -> AgentState:
        """Post-processing hook for agents."""
        self.logger.info(
            "agent_completed",
            agent=self.name,
            session_id=state.session_id,
            has_errors=state.has_errors(),
            error_count=len(state.errors),
            has_fraud_analysis=state.fraud_analysis_result is not None,
        )

        # Log fraud detection results if present
        if state.fraud_analysis_result:
            self._log_fraud_detection(state)

        return state

    def _log_fraud_detection(self, state: AgentState) -> None:
        """Log fraud detection results for audit trail."""
        try:
            fraud_result = state.fraud_analysis_result
            if fraud_result and isinstance(fraud_result, dict):
                scenarios = fraud_result.get('scenarios', [])
                self.logger.info(
                    "fraud_detection_logged",
                    agent=self.name,
                    session_id=state.session_id,
                    scenarios_detected=len(scenarios),
                    fraud_categories=[s.get('category') for s in scenarios if s.get('category')],
                    highest_risk=max([s.get('risk_level', 'low') for s in scenarios], default='low'),
                )
        except Exception as e:
            self.logger.warning("Failed to log fraud detection", error=str(e))
    
    async def run(self, state: AgentState) -> AgentState:
        """Run the agent with pre and post processing."""
        try:
            state = await self.pre_process(state)
            state = await self.process(state)
            state = await self.post_process(state)
            return state
        except Exception as e:
            self.logger.error(
                "agent_error",
                agent=self.name,
                session_id=state.session_id,
                error=str(e),
                exc_info=True,
            )
            state.add_error(f"{self.name} agent error: {e}")
            return state
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "description": self.__doc__ or "No description available",
        } 