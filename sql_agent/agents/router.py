"""Router Agent for SQL Agent."""

from typing import Dict, List, Any, Optional
from langchain.schema import HumanMessage, SystemMessage
from .base import BaseAgent
from ..core.state import AgentState, SchemaContext
from ..utils.logging import log_agent_decision


class RouterAgent(BaseAgent):
    """Router agent that determines which agent should handle the query."""
    
    def __init__(self, llm_provider):
        super().__init__("router", llm_provider)
        
        # Define routing patterns and intents
        self.intent_patterns = {
            "sql_generation": [
                "show me", "get", "find", "select", "query", "what is", "how many",
                "list", "display", "retrieve", "fetch", "extract", "search"
            ],
            "analysis": [
                "analyze", "compare", "trend", "pattern", "insight", "summary",
                "statistics", "average", "total", "percentage", "growth", "performance"
            ],
            "visualization": [
                "chart", "graph", "plot", "visualize", "bar chart", "line chart",
                "pie chart", "scatter plot", "dashboard", "report", "visual"
            ]
        }
    
    async def process(self, state: AgentState) -> AgentState:
        """Process the query and determine routing."""
        self.logger.info("router_processing", query=state.query)
        
        # Analyze query intent
        intent_analysis = await self._analyze_intent(state.query)
        
        # Determine routing decision
        routing_decision = await self._determine_routing(state.query, intent_analysis)
        
        # Log the decision
        log_agent_decision(
            self.logger,
            agent=self.name,
            decision=routing_decision["primary_agent"],
            reasoning=routing_decision["reasoning"],
            metadata={
                "confidence": routing_decision["confidence"],
                "intents": intent_analysis["intents"],
                "secondary_agents": routing_decision["secondary_agents"]
            }
        )
        
        # Update state with routing information
        state.metadata["routing"] = routing_decision
        state.metadata["intent_analysis"] = intent_analysis
        
        # Set next agent
        state.metadata["next_agent"] = routing_decision["primary_agent"]
        
        return state
    
    async def _analyze_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the intent of the query using LLM."""
        system_prompt = """You are an intent analysis expert. Analyze the given query and identify the primary and secondary intents.

Available intents:
- sql_generation: User wants to retrieve or query data
- analysis: User wants insights, trends, or statistical analysis
- visualization: User wants charts, graphs, or visual representations

For each intent, provide a confidence score (0-1) and reasoning.

Respond in JSON format:
{
    "intents": {
        "sql_generation": {"confidence": 0.8, "reasoning": "..."},
        "analysis": {"confidence": 0.3, "reasoning": "..."},
        "visualization": {"confidence": 0.1, "reasoning": "..."}
    },
    "primary_intent": "sql_generation",
    "overall_confidence": 0.8
}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Analyze this query: {query}")
        ]
        
        try:
            response = await self.llm.generate(messages)
            # Parse JSON response (simplified - in production, use proper JSON parsing)
            return self._parse_intent_response(response)
        except Exception as e:
            self.logger.error("intent_analysis_failed", error=str(e))
            # Fallback to pattern-based analysis
            return self._fallback_intent_analysis(query)
    
    def _parse_intent_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response for intent analysis."""
        # Simplified parsing - in production, use proper JSON parsing with error handling
        try:
            # This is a simplified implementation
            # In production, you'd use json.loads() with proper error handling
            return {
                "intents": {
                    "sql_generation": {"confidence": 0.7, "reasoning": "Query appears to request data"},
                    "analysis": {"confidence": 0.2, "reasoning": "Some analysis may be needed"},
                    "visualization": {"confidence": 0.1, "reasoning": "Visualization not explicitly requested"}
                },
                "primary_intent": "sql_generation",
                "overall_confidence": 0.7
            }
        except Exception as e:
            self.logger.error("intent_parsing_failed", error=str(e))
            return self._fallback_intent_analysis("")
    
    def _fallback_intent_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback intent analysis using pattern matching."""
        query_lower = query.lower()
        
        intent_scores = {
            "sql_generation": 0.0,
            "analysis": 0.0,
            "visualization": 0.0
        }
        
        # Pattern-based scoring
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    intent_scores[intent] += 0.3
        
        # Normalize scores
        total_score = sum(intent_scores.values()) or 1.0
        intent_scores = {k: min(v / total_score, 1.0) for k, v in intent_scores.items()}
        
        # Determine primary intent
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return {
            "intents": {
                intent: {
                    "confidence": score,
                    "reasoning": f"Pattern-based analysis: {score:.2f} confidence"
                }
                for intent, score in intent_scores.items()
            },
            "primary_intent": primary_intent,
            "overall_confidence": intent_scores[primary_intent]
        }
    
    async def _determine_routing(self, query: str, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the routing decision based on intent analysis."""
        primary_intent = intent_analysis["primary_intent"]
        confidence = intent_analysis["overall_confidence"]
        
        # Define agent mapping
        agent_mapping = {
            "sql_generation": "sql",
            "analysis": "analysis", 
            "visualization": "visualization"
        }
        
        primary_agent = agent_mapping.get(primary_intent, "sql")
        
        # Determine secondary agents based on other intents
        secondary_agents = []
        for intent, data in intent_analysis["intents"].items():
            if intent != primary_intent and data["confidence"] > 0.3:
                secondary_agents.append(agent_mapping.get(intent, "sql"))
        
        # Generate reasoning
        reasoning = f"Primary intent: {primary_intent} (confidence: {confidence:.2f})"
        if secondary_agents:
            reasoning += f". Secondary agents: {', '.join(secondary_agents)}"
        
        return {
            "primary_agent": primary_agent,
            "secondary_agents": secondary_agents,
            "confidence": confidence,
            "reasoning": reasoning,
            "intent_analysis": intent_analysis
        }
    
    def get_routing_rules(self) -> Dict[str, Any]:
        """Get the current routing rules and patterns."""
        return {
            "intent_patterns": self.intent_patterns,
            "agent_mapping": {
                "sql_generation": "sql",
                "analysis": "analysis",
                "visualization": "visualization"
            }
        } 