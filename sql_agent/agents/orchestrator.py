"""Enhanced Agent orchestrator for SQL Agent."""

import uuid
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
import time
from contextlib import asynccontextmanager

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel

from .base import BaseAgent
from .router import RouterAgent
from .sql import SQLAgent
from .analysis import AnalysisAgent
from .viz import VisualizationAgent
from ..core.state import AgentState, QueryResult, AnalysisResult, VisualizationConfig
from ..core.llm import LLMFactory
from ..core.database import db_manager
from ..rag import context_manager
from ..utils.logging import get_logger


class WorkflowMode(str, Enum):
    """Workflow execution modes."""
    SEQUENTIAL = "sequential"  # Router -> SQL -> Analysis -> Visualization
    PARALLEL = "parallel"     # SQL + Analysis + Visualization in parallel
    ADAPTIVE = "adaptive"     # Dynamic based on query type
    CUSTOM = "custom"         # User-defined sequence


class ExecutionStrategy(str, Enum):
    """Execution strategies for different query types."""
    FAST = "fast"           # Skip analysis and visualization for simple queries
    COMPREHENSIVE = "comprehensive"  # Full pipeline
    ANALYSIS_FOCUSED = "analysis_focused"  # Emphasis on analysis
    VISUALIZATION_FOCUSED = "visualization_focused"  # Emphasis on visualization


class AgentOrchestrator:
    """Enhanced orchestrator for managing agent workflows with advanced features."""
    
    def __init__(self, llm_provider_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger("orchestrator")
        self.config = config or {}
        
        # Create LLM provider
        self.llm_provider = LLMFactory.create_provider(llm_provider_name)
        
        # Initialize agents
        self.router_agent = RouterAgent(self.llm_provider)
        self.sql_agent = SQLAgent(self.llm_provider)
        self.analysis_agent = AnalysisAgent(self.llm_provider)
        self.visualization_agent = VisualizationAgent(self.llm_provider)
        
        # Workflow management
        self.workflows: Dict[WorkflowMode, CompiledStateGraph] = {}
        self.current_workflow_mode = WorkflowMode.ADAPTIVE
        
        # Performance tracking
        self.execution_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "average_processing_time": 0.0,
            "agent_performance": {
                "router": {"calls": 0, "avg_time": 0.0, "errors": 0},
                "sql": {"calls": 0, "avg_time": 0.0, "errors": 0},
                "analysis": {"calls": 0, "avg_time": 0.0, "errors": 0},
                "visualization": {"calls": 0, "avg_time": 0.0, "errors": 0}
            }
        }
        
        # Circuit breaker for agent failures
        self.circuit_breakers = {
            "sql": {"failures": 0, "last_failure": None, "threshold": 5},
            "analysis": {"failures": 0, "last_failure": None, "threshold": 3},
            "visualization": {"failures": 0, "last_failure": None, "threshold": 3}
        }
        
        # RAG and context management
        self._rag_initialized = False
        self._context_cache = {}
        
        # Create workflows
        self._create_workflows()
    
    async def initialize(self) -> None:
        """Initialize the orchestrator and all components."""
        try:
            self.logger.info("initializing_enhanced_orchestrator")
            
            # Initialize database manager
            try:
                if hasattr(db_manager, 'initialize'):
                    await db_manager.initialize()
                self.logger.info("database_manager_initialized")
            except Exception as e:
                self.logger.error("database_manager_initialization_failed", error=repr(e))
                # Don't fail initialization if DB manager fails
            
            # Initialize RAG components
            try:
                await context_manager.initialize()
                self._rag_initialized = True
                self.logger.info("rag_context_manager_initialized")
            except Exception as e:
                self.logger.error("rag_context_manager_initialization_failed", error=repr(e))
                # Continue without RAG if it fails
                self._rag_initialized = False
            
            # Initialize all agents
            for agent_name, agent in self._get_all_agents().items():
                try:
                    if hasattr(agent, 'initialize'):
                        await agent.initialize()
                    self.logger.info(f"{agent_name}_agent_initialized")
                except Exception as e:
                    self.logger.error(f"{agent_name}_agent_initialization_failed", error=repr(e))
            
            # Test LLM provider
            try:
                await self._test_llm_provider()
                self.logger.info("llm_provider_tested_successfully")
            except Exception as e:
                self.logger.error("llm_provider_test_failed", error=repr(e))
                raise RuntimeError(f"LLM provider test failed: {e}")
            
            self.logger.info("enhanced_orchestrator_initialized", 
                           rag_initialized=self._rag_initialized,
                           workflow_modes=list(self.workflows.keys()))
            
        except Exception as e:
            self.logger.error("orchestrator_initialization_failed", error=repr(e), exc_info=True)
            raise
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self.logger.info("cleaning_up_orchestrator")
            
            # Cleanup agents
            for agent_name, agent in self._get_all_agents().items():
                try:
                    if hasattr(agent, 'cleanup'):
                        await agent.cleanup()
                except Exception as e:
                    self.logger.warning(f"cleanup_failed_for_{agent_name}", error=str(e))
            
            # Clear caches
            self._context_cache.clear()
            
            self.logger.info("orchestrator_cleanup_complete")
            
        except Exception as e:
            self.logger.error("orchestrator_cleanup_failed", error=str(e))
    
    def _create_workflows(self) -> None:
        """Create different workflow configurations."""
        # Sequential workflow (original)
        self.workflows[WorkflowMode.SEQUENTIAL] = self._create_sequential_workflow()
        
        # Parallel workflow
        self.workflows[WorkflowMode.PARALLEL] = self._create_parallel_workflow()
        
        # Adaptive workflow
        self.workflows[WorkflowMode.ADAPTIVE] = self._create_adaptive_workflow()
    
    def _create_sequential_workflow(self) -> CompiledStateGraph:
        """Create the traditional sequential workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self._run_router)
        workflow.add_node("sql", self._run_sql)
        workflow.add_node("analysis", self._run_analysis)
        workflow.add_node("visualization", self._run_visualization)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional edges based on routing
        workflow.add_conditional_edges(
            "router",
            self._route_to_agent,
            {
                "sql": "sql",
                "analysis": "analysis",
                "visualization": "visualization",
                "end": END
            }
        )
        
        # Sequential flow
        workflow.add_edge("sql", "analysis")
        workflow.add_edge("analysis", "visualization")
        workflow.add_edge("visualization", END)
        
        return workflow.compile()
    
    def _create_parallel_workflow(self) -> CompiledStateGraph:
        """Create a parallel execution workflow."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("router", self._run_router)
        workflow.add_node("sql", self._run_sql)
        workflow.add_node("parallel_processor", self._run_parallel_analysis_viz)
        workflow.add_node("aggregator", self._aggregate_results)
        
        workflow.set_entry_point("router")
        
        workflow.add_conditional_edges(
            "router",
            self._route_to_agent,
            {
                "sql": "sql",
                "parallel": "parallel_processor",
                "end": END
            }
        )
        
        workflow.add_edge("sql", "parallel_processor")
        workflow.add_edge("parallel_processor", "aggregator")
        workflow.add_edge("aggregator", END)
        
        return workflow.compile()
    
    def _create_adaptive_workflow(self) -> CompiledStateGraph:
        """Create an adaptive workflow that adjusts based on query characteristics."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("router", self._run_enhanced_router)
        workflow.add_node("sql", self._run_sql)
        workflow.add_node("analysis", self._run_analysis)
        workflow.add_node("visualization", self._run_visualization)
        workflow.add_node("quality_check", self._run_quality_check)
        
        workflow.set_entry_point("router")
        
        workflow.add_conditional_edges(
            "router",
            self._adaptive_route,
            {
                "sql_only": "sql",
                "sql_analysis": "sql",
                "sql_viz": "sql",
                "full_pipeline": "sql",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "sql",
            self._determine_next_step,
            {
                "analysis": "analysis",
                "visualization": "visualization",
                "quality_check": "quality_check",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "analysis",
            self._post_analysis_route,
            {
                "visualization": "visualization",
                "quality_check": "quality_check",
                "end": END
            }
        )
        
        workflow.add_edge("visualization", "quality_check")
        workflow.add_edge("quality_check", END)
        
        return workflow.compile()
    
    async def _run_router(self, state: AgentState) -> AgentState:
        """Run the router agent with performance tracking."""
        return await self._run_agent_with_tracking("router", self.router_agent, state)
    
    async def _run_enhanced_router(self, state: AgentState) -> AgentState:
        """Enhanced router with context enrichment."""
        start_time = time.time()
        
        try:
            # Enrich state with context if RAG is available
            if self._rag_initialized:
                state = await self._enrich_with_context(state)
            
            # Run router
            state = await self.router_agent.run(state)
            
            # Enhance routing decision with query characteristics
            state = await self._analyze_query_characteristics(state)
            
            return state
            
        except Exception as e:
            self.logger.error("enhanced_router_failed", error=str(e), session_id=state.session_id)
            state.add_error(f"Enhanced router failed: {e}")
            return state
        finally:
            execution_time = time.time() - start_time
            self._update_agent_stats("router", execution_time, state.has_errors())
    
    async def _run_sql(self, state: AgentState) -> AgentState:
        """Run SQL agent with circuit breaker and retry logic."""
        if self._is_circuit_open("sql"):
            state.add_error("SQL agent circuit breaker is open")
            return state
        
        return await self._run_agent_with_tracking("sql", self.sql_agent, state)
    
    async def _run_analysis(self, state: AgentState) -> AgentState:
        """Run analysis agent with conditional execution."""
        # Skip analysis if no data or if circuit is open
        if not state.query_result or not state.query_result.data:
            self.logger.info("skipping_analysis_no_data", session_id=state.session_id)
            return state
        
        if self._is_circuit_open("analysis"):
            state.add_error("Analysis agent circuit breaker is open")
            return state
        
        return await self._run_agent_with_tracking("analysis", self.analysis_agent, state)
    
    async def _run_visualization(self, state: AgentState) -> AgentState:
        """Run visualization agent with conditional execution."""
        # Skip visualization if no data or if circuit is open
        if not state.query_result or not state.query_result.data:
            self.logger.info("skipping_visualization_no_data", session_id=state.session_id)
            return state
        
        if self._is_circuit_open("visualization"):
            state.add_error("Visualization agent circuit breaker is open")
            return state
        
        return await self._run_agent_with_tracking("visualization", self.visualization_agent, state)
    
    async def _run_parallel_analysis_viz(self, state: AgentState) -> AgentState:
        """Run analysis and visualization in parallel."""
        if not state.query_result or not state.query_result.data:
            return state
        
        tasks = []
        
        if not self._is_circuit_open("analysis"):
            tasks.append(self._run_agent_with_tracking("analysis", self.analysis_agent, state.copy()))
        
        if not self._is_circuit_open("visualization"):
            tasks.append(self._run_agent_with_tracking("visualization", self.visualization_agent, state.copy()))
        
        if not tasks:
            return state
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Merge results back to original state
            for result in results:
                if isinstance(result, AgentState):
                    if result.analysis_result:
                        state.analysis_result = result.analysis_result
                    if result.visualization_config:
                        state.visualization_config = result.visualization_config
                    state.errors.extend(result.errors)
                elif isinstance(result, Exception):
                    state.add_error(f"Parallel execution error: {result}")
            
            return state
            
        except Exception as e:
            self.logger.error("parallel_execution_failed", error=str(e), session_id=state.session_id)
            state.add_error(f"Parallel execution failed: {e}")
            return state
    
    async def _run_quality_check(self, state: AgentState) -> AgentState:
        """Run quality checks on the results."""
        try:
            quality_score = 1.0
            quality_issues = []
            
            # Check SQL quality
            if state.query_result:
                if not state.query_result.data:
                    quality_issues.append("No data returned from query")
                    quality_score -= 0.3
                elif len(state.query_result.data) < 2:
                    quality_issues.append("Very few rows returned")
                    quality_score -= 0.1
            
            # Check analysis quality
            if state.analysis_result:
                if not state.analysis_result.insights:
                    quality_issues.append("No insights generated")
                    quality_score -= 0.2
            
            # Check visualization quality
            if state.visualization_config:
                if not state.visualization_config.chart_type:
                    quality_issues.append("No chart type determined")
                    quality_score -= 0.2
            
            # Add quality metadata
            state.metadata["quality_score"] = max(0.0, quality_score)
            state.metadata["quality_issues"] = quality_issues
            
            if quality_score < 0.5:
                self.logger.warning("low_quality_result", 
                                  session_id=state.session_id,
                                  quality_score=quality_score,
                                  issues=quality_issues)
            
            return state
            
        except Exception as e:
            self.logger.error("quality_check_failed", error=str(e), session_id=state.session_id)
            state.add_error(f"Quality check failed: {e}")
            return state
    
    async def _aggregate_results(self, state: AgentState) -> AgentState:
        """Aggregate and finalize results."""
        try:
            # Calculate overall confidence score
            confidence_scores = []
            
            if state.metadata.get("routing", {}).get("confidence"):
                confidence_scores.append(state.metadata["routing"]["confidence"])
            
            if state.metadata.get("quality_score"):
                confidence_scores.append(state.metadata["quality_score"])
            
            if confidence_scores:
                state.metadata["overall_confidence"] = sum(confidence_scores) / len(confidence_scores)
            
            # Add execution summary
            state.metadata["execution_summary"] = {
                "agents_executed": [],
                "total_processing_time": state.processing_time,
                "error_count": len(state.errors),
                "has_sql_result": bool(state.query_result),
                "has_analysis": bool(state.analysis_result),
                "has_visualization": bool(state.visualization_config)
            }
            
            return state
            
        except Exception as e:
            self.logger.error("result_aggregation_failed", error=str(e), session_id=state.session_id)
            state.add_error(f"Result aggregation failed: {e}")
            return state
    
    async def _run_agent_with_tracking(self, agent_name: str, agent: BaseAgent, state: AgentState) -> AgentState:
        """Run an agent with performance tracking and error handling."""
        start_time = time.time()
        
        try:
            self.logger.info(f"orchestrator_{agent_name}_start", session_id=state.session_id)
            
            # Run the agent with timeout
            result_state = await asyncio.wait_for(
                agent.run(state),
                timeout=self.config.get(f"{agent_name}_timeout", 120)  # Default 2 minutes
            )
            
            execution_time = time.time() - start_time
            self._update_agent_stats(agent_name, execution_time, result_state.has_errors())
            
            if result_state.has_errors():
                self._record_agent_failure(agent_name)
                self.logger.warning(f"orchestrator_{agent_name}_errors",
                                  session_id=state.session_id,
                                  errors=result_state.errors)
            else:
                self._reset_circuit_breaker(agent_name)
            
            self.logger.info(f"orchestrator_{agent_name}_complete",
                           session_id=state.session_id,
                           execution_time=execution_time,
                           has_errors=result_state.has_errors())
            
            return result_state
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            error_msg = f"{agent_name} agent timed out after {execution_time:.2f} seconds"
            self.logger.error(f"orchestrator_{agent_name}_timeout", 
                            session_id=state.session_id,
                            execution_time=execution_time)
            
            self._record_agent_failure(agent_name)
            state.add_error(error_msg)
            return state
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{agent_name} agent failed: {str(e)}"
            self.logger.error(f"orchestrator_{agent_name}_failed",
                            session_id=state.session_id,
                            error=str(e),
                            execution_time=execution_time,
                            exc_info=True)
            
            self._record_agent_failure(agent_name)
            self._update_agent_stats(agent_name, execution_time, True)
            state.add_error(error_msg)
            return state
    
    def _route_to_agent(self, state: AgentState) -> str:
        """Route to the appropriate agent based on router decision."""
        routing = state.metadata.get("routing", {})
        primary_agent = routing.get("primary_agent", "sql")
        
        self.logger.info("orchestrator_routing",
                        session_id=state.session_id,
                        primary_agent=primary_agent,
                        confidence=routing.get("confidence", 0.0))
        
        return primary_agent
    
    def _adaptive_route(self, state: AgentState) -> str:
        """Adaptive routing based on query characteristics and system state."""
        routing = state.metadata.get("routing", {})
        query_chars = state.metadata.get("query_characteristics", {})
        
        # Check system health
        if self._get_system_health_score() < 0.5:
            return "sql_only"  # Fallback to minimal processing
        
        # Route based on query complexity and type
        complexity = query_chars.get("complexity", "medium")
        query_type = routing.get("primary_agent", "sql")
        
        if complexity == "simple" and query_type == "sql":
            return "sql_only"
        elif query_type == "analysis" or complexity == "complex":
            return "full_pipeline"
        elif query_type == "visualization":
            return "sql_viz"
        else:
            return "sql_analysis"
    
    def _determine_next_step(self, state: AgentState) -> str:
        """Determine next step after SQL execution."""
        if not state.query_result or not state.query_result.data:
            return "end"
        
        routing = state.metadata.get("routing", {})
        execution_strategy = state.metadata.get("execution_strategy", ExecutionStrategy.COMPREHENSIVE)
        
        if execution_strategy == ExecutionStrategy.FAST:
            return "end"
        elif execution_strategy == ExecutionStrategy.ANALYSIS_FOCUSED:
            return "analysis"
        elif execution_strategy == ExecutionStrategy.VISUALIZATION_FOCUSED:
            return "visualization"
        else:
            # Default comprehensive
            return "analysis"
    
    def _post_analysis_route(self, state: AgentState) -> str:
        """Determine routing after analysis."""
        execution_strategy = state.metadata.get("execution_strategy", ExecutionStrategy.COMPREHENSIVE)
        
        if execution_strategy == ExecutionStrategy.ANALYSIS_FOCUSED:
            return "quality_check"
        else:
            return "visualization"
    
    async def _enrich_with_context(self, state: AgentState) -> AgentState:
        """Enrich state with RAG context."""
        try:
            if state.database_name and self._rag_initialized:
                # Get schema context from cache or RAG
                cache_key = f"schema_context_{state.database_name}"
                
                if cache_key in self._context_cache:
                    context = self._context_cache[cache_key]
                else:
                    context = await context_manager.get_context(
                        query=state.query,
                        database_name=state.database_name
                    )
                    # Cache for 10 minutes
                    self._context_cache[cache_key] = context
                    asyncio.create_task(self._expire_cache_entry(cache_key, 600))
                
                state.schema_context = context
                
            return state
            
        except Exception as e:
            self.logger.warning("context_enrichment_failed", 
                              error=str(e), 
                              session_id=state.session_id)
            return state
    
    async def _analyze_query_characteristics(self, state: AgentState) -> AgentState:
        """Analyze query characteristics for better routing."""
        try:
            query = state.query.lower()
            characteristics = {
                "length": len(state.query),
                "word_count": len(state.query.split()),
                "complexity": "simple",
                "has_aggregation": any(word in query for word in ["sum", "count", "avg", "group by", "having"]),
                "has_joins": any(word in query for word in ["join", "inner", "outer", "left", "right"]),
                "has_subqueries": "select" in query and query.count("select") > 1,
                "has_window_functions": any(word in query for word in ["over", "partition", "row_number"]),
                "is_analytical": any(word in query for word in ["trend", "analyze", "insight", "pattern"]),
                "is_visual": any(word in query for word in ["chart", "graph", "plot", "visualize", "show"]),
                "execution_strategy": ExecutionStrategy.COMPREHENSIVE
            }
            
            # Determine complexity
            complexity_score = 0
            if characteristics["has_aggregation"]: complexity_score += 1
            if characteristics["has_joins"]: complexity_score += 2
            if characteristics["has_subqueries"]: complexity_score += 2
            if characteristics["has_window_functions"]: complexity_score += 3
            if characteristics["word_count"] > 20: complexity_score += 1
            
            if complexity_score == 0:
                characteristics["complexity"] = "simple"
                characteristics["execution_strategy"] = ExecutionStrategy.FAST
            elif complexity_score <= 3:
                characteristics["complexity"] = "medium"
            else:
                characteristics["complexity"] = "complex"
            
            # Adjust strategy based on query intent
            if characteristics["is_analytical"]:
                characteristics["execution_strategy"] = ExecutionStrategy.ANALYSIS_FOCUSED
            elif characteristics["is_visual"]:
                characteristics["execution_strategy"] = ExecutionStrategy.VISUALIZATION_FOCUSED
            
            state.metadata["query_characteristics"] = characteristics
            
            return state
            
        except Exception as e:
            self.logger.warning("query_analysis_failed", 
                              error=str(e), 
                              session_id=state.session_id)
            return state
    
    async def _expire_cache_entry(self, cache_key: str, delay: int):
        """Expire cache entry after delay."""
        await asyncio.sleep(delay)
        self._context_cache.pop(cache_key, None)
    
    async def _test_llm_provider(self):
        """Test LLM provider connectivity."""
        test_messages = [{"role": "user", "content": "Hello, this is a connectivity test."}]
        await self.llm_provider.generate(test_messages)
    
    def _get_all_agents(self) -> Dict[str, BaseAgent]:
        """Get all agents."""
        return {
            "router": self.router_agent,
            "sql": self.sql_agent,
            "analysis": self.analysis_agent,
            "visualization": self.visualization_agent
        }
    
    def _is_circuit_open(self, agent_name: str) -> bool:
        """Check if circuit breaker is open for an agent."""
        breaker = self.circuit_breakers.get(agent_name, {})
        failures = breaker.get("failures", 0)
        threshold = breaker.get("threshold", 5)
        last_failure = breaker.get("last_failure")
        
        if failures >= threshold:
            if last_failure and datetime.now() - last_failure < timedelta(minutes=5):
                return True
            else:
                # Reset after 5 minutes
                self._reset_circuit_breaker(agent_name)
        
        return False
    
    def _record_agent_failure(self, agent_name: str):
        """Record an agent failure for circuit breaker."""
        if agent_name in self.circuit_breakers:
            self.circuit_breakers[agent_name]["failures"] += 1
            self.circuit_breakers[agent_name]["last_failure"] = datetime.now()
    
    def _reset_circuit_breaker(self, agent_name: str):
        """Reset circuit breaker for an agent."""
        if agent_name in self.circuit_breakers:
            self.circuit_breakers[agent_name]["failures"] = 0
            self.circuit_breakers[agent_name]["last_failure"] = None
    
    def _update_agent_stats(self, agent_name: str, execution_time: float, has_errors: bool):
        """Update agent performance statistics."""
        stats = self.execution_stats["agent_performance"].get(agent_name, {})
        
        calls = stats.get("calls", 0) + 1
        total_time = stats.get("avg_time", 0) * stats.get("calls", 0) + execution_time
        avg_time = total_time / calls
        errors = stats.get("errors", 0) + (1 if has_errors else 0)
        
        self.execution_stats["agent_performance"][agent_name] = {
            "calls": calls,
            "avg_time": avg_time,
            "errors": errors
        }
    
    def _get_system_health_score(self) -> float:
        """Calculate overall system health score."""
        total_score = 0.0
        agent_count = 0
        
        for agent_name, stats in self.execution_stats["agent_performance"].items():
            if stats["calls"] > 0:
                error_rate = stats["errors"] / stats["calls"]
                agent_score = max(0.0, 1.0 - error_rate)
                total_score += agent_score
                agent_count += 1
        
        if agent_count == 0:
            return 1.0  # No data, assume healthy
        
        return total_score / agent_count
    
    async def process_query(
        self, 
        query: str, 
        database_name: Optional[str] = None, 
        context: Optional[Dict[str, Any]] = None,
        workflow_mode: Optional[WorkflowMode] = None,
        execution_strategy: Optional[ExecutionStrategy] = None
    ) -> AgentState:
        """Enhanced query processing with configurable workflows."""
        # Ensure RAG is initialized
        if not self._rag_initialized:
            try:
                await self.initialize()
            except Exception as e:
                self.logger.warning("initialization_during_query_failed", error=str(e))
        
        # Create initial state
        session_id = str(uuid.uuid4())
        state = AgentState(
            query=query,
            session_id=session_id,
            database_name=database_name,
            start_time=datetime.utcnow()
        )
        
        # Add context to metadata
        if context:
            state.metadata.update(context)
        
        # Set execution strategy
        if execution_strategy:
            state.metadata["execution_strategy"] = execution_strategy
        
        # Update stats
        self.execution_stats["total_queries"] += 1
        
        # Select workflow
        workflow_mode = workflow_mode or self.current_workflow_mode
        workflow = self.workflows.get(workflow_mode, self.workflows[WorkflowMode.ADAPTIVE])
        
        self.logger.info("orchestrator_enhanced_query_start",
                        session_id=session_id,
                        query=query[:100],  # Truncate for logging
                        database_name=database_name,
                        workflow_mode=workflow_mode.value,
                        execution_strategy=execution_strategy.value if execution_strategy else None,
                        rag_initialized=self._rag_initialized)
        
        try:
            # Run the workflow
            final_state = await workflow.ainvoke(state)
            
            # If the result is a dict, convert to AgentState
            if isinstance(final_state, dict):
                final_state = AgentState(**final_state)
            
            # Calculate processing time
            final_state.end_time = datetime.utcnow()
            if final_state.start_time:
                final_state.processing_time = (final_state.end_time - final_state.start_time).total_seconds()
            
            # Update stats
            if final_state.has_errors():
                self.execution_stats["failed_queries"] += 1
            else:
                self.execution_stats["successful_queries"] += 1
            
            # Update average processing time
            total_queries = self.execution_stats["total_queries"]
            current_avg = self.execution_stats["average_processing_time"]
            new_avg = ((current_avg * (total_queries - 1)) + final_state.processing_time) / total_queries
            self.execution_stats["average_processing_time"] = new_avg
            
            self.logger.info("orchestrator_enhanced_query_complete",
                           session_id=session_id,
                           processing_time=final_state.processing_time,
                           has_errors=final_state.has_errors(),
                           error_count=len(final_state.errors),
                           workflow_mode=workflow_mode.value,
                           rag_context_count=len(final_state.schema_context) if final_state.schema_context else 0,
                           quality_score=final_state.metadata.get("quality_score"),
                           overall_confidence=final_state.metadata.get("overall_confidence"))
            
            return final_state
            
        except Exception as e:
            self.execution_stats["failed_queries"] += 1
            
            self.logger.error("orchestrator_enhanced_query_failed",
                            session_id=session_id,
                            error=str(e),
                            workflow_mode=workflow_mode.value,
                            exc_info=True)
            
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
        database_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentState:
        """Process a query with a custom agent sequence."""
        session_id = str(uuid.uuid4())
        state = AgentState(
            query=query,
            session_id=session_id,
            database_name=database_name,
            start_time=datetime.utcnow()
        )
        
        if context:
            state.metadata.update(context)
        
        self.logger.info("orchestrator_custom_flow_start",
                        session_id=session_id,
                        agent_sequence=agent_sequence)
        
        try:
            # Run agents in custom sequence
            for i, agent_name in enumerate(agent_sequence):
                agent = self._get_agent_by_name(agent_name)
                if agent:
                    self.logger.info(f"orchestrator_custom_step_{i+1}",
                                   session_id=session_id,
                                   agent=agent_name,
                                   step=f"{i+1}/{len(agent_sequence)}")
                    
                    state = await self._run_agent_with_tracking(agent_name, agent, state)
                    
                    # Check for critical errors that should stop the flow
                    if state.has_errors() and any("circuit breaker" in error for error in state.errors):
                        self.logger.warning("orchestrator_custom_flow_circuit_break",
                                          session_id=session_id,
                                          agent=agent_name,
                                          errors=state.errors)
                        break
                else:
                    error_msg = f"Unknown agent: {agent_name}"
                    state.add_error(error_msg)
                    self.logger.error("orchestrator_custom_flow_unknown_agent",
                                    session_id=session_id,
                                    agent=agent_name)
                    break
            
            # Run final quality check
            state = await self._run_quality_check(state)
            state = await self._aggregate_results(state)
            
            # Calculate processing time
            state.end_time = datetime.utcnow()
            if state.start_time:
                state.processing_time = (state.end_time - state.start_time).total_seconds()
            
            return state
            
        except Exception as e:
            self.logger.error("orchestrator_custom_flow_failed",
                            session_id=session_id,
                            error=str(e),
                            exc_info=True)
            
            state.add_error(f"Custom flow execution failed: {e}")
            state.end_time = datetime.utcnow()
            if state.start_time:
                state.processing_time = (state.end_time - state.start_time).total_seconds()
            
            return state
    
    async def validate_sql_query(self, query: str, database_name: Optional[str] = None) -> AgentState:
        """Validate SQL query without execution."""
        session_id = str(uuid.uuid4())
        state = AgentState(
            query=query,
            session_id=session_id,
            database_name=database_name,
            start_time=datetime.utcnow()
        )
        
        try:
            # Only run router and SQL generation (no execution)
            state = await self._run_router(state)
            
            # Set SQL agent to validation mode
            original_mode = getattr(self.sql_agent, 'execution_mode', None)
            if hasattr(self.sql_agent, 'execution_mode'):
                self.sql_agent.execution_mode = 'validate'
            
            try:
                state = await self._run_sql(state)
            finally:
                # Restore original mode
                if hasattr(self.sql_agent, 'execution_mode') and original_mode:
                    self.sql_agent.execution_mode = original_mode
            
            state.end_time = datetime.utcnow()
            if state.start_time:
                state.processing_time = (state.end_time - state.start_time).total_seconds()
            
            return state
            
        except Exception as e:
            self.logger.error("sql_validation_failed",
                            session_id=session_id,
                            error=str(e))
            
            state.add_error(f"SQL validation failed: {e}")
            state.end_time = datetime.utcnow()
            if state.start_time:
                state.processing_time = (state.end_time - state.start_time).total_seconds()
            
            return state
    
    async def generate_table_description(
        self, 
        database_id: str, 
        table_name: str, 
        regenerate: bool = False
    ) -> AgentState:
        """Generate AI description for a table."""
        session_id = str(uuid.uuid4())
        state = AgentState(
            query=f"Describe the {table_name} table structure and purpose",
            session_id=session_id,
            database_name=database_id,
            start_time=datetime.utcnow()
        )
        
        state.metadata["table_description_request"] = {
            "table_name": table_name,
            "regenerate": regenerate
        }
        
        try:
            # Check cache first if not regenerating
            if not regenerate:
                cache_key = f"table_desc_{database_id}_{table_name}"
                if cache_key in self._context_cache:
                    cached_desc = self._context_cache[cache_key]
                    state.metadata["table_description"] = {
                        "text": cached_desc,
                        "generated_at": datetime.utcnow(),
                        "cached": True
                    }
                    return state
            
            # Use analysis agent to generate description
            state = await self._run_analysis(state)
            
            # Cache the result
            if state.analysis_result and hasattr(state.analysis_result, 'description'):
                cache_key = f"table_desc_{database_id}_{table_name}"
                self._context_cache[cache_key] = state.analysis_result.description
                asyncio.create_task(self._expire_cache_entry(cache_key, 3600))  # 1 hour cache
            
            state.end_time = datetime.utcnow()
            if state.start_time:
                state.processing_time = (state.end_time - state.start_time).total_seconds()
            
            return state
            
        except Exception as e:
            self.logger.error("table_description_generation_failed",
                            session_id=session_id,
                            table_name=table_name,
                            error=str(e))
            
            state.add_error(f"Table description generation failed: {e}")
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
    
    def set_workflow_mode(self, mode: WorkflowMode) -> None:
        """Set the current workflow mode."""
        if mode in self.workflows:
            self.current_workflow_mode = mode
            self.logger.info("workflow_mode_changed", 
                           old_mode=self.current_workflow_mode.value,
                           new_mode=mode.value)
        else:
            raise ValueError(f"Unknown workflow mode: {mode}")
    
    def get_agent_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all agents."""
        info = {}
        for agent_name, agent in self._get_all_agents().items():
            try:
                agent_info = agent.get_agent_info()
                agent_info["performance"] = self.execution_stats["agent_performance"].get(agent_name, {})
                agent_info["circuit_breaker"] = self.circuit_breakers.get(agent_name, {})
                info[agent_name] = agent_info
            except Exception as e:
                info[agent_name] = {"error": str(e)}
        
        return info
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the workflow."""
        return {
            "available_modes": [mode.value for mode in WorkflowMode],
            "current_mode": self.current_workflow_mode.value,
            "execution_strategies": [strategy.value for strategy in ExecutionStrategy],
            "agents": list(self._get_all_agents().keys()),
            "default_flow": ["router", "sql", "analysis", "visualization"],
            "llm_provider": self.llm_provider.model_name,
            "rag_initialized": self._rag_initialized,
            "system_health_score": self._get_system_health_score()
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        stats = self.execution_stats.copy()
        
        # Add derived metrics
        if stats["total_queries"] > 0:
            stats["success_rate"] = stats["successful_queries"] / stats["total_queries"]
            stats["failure_rate"] = stats["failed_queries"] / stats["total_queries"]
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0
        
        stats["system_health_score"] = self._get_system_health_score()
        stats["circuit_breaker_status"] = {
            agent: {
                "is_open": self._is_circuit_open(agent),
                "failures": breaker.get("failures", 0),
                "threshold": breaker.get("threshold", 5)
            }
            for agent, breaker in self.circuit_breakers.items()
        }
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check on all components."""
        health_status = {
            "orchestrator": "healthy",
            "agents": {},
            "llm_provider": "unknown",
            "rag_system": "unknown",
            "database": "unknown",
            "overall_status": "healthy",
            "system_health_score": self._get_system_health_score(),
            "performance_stats": self.get_performance_stats(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        issues = []
        
        # Check LLM provider
        try:
            await self._test_llm_provider()
            health_status["llm_provider"] = "healthy"
        except Exception as e:
            health_status["llm_provider"] = f"unhealthy: {str(e)}"
            issues.append(f"LLM provider: {str(e)}")
        
        # Check RAG system
        try:
            if self._rag_initialized:
                # Simple test query
                test_context = await context_manager.get_context("test query")
                health_status["rag_system"] = "healthy"
            else:
                health_status["rag_system"] = "not_initialized"
        except Exception as e:
            health_status["rag_system"] = f"unhealthy: {str(e)}"
            issues.append(f"RAG system: {str(e)}")
        
        # Check database
        try:
            if hasattr(db_manager, 'test_connection'):
                await db_manager.test_connection()
                health_status["database"] = "healthy"
            else:
                health_status["database"] = "unknown"
        except Exception as e:
            health_status["database"] = f"unhealthy: {str(e)}"
            issues.append(f"Database: {str(e)}")
        
        # Check each agent
        for agent_name, agent in self._get_all_agents().items():
            try:
                agent_info = agent.get_agent_info()
                
                # Check circuit breaker status
                if self._is_circuit_open(agent_name):
                    health_status["agents"][agent_name] = "circuit_breaker_open"
                    issues.append(f"{agent_name} agent circuit breaker is open")
                else:
                    health_status["agents"][agent_name] = "healthy"
                    
            except Exception as e:
                health_status["agents"][agent_name] = f"unhealthy: {str(e)}"
                issues.append(f"{agent_name} agent: {str(e)}")
        
        # Determine overall status
        if issues:
            if len(issues) >= len(self._get_all_agents()) / 2:  # More than half have issues
                health_status["overall_status"] = "unhealthy"
            else:
                health_status["overall_status"] = "degraded"
            
            health_status["issues"] = issues
        
        return health_status
    
    @asynccontextmanager
    async def temporary_config(self, **config_overrides):
        """Context manager for temporary configuration changes."""
        original_config = self.config.copy()
        try:
            self.config.update(config_overrides)
            yield
        finally:
            self.config = original_config
    
    def reset_performance_stats(self) -> None:
        """Reset all performance statistics."""
        self.execution_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "average_processing_time": 0.0,
            "agent_performance": {
                "router": {"calls": 0, "avg_time": 0.0, "errors": 0},
                "sql": {"calls": 0, "avg_time": 0.0, "errors": 0},
                "analysis": {"calls": 0, "avg_time": 0.0, "errors": 0},
                "visualization": {"calls": 0, "avg_time": 0.0, "errors": 0}
            }
        }
        
        # Reset circuit breakers
        for agent_name in self.circuit_breakers:
            self._reset_circuit_breaker(agent_name)
        
        self.logger.info("performance_stats_reset")
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self._context_cache.clear()
        self.logger.info("cache_cleared")
    
    async def warmup(self) -> None:
        """Warm up the orchestrator with test queries."""
        try:
            self.logger.info("orchestrator_warmup_start")
            
            # Test queries for different types
            test_queries = [
                ("SELECT 1", "simple_sql"),
                ("Show me sales data", "analysis"),
                ("Create a chart of revenue", "visualization")
            ]
            
            for query, query_type in test_queries:
                try:
                    await self.process_query(
                        query=query,
                        execution_strategy=ExecutionStrategy.FAST
                    )
                    self.logger.info(f"warmup_{query_type}_success")
                except Exception as e:
                    self.logger.warning(f"warmup_{query_type}_failed", error=str(e))
            
            self.logger.info("orchestrator_warmup_complete")
            
        except Exception as e:
            self.logger.error("orchestrator_warmup_failed", error=str(e))