"""
Enhanced Router Agent for SQL Agent with Vector-Based Table Selection.

This router intelligently decides between traditional RAG and vector store based on
database complexity, and pre-selects relevant tables for downstream agents.
"""

from typing import Dict, List, Any, Optional, Tuple
import time
from datetime import datetime

from langchain.schema import HumanMessage, SystemMessage
from .base import BaseAgent
from ..core.state import AgentState, SchemaContext
from ..rag import context_manager
from ..rag.vector_store import vector_store
from ..rag.schema import schema_processor
from ..utils.logging import log_agent_decision


class RouterAgent(BaseAgent):
    """Enhanced router agent with vector-based table selection for large schemas."""
    
    def __init__(self, llm_provider):
        super().__init__("router", llm_provider)
        
        # Enhanced configuration
        self.config = {
            "vector_db_threshold": 10,      # Use vector DB when tables >= 10
            "enterprise_threshold": 50,     # Enterprise mode when tables >= 50
            "max_tables_selection": 5,      # Maximum tables to pre-select
            "context_relevance_threshold": 0.5,  # Minimum relevance for context
            "cache_ttl_minutes": 30,        # Cache routing decisions
            "enable_business_domain_routing": True,
            "enable_relationship_awareness": True
        }
        
        # Routing decision cache
        self._routing_cache: Dict[str, Tuple[Dict[str, Any], datetime]] = {}
        
        # Enhanced intent patterns with business domains
        self.intent_patterns = {
            "sql_generation": {
                "keywords": [
                    "show me", "get", "find", "select", "query", "what is", "how many",
                    "list", "display", "retrieve", "fetch", "extract", "search", "count"
                ],
                "business_domains": ["data_retrieval", "reporting", "lookup"]
            },
            "analysis": {
                "keywords": [
                    "analyze", "compare", "trend", "pattern", "insight", "summary",
                    "statistics", "average", "total", "percentage", "growth", "performance",
                    "correlation", "distribution", "variance", "insights"
                ],
                "business_domains": ["analytics", "business_intelligence", "metrics"]
            },
            "visualization": {
                "keywords": [
                    "chart", "graph", "plot", "visualize", "bar chart", "line chart",
                    "pie chart", "scatter plot", "dashboard", "report", "visual", "diagram"
                ],
                "business_domains": ["visualization", "reporting", "dashboard"]
            },
            "fraud_detection": {
                "keywords": [
                    "fraud", "suspicious", "anomaly", "unusual", "vulnerability",
                    "risk", "threat", "detection", "pattern", "outlier", "abnormal",
                    "irregular", "compliance", "audit", "security", "breach", "attack"
                ],
                "business_domains": ["fraud_detection", "security", "compliance", "audit"]
            }
        }
        
        # Business domain concepts for enhanced routing
        self.business_domains = {
            "customer_management": ["customer", "client", "user", "account", "subscriber"],
            "product_catalog": ["product", "item", "inventory", "catalog", "merchandise"],
            "order_processing": ["order", "transaction", "purchase", "sale", "payment"],
            "financial": ["revenue", "profit", "cost", "budget", "finance", "accounting"],
            "hr_management": ["employee", "staff", "hr", "personnel", "payroll"],
            "marketing": ["campaign", "promotion", "lead", "conversion", "marketing"],
            "operations": ["logistics", "warehouse", "shipping", "supply", "operations"],
            "fraud_detection": ["fraud", "suspicious", "anomaly", "risk", "threat", "vulnerability", "compliance", "audit"]
        }
    
    async def process(self, state: AgentState) -> AgentState:
        """Enhanced processing with vector-based table selection."""
        start_time = time.time()
        
        self.logger.info(
            "enhanced_router_processing", 
            query=state.query[:100],
            database=state.database_name,
            session_id=state.session_id
        )
        
        try:
            # Step 1: Check if we should use vector store or traditional RAG
            routing_strategy = await self._determine_routing_strategy(state)
            print(f"\n[ROUTER] Step 1: Routing strategy = {routing_strategy.get('strategy')}, use_vector_store = {routing_strategy.get('use_vector_store')}")

            # Step 2: Get relevant schema context using appropriate strategy
            if routing_strategy["use_vector_store"]:
                print(f"[ROUTER] Step 2: Using vector-based context...")
                schema_context, selected_tables = await self._get_vector_based_context(state)
                print(f"[ROUTER] Step 2: Got {len(schema_context)} schema contexts, {len(selected_tables)} selected tables: {selected_tables}")
            else:
                print(f"[ROUTER] Step 2: Using traditional context...")
                schema_context = await self._get_traditional_context(state)
                selected_tables = self._extract_tables_from_context(schema_context)
                print(f"[ROUTER] Step 2: Got {len(schema_context)} schema contexts, {len(selected_tables)} selected tables: {selected_tables}")

            # CRITICAL FIX: If no tables selected, try to infer from query or use all tables
            if not selected_tables:
                print(f"[ROUTER] Step 2b: No tables selected! Attempting fallback table selection...")
                selected_tables = await self._fallback_table_selection(state.query, state.database_name)
                print(f"[ROUTER] Step 2b: Fallback selected {len(selected_tables)} tables: {selected_tables}")

            # Step 3: Analyze query intent with enhanced context
            intent_analysis = await self._analyze_intent_enhanced(
                state.query, schema_context, selected_tables, routing_strategy
            )
            
            # Step 4: Determine routing decision with business domain awareness
            routing_decision = await self._determine_enhanced_routing(
                state.query, intent_analysis, schema_context, selected_tables, routing_strategy
            )
            
            # Step 5: Enrich context for downstream agents
            print(f"\n{'='*80}")
            print(f"DEBUG ROUTER - Before enrichment:")
            print(f"  Selected tables: {selected_tables}")
            print(f"  Schema context count: {len(schema_context)}")
            print(f"  Database name: {state.database_name}")
            print(f"{'='*80}\n")

            enriched_context = await self._enrich_context_for_agents(
                selected_tables, schema_context, state.database_name
            )

            print(f"\n{'='*80}")
            print(f"DEBUG ROUTER - After enrichment:")
            print(f"  Enriched context keys: {enriched_context.keys()}")
            print(f"  Column contexts keys: {enriched_context.get('column_contexts', {}).keys()}")
            if enriched_context.get('column_contexts'):
                for table, cols in enriched_context['column_contexts'].items():
                    print(f"  Table '{table}' has {len(cols)} columns")
                    if cols:
                        print(f"    First column sample: {cols[0]}")
            else:
                print(f"  WARNING: No column_contexts in enriched_context!")
            print(f"{'='*80}\n")

            processing_time = time.time() - start_time
            
            # Log comprehensive routing decision
            log_agent_decision(
                self.logger,
                agent=self.name,
                decision=routing_decision["primary_agent"],
                reasoning=routing_decision["reasoning"],
                metadata={
                    "confidence": routing_decision["confidence"],
                    "selected_tables": selected_tables,
                    "routing_strategy": routing_strategy["strategy"],
                    "business_domains": routing_decision.get("business_domains", []),
                    "processing_time": processing_time,
                    "context_count": len(schema_context),
                    "use_vector_store": routing_strategy["use_vector_store"]
                }
            )
            
            # Update state with comprehensive routing information
            state.metadata.update({
                "routing": routing_decision,
                "intent_analysis": intent_analysis,
                "selected_tables": selected_tables,
                "enriched_context": enriched_context,
                "routing_strategy": routing_strategy,
                "processing_time": processing_time
            })
            
            # Store schema context for downstream agents
            state.schema_context = schema_context
            
            # Set next agent
            state.metadata["next_agent"] = routing_decision["primary_agent"]
            
            return state
            
        except Exception as e:
            self.logger.error(
                "enhanced_router_processing_failed",
                query=state.query[:100],
                error=str(e),
                processing_time=time.time() - start_time,
                exc_info=True
            )
            
            # Fallback to basic routing
            return await self._fallback_routing(state)
    
    async def _determine_routing_strategy(self, state: AgentState) -> Dict[str, Any]:
        """Determine whether to use vector store or traditional RAG."""
        try:
            # Get database schema information
            schema_data = await schema_processor.get_database_schema(state.database_name)
            table_count = len(schema_data.get("tables", []))
            
            # Determine strategy based on table count and complexity
            if table_count >= self.config["enterprise_threshold"]:
                strategy = "enterprise_vector"
                use_vector_store = True
                reasoning = f"Enterprise schema with {table_count} tables - using advanced vector search"
            elif table_count >= self.config["vector_db_threshold"]:
                strategy = "vector_enhanced"
                use_vector_store = True
                reasoning = f"Medium schema with {table_count} tables - using vector-based selection"
            else:
                strategy = "traditional_rag"
                use_vector_store = False
                reasoning = f"Small schema with {table_count} tables - using traditional RAG"
            
            self.logger.info(
                "routing_strategy_determined",
                database=state.database_name,
                table_count=table_count,
                strategy=strategy,
                use_vector_store=use_vector_store
            )
            
            return {
                "strategy": strategy,
                "use_vector_store": use_vector_store,
                "table_count": table_count,
                "reasoning": reasoning,
                "max_tables": self.config["max_tables_selection"]
            }
            
        except Exception as e:
            self.logger.warning("routing_strategy_determination_failed", error=str(e))
            # Default to traditional RAG on error
            return {
                "strategy": "traditional_rag_fallback",
                "use_vector_store": False,
                "table_count": 0,
                "reasoning": f"Fallback to traditional RAG due to error: {str(e)}",
                "max_tables": self.config["max_tables_selection"]
            }
    
    async def _get_vector_based_context(self, state: AgentState) -> Tuple[List[SchemaContext], List[str]]:
        """Get context using vector store for intelligent table selection."""
        try:
            print(f"[ROUTER-VECTOR] Querying vector store for tables...")
            print(f"[ROUTER-VECTOR]   Query: {state.query}")
            print(f"[ROUTER-VECTOR]   Database: {state.database_name}")
            print(f"[ROUTER-VECTOR]   Limit: {self.config['max_tables_selection']}")

            # Get relevant tables using vector store
            selected_tables = await vector_store.get_tables_for_query(
                query=state.query,
                database_name=state.database_name,
                limit=self.config["max_tables_selection"]
            )

            print(f"[ROUTER-VECTOR] Vector store returned {len(selected_tables)} tables: {selected_tables}")

            if not selected_tables:
                self.logger.warning("no_tables_selected_from_vector_store", query=state.query[:100])
                print(f"[ROUTER-VECTOR] WARNING: No tables selected! Returning empty")
                return [], []
            
            # Get detailed context for selected tables
            schema_contexts = []
            for table_name in selected_tables:
                try:
                    table_context_data = await vector_store.get_table_context(
                        table_name=table_name,
                        database_name=state.database_name
                    )
                    
                    # Convert to SchemaContext objects
                    if table_context_data.get("table_context"):
                        context = self._create_schema_context_from_vector_data(
                            table_name, table_context_data
                        )
                        schema_contexts.append(context)
                        
                except Exception as e:
                    self.logger.warning(
                        "table_context_retrieval_failed", 
                        table=table_name, 
                        error=str(e)
                    )
                    continue
            
            self.logger.info(
                "vector_based_context_retrieved",
                query=state.query[:100],
                selected_tables=selected_tables,
                context_count=len(schema_contexts)
            )
            
            return schema_contexts, selected_tables
            
        except Exception as e:
            self.logger.error("vector_based_context_failed", error=str(e))
            return [], []
    
    async def _get_traditional_context(self, state: AgentState) -> List[SchemaContext]:
        """Get context using traditional RAG context manager."""
        try:
            contexts = await context_manager.retrieve_schema_context(
                query=state.query,
                limit=self.config["max_tables_selection"],
                min_similarity=self.config["context_relevance_threshold"]
            )
            
            self.logger.info(
                "traditional_context_retrieved",
                query=state.query[:100],
                context_count=len(contexts)
            )
            
            return contexts
            
        except Exception as e:
            self.logger.error("traditional_context_failed", error=str(e))
            return []
    
    def _extract_tables_from_context(self, schema_context: List[SchemaContext]) -> List[str]:
        """Extract unique table names from schema context."""
        tables = []
        seen_tables = set()
        
        for context in schema_context:
            if context.table_name and context.table_name not in seen_tables:
                tables.append(context.table_name)
                seen_tables.add(context.table_name)
        
        return tables
    
    def _create_schema_context_from_vector_data(
        self, 
        table_name: str, 
        table_context_data: Dict[str, Any]
    ) -> SchemaContext:
        """Create SchemaContext from vector store data."""
        table_context = table_context_data.get("table_context", {})
        metadata = table_context.get("metadata", {})
        
        return SchemaContext(
            table_name=table_name,
            description=table_context.get("content", ""),
            sample_values=[],  # Could be enhanced with actual sample values
            relationships=[], # Could be enhanced with relationship data
            embedding=None    # Not needed for routing
        )
    
    async def _analyze_intent_enhanced(
        self, 
        query: str, 
        schema_context: List[SchemaContext], 
        selected_tables: List[str],
        routing_strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhanced intent analysis with business domain awareness."""
        try:
            # Extract business domains from query and context
            business_domains = self._extract_business_domains(query, selected_tables)
            
            # Build enhanced schema context string
            schema_context_str = self._build_enhanced_schema_context_string(
                schema_context, selected_tables, business_domains
            )
            
            # Create enhanced system prompt
            system_prompt = self._create_enhanced_intent_prompt(
                schema_context_str, business_domains, routing_strategy
            )
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Analyze this query with business context: {query}")
            ]
            
            response = await self.llm.generate(messages)
            return self._parse_enhanced_intent_response(
                response, schema_context, selected_tables, business_domains
            )
            
        except Exception as e:
            self.logger.error("enhanced_intent_analysis_failed", error=str(e))
            return self._fallback_intent_analysis_enhanced(query, selected_tables)
    
    def _extract_business_domains(self, query: str, selected_tables: List[str]) -> List[str]:
        """Extract business domains from query and table context."""
        query_lower = query.lower()
        domains = []
        
        # Check query text for business domain indicators
        for domain, keywords in self.business_domains.items():
            if any(keyword in query_lower for keyword in keywords):
                domains.append(domain)
        
        # Check table names for business domain indicators
        for table_name in selected_tables:
            table_lower = table_name.lower()
            for domain, keywords in self.business_domains.items():
                if any(keyword in table_lower for keyword in keywords):
                    if domain not in domains:
                        domains.append(domain)
        
        return domains
    
    def _build_enhanced_schema_context_string(
        self, 
        schema_context: List[SchemaContext], 
        selected_tables: List[str],
        business_domains: List[str]
    ) -> str:
        """Build enhanced schema context string with business intelligence."""
        if not schema_context and not selected_tables:
            return "No schema context available"
        
        context_parts = []
        
        # Add business domain context
        if business_domains:
            context_parts.append(f"Business Domains: {', '.join(business_domains)}")
        
        # Add selected tables summary
        if selected_tables:
            context_parts.append(f"Pre-selected Relevant Tables: {', '.join(selected_tables)}")
        
        # Add detailed schema context
        for context in schema_context:
            context_parts.append(f"\nTable: {context.table_name}")
            if context.description:
                context_parts.append(f"  Purpose: {context.description}")
            if context.relationships:
                context_parts.append(f"  Relationships: {', '.join(context.relationships)}")
        
        return "\n".join(context_parts)
    
    def _create_enhanced_intent_prompt(
        self, 
        schema_context_str: str, 
        business_domains: List[str],
        routing_strategy: Dict[str, Any]
    ) -> str:
        """Create enhanced system prompt for intent analysis."""
        return f"""You are an advanced intent analysis expert for database queries with business intelligence.

Available intents and their business contexts:
- sql_generation: Direct data retrieval, reporting, lookups
- analysis: Statistical analysis, trends, insights, business intelligence
- visualization: Charts, graphs, dashboards, visual reports

Business Context:
{schema_context_str}

Routing Strategy: {routing_strategy['strategy']}
Available Tables: {routing_strategy.get('table_count', 'unknown')} tables

Business Domains Detected: {', '.join(business_domains) if business_domains else 'None'}

Consider:
1. Query complexity and data requirements
2. Business domain context
3. Available table relationships
4. Visualization potential of the data
5. Analysis depth required

Respond in JSON format:
{{
    "intents": {{
        "sql_generation": {{"confidence": 0.0-1.0, "reasoning": "detailed reasoning"}},
        "analysis": {{"confidence": 0.0-1.0, "reasoning": "detailed reasoning"}},
        "visualization": {{"confidence": 0.0-1.0, "reasoning": "detailed reasoning"}}
    }},
    "primary_intent": "intent_name",
    "secondary_intents": ["intent1", "intent2"],
    "overall_confidence": 0.0-1.0,
    "business_context": "business domain analysis",
    "complexity_assessment": "simple|medium|complex",
    "requires_joins": true|false
}}"""
    
    def _parse_enhanced_intent_response(
        self, 
        response: str, 
        schema_context: List[SchemaContext],
        selected_tables: List[str],
        business_domains: List[str]
    ) -> Dict[str, Any]:
        """Parse enhanced LLM response for intent analysis."""
        try:
            import json
            
            # Try to extract JSON from response
            response_clean = response.strip()
            if "```json" in response_clean:
                start = response_clean.find("```json") + 7
                end = response_clean.find("```", start)
                response_clean = response_clean[start:end].strip()
            elif "```" in response_clean:
                start = response_clean.find("```") + 3
                end = response_clean.rfind("```")
                response_clean = response_clean[start:end].strip()
            
            parsed = json.loads(response_clean)
            
            # Enhance with additional context
            parsed["business_domains"] = business_domains
            parsed["selected_tables"] = selected_tables
            parsed["context_count"] = len(schema_context)
            
            return parsed
            
        except Exception as e:
            self.logger.warning("enhanced_intent_parsing_failed", error=str(e))
            return self._fallback_intent_analysis_enhanced("", selected_tables)
    
    def _fallback_intent_analysis_enhanced(self, query: str, selected_tables: List[str]) -> Dict[str, Any]:
        """Enhanced fallback intent analysis with business domain awareness."""
        query_lower = query.lower()
        
        # Business domain aware scoring
        intent_scores = {
            "sql_generation": 0.3,  # Base score for any query
            "analysis": 0.0,
            "visualization": 0.0
        }
        
        # Enhanced pattern-based scoring
        for intent, config in self.intent_patterns.items():
            for keyword in config["keywords"]:
                if keyword in query_lower:
                    intent_scores[intent] += 0.2
        
        # Business domain bonus scoring
        business_domains = self._extract_business_domains(query, selected_tables)
        if business_domains:
            # Analytics domains boost analysis intent
            if any(domain in ["financial", "hr_management"] for domain in business_domains):
                intent_scores["analysis"] += 0.3
            
            # Visual domains boost visualization intent
            if "dashboard" in query_lower or "report" in query_lower:
                intent_scores["visualization"] += 0.4
        
        # Normalize scores
        max_score = max(intent_scores.values())
        if max_score > 0:
            intent_scores = {k: v / max_score for k, v in intent_scores.items()}
        
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return {
            "intents": {
                intent: {
                    "confidence": score,
                    "reasoning": f"Enhanced pattern analysis: {score:.2f} confidence"
                }
                for intent, score in intent_scores.items()
            },
            "primary_intent": primary_intent,
            "secondary_intents": [
                intent for intent, score in intent_scores.items() 
                if intent != primary_intent and score > 0.3
            ],
            "overall_confidence": intent_scores[primary_intent],
            "business_domains": business_domains,
            "complexity_assessment": "medium" if len(selected_tables) > 2 else "simple",
            "requires_joins": len(selected_tables) > 1
        }
    
    async def _determine_enhanced_routing(
        self, 
        query: str, 
        intent_analysis: Dict[str, Any], 
        schema_context: List[SchemaContext],
        selected_tables: List[str],
        routing_strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine enhanced routing decision with business intelligence."""
        primary_intent = intent_analysis["primary_intent"]
        confidence = intent_analysis["overall_confidence"]
        
        # Enhanced agent mapping with business context
        agent_mapping = {
            "sql_generation": "sql",
            "analysis": "analysis",
            "visualization": "visualization"
        }
        
        primary_agent = agent_mapping.get(primary_intent, "sql")
        
        # Determine secondary agents based on business context
        secondary_agents = []
        secondary_intents = intent_analysis.get("secondary_intents", [])
        
        for intent in secondary_intents:
            if intent in agent_mapping:
                secondary_agents.append(agent_mapping[intent])
        
        # Business domain aware routing adjustments
        business_domains = intent_analysis.get("business_domains", [])
        routing_adjustments = []
        
        if "financial" in business_domains and primary_agent == "sql":
            routing_adjustments.append("Consider analysis for financial insights")
            if "analysis" not in secondary_agents:
                secondary_agents.append("analysis")
        
        if intent_analysis.get("complexity_assessment") == "complex":
            routing_adjustments.append("Complex query - enhanced processing recommended")
        
        # Generate comprehensive reasoning
        reasoning_parts = [
            f"Primary intent: {primary_intent} (confidence: {confidence:.2f})",
            f"Selected tables: {', '.join(selected_tables[:3])}{'...' if len(selected_tables) > 3 else ''}",
            f"Strategy: {routing_strategy['strategy']}"
        ]
        
        if business_domains:
            reasoning_parts.append(f"Business domains: {', '.join(business_domains)}")
        
        if secondary_agents:
            reasoning_parts.append(f"Secondary agents: {', '.join(secondary_agents)}")
        
        if routing_adjustments:
            reasoning_parts.extend(routing_adjustments)
        
        return {
            "primary_agent": primary_agent,
            "secondary_agents": secondary_agents,
            "confidence": confidence,
            "reasoning": ". ".join(reasoning_parts),
            "intent_analysis": intent_analysis,
            "business_domains": business_domains,
            "selected_tables": selected_tables,
            "routing_strategy": routing_strategy["strategy"],
            "complexity": intent_analysis.get("complexity_assessment", "medium"),
            "requires_joins": intent_analysis.get("requires_joins", False)
        }
    
    async def _enrich_context_for_agents(
        self, 
        selected_tables: List[str], 
        schema_context: List[SchemaContext],
        database_name: str
    ) -> Dict[str, Any]:
        """Enrich context with additional information for downstream agents."""
        try:
            enriched_context = {
                "selected_tables": selected_tables,
                "table_count": len(selected_tables),
                "schema_contexts": schema_context,
                "database_name": database_name
            }
            
            # Add relationship insights if multiple tables
            if len(selected_tables) > 1:
                try:
                    relationship_insights = await vector_store.get_relationship_insights(
                        table_names=selected_tables,
                        database_name=database_name
                    )
                    enriched_context["relationships"] = relationship_insights
                except Exception as e:
                    self.logger.warning("relationship_insights_failed", error=str(e))
            
            # Add column context for detailed SQL generation with data types, statistics, and sample data
            if selected_tables:
                print(f"\n[ENRICH] Starting column context enrichment for {len(selected_tables)} tables")
                try:
                    from sql_agent.core.database import db_manager

                    # Get comprehensive column statistics and metadata
                    column_contexts = {}
                    for table_name in selected_tables[:3]:  # Limit to top 3 tables
                        print(f"\n[ENRICH] Processing table: {table_name}")
                        enriched_columns = []

                        try:
                            # Get detailed column statistics including min/max/count/distinct
                            print(f"[ENRICH]   Fetching column statistics...")
                            col_stats = await db_manager.get_column_statistics(table_name)
                            print(f"[ENRICH]   Got {len(col_stats)} columns from statistics")

                            if "error" not in col_stats:
                                print(f"[ENRICH]   Column statistics successful, fetching vector context...")
                                # Try to get additional context from vector store
                                table_context = await vector_store.get_table_context(table_name, database_name)
                                vector_columns = {}

                                if table_context.get("column_contexts"):
                                    print(f"[ENRICH]   Vector store has {len(table_context['column_contexts'])} column contexts")
                                    for col_ctx in table_context["column_contexts"]:
                                        col_name = col_ctx.get("metadata", {}).get("column_name", col_ctx.get("column_name", ""))
                                        if col_name:
                                            vector_columns[col_name] = col_ctx.get("metadata", {})
                                else:
                                    print(f"[ENRICH]   No column contexts in vector store")

                                # Build enriched column information
                                print(f"[ENRICH]   Building enriched columns from {len(col_stats)} statistics...")
                                for col_name, stats in col_stats.items():
                                    vector_meta = vector_columns.get(col_name, {})

                                    enriched_col = {
                                        "column_name": col_name,
                                        "data_type": stats.get("data_type", "unknown"),
                                        "nullable": stats.get("is_nullable", True),
                                        "primary_key": vector_meta.get("is_primary_key", False),
                                        "foreign_key": vector_meta.get("is_foreign_key", False),
                                        "business_concept": vector_meta.get("business_concept", ""),
                                        # Statistics
                                        "total_count": stats.get("total_count"),
                                        "distinct_count": stats.get("distinct_count"),
                                        "null_count": stats.get("null_count"),
                                        "min_value": stats.get("min_value"),
                                        "max_value": stats.get("max_value"),
                                        "avg_value": stats.get("avg_value"),
                                        "sample_values": stats.get("sample_values", [])
                                    }

                                    # Remove None values for cleaner context
                                    enriched_col = {k: v for k, v in enriched_col.items() if v is not None}
                                    enriched_columns.append(enriched_col)

                                print(f"[ENRICH]   Built {len(enriched_columns)} enriched columns")
                                if enriched_columns:
                                    print(f"[ENRICH]   Sample enriched column: {enriched_columns[0]}")

                            else:
                                print(f"[ENRICH]   Column statistics had error: {col_stats.get('error')}")
                                print(f"[ENRICH]   Falling back to vector store only...")
                                # Fallback: Try to get basic column info from vector store
                                table_context = await vector_store.get_table_context(table_name, database_name)
                                if table_context.get("column_contexts"):
                                    for col_ctx in table_context["column_contexts"]:
                                        metadata = col_ctx.get("metadata", {})
                                        enriched_col = {
                                            "column_name": metadata.get("column_name", col_ctx.get("column_name", "")),
                                            "data_type": metadata.get("data_type", "unknown"),
                                            "nullable": metadata.get("is_nullable", True),
                                            "primary_key": metadata.get("is_primary_key", False),
                                            "foreign_key": metadata.get("is_foreign_key", False),
                                        }
                                        enriched_columns.append(enriched_col)

                                # Also try to fetch sample data
                                try:
                                    sample_data = await db_manager.get_sample_data(table_name, limit=3)
                                    if sample_data and sample_data.get("data"):
                                        sample_rows = sample_data["data"]
                                        columns = sample_data.get("columns", [])

                                        for enriched_col in enriched_columns:
                                            col_name = enriched_col["column_name"]
                                            if col_name in columns:
                                                col_idx = columns.index(col_name)
                                                sample_values = [
                                                    row[col_idx] for row in sample_rows
                                                    if col_idx < len(row) and row[col_idx] is not None
                                                ]
                                                enriched_col["sample_values"] = [str(v) for v in sample_values[:3]]
                                except Exception:
                                    pass

                            if enriched_columns:
                                column_contexts[table_name] = enriched_columns
                                print(f"[ENRICH]   Added {len(enriched_columns)} columns for table '{table_name}'")
                                self.logger.info("column_context_enriched",
                                               table=table_name,
                                               column_count=len(enriched_columns))
                            else:
                                print(f"[ENRICH]   WARNING: No enriched columns for table '{table_name}'")

                        except Exception as table_err:
                            print(f"[ENRICH]   ERROR enriching table '{table_name}': {table_err}")
                            import traceback
                            traceback.print_exc()
                            self.logger.warning("table_context_enrichment_failed",
                                              table=table_name,
                                              error=str(table_err))

                    print(f"\n[ENRICH] Final column_contexts has {len(column_contexts)} tables")
                    if column_contexts:
                        enriched_context["column_contexts"] = column_contexts
                        print(f"[ENRICH] Successfully added column_contexts to enriched_context")
                        self.logger.info("enriched_context_created",
                                       tables=list(column_contexts.keys()),
                                       total_columns=sum(len(cols) for cols in column_contexts.values()))
                    else:
                        print(f"[ENRICH] WARNING: No column_contexts created for any table!")
                        self.logger.warning("no_column_contexts_created",
                                          selected_tables=selected_tables)

                except Exception as e:
                    self.logger.error("column_context_enrichment_failed", error=str(e), exc_info=True)
            
            return enriched_context
            
        except Exception as e:
            self.logger.error("context_enrichment_failed", error=str(e))
            return {
                "selected_tables": selected_tables,
                "table_count": len(selected_tables),
                "schema_contexts": schema_context,
                "database_name": database_name
            }
    
    async def _fallback_routing(self, state: AgentState) -> AgentState:
        """Fallback routing when enhanced processing fails."""
        try:
            self.logger.warning("using_fallback_routing", query=state.query[:100])
            
            # Basic intent analysis
            intent_analysis = self._fallback_intent_analysis_enhanced(state.query, [])
            
            # Basic routing decision
            routing_decision = {
                "primary_agent": "sql",
                "secondary_agents": [],
                "confidence": 0.5,
                "reasoning": "Fallback routing due to processing error",
                "intent_analysis": intent_analysis,
                "selected_tables": [],
                "routing_strategy": "fallback"
            }
            
            # Update state
            state.metadata.update({
                "routing": routing_decision,
                "intent_analysis": intent_analysis,
                "selected_tables": [],
                "routing_strategy": {"strategy": "fallback", "use_vector_store": False}
            })
            
            state.metadata["next_agent"] = "sql"
            
            return state
            
        except Exception as e:
            self.logger.error("fallback_routing_failed", error=str(e))
            # Ultimate fallback
            state.metadata["next_agent"] = "sql"
            return state
    
    # Public API methods
    
    async def route(self, state: AgentState) -> Dict[str, Any]:
        """Public routing method for orchestrator integration."""
        processed_state = await self.process(state)
        return processed_state.metadata.get("routing", {})
    
    def get_routing_config(self) -> Dict[str, Any]:
        """Get current routing configuration."""
        return {
            "config": self.config,
            "intent_patterns": self.intent_patterns,
            "business_domains": self.business_domains,
            "agent_mapping": {
                "sql_generation": "sql",
                "analysis": "analysis", 
                "visualization": "visualization"
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for router agent."""
        health_status = {
            "agent": self.name,
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "config": self.config
        }
        
        try:
            # Test vector store connectivity
            vector_health = await vector_store.health_check()
            health_status["vector_store"] = vector_health["status"]
            
            # Test schema processor
            try:
                await schema_processor.get_cache_status()
                health_status["schema_processor"] = "healthy"
            except Exception:
                health_status["schema_processor"] = "degraded"
            
            # Test traditional RAG
            try:
                await context_manager.retrieve_schema_context("test", limit=1)
                health_status["rag_context_manager"] = "healthy"
            except Exception:
                health_status["rag_context_manager"] = "degraded"
                
        except Exception as e:
            health_status["status"] = "degraded"
            health_status["error"] = str(e)

        return health_status

    async def _fallback_table_selection(self, query: str, database_name: str) -> List[str]:
        """Fallback table selection when RAG/vector store returns no tables.

        Uses multiple strategies:
        1. Extract table names from query text
        2. Get all available tables from database
        3. Use LLM to match query to tables
        """
        try:
            from sql_agent.core.database import db_manager
            import re

            # Strategy 1: Extract table names from query text using simple pattern matching
            query_lower = query.lower()
            potential_tables = []

            # Get all available tables from database
            try:
                schema_data = await db_manager.get_database_schema()
                all_tables = [table["name"] for table in schema_data.get("tables", [])]
                print(f"[FALLBACK] Found {len(all_tables)} total tables in database: {all_tables}")

                # Check if any table name appears in the query
                for table_name in all_tables:
                    if table_name.lower() in query_lower:
                        potential_tables.append(table_name)
                        print(f"[FALLBACK] Found table '{table_name}' mentioned in query")

                if potential_tables:
                    print(f"[FALLBACK] Strategy 1 success: Found {len(potential_tables)} tables from query text")
                    return potential_tables[:3]  # Limit to 3 tables

                # Strategy 2: If no tables found in query, return all tables (let LLM decide)
                if all_tables:
                    print(f"[FALLBACK] Strategy 2: No tables found in query text, using all {len(all_tables)} tables")
                    return all_tables[:3]  # Limit to 3 tables to avoid context overflow

            except Exception as db_error:
                print(f"[FALLBACK] Database query failed: {db_error}")

            # Ultimate fallback: return empty (will trigger SQL agent's own fallback)
            print(f"[FALLBACK] All strategies failed, returning empty")
            return []

        except Exception as e:
            self.logger.error("fallback_table_selection_failed", error=str(e), exc_info=True)
            return []