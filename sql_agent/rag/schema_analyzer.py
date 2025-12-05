"""
Schema Analyzer - LLM-Powered Discovery Intelligence Engine

This module provides the core intelligence for understanding database schemas
through discovery rather than classification. Uses LLM analysis to understand
business context, discover domains, and generate rich semantic understanding.

Design Philosophy:
- DISCOVERY over classification
- EMERGENCE over predefined categories
- CONTEXT over rules
- INTELLIGENCE over pattern matching

Architecture:
- Uses schema_utils for data preparation and pattern detection
- Focuses on LLM interactions and semantic analysis
- Provides rich business intelligence discovery
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict

from ..core.llm import LLMFactory
from ..core.config import settings
from ..utils.schema_utils import (
    prepare_analysis_context,
    extract_business_keywords,
    detect_naming_patterns,
    detect_potential_relationships,
    calculate_basic_quality_score,
    smart_sample_data,
    prepare_table_metadata,
    generate_cache_key,
)
from ..utils.logging import get_logger


@dataclass
class BusinessDomain:
    """Discovered business domain (not predefined)."""

    name: str
    description: str
    confidence: float
    evidence: List[str]
    related_processes: List[str]
    criticality: str  # "core", "supporting", "peripheral"


@dataclass
class TableBusinessContext:
    """Rich business context for a table."""

    primary_purpose: str
    business_role: str
    data_characteristics: str
    user_personas: List[str]
    business_processes: List[str]
    criticality_assessment: str
    optimization_opportunities: List[str]
    regulatory_considerations: List[str]


@dataclass
class DatabaseIntelligence:
    """Comprehensive database intelligence."""

    business_purpose: str
    industry_domain: str
    discovered_domains: List[BusinessDomain]
    business_architecture: Dict[str, Any]
    data_flow_patterns: List[str]
    critical_entities: List[str]
    compliance_requirements: List[str]
    optimization_priorities: List[str]


@dataclass
class AnalysisResult:
    """Complete analysis result with confidence metrics."""

    database_intelligence: DatabaseIntelligence
    table_contexts: Dict[str, TableBusinessContext]
    semantic_relationships: List[Dict[str, Any]]
    business_workflow_map: Dict[str, List[str]]
    confidence_metrics: Dict[str, float]
    analysis_metadata: Dict[str, Any]


class AnalysisLevel(Enum):
    """Analysis depth levels."""

    QUICK = "quick"  # Fast, surface-level analysis
    STANDARD = "standard"  # Balanced depth and speed
    DEEP = "deep"  # Comprehensive analysis
    RESEARCH = "research"  # Maximum depth for research


class SchemaAnalyzer:
    """LLM-powered schema discovery and intelligence engine."""

    def __init__(self):
        self.logger = get_logger("rag.schema_analyzer")

        # Initialize LLM provider
        self._llm_provider = None

        # Analysis configuration
        self.config = {
            "default_analysis_level": AnalysisLevel.STANDARD,
            "max_tables_per_batch": 10,
            "confidence_threshold": 0.7,
            "enable_caching": True,
            "cache_ttl_hours": 24,
            "temperature": 0.1,  # Low temperature for consistent analysis
            "max_tokens": 4000,
        }

        # Analysis cache
        self._analysis_cache: Dict[str, Tuple[Any, datetime]] = {}
        self._initialize_llm()

        # Initialize prompt templates
        self._initialize_prompt_templates()

    def _initialize_llm(self) -> None:
        """Initialize LLM provider for analysis."""
        try:
            if settings.enable_business_intelligence:
                self._llm_provider = LLMFactory.create_provider(
                    settings.effective_llm_provider,
                    temperature=self.config["temperature"],
                )
                self.logger.info(
                    "llm_provider_initialized", provider=settings.effective_llm_provider
                )
            else:
                self.logger.warning("business_intelligence_disabled")
        except Exception as e:
            self.logger.error("llm_initialization_failed", error=str(e))
            self._llm_provider = None

    def _initialize_prompt_templates(self) -> None:
        """Initialize prompt templates for different analysis types."""
        self.prompts = {
            "database_discovery": {
                "system": """You are a senior data architect and business analyst. Your task is to analyze database schemas and discover their business purpose through intelligent observation.

ANALYSIS APPROACH:
1. OBSERVE the actual schema structure, table names, relationships
2. INFER business purpose from patterns and naming
3. DISCOVER business domains organically (don't use predefined categories)
4. UNDERSTAND how data flows through business processes
5. ASSESS business criticality and optimization opportunities

RESPONSE FORMAT: Structured JSON with confidence scores and evidence.""",
                "user_template": """Analyze this database schema and discover its business intelligence:

DATABASE CONTEXT:
{database_context}

BUSINESS KEYWORDS IDENTIFIED:
{business_keywords}

NAMING PATTERNS DETECTED:
{naming_patterns}

RELATIONSHIP PATTERNS:
{relationship_patterns}

KEY TABLE SAMPLE:
{key_tables_analysis}

DISCOVER AND PROVIDE (JSON format):
{{
  "business_purpose": "What does this database system do?",
  "industry_domain": "What industry/sector does this serve?",
  "discovered_domains": [
    {{
      "name": "domain_name",
      "description": "domain description", 
      "confidence": 0.85,
      "evidence": ["evidence1", "evidence2"],
      "related_processes": ["process1", "process2"],
      "criticality": "core|supporting|peripheral"
    }}
  ],
  "business_architecture": {{"how domains work together": "description"}},
  "critical_entities": ["entity1", "entity2"],
  "data_flow_patterns": ["pattern1", "pattern2"],
  "compliance_requirements": ["requirement1", "requirement2"],
  "optimization_priorities": ["priority1", "priority2"]
}}""",
            },
            "table_analysis": {
    "system": """You are a business analyst specializing in understanding data from a business perspective. Analyze tables to discover their role in business operations.

ANALYSIS APPROACH:
1. OBSERVE the table structure, columns, and data patterns
2. INFER business purpose from naming and relationships
3. DISCOVER business processes and user personas
4. ASSESS criticality and optimization opportunities

RESPONSE FORMAT: Structured JSON with confidence scores and evidence.""",
    
    "user_template": """Analyze this table and discover its business context:

TABLE ANALYSIS CONTEXT:
{table_context}

BUSINESS KEYWORDS:
{business_keywords}

NAMING PATTERNS:
{naming_patterns}

DATA QUALITY ASSESSMENT:
{quality_metrics}

DISCOVER AND PROVIDE (JSON format):
{{
  "primary_purpose": "What business function does this serve?",
  "business_role": "How does this fit in business operations?",
  "data_characteristics": "What kind of business data is this?",
  "user_personas": ["role1", "role2"],
  "business_processes": ["process1", "process2"],
  "criticality": "Critical|High|Medium|Low",
  "optimization_opportunities": ["opportunity1", "opportunity2"],
  "regulatory_considerations": ["consideration1", "consideration2"]
}}""",
            },
            "relationship_discovery": {
                "system": """You are a business process analyst. Analyze table relationships to understand business workflows and data dependencies.

FOCUS ON:
- Business process flows
- Data dependencies
- Workflow sequences
- Business rules implied by relationships
- Process bottlenecks or optimization opportunities""",
                "user_template": """Analyze these relationships to discover business workflows:

DETECTED RELATIONSHIPS:
{relationships}

TABLE BUSINESS CONTEXTS:
{table_contexts}

DISCOVER (JSON format):
{{
  "business_workflows": {{"workflow_name": ["table1", "table2", "table3"]}},
  "data_flow_patterns": ["pattern1", "pattern2"],
  "process_dependencies": ["dependency1", "dependency2"],
  "business_rules": ["rule1", "rule2"],
  "workflow_optimization": ["opportunity1", "opportunity2"]
}}""",
            },
        }

    # ==================== MAIN ANALYSIS METHODS ====================

    async def analyze_database_intelligence(
        self,
        schema_data: Dict[str, Any],
        analysis_level: AnalysisLevel = AnalysisLevel.STANDARD,
    ) -> AnalysisResult:
        """
        Comprehensive database intelligence analysis using utilities foundation.
        """
        try:
            self.logger.info(
                "database_intelligence_analysis_start",
                database=schema_data.get("database_name"),
                analysis_level=analysis_level.value,
                table_count=len(schema_data.get("tables", [])),
            )

            if not self._llm_provider:
                return self._create_fallback_analysis(schema_data)

            # Phase 1: Database-level discovery using utilities
            database_intelligence = await self._discover_database_purpose(schema_data)

            # Phase 2: Table-level analysis using utilities
            table_contexts = await self._analyze_table_contexts(
                schema_data.get("tables", []), analysis_level
            )

            # Phase 3: Relationship discovery using utilities
            semantic_relationships = await self._discover_semantic_relationships(
                schema_data, table_contexts
            )

            # Phase 4: Business workflow mapping
            workflow_map = self._map_business_workflows(
                database_intelligence, table_contexts, semantic_relationships
            )

            # Phase 5: Generate confidence metrics
            confidence_metrics = self._calculate_confidence_metrics(
                database_intelligence, table_contexts, semantic_relationships
            )

            # Create comprehensive result
            result = AnalysisResult(
                database_intelligence=database_intelligence,
                table_contexts=table_contexts,
                semantic_relationships=semantic_relationships,
                business_workflow_map=workflow_map,
                confidence_metrics=confidence_metrics,
                analysis_metadata={
                    "analysis_level": analysis_level.value,
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                    "llm_provider": settings.effective_llm_provider,
                    "table_count": len(schema_data.get("tables", [])),
                    "relationship_count": len(semantic_relationships),
                    "utilities_used": True,
                },
            )

            self.logger.info(
                "database_intelligence_analysis_complete",
                database=schema_data.get("database_name"),
                discovered_domains=len(database_intelligence.discovered_domains),
                analyzed_tables=len(table_contexts),
                avg_confidence=confidence_metrics.get("overall_confidence", 0),
            )

            return result

        except Exception as e:
            self.logger.error(
                "database_intelligence_analysis_failed",
                database=schema_data.get("database_name"),
                error=str(e),
                exc_info=True,
            )
            return self._create_error_analysis(schema_data, str(e))

    async def _discover_database_purpose(
        self, schema_data: Dict[str, Any]
    ) -> DatabaseIntelligence:
        """Discover database purpose using utilities for data preparation."""
        try:
            tables = schema_data.get("tables", [])

            # Use utilities to prepare comprehensive context
            database_context = self._prepare_database_context(tables)
            business_keywords = self._extract_database_keywords(tables)
            naming_patterns = self._analyze_database_naming_patterns(tables)
            relationship_patterns = self._analyze_relationship_patterns(tables)
            key_tables_analysis = await self._analyze_key_tables(tables)

            # Create discovery prompt using prepared data
            prompt_data = {
                "database_context": database_context,
                "business_keywords": ", ".join(business_keywords),
                "naming_patterns": json.dumps(naming_patterns, indent=2),
                "relationship_patterns": relationship_patterns,
                "key_tables_analysis": key_tables_analysis,
            }

            # Execute LLM analysis
            discovery_result = await self._execute_llm_analysis(
                "database_discovery", prompt_data
            )

            # Parse and structure result
            return self._parse_database_intelligence(discovery_result)

        except Exception as e:
            self.logger.error("discover_database_purpose_failed", error=str(e))
            return self._create_minimal_database_intelligence(schema_data)

    async def _analyze_table_contexts(
        self, tables: List[Dict[str, Any]], analysis_level: AnalysisLevel
    ) -> Dict[str, TableBusinessContext]:
        """Analyze table contexts using utilities for data preparation."""
        table_contexts = {}

        # Process tables in batches
        batch_size = self.config["max_tables_per_batch"]

        for i in range(0, len(tables), batch_size):
            batch = tables[i : i + batch_size]
            batch_results = await self._process_table_batch(batch, analysis_level)
            table_contexts.update(batch_results)

        return table_contexts

    async def _process_table_batch(
        self, tables: List[Dict[str, Any]], analysis_level: AnalysisLevel
    ) -> Dict[str, TableBusinessContext]:
        """Process table batch using utilities."""
        batch_results = {}

        # Create analysis tasks
        for table in tables:
            try:
                table_name = table.get("name", "")
                context = await self._analyze_single_table_context(
                    table, analysis_level
                )
                batch_results[table_name] = context
            except Exception as e:
                self.logger.warning(
                    "table_context_analysis_failed",
                    table=table.get("name"),
                    error=str(e),
                )
                batch_results[table.get("name", "")] = (
                    self._create_minimal_table_context(table.get("name", ""))
                )

        return batch_results

    async def _analyze_single_table_context(
        self, table_data: Dict[str, Any], analysis_level: AnalysisLevel
    ) -> TableBusinessContext:
        """Analyze single table using utilities for preparation."""
        try:
            table_name = table_data.get("name", "")

            # Use utilities to prepare analysis context
            table_context = prepare_analysis_context(
                table_data,
                table_data.get("sample_data", {}),
                [],  # related_tables - could be enhanced
            )

            # Extract business keywords using utilities
            business_keywords = extract_business_keywords(
                table_name, table_data.get("columns", [])
            )

            # Detect naming patterns using utilities
            naming_patterns = detect_naming_patterns(
                table_name, table_data.get("columns", [])
            )

            # Calculate quality metrics using utilities
            quality_metrics = calculate_basic_quality_score(
                table_data.get("sample_data", {})
            )

            # Create analysis prompt using utility-prepared data
            prompt_data = {
                "table_context": json.dumps(table_context, indent=2, default=str),
                "business_keywords": ", ".join(business_keywords),
                "naming_patterns": json.dumps(naming_patterns, indent=2),
                "quality_metrics": json.dumps(quality_metrics, indent=2),
            }

            # Execute LLM analysis
            analysis_result = await self._execute_llm_analysis(
                "table_analysis", prompt_data
            )

            # Parse result
            return self._parse_table_business_context(analysis_result)

        except Exception as e:
            self.logger.error(
                "analyze_single_table_context_failed",
                table=table_data.get("name"),
                error=str(e),
            )
            return self._create_minimal_table_context(table_data.get("name", ""))

    async def _discover_semantic_relationships(
        self,
        schema_data: Dict[str, Any],
        table_contexts: Dict[str, TableBusinessContext],
    ) -> List[Dict[str, Any]]:
        """Discover semantic relationships using utilities."""
        try:
            tables = schema_data.get("tables", [])
            all_relationships = []

            # Use utilities to detect potential relationships
            for i, source_table in enumerate(tables):
                target_tables = tables[i + 1 :]  # Avoid duplicate pairs

                relationships = detect_potential_relationships(
                    source_table, target_tables
                )
                all_relationships.extend(relationships)

            # Enhance with business context analysis
            enhanced_relationships = await self._enhance_relationships_with_context(
                all_relationships, table_contexts
            )

            return enhanced_relationships

        except Exception as e:
            self.logger.error("discover_semantic_relationships_failed", error=str(e))
            return []

    # ==================== UTILITY-BASED HELPER METHODS ====================

    def _prepare_database_context(self, tables: List[Dict[str, Any]]) -> str:
        """Prepare database context using utility functions."""
        context_parts = []

        # Use utility to prepare metadata for each table
        for table in tables[:10]:  # Limit for prompt size
            metadata = prepare_table_metadata(table)
            context_parts.append(
                f"- {metadata['name']}: {metadata['column_count']} columns, "
                f"{metadata['relationship_count']} relationships, "
                f"~{metadata['estimated_rows']} rows"
            )

        return "\n".join(context_parts)

    def _extract_database_keywords(self, tables: List[Dict[str, Any]]) -> List[str]:
        """Extract database-level keywords using utilities."""
        all_keywords = set()

        for table in tables:
            table_keywords = extract_business_keywords(
                table.get("name", ""), table.get("columns", [])
            )
            all_keywords.update(table_keywords)

        # Return most common keywords
        return list(all_keywords)[:20]

    def _analyze_database_naming_patterns(
        self, tables: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze database-wide naming patterns using utilities."""
        all_patterns = []

        for table in tables:
            patterns = detect_naming_patterns(
                table.get("name", ""), table.get("columns", [])
            )
            all_patterns.append(patterns)

        # Aggregate patterns
        return {
            "table_count": len(tables),
            "sample_patterns": all_patterns[:3],  # First 3 as sample
            "consistency_trend": "mixed",  # Could be calculated from patterns
        }

    def _analyze_relationship_patterns(self, tables: List[Dict[str, Any]]) -> str:
        """Analyze relationship patterns using utilities."""
        total_relationships = 0
        relationship_descriptions = []

        for table in tables:
            foreign_keys = table.get("foreign_keys", [])
            total_relationships += len(foreign_keys)

            for fk in foreign_keys[:2]:  # Sample first 2
                rel_desc = f"{table.get('name', '')}.{fk.get('column', '')} -> {fk.get('references_table', '')}"
                relationship_descriptions.append(rel_desc)

        density = (
            "high"
            if total_relationships > len(tables) * 1.5
            else "medium" if total_relationships > len(tables) * 0.5 else "low"
        )

        return f"Relationship density: {density}. Sample relationships: {'; '.join(relationship_descriptions[:5])}"

    async def _analyze_key_tables(self, tables: List[Dict[str, Any]]) -> str:
        """Analyze key tables using utility patterns."""
        key_table_analysis = []

        for table in tables[:5]:  # Analyze top 5 tables
            table_name = table.get("name", "")
            columns = table.get("columns", [])

            # Use utility to extract business keywords
            keywords = extract_business_keywords(table_name, columns)

            # Use utility to prepare metadata
            metadata = prepare_table_metadata(table)

            analysis = (
                f"Table: {table_name} | "
                f"Columns: {metadata['column_count']} | "
                f"Relationships: {metadata['relationship_count']} | "
                f"Keywords: {', '.join(keywords[:3])}"
            )
            key_table_analysis.append(analysis)

        return "\n".join(key_table_analysis)

    async def _enhance_relationships_with_context(
        self,
        relationships: List[Dict[str, Any]],
        table_contexts: Dict[str, TableBusinessContext],
    ) -> List[Dict[str, Any]]:
        """Enhance relationships with business context."""
        enhanced_relationships = []

        for rel in relationships:
            source_table = rel.get("source_table", "")
            target_table = rel.get("target_table", "")

            # Add business context if available
            enhanced_rel = rel.copy()

            if source_table in table_contexts and target_table in table_contexts:
                source_context = table_contexts[source_table]
                target_context = table_contexts[target_table]

                # Find common business processes
                common_processes = set(source_context.business_processes) & set(
                    target_context.business_processes
                )

                if common_processes:
                    enhanced_rel["business_context"] = {
                        "common_processes": list(common_processes),
                        "business_meaning": f"{source_table} and {target_table} both support: {', '.join(common_processes)}",
                    }

            enhanced_relationships.append(enhanced_rel)

        return enhanced_relationships

    def _map_business_workflows(
        self,
        database_intelligence: DatabaseIntelligence,
        table_contexts: Dict[str, TableBusinessContext],
        relationships: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """Map business workflows from discovered intelligence."""
        workflow_map = defaultdict(list)

        # Group tables by business processes
        for table_name, context in table_contexts.items():
            for process in context.business_processes:
                workflow_map[process].append(table_name)

        # Convert to regular dict and limit workflow size
        return {
            process: tables[:10]  # Limit to 10 tables per workflow
            for process, tables in workflow_map.items()
            if len(tables) > 1  # Only include workflows with multiple tables
        }

    # ==================== LLM INTERACTION METHODS ====================

    async def _execute_llm_analysis(
        self, analysis_type: str, prompt_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute LLM analysis with error handling."""
        try:
            # Get prompt template
            prompt_config = self.prompts[analysis_type]

            # Format prompts
            system_prompt = prompt_config["system"]
            user_prompt = prompt_config["user_template"].format(**prompt_data)

            # Execute LLM call
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = await self._llm_provider.generate(messages)

            # Parse JSON response
            return self._parse_llm_response(response)

        except Exception as e:
            self.logger.error(
                "execute_llm_analysis_failed", analysis_type=analysis_type, error=str(e)
            )
            return {"error": str(e), "analysis_type": analysis_type}

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response with error handling."""
        try:
            # Clean response text
            response_clean = response.strip()

            print(response_clean)

            # Handle markdown code blocks
            if "```json" in response_clean:
                start = response_clean.find("```json") + 7
                end = response_clean.find("```", start)
                response_clean = response_clean[start:end].strip()
            elif "```" in response_clean:
                start = response_clean.find("```") + 3
                end = response_clean.rfind("```")
                response_clean = response_clean[start:end].strip()

            # Parse JSON
            parsed_result = json.loads(response_clean)
            return parsed_result

        except json.JSONDecodeError as e:
            self.logger.warning("llm_response_json_parse_failed", error=str(e))
            return {
                "error": f"JSON parse failed: {str(e)}",
                "raw_response": response[:500],
            }
        except Exception as e:
            self.logger.error("parse_llm_response_failed", error=str(e))
            return {"error": str(e), "raw_response": response[:500]}

    # ==================== RESULT PARSING METHODS ====================

    def _parse_database_intelligence(
        self, analysis_result: Dict[str, Any]
    ) -> DatabaseIntelligence:
        """Parse database intelligence from LLM analysis."""
        try:
            # Extract discovered domains
            discovered_domains = []
            domains_data = analysis_result.get("discovered_domains", [])

            if isinstance(domains_data, list):
                for domain_info in domains_data:
                    if isinstance(domain_info, dict):
                        domain = BusinessDomain(
                            name=domain_info.get("name", "unknown"),
                            description=domain_info.get("description", ""),
                            confidence=domain_info.get("confidence", 0.5),
                            evidence=domain_info.get("evidence", []),
                            related_processes=domain_info.get("related_processes", []),
                            criticality=domain_info.get("criticality", "supporting"),
                        )
                        discovered_domains.append(domain)

            return DatabaseIntelligence(
                business_purpose=analysis_result.get(
                    "business_purpose", "Database system"
                ),
                industry_domain=analysis_result.get("industry_domain", "General"),
                discovered_domains=discovered_domains,
                business_architecture=analysis_result.get("business_architecture", {}),
                data_flow_patterns=analysis_result.get("data_flow_patterns", []),
                critical_entities=analysis_result.get("critical_entities", []),
                compliance_requirements=analysis_result.get(
                    "compliance_requirements", []
                ),
                optimization_priorities=analysis_result.get(
                    "optimization_priorities", []
                ),
            )

        except Exception as e:
            self.logger.error("parse_database_intelligence_failed", error=str(e))
            return self._create_minimal_database_intelligence({})

    def _parse_table_business_context(
        self, analysis_result: Dict[str, Any]
    ) -> TableBusinessContext:
        """Parse table business context from LLM analysis."""
        try:
            return TableBusinessContext(
                primary_purpose=analysis_result.get("primary_purpose", "Data storage"),
                business_role=analysis_result.get("business_role", "Supporting"),
                data_characteristics=analysis_result.get(
                    "data_characteristics", "Mixed data"
                ),
                user_personas=analysis_result.get("user_personas", []),
                business_processes=analysis_result.get("business_processes", []),
                criticality_assessment=analysis_result.get("criticality", "Medium"),
                optimization_opportunities=analysis_result.get(
                    "optimization_opportunities", []
                ),
                regulatory_considerations=analysis_result.get(
                    "regulatory_considerations", []
                ),
            )

        except Exception as e:
            self.logger.error("parse_table_business_context_failed", error=str(e))
            return self._create_minimal_table_context("unknown")

    def _calculate_confidence_metrics(
        self,
        database_intelligence: DatabaseIntelligence,
        table_contexts: Dict[str, TableBusinessContext],
        relationships: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculate confidence metrics for the analysis."""
        metrics = {}

        # Database-level confidence
        db_confidence = 0.7  # Base confidence
        if len(database_intelligence.discovered_domains) > 0:
            avg_domain_confidence = sum(
                d.confidence for d in database_intelligence.discovered_domains
            ) / len(database_intelligence.discovered_domains)
            db_confidence = avg_domain_confidence

        metrics["database_confidence"] = db_confidence

        # Table-level confidence
        if table_contexts:
            table_confidences = []
            for context in table_contexts.values():
                table_conf = 0.7  # Base confidence
                if context.primary_purpose != "Data storage":
                    table_conf += 0.1
                if len(context.business_processes) > 0:
                    table_conf += 0.1
                if len(context.user_personas) > 0:
                    table_conf += 0.1
                table_confidences.append(min(table_conf, 1.0))

            metrics["avg_table_confidence"] = sum(table_confidences) / len(
                table_confidences
            )
        else:
            metrics["avg_table_confidence"] = 0.5

        # Relationship confidence
        metrics["relationship_confidence"] = 0.8 if relationships else 0.3

        # Overall confidence
        metrics["overall_confidence"] = (
            metrics["database_confidence"] * 0.4
            + metrics["avg_table_confidence"] * 0.4
            + metrics["relationship_confidence"] * 0.2
        )

        return metrics

    # ==================== FALLBACK AND ERROR HANDLING ====================

    def _create_fallback_analysis(self, schema_data: Dict[str, Any]) -> AnalysisResult:
        """Create fallback analysis using utilities when LLM unavailable."""
        self.logger.warning("creating_fallback_analysis_no_llm")

        tables = schema_data.get("tables", [])

        # Use utilities for fallback analysis
        database_keywords = self._extract_database_keywords(tables)

        # Create minimal database intelligence
        database_intelligence = DatabaseIntelligence(
            business_purpose=f"Database system with {len(tables)} tables (LLM analysis unavailable)",
            industry_domain="General",
            discovered_domains=[],
            business_architecture={},
            data_flow_patterns=[],
            critical_entities=database_keywords[:5],  # Use keywords as entities
            compliance_requirements=[],
            optimization_priorities=[],
        )

        # Create minimal table contexts using utilities
        table_contexts = {}
        for table in tables:
            table_name = table.get("name", "")
            table_contexts[table_name] = self._create_minimal_table_context(table_name)

        return AnalysisResult(
            database_intelligence=database_intelligence,
            table_contexts=table_contexts,
            semantic_relationships=[],
            business_workflow_map={},
            confidence_metrics={"overall_confidence": 0.3},
            analysis_metadata={
                "analysis_type": "fallback",
                "llm_available": False,
                "utilities_used": True,
                "analysis_timestamp": datetime.utcnow().isoformat(),
            },
        )

    def _create_error_analysis(
        self, schema_data: Dict[str, Any], error: str
    ) -> AnalysisResult:
        """Create error analysis result."""
        return AnalysisResult(
            database_intelligence=self._create_minimal_database_intelligence(
                schema_data
            ),
            table_contexts={},
            semantic_relationships=[],
            business_workflow_map={},
            confidence_metrics={"overall_confidence": 0.1, "error": error},
            analysis_metadata={
                "analysis_type": "error",
                "error": error,
                "utilities_used": True,
                "analysis_timestamp": datetime.utcnow().isoformat(),
            },
        )

    def _create_minimal_database_intelligence(
        self, schema_data: Dict[str, Any]
    ) -> DatabaseIntelligence:
        """Create minimal database intelligence using utilities."""
        tables = schema_data.get("tables", [])

        # Use utilities to extract some basic intelligence
        keywords = self._extract_database_keywords(tables) if tables else []

        return DatabaseIntelligence(
            business_purpose="Database system (detailed analysis unavailable)",
            industry_domain="General",
            discovered_domains=[],
            business_architecture={},
            data_flow_patterns=[],
            critical_entities=keywords[:5],  # Use extracted keywords
            compliance_requirements=[],
            optimization_priorities=[],
        )

    def _create_minimal_table_context(self, table_name: str) -> TableBusinessContext:
        """Create minimal table context for fallback cases."""
        return TableBusinessContext(
            primary_purpose=f"Data storage table: {table_name}",
            business_role="Supporting",
            data_characteristics="Mixed data types",
            user_personas=["System users"],
            business_processes=["Data management"],
            criticality_assessment="Medium",
            optimization_opportunities=[],
            regulatory_considerations=[],
        )

    # ==================== PUBLIC API METHODS ====================

    async def quick_table_analysis(
        self, table_data: Dict[str, Any]
    ) -> TableBusinessContext:
        """Quick analysis of a single table using utilities."""
        return await self._analyze_single_table_context(table_data, AnalysisLevel.QUICK)

    async def discover_business_domains(
        self, schema_data: Dict[str, Any]
    ) -> List[BusinessDomain]:
        """Discover business domains in the database."""
        analysis_result = await self.analyze_database_intelligence(
            schema_data, AnalysisLevel.STANDARD
        )
        return analysis_result.database_intelligence.discovered_domains

    async def analyze_business_workflows(
        self, schema_data: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Analyze business workflows from schema."""
        analysis_result = await self.analyze_database_intelligence(
            schema_data, AnalysisLevel.STANDARD
        )
        return analysis_result.business_workflow_map

    def get_analysis_summary(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """Get a summary of analysis results."""
        return {
            "business_purpose": analysis_result.database_intelligence.business_purpose,
            "industry_domain": analysis_result.database_intelligence.industry_domain,
            "discovered_domains_count": len(
                analysis_result.database_intelligence.discovered_domains
            ),
            "analyzed_tables_count": len(analysis_result.table_contexts),
            "relationship_count": len(analysis_result.semantic_relationships),
            "workflow_count": len(analysis_result.business_workflow_map),
            "overall_confidence": analysis_result.confidence_metrics.get(
                "overall_confidence", 0
            ),
            "analysis_timestamp": analysis_result.analysis_metadata.get(
                "analysis_timestamp"
            ),
            "llm_provider": analysis_result.analysis_metadata.get("llm_provider"),
            "utilities_used": analysis_result.analysis_metadata.get(
                "utilities_used", False
            ),
        }

    def export_for_vector_storage(
        self, analysis_result: AnalysisResult
    ) -> Dict[str, Any]:
        """Export analysis results in format suitable for vector storage."""
        vector_data = {}

        # Database-level context
        db_intel = analysis_result.database_intelligence
        vector_data["database_context"] = {
            "business_purpose": db_intel.business_purpose,
            "industry_domain": db_intel.industry_domain,
            "discovered_domains": [
                {
                    "name": domain.name,
                    "description": domain.description,
                    "confidence": domain.confidence,
                    "evidence": domain.evidence,
                    "processes": domain.related_processes,
                }
                for domain in db_intel.discovered_domains
            ],
            "critical_entities": db_intel.critical_entities,
            "data_flow_patterns": db_intel.data_flow_patterns,
        }

        # Table-level contexts for vector storage
        vector_data["table_contexts"] = {}
        for table_name, context in analysis_result.table_contexts.items():

            # Create rich description for vector embedding
            rich_description = (
                f"{context.primary_purpose}. "
                f"Business role: {context.business_role}. "
                f"Used by: {', '.join(context.user_personas) if context.user_personas else 'system users'}. "
                f"Supports business processes: {', '.join(context.business_processes) if context.business_processes else 'data operations'}. "
                f"Criticality: {context.criticality_assessment}. "
                f"Data characteristics: {context.data_characteristics}."
            )

            # Create business keywords string for search
            # FIX: Ensure all items are strings before joining
            business_keywords_list = [
                *context.business_processes,
                *context.user_personas,
                context.business_role,
                context.criticality_assessment,
            ]
            business_keywords_str = ",".join(
                str(item) for item in business_keywords_list if item
            )

            vector_data["table_contexts"][table_name] = {
                "primary_purpose": context.primary_purpose,
                "business_role": context.business_role,
                "business_processes": context.business_processes,
                "user_personas": context.user_personas,
                "criticality": context.criticality_assessment,
                "rich_description": rich_description,
                "business_keywords_str": business_keywords_str,  # String version for vector storage
                "optimization_opportunities": context.optimization_opportunities,
                "regulatory_considerations": context.regulatory_considerations,
            }

        # Semantic relationships
        vector_data["semantic_relationships"] = analysis_result.semantic_relationships

        # Business workflows
        vector_data["business_workflows"] = analysis_result.business_workflow_map

        # Analysis metadata
        vector_data["analysis_metadata"] = {
            "confidence_metrics": analysis_result.confidence_metrics,
            "analysis_timestamp": analysis_result.analysis_metadata.get(
                "analysis_timestamp"
            ),
            "llm_provider": analysis_result.analysis_metadata.get("llm_provider"),
            "utilities_used": analysis_result.analysis_metadata.get(
                "utilities_used", False
            ),
        }

        return vector_data

    async def enhance_existing_schema(
        self, existing_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance existing schema data with intelligence using utilities."""
        try:
            # Run full analysis using utilities
            analysis_result = await self.analyze_database_intelligence(existing_schema)

            # Enhance schema with intelligence
            enhanced_schema = existing_schema.copy()

            # Add database-level intelligence
            enhanced_schema["business_intelligence"] = {
                "business_purpose": analysis_result.database_intelligence.business_purpose,
                "industry_domain": analysis_result.database_intelligence.industry_domain,
                "discovered_domains": [
                    asdict(domain)
                    for domain in analysis_result.database_intelligence.discovered_domains
                ],
                "confidence_metrics": analysis_result.confidence_metrics,
                "data_flow_patterns": analysis_result.database_intelligence.data_flow_patterns,
                "critical_entities": analysis_result.database_intelligence.critical_entities,
            }

            # Enhance each table with business context
            enhanced_tables = []
            for table in enhanced_schema.get("tables", []):
                table_name = table.get("name", "")

                if table_name in analysis_result.table_contexts:
                    context = analysis_result.table_contexts[table_name]

                    # Add business intelligence to table
                    table["business_context"] = asdict(context)

                    # Create rich description for vector storage (string format)
                    table["rich_description"] = (
                        f"{context.primary_purpose}. Business role: {context.business_role}. "
                        f"Used by {', '.join(context.user_personas) if context.user_personas else 'system users'}. "
                        f"Supports {', '.join(context.business_processes) if context.business_processes else 'data operations'}. "
                        f"Criticality: {context.criticality_assessment}. "
                        f"Data characteristics: {context.data_characteristics}."
                    )

                    # Add business keywords for search (string format)
                    # FIX: Ensure all items are strings before joining
                    business_keywords = [
                        *context.business_processes,
                        *context.user_personas,
                        context.business_role,
                        context.criticality_assessment,
                    ]
                    # Convert all items to strings and filter out None/empty values
                    table["business_keywords_str"] = ",".join(
                        str(item) for item in business_keywords if item
                    )

                    # Also keep array versions for API compatibility
                    table["business_concepts"] = context.business_processes
                    table["semantic_tags"] = context.user_personas

                enhanced_tables.append(table)

            enhanced_schema["tables"] = enhanced_tables

            # Add analysis metadata
            enhanced_schema["intelligence_metadata"] = analysis_result.analysis_metadata

            self.logger.info(
                "schema_enhancement_complete",
                table_count=len(enhanced_tables),
                domains_discovered=len(
                    analysis_result.database_intelligence.discovered_domains
                ),
            )

            return enhanced_schema

        except Exception as e:
            self.logger.error("enhance_existing_schema_failed", error=str(e))
            return existing_schema

    def get_config(self) -> Dict[str, Any]:
        """Get current analyzer configuration."""
        return {
            **self.config,
            "llm_available": self._llm_provider is not None,
            "llm_provider": (
                settings.effective_llm_provider if self._llm_provider else None
            ),
            "utilities_integrated": True,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the analyzer."""
        health_status = {
            "status": "healthy",
            "llm_available": self._llm_provider is not None,
            "llm_provider": (
                settings.effective_llm_provider if self._llm_provider else None
            ),
            "cache_size": len(self._analysis_cache),
            "utilities_integrated": True,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Test LLM connectivity
        if self._llm_provider:
            try:
                test_messages = [
                    {"role": "system", "content": "You are a test assistant."},
                    {
                        "role": "user",
                        "content": "Respond with 'OK' to confirm connectivity.",
                    },
                ]
                response = await self._llm_provider.generate(test_messages)
                health_status["llm_test"] = "passed"
                health_status["llm_response_preview"] = (
                    response[:50] if response else "empty"
                )
            except Exception as e:
                health_status["llm_test"] = f"failed: {str(e)}"
                health_status["status"] = "degraded"
        else:
            health_status["llm_test"] = "not_available"
            health_status["status"] = "limited"

        # Test utilities integration
        try:
            test_keywords = extract_business_keywords(
                "test_table", ["id", "name", "email"]
            )
            health_status["utilities_test"] = "passed"
            health_status["sample_keywords"] = test_keywords[:3]
        except Exception as e:
            health_status["utilities_test"] = f"failed: {str(e)}"
            health_status["status"] = "degraded"

        return health_status


# Global schema analyzer instance
schema_analyzer = SchemaAnalyzer()
