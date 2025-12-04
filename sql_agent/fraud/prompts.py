"""LLM prompts for fraud detection."""

from typing import Dict, List, Any, Optional


class FraudDetectionPrompts:
    """Prompt templates for fraud detection with LLM."""

    @staticmethod
    def get_fraud_scenario_generation_prompt(
        table_name: str,
        table_schema: Dict[str, Any],
        business_context: Optional[Dict[str, Any]] = None,
        detected_patterns: Optional[List[Dict]] = None,
        industry: Optional[str] = None
    ) -> str:
        """Generate prompt for LLM to identify fraud scenarios."""

        columns_info = []
        for col_name, col_detail in table_schema.get("column_details", {}).items():
            col_type = col_detail.get("type", "unknown")
            nullable = col_detail.get("nullable", True)
            columns_info.append(f"  - {col_name} ({col_type}, {'NULL' if nullable else 'NOT NULL'})")

        columns_str = "\n".join(columns_info) if columns_info else "  No column details available"

        business_context_str = ""
        if business_context:
            domains = business_context.get("business_domains", [])
            purpose = business_context.get("business_purpose", "")
            if domains or purpose:
                business_context_str = f"\n\nBusiness Context:\n- Purpose: {purpose}\n- Domains: {', '.join(domains)}"

        detected_patterns_str = ""
        if detected_patterns:
            pattern_list = [f"  - {p.get('title', 'Unknown')}: {p.get('risk_level', 'medium')} risk"
                          for p in detected_patterns[:5]]
            detected_patterns_str = f"\n\nDetected Pattern Types:\n" + "\n".join(pattern_list)

        industry_str = f"\n\nIndustry Context: {industry}" if industry else ""

        return f"""You are an expert fraud analyst and database security specialist. Analyze the following database table and identify ALL possible fraud scenarios that could occur.

Table: {table_name}

Schema:
{columns_str}
{business_context_str}
{industry_str}
{detected_patterns_str}

Your task:
1. Analyze the table structure and identify fraud vulnerabilities
2. Consider the business context and typical fraud patterns in this domain
3. Think creatively about how fraudsters might exploit this table
4. For each fraud scenario, provide:
   - A clear title
   - Detailed description
   - Risk level (low, medium, high, critical)
   - Why this is a fraud risk (reasoning)
   - Which columns are involved
   - How to detect it (SQL query if possible)
   - How to prevent it

Consider these fraud categories:
- Transaction Anomalies (duplicates, round amounts, velocity, structuring)
- Data Quality Risks (missing audit fields, orphaned records, nulls)
- Temporal Anomalies (backdated records, future dates, timestamp tampering)
- Relationship Integrity (circular references, orphaned data)
- Schema Vulnerabilities (missing constraints, no primary keys)
- Statistical Anomalies (Benford's Law violations, outliers)
- Access Pattern Risks (off-hours activity, privilege escalation)

Response format (JSON):
{{
    "fraud_scenarios": [
        {{
            "title": "Clear, specific title",
            "category": "transaction_anomaly|data_quality_risk|temporal_anomaly|relationship_integrity|schema_vulnerability|statistical_anomaly|access_pattern_risk",
            "risk_level": "low|medium|high|critical",
            "description": "Detailed description of the fraud scenario",
            "reasoning": "Why this is a fraud risk for this specific table",
            "affected_columns": ["column1", "column2"],
            "detection_sql": "SELECT ... FROM {table_name} WHERE ...",
            "prevention_recommendations": ["recommendation1", "recommendation2"],
            "likelihood": 0.7,
            "impact_severity": "low|medium|high|critical",
            "real_world_examples": ["Example 1", "Example 2"]
        }}
    ],
    "overall_assessment": "Brief overall risk assessment",
    "confidence": 0.85
}}

Generate at least 10-15 unique fraud scenarios. Be specific to this table structure and business context."""

    @staticmethod
    def get_schema_vulnerability_prompt(
        table_name: str,
        table_schema: Dict[str, Any]
    ) -> str:
        """Generate prompt for schema vulnerability analysis."""

        has_pk = bool(table_schema.get("primary_keys"))
        has_fk = bool(table_schema.get("foreign_keys"))
        has_indexes = bool(table_schema.get("indexes"))

        columns_info = []
        for col_name, col_detail in table_schema.get("column_details", {}).items():
            nullable = "NULL" if col_detail.get("nullable", True) else "NOT NULL"
            has_default = col_detail.get("default") is not None
            columns_info.append(f"  - {col_name}: {col_detail.get('type')} {nullable} {'(has default)' if has_default else ''}")

        return f"""Analyze this database table schema for security vulnerabilities that could enable fraud:

Table: {table_name}

Schema Details:
- Primary Key: {'Yes' if has_pk else 'NO - VULNERABLE'}
- Foreign Keys: {'Yes' if has_fk else 'NO - VULNERABLE'}
- Indexes: {'Yes' if has_indexes else 'NO - MAY IMPACT DETECTION'}

Columns:
{chr(10).join(columns_info)}

Identify schema-level vulnerabilities:
1. Missing constraints that enable fraud
2. Overly permissive nullable fields
3. Missing audit columns (created_at, created_by, updated_at, updated_by)
4. Weak data types that allow invalid data
5. Missing indexes on fraud-detection columns

Response format (JSON):
{{
    "vulnerabilities": [
        {{
            "vulnerability_type": "missing_primary_key|missing_foreign_key|missing_audit_fields|overly_permissive_nulls|weak_data_types",
            "severity": "low|medium|high|critical",
            "description": "What is vulnerable",
            "affected_columns": ["column1"],
            "remediation": "How to fix it",
            "sql_fix": "ALTER TABLE ... SQL statement",
            "exploitability": "How easily this can be exploited"
        }}
    ],
    "risk_score": 0.75,
    "confidence": 0.9
}}"""

    @staticmethod
    def get_business_context_prompt(
        table_name: str,
        columns: List[str],
        sample_data: Optional[List[Dict]] = None
    ) -> str:
        """Generate prompt to understand table's business purpose."""

        sample_str = ""
        if sample_data and len(sample_data) > 0:
            sample_str = "\n\nSample data (first 3 rows):\n"
            for i, row in enumerate(sample_data[:3], 1):
                sample_str += f"Row {i}: {row}\n"

        return f"""Analyze this database table and determine its business purpose and fraud risk profile:

Table: {table_name}
Columns: {', '.join(columns)}
{sample_str}

Determine:
1. What business entity/process does this table represent?
2. What industry/domain is this likely from?
3. What types of fraud are most common for this business entity?
4. What are the critical columns that need protection?
5. What compliance requirements might apply (PCI-DSS, SOX, GDPR, HIPAA)?

Response format (JSON):
{{
    "business_entity": "What this table represents (e.g., financial_transactions, customer_orders)",
    "industry": "finance|healthcare|retail|e-commerce|generic",
    "business_purpose": "Brief description of purpose",
    "critical_columns": ["column1", "column2"],
    "fraud_risk_profile": "high|medium|low",
    "common_fraud_types": ["fraud type 1", "fraud type 2"],
    "compliance_requirements": ["PCI-DSS", "SOX"],
    "confidence": 0.8
}}"""

    @staticmethod
    def get_data_quality_analysis_prompt(
        table_name: str,
        quality_stats: Dict[str, Any]
    ) -> str:
        """Generate prompt for data quality issue analysis."""

        stats_str = "\n".join([f"  - {k}: {v}" for k, v in quality_stats.items()])

        return f"""Analyze these data quality statistics and identify issues that could indicate or enable fraud:

Table: {table_name}

Quality Statistics:
{stats_str}

Identify data quality issues:
1. Missing critical data (NULL values where shouldn't be)
2. Data inconsistencies (e.g., updated_at < created_at)
3. Orphaned records (foreign keys pointing to non-existent records)
4. Suspicious patterns (too many duplicates, unrealistic values)
5. Incomplete audit trails

Response format (JSON):
{{
    "data_quality_issues": [
        {{
            "issue_type": "missing_data|inconsistent_data|orphaned_records|suspicious_patterns|incomplete_audit",
            "severity": "low|medium|high|critical",
            "description": "What is the issue",
            "affected_columns": ["column1"],
            "affected_rows_estimate": 1000,
            "example_sql": "SQL query to find examples",
            "impact": "How this enables or indicates fraud",
            "remediation": "How to fix"
        }}
    ],
    "overall_quality_score": 0.65,
    "confidence": 0.85
}}"""

    @staticmethod
    def get_fraud_scenario_refinement_prompt(
        scenarios: List[Dict],
        table_schema: Dict[str, Any]
    ) -> str:
        """Refine and enhance fraud scenarios."""

        scenarios_str = "\n".join([
            f"{i+1}. {s.get('title', 'Unknown')}: {s.get('description', '')[:100]}..."
            for i, s in enumerate(scenarios[:10])
        ])

        return f"""Review and enhance these fraud scenarios for completeness and accuracy:

Table Schema: {table_schema.get('name')}
Columns: {', '.join(table_schema.get('columns', []))}

Initial Scenarios:
{scenarios_str}

For each scenario:
1. Verify it's technically feasible given the schema
2. Add specific SQL detection queries where missing
3. Enhance prevention recommendations
4. Add real-world examples if applicable
5. Adjust risk levels based on likelihood and impact
6. Remove duplicate or overlapping scenarios

Response format (JSON):
{{
    "refined_scenarios": [
        {{
            "title": "...",
            "category": "...",
            "risk_level": "...",
            "description": "Enhanced description",
            "reasoning": "Refined reasoning",
            "affected_columns": [...],
            "detection_sql": "Verified SQL query",
            "prevention_recommendations": ["Enhanced recommendations"],
            "detection_difficulty": "easy|medium|hard",
            "likelihood": 0.7,
            "impact_severity": "low|medium|high|critical",
            "real_world_examples": ["Example 1"]
        }}
    ],
    "scenarios_removed": ["Reason for removal"],
    "confidence": 0.9
}}"""

    @staticmethod
    def get_system_prompt() -> str:
        """Get system prompt for fraud detection."""
        return """You are an expert fraud analyst and database security specialist with deep knowledge of:
- Financial fraud patterns (transaction fraud, money laundering, structuring)
- Data manipulation techniques (backdating, tampering, deletion)
- Statistical fraud detection (Benford's Law, outlier detection, distribution analysis)
- Compliance requirements (SOX, PCI-DSS, GDPR, HIPAA, AML/KYC)
- Database security best practices
- Real-world fraud case studies

Your role is to:
1. Identify ALL possible fraud scenarios for database tables
2. Provide specific, actionable detection methods
3. Explain risks in both technical and business terms
4. Recommend practical prevention measures
5. Consider industry-specific fraud patterns

Always respond in valid JSON format as specified in the prompts.
Be thorough, specific, and realistic about fraud risks."""


fraud_prompts = FraudDetectionPrompts()
