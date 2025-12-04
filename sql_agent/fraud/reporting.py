"""Fraud analysis report generation and formatting."""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from .models import FraudAnalysisReport, RiskLevel, FraudScenario


class FraudReportGenerator:
    """Generate formatted fraud analysis reports."""

    def generate_text_report(self, report: FraudAnalysisReport) -> str:
        """Generate a human-readable text report."""
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("FRAUD ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append(f"Table: {report.table_name}")
        lines.append(f"Database: {report.database_name}")
        lines.append(f"Analysis Date: {report.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Analysis Depth: {report.analysis_depth}")
        lines.append("")

        # Executive Summary
        lines.append("-" * 80)
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Overall Risk Score: {report.overall_risk_score:.2f} / 1.00")
        lines.append(f"Total Fraud Scenarios Identified: {report.total_scenarios}")
        lines.append(f"  - Critical: {report.critical_count}")
        lines.append(f"  - High: {report.high_risk_count}")
        lines.append(f"  - Medium: {report.medium_risk_count}")
        lines.append(f"  - Low: {report.low_risk_count}")
        lines.append(f"Schema Vulnerabilities: {len(report.schema_vulnerabilities)}")
        lines.append(f"Data Quality Issues: {len(report.data_quality_issues)}")
        lines.append(f"Statistical Anomalies: {len(report.statistical_anomalies)}")
        lines.append("")

        # Immediate Actions
        if report.recommended_immediate_actions:
            lines.append("-" * 80)
            lines.append("RECOMMENDED IMMEDIATE ACTIONS")
            lines.append("-" * 80)
            for i, action in enumerate(report.recommended_immediate_actions, 1):
                lines.append(f"{i}. {action}")
            lines.append("")

        # Critical & High Risk Scenarios
        high_priority = report.get_high_priority_items()

        if high_priority["critical_scenarios"]:
            lines.append("-" * 80)
            lines.append("CRITICAL RISK SCENARIOS")
            lines.append("-" * 80)
            for scenario in high_priority["critical_scenarios"]:
                lines.extend(self._format_scenario(scenario))
            lines.append("")

        if high_priority["high_risk_scenarios"]:
            lines.append("-" * 80)
            lines.append("HIGH RISK SCENARIOS")
            lines.append("-" * 80)
            for scenario in high_priority["high_risk_scenarios"]:
                lines.extend(self._format_scenario(scenario))
            lines.append("")

        # Schema Vulnerabilities
        if report.schema_vulnerabilities:
            lines.append("-" * 80)
            lines.append("SCHEMA VULNERABILITIES")
            lines.append("-" * 80)
            for vuln in report.schema_vulnerabilities:
                lines.append(f"• {vuln.vulnerability_type.upper()} - {vuln.severity.value.upper()}")
                lines.append(f"  Description: {vuln.description}")
                if vuln.affected_columns:
                    lines.append(f"  Affected Columns: {', '.join(vuln.affected_columns)}")
                lines.append(f"  Remediation: {vuln.remediation}")
                if vuln.sql_fix:
                    lines.append(f"  SQL Fix: {vuln.sql_fix}")
                lines.append("")

        # Data Quality Issues
        if report.data_quality_issues:
            lines.append("-" * 80)
            lines.append("DATA QUALITY ISSUES")
            lines.append("-" * 80)
            for issue in report.data_quality_issues:
                lines.append(f"• {issue.issue_type.upper()} - {issue.severity.value.upper()}")
                lines.append(f"  Description: {issue.description}")
                if issue.affected_rows_estimate:
                    lines.append(f"  Estimated Affected Rows: {issue.affected_rows_estimate:,}")
                lines.append(f"  Impact: {issue.impact}")
                lines.append(f"  Remediation: {issue.remediation}")
                lines.append("")

        # All Fraud Scenarios (grouped by category)
        lines.append("-" * 80)
        lines.append("ALL FRAUD SCENARIOS")
        lines.append("-" * 80)

        scenarios_by_category = self._group_scenarios_by_category(report.fraud_scenarios)
        for category, scenarios in scenarios_by_category.items():
            lines.append(f"\n{category.upper()}")
            lines.append("-" * 40)
            for scenario in scenarios:
                lines.extend(self._format_scenario(scenario, compact=True))

        # Long-term Recommendations
        if report.long_term_recommendations:
            lines.append("-" * 80)
            lines.append("LONG-TERM RECOMMENDATIONS")
            lines.append("-" * 80)
            for i, rec in enumerate(report.long_term_recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        # Footer
        lines.append("=" * 80)
        lines.append(f"Report generated in {report.estimated_analysis_time:.2f} seconds")
        lines.append(f"LLM Confidence: {report.llm_confidence:.2%}")
        lines.append("=" * 80)

        return "\n".join(lines)

    def _format_scenario(self, scenario: FraudScenario, compact: bool = False) -> List[str]:
        """Format a fraud scenario for text output."""
        lines = []

        if compact:
            lines.append(f"• {scenario.title} ({scenario.risk_level.value.upper()})")
            lines.append(f"  {scenario.description[:150]}...")
        else:
            lines.append(f"Title: {scenario.title}")
            lines.append(f"Risk Level: {scenario.risk_level.value.upper()}")
            lines.append(f"Category: {scenario.category.value}")
            lines.append(f"Description: {scenario.description}")
            lines.append(f"Reasoning: {scenario.reasoning}")

            if scenario.affected_columns:
                lines.append(f"Affected Columns: {', '.join(scenario.affected_columns)}")

            lines.append(f"Detection Difficulty: {scenario.detection_difficulty}")
            lines.append(f"Likelihood: {scenario.likelihood:.0%}")
            lines.append(f"Impact Severity: {scenario.impact_severity}")

            if scenario.detection_sql:
                lines.append(f"Detection SQL:")
                lines.append(f"  {scenario.detection_sql.strip()}")

            if scenario.prevention_recommendations:
                lines.append("Prevention Recommendations:")
                for rec in scenario.prevention_recommendations:
                    lines.append(f"  - {rec}")

            if scenario.real_world_examples:
                lines.append("Real-world Examples:")
                for ex in scenario.real_world_examples:
                    lines.append(f"  - {ex}")

        lines.append("")
        return lines

    def _group_scenarios_by_category(self, scenarios: List[FraudScenario]) -> Dict[str, List[FraudScenario]]:
        """Group scenarios by category."""
        grouped = {}
        for scenario in scenarios:
            category = scenario.category.value
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(scenario)
        return grouped

    def generate_json_report(self, report: FraudAnalysisReport) -> str:
        """Generate JSON report."""
        return report.model_dump_json(indent=2)

    def generate_summary(self, report: FraudAnalysisReport) -> Dict[str, Any]:
        """Generate executive summary."""
        return {
            "table_name": report.table_name,
            "database_name": report.database_name,
            "analysis_timestamp": report.analysis_timestamp.isoformat(),
            "overall_risk_score": report.overall_risk_score,
            "risk_category": self._get_risk_category(report.overall_risk_score),
            "total_scenarios": report.total_scenarios,
            "risk_breakdown": {
                "critical": report.critical_count,
                "high": report.high_risk_count,
                "medium": report.medium_risk_count,
                "low": report.low_risk_count
            },
            "vulnerabilities_count": len(report.schema_vulnerabilities),
            "data_quality_issues_count": len(report.data_quality_issues),
            "top_3_risks": self._get_top_risks(report),
            "immediate_actions_count": len(report.recommended_immediate_actions),
            "analysis_time_seconds": report.estimated_analysis_time
        }

    def _get_risk_category(self, risk_score: float) -> str:
        """Convert risk score to category."""
        if risk_score >= 0.8:
            return "CRITICAL"
        elif risk_score >= 0.6:
            return "HIGH"
        elif risk_score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_top_risks(self, report: FraudAnalysisReport) -> List[Dict[str, str]]:
        """Get top 3 highest risks."""
        all_scenarios = sorted(
            report.fraud_scenarios,
            key=lambda x: (
                4 if x.risk_level == RiskLevel.CRITICAL else
                3 if x.risk_level == RiskLevel.HIGH else
                2 if x.risk_level == RiskLevel.MEDIUM else 1,
                x.likelihood
            ),
            reverse=True
        )

        return [
            {
                "title": s.title,
                "risk_level": s.risk_level.value,
                "category": s.category.value,
                "likelihood": f"{s.likelihood:.0%}"
            }
            for s in all_scenarios[:3]
        ]

    def generate_detection_sql_file(self, report: FraudAnalysisReport) -> str:
        """Generate a SQL file with all detection queries."""
        lines = []

        lines.append("-- Fraud Detection SQL Queries")
        lines.append(f"-- Table: {report.table_name}")
        lines.append(f"-- Generated: {report.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        for i, scenario in enumerate(report.fraud_scenarios, 1):
            if scenario.detection_sql:
                lines.append(f"-- Scenario {i}: {scenario.title}")
                lines.append(f"-- Risk Level: {scenario.risk_level.value.upper()}")
                lines.append(f"-- Category: {scenario.category.value}")
                lines.append(scenario.detection_sql.strip())
                lines.append("")
                lines.append("")

        return "\n".join(lines)

    def generate_html_report(self, report: FraudAnalysisReport) -> str:
        """Generate HTML report."""
        risk_color = {
            RiskLevel.CRITICAL: "#dc3545",
            RiskLevel.HIGH: "#fd7e14",
            RiskLevel.MEDIUM: "#ffc107",
            RiskLevel.LOW: "#28a745"
        }

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Fraud Analysis Report - {report.table_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
        .summary {{ background: #e7f3ff; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .risk-score {{ font-size: 48px; font-weight: bold; color: {self._get_risk_color(report.overall_risk_score)}; }}
        .scenario {{ background: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #ddd; }}
        .critical {{ border-left-color: {risk_color[RiskLevel.CRITICAL]}; }}
        .high {{ border-left-color: {risk_color[RiskLevel.HIGH]}; }}
        .medium {{ border-left-color: {risk_color[RiskLevel.MEDIUM]}; }}
        .low {{ border-left-color: {risk_color[RiskLevel.LOW]}; }}
        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 3px; color: white; font-size: 12px; margin-right: 5px; }}
        pre {{ background: #f4f4f4; padding: 10px; border-radius: 3px; overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #007bff; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Fraud Analysis Report</h1>

        <div class="summary">
            <table>
                <tr>
                    <td><strong>Table:</strong> {report.table_name}</td>
                    <td><strong>Database:</strong> {report.database_name}</td>
                </tr>
                <tr>
                    <td><strong>Analysis Date:</strong> {report.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td>
                    <td><strong>Analysis Depth:</strong> {report.analysis_depth}</td>
                </tr>
                <tr>
                    <td colspan="2">
                        <strong>Overall Risk Score:</strong>
                        <span class="risk-score">{report.overall_risk_score:.2f}</span> / 1.00
                    </td>
                </tr>
            </table>
        </div>

        <h2>Risk Breakdown</h2>
        <table>
            <tr>
                <th>Risk Level</th>
                <th>Count</th>
            </tr>
            <tr>
                <td><span class="badge" style="background: {risk_color[RiskLevel.CRITICAL]}">Critical</span></td>
                <td>{report.critical_count}</td>
            </tr>
            <tr>
                <td><span class="badge" style="background: {risk_color[RiskLevel.HIGH]}">High</span></td>
                <td>{report.high_risk_count}</td>
            </tr>
            <tr>
                <td><span class="badge" style="background: {risk_color[RiskLevel.MEDIUM]}">Medium</span></td>
                <td>{report.medium_risk_count}</td>
            </tr>
            <tr>
                <td><span class="badge" style="background: {risk_color[RiskLevel.LOW]}">Low</span></td>
                <td>{report.low_risk_count}</td>
            </tr>
        </table>
"""

        # Add scenarios
        if report.fraud_scenarios:
            html += "\n        <h2>Fraud Scenarios</h2>\n"
            for scenario in sorted(report.fraud_scenarios,
                                 key=lambda x: (4 if x.risk_level == RiskLevel.CRITICAL else
                                              3 if x.risk_level == RiskLevel.HIGH else
                                              2 if x.risk_level == RiskLevel.MEDIUM else 1),
                                 reverse=True):
                risk_class = scenario.risk_level.value
                html += f"""
        <div class="scenario {risk_class}">
            <h3>{scenario.title} <span class="badge" style="background: {risk_color[scenario.risk_level]}">{scenario.risk_level.value.upper()}</span></h3>
            <p><strong>Category:</strong> {scenario.category.value}</p>
            <p><strong>Description:</strong> {scenario.description}</p>
            <p><strong>Reasoning:</strong> {scenario.reasoning}</p>
"""
                if scenario.detection_sql:
                    html += f"            <p><strong>Detection SQL:</strong></p>\n"
                    html += f"            <pre>{scenario.detection_sql}</pre>\n"

                html += "        </div>\n"

        html += """
    </div>
</body>
</html>"""

        return html

    def _get_risk_color(self, risk_score: float) -> str:
        """Get color for risk score."""
        if risk_score >= 0.8:
            return "#dc3545"  # red
        elif risk_score >= 0.6:
            return "#fd7e14"  # orange
        elif risk_score >= 0.4:
            return "#ffc107"  # yellow
        else:
            return "#28a745"  # green


report_generator = FraudReportGenerator()
