"""Temporal anomaly detection for fraud."""

from typing import Dict, List, Any, Optional
from ..models import FraudScenario, RiskLevel, FraudCategory


class TemporalAnomalyDetector:
    """Detect temporal fraud patterns and anomalies."""

    async def detect_temporal_anomalies(
        self,
        table_name: str,
        table_schema: Dict[str, Any],
        db_manager
    ) -> List[FraudScenario]:
        """Detect temporal anomalies."""
        scenarios = []

        # Detect backdated records
        backdated = await self._detect_backdated_records(table_name, table_schema, db_manager)
        if backdated:
            scenarios.append(backdated)

        # Detect future-dated records
        future = await self._detect_future_dated_records(table_name, table_schema, db_manager)
        if future:
            scenarios.append(future)

        # Detect timestamp tampering
        tampering = await self._detect_timestamp_tampering(table_name, table_schema, db_manager)
        if tampering:
            scenarios.append(tampering)

        return scenarios

    async def _detect_backdated_records(
        self,
        table_name: str,
        table_schema: Dict[str, Any],
        db_manager
    ) -> Optional[FraudScenario]:
        """Detect backdated records."""
        columns = table_schema.get("columns", [])

        created_at = self._find_column(columns, ["created_at", "created_date"])
        inserted_at = self._find_column(columns, ["inserted_at", "sys_created"])

        if not created_at:
            return None

        sql = f"""
            SELECT COUNT(*) as backdated_count,
                   MIN({created_at}) as earliest_backdate,
                   AVG(EXTRACT(EPOCH FROM (NOW() - {created_at}))/86400) as avg_days_back
            FROM {table_name}
            WHERE {created_at} < NOW() - INTERVAL '7 days'
              AND {created_at} > NOW() - INTERVAL '1 year'
            HAVING COUNT(*) > 10;
        """

        try:
            result = await db_manager.execute_query(sql, timeout=30)
            if result.row_count > 0:
                data = result.data[0]
                backdated_count = data.get("backdated_count", 0)

                if backdated_count > 10:
                    return FraudScenario(
                        scenario_id="backdated_records_001",
                        category=FraudCategory.TEMPORAL_ANOMALY,
                        risk_level=RiskLevel.HIGH,
                        title="Backdated Records Detected",
                        description=f"Found {backdated_count} records with creation dates in the past",
                        reasoning="Backdating records can hide fraudulent activity, manipulate financial reports, or bypass controls",
                        affected_columns=[created_at],
                        detection_sql=sql,
                        prevention_recommendations=[
                            "Use database triggers to validate timestamps against system time",
                            "Require approval for any backdated entries",
                            "Log all backdated entries for audit review",
                            "Add alerts for records with timestamps > 24 hours old"
                        ],
                        detection_difficulty="easy",
                        impact_severity="high",
                        likelihood=0.7,
                        real_world_examples=[
                            "2015: Enron scandal involved backdating documents to hide losses",
                            "Stock options backdating fraud to increase executive compensation"
                        ],
                        compliance_violations=["SOX: Transaction dating", "SEC: Financial reporting"],
                        confidence_score=0.85
                    )
        except Exception:
            return None

        return None

    async def _detect_future_dated_records(
        self,
        table_name: str,
        table_schema: Dict[str, Any],
        db_manager
    ) -> Optional[FraudScenario]:
        """Detect future-dated records."""
        columns = table_schema.get("columns", [])
        timestamp_col = self._find_column(columns, ["timestamp", "created_at", "date", "scheduled_date"])

        if not timestamp_col:
            return None

        sql = f"""
            SELECT COUNT(*) as future_count,
                   MAX({timestamp_col}) as furthest_future,
                   AVG(EXTRACT(EPOCH FROM ({timestamp_col} - NOW()))/86400) as avg_days_ahead
            FROM {table_name}
            WHERE {timestamp_col} > NOW()
            HAVING COUNT(*) > 5;
        """

        try:
            result = await db_manager.execute_query(sql, timeout=30)
            if result.row_count > 0:
                data = result.data[0]
                future_count = data.get("future_count", 0)

                if future_count > 5:
                    return FraudScenario(
                        scenario_id="future_dated_records_001",
                        category=FraudCategory.TEMPORAL_ANOMALY,
                        risk_level=RiskLevel.MEDIUM,
                        title="Future-Dated Records Detected",
                        description=f"Found {future_count} records with timestamps in the future",
                        reasoning="Future dates may indicate system manipulation, fraudulent scheduling, or timestamp errors",
                        affected_columns=[timestamp_col],
                        detection_sql=sql,
                        prevention_recommendations=[
                            "Add check constraint to prevent future timestamps",
                            "Validate timestamps at application level",
                            "Log and alert on future-dated entries"
                        ],
                        detection_difficulty="easy",
                        impact_severity="medium",
                        likelihood=0.5,
                        real_world_examples=[
                            "Insurance fraud with future-dated policies to cover past events",
                            "Revenue recognition fraud with future-dated sales"
                        ],
                        compliance_violations=["SOX: Accurate dating", "Revenue recognition rules"],
                        confidence_score=0.8
                    )
        except Exception:
            return None

        return None

    async def _detect_timestamp_tampering(
        self,
        table_name: str,
        table_schema: Dict[str, Any],
        db_manager
    ) -> Optional[FraudScenario]:
        """Detect timestamp tampering (updated_at < created_at)."""
        columns = table_schema.get("columns", [])

        created_at = self._find_column(columns, ["created_at", "created_date"])
        updated_at = self._find_column(columns, ["updated_at", "updated_date", "modified_at"])

        if not created_at or not updated_at:
            return None

        sql = f"""
            SELECT COUNT(*) as tampered_count,
                   AVG(EXTRACT(EPOCH FROM ({created_at} - {updated_at}))/3600) as avg_hours_diff
            FROM {table_name}
            WHERE {updated_at} < {created_at}
            HAVING COUNT(*) > 0;
        """

        try:
            result = await db_manager.execute_query(sql, timeout=30)
            if result.row_count > 0:
                data = result.data[0]
                tampered_count = data.get("tampered_count", 0)

                if tampered_count > 0:
                    return FraudScenario(
                        scenario_id="timestamp_tampering_001",
                        category=FraudCategory.TEMPORAL_ANOMALY,
                        risk_level=RiskLevel.CRITICAL,
                        title="Timestamp Tampering Detected",
                        description=f"Found {tampered_count} records where updated_at < created_at (impossible condition)",
                        reasoning="This indicates direct database manipulation, bypassing application controls",
                        affected_columns=[created_at, updated_at],
                        detection_sql=sql,
                        prevention_recommendations=[
                            "Add database check constraint: updated_at >= created_at",
                            "Use triggers to prevent timestamp manipulation",
                            "Implement immutable audit log",
                            "Alert security team immediately on detection"
                        ],
                        detection_difficulty="easy",
                        impact_severity="critical",
                        likelihood=0.9,
                        real_world_examples=[
                            "Insider fraud with direct database access",
                            "SQL injection attacks modifying timestamps",
                            "Privilege escalation to hide fraudulent changes"
                        ],
                        compliance_violations=[
                            "SOX: Data integrity",
                            "GDPR: Audit trail accuracy",
                            "PCI-DSS: System integrity"
                        ],
                        confidence_score=0.95
                    )
        except Exception:
            return None

        return None

    def _find_column(self, columns: List[str], keywords: List[str]) -> Optional[str]:
        """Find column matching keywords."""
        for col in columns:
            col_lower = col.lower()
            for keyword in keywords:
                if keyword in col_lower:
                    return col
        return None


temporal_anomaly_detector = TemporalAnomalyDetector()
