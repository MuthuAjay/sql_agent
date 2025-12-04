"""Transaction fraud pattern detection."""

from typing import Dict, List, Any, Optional
from datetime import datetime
from ..models import FraudScenario, RiskLevel, FraudCategory


class TransactionFraudDetector:
    """Detect transaction-based fraud patterns."""

    async def detect_fraud_patterns(
        self,
        table_name: str,
        table_schema: Dict[str, Any],
        db_manager
    ) -> List[FraudScenario]:
        """Detect transaction fraud patterns."""
        scenarios = []

        # Check for duplicate transactions
        dup_scenario = await self._detect_duplicate_transactions(
            table_name, table_schema, db_manager
        )
        if dup_scenario:
            scenarios.append(dup_scenario)

        # Check for round amount fraud
        round_scenario = await self._detect_round_amounts(
            table_name, table_schema, db_manager
        )
        if round_scenario:
            scenarios.append(round_scenario)

        # Check for velocity fraud
        velocity_scenario = await self._detect_velocity_fraud(
            table_name, table_schema, db_manager
        )
        if velocity_scenario:
            scenarios.append(velocity_scenario)

        return scenarios

    async def _detect_duplicate_transactions(
        self,
        table_name: str,
        table_schema: Dict[str, Any],
        db_manager
    ) -> Optional[FraudScenario]:
        """Detect duplicate transactions."""
        columns = table_schema.get("columns", [])

        # Find amount and timestamp columns
        amount_col = self._find_column(columns, ["amount", "price", "total", "value"])
        timestamp_col = self._find_column(columns, ["timestamp", "created_at", "date", "time"])

        if not amount_col or not timestamp_col:
            return None

        sql = f"""
            SELECT {amount_col}, {timestamp_col}, COUNT(*) as occurrences
            FROM {table_name}
            GROUP BY {amount_col}, {timestamp_col}
            HAVING COUNT(*) > 1
            LIMIT 5;
        """

        try:
            result = await db_manager.execute_query(sql, timeout=30)
            if result.row_count > 0:
                return FraudScenario(
                    scenario_id="duplicate_transactions_001",
                    category=FraudCategory.TRANSACTION_ANOMALY,
                    risk_level=RiskLevel.HIGH,
                    title="Duplicate Transactions Detected",
                    description=f"Found {result.row_count} sets of duplicate transactions with identical amounts and timestamps",
                    reasoning="Duplicate transactions may indicate replay attacks, system errors, or fraudulent double-charging",
                    affected_columns=[amount_col, timestamp_col],
                    detection_sql=sql,
                    prevention_recommendations=[
                        "Add unique constraint on transaction_id",
                        "Implement idempotency keys for transaction processing",
                        "Add duplicate detection before transaction commit"
                    ],
                    detection_difficulty="easy",
                    impact_severity="high",
                    likelihood=0.8,
                    real_world_examples=[
                        "2019: Major retailer lost $2M due to duplicate payment processing bug",
                        "Payment fraud ring exploited duplicate transaction vulnerabilities"
                    ],
                    compliance_violations=["PCI-DSS: Transaction integrity"],
                    confidence_score=0.9
                )
        except Exception as e:
            return None

        return None

    async def _detect_round_amounts(
        self,
        table_name: str,
        table_schema: Dict[str, Any],
        db_manager
    ) -> Optional[FraudScenario]:
        """Detect excessive round amount transactions."""
        columns = table_schema.get("columns", [])
        amount_col = self._find_column(columns, ["amount", "price", "total", "value"])

        if not amount_col:
            return None

        sql = f"""
            SELECT
                COUNT(*) as total_transactions,
                SUM(CASE WHEN {amount_col}::numeric % 100 = 0 THEN 1 ELSE 0 END) as round_100,
                SUM(CASE WHEN {amount_col}::numeric % 1000 = 0 THEN 1 ELSE 0 END) as round_1000,
                (SUM(CASE WHEN {amount_col}::numeric % 100 = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as round_percentage
            FROM {table_name}
            WHERE {amount_col} IS NOT NULL;
        """

        try:
            result = await db_manager.execute_query(sql, timeout=30)
            if result.row_count > 0:
                data = result.data[0]
                round_pct = float(data.get("round_percentage", 0))

                if round_pct > 20:  # More than 20% are round numbers
                    return FraudScenario(
                        scenario_id="round_amount_fraud_001",
                        category=FraudCategory.TRANSACTION_ANOMALY,
                        risk_level=RiskLevel.MEDIUM if round_pct < 40 else RiskLevel.HIGH,
                        title="Suspicious Round Amount Pattern",
                        description=f"{round_pct:.1f}% of transactions are round numbers, which is unusually high",
                        reasoning="Legitimate transactions tend to have irregular amounts; fraudsters often use round numbers",
                        affected_columns=[amount_col],
                        detection_sql=sql,
                        prevention_recommendations=[
                            "Monitor and flag transactions with round amounts for review",
                            "Implement risk scoring that increases for round amounts",
                            "Require additional verification for high-value round transactions"
                        ],
                        detection_difficulty="easy",
                        impact_severity="medium",
                        likelihood=0.6,
                        real_world_examples=[
                            "Employee expense fraud often uses round numbers ($500, $1000)",
                            "Money laundering operations frequently use round amounts"
                        ],
                        compliance_violations=["AML: Unusual transaction patterns"],
                        confidence_score=0.85
                    )
        except Exception:
            return None

        return None

    async def _detect_velocity_fraud(
        self,
        table_name: str,
        table_schema: Dict[str, Any],
        db_manager
    ) -> Optional[FraudScenario]:
        """Detect high velocity fraud patterns."""
        columns = table_schema.get("columns", [])

        user_col = self._find_column(columns, ["user_id", "customer_id", "account_id"])
        timestamp_col = self._find_column(columns, ["timestamp", "created_at", "date"])

        if not user_col or not timestamp_col:
            return None

        sql = f"""
            WITH velocity AS (
                SELECT {user_col},
                       COUNT(*) as tx_count,
                       EXTRACT(EPOCH FROM (MAX({timestamp_col}) - MIN({timestamp_col})))/60 as time_window_minutes
                FROM {table_name}
                WHERE {timestamp_col} >= NOW() - INTERVAL '1 hour'
                GROUP BY {user_col}
                HAVING COUNT(*) > 10
            )
            SELECT * FROM velocity
            WHERE time_window_minutes < 60
            ORDER BY tx_count DESC
            LIMIT 5;
        """

        try:
            result = await db_manager.execute_query(sql, timeout=30)
            if result.row_count > 0:
                return FraudScenario(
                    scenario_id="velocity_fraud_001",
                    category=FraudCategory.TRANSACTION_ANOMALY,
                    risk_level=RiskLevel.HIGH,
                    title="High Transaction Velocity Detected",
                    description=f"Found {result.row_count} users with abnormally high transaction frequency",
                    reasoning="Rapid-fire transactions often indicate compromised accounts, bots, or automated fraud",
                    affected_columns=[user_col, timestamp_col],
                    detection_sql=sql,
                    prevention_recommendations=[
                        "Implement rate limiting per user",
                        "Add velocity checks before transaction approval",
                        "Require CAPTCHA or 2FA for high-velocity patterns",
                        "Set maximum transactions per time window"
                    ],
                    detection_difficulty="medium",
                    impact_severity="high",
                    likelihood=0.75,
                    real_world_examples=[
                        "2020: Card testing bots hit e-commerce sites with 100+ transactions/minute",
                        "Account takeover attacks show 50+ transactions in 5 minutes"
                    ],
                    compliance_violations=["PCI-DSS: Velocity controls", "SOX: Transaction monitoring"],
                    confidence_score=0.9
                )
        except Exception:
            return None

        return None

    def _find_column(self, columns: List[str], keywords: List[str]) -> Optional[str]:
        """Find a column matching keywords."""
        for col in columns:
            col_lower = col.lower()
            for keyword in keywords:
                if keyword in col_lower:
                    return col
        return None


transaction_fraud_detector = TransactionFraudDetector()
