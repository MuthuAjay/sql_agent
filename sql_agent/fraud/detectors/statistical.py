"""Statistical anomaly detection for fraud."""

from typing import Dict, List, Any, Optional
from ..models import StatisticalAnomaly, RiskLevel
import math


class StatisticalAnomalyDetector:
    """Detect statistical anomalies in data."""

    async def detect_anomalies(
        self,
        table_name: str,
        table_schema: Dict[str, Any],
        db_manager
    ) -> List[StatisticalAnomaly]:
        """Detect statistical anomalies."""
        anomalies = []

        # Benford's Law check
        benford = await self._check_benfords_law(table_name, table_schema, db_manager)
        if benford:
            anomalies.append(benford)

        # Outlier detection
        outliers = await self._detect_outliers(table_name, table_schema, db_manager)
        anomalies.extend(outliers)

        return anomalies

    async def _check_benfords_law(
        self,
        table_name: str,
        table_schema: Dict[str, Any],
        db_manager
    ) -> Optional[StatisticalAnomaly]:
        """Check if numeric data follows Benford's Law."""
        columns = table_schema.get("columns", [])
        column_details = table_schema.get("column_details", {})

        # Find numeric columns
        numeric_cols = [
            col for col, details in column_details.items()
            if any(t in details.get("type", "").lower() for t in ["numeric", "decimal", "integer", "money"])
            and any(k in col.lower() for k in ["amount", "price", "total", "value"])
        ]

        if not numeric_cols:
            return None

        amount_col = numeric_cols[0]

        sql = f"""
            WITH first_digits AS (
                SELECT LEFT(CAST(ABS({amount_col}) AS TEXT), 1) as digit,
                       COUNT(*) as frequency
                FROM {table_name}
                WHERE {amount_col} > 0 AND {amount_col} IS NOT NULL
                GROUP BY LEFT(CAST(ABS({amount_col}) AS TEXT), 1)
            )
            SELECT digit,
                   frequency,
                   frequency * 100.0 / SUM(frequency) OVER () as percentage
            FROM first_digits
            WHERE digit BETWEEN '1' AND '9'
            ORDER BY digit;
        """

        try:
            result = await db_manager.execute_query(sql, timeout=30)
            if result.row_count >= 7:  # Need at least 7 digits for meaningful test
                # Expected Benford's Law distribution
                benford_expected = {
                    '1': 30.1, '2': 17.6, '3': 12.5, '4': 9.7, '5': 7.9,
                    '6': 6.7, '7': 5.8, '8': 5.1, '9': 4.6
                }

                # Calculate chi-square statistic
                chi_square = 0
                total_count = sum(row.get("frequency", 0) for row in result.data)

                for row in result.data:
                    digit = row.get("digit")
                    observed = row.get("frequency", 0)
                    expected = benford_expected.get(digit, 0) * total_count / 100

                    if expected > 0:
                        chi_square += ((observed - expected) ** 2) / expected

                # Chi-square critical value for 8 degrees of freedom at p=0.05 is ~15.51
                p_value = self._chi_square_to_p_value(chi_square, 8)

                if chi_square > 15.51:  # Significant deviation
                    return StatisticalAnomaly(
                        anomaly_id=f"{table_name}_{amount_col}_benford",
                        anomaly_type="benfords_law_violation",
                        severity=RiskLevel.MEDIUM if chi_square < 25 else RiskLevel.HIGH,
                        description=f"Column '{amount_col}' violates Benford's Law (χ²={chi_square:.2f})",
                        affected_column=amount_col,
                        statistical_test="Chi-square test for Benford's Law",
                        test_statistic=chi_square,
                        p_value=p_value,
                        threshold=15.51,
                        sample_values=[],
                        detection_sql=sql
                    )
        except Exception:
            return None

        return None

    async def _detect_outliers(
        self,
        table_name: str,
        table_schema: Dict[str, Any],
        db_manager
    ) -> List[StatisticalAnomaly]:
        """Detect statistical outliers using Z-score."""
        anomalies = []
        columns = table_schema.get("columns", [])
        column_details = table_schema.get("column_details", {})

        # Find numeric columns
        numeric_cols = [
            col for col, details in column_details.items()
            if any(t in details.get("type", "").lower() for t in ["numeric", "decimal", "integer", "money"])
            and any(k in col.lower() for k in ["amount", "price", "total", "value", "quantity"])
        ]

        for col in numeric_cols[:3]:  # Check top 3 numeric columns
            sql = f"""
                WITH stats AS (
                    SELECT AVG({col}) as mean,
                           STDDEV({col}) as stddev,
                           COUNT(*) as total_count
                    FROM {table_name}
                    WHERE {col} IS NOT NULL
                ),
                outliers AS (
                    SELECT COUNT(*) as outlier_count,
                           MAX(ABS(t.{col} - s.mean) / NULLIF(s.stddev, 0)) as max_z_score,
                           MIN(t.{col}) as min_value,
                           MAX(t.{col}) as max_value
                    FROM {table_name} t, stats s
                    WHERE ABS(t.{col} - s.mean) / NULLIF(s.stddev, 0) > 3
                      AND s.stddev > 0
                )
                SELECT o.*, s.total_count,
                       (o.outlier_count * 100.0 / s.total_count) as outlier_percentage
                FROM outliers o, stats s
                WHERE o.outlier_count > 0;
            """

            try:
                result = await db_manager.execute_query(sql, timeout=30)
                if result.row_count > 0:
                    data = result.data[0]
                    outlier_count = data.get("outlier_count", 0)
                    outlier_pct = data.get("outlier_percentage", 0)

                    if outlier_count > 5 and outlier_pct > 1:  # More than 1% outliers
                        anomalies.append(StatisticalAnomaly(
                            anomaly_id=f"{table_name}_{col}_outliers",
                            anomaly_type="statistical_outliers",
                            severity=RiskLevel.MEDIUM if outlier_pct < 5 else RiskLevel.HIGH,
                            description=f"Found {outlier_count} statistical outliers in '{col}' ({outlier_pct:.2f}%)",
                            affected_column=col,
                            statistical_test="Z-score outlier detection",
                            test_statistic=data.get("max_z_score"),
                            p_value=None,
                            threshold=3.0,
                            sample_values=[data.get("min_value"), data.get("max_value")],
                            detection_sql=sql
                        ))
            except Exception:
                continue

        return anomalies

    def _chi_square_to_p_value(self, chi_square: float, df: int) -> float:
        """Approximate p-value from chi-square statistic."""
        # Simplified approximation
        if chi_square > 30:
            return 0.001
        elif chi_square > 20:
            return 0.01
        elif chi_square > 15.51:
            return 0.05
        else:
            return 0.10


statistical_anomaly_detector = StatisticalAnomalyDetector()
