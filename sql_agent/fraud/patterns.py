"""Fraud pattern library with pre-built detection scenarios."""

from typing import Dict, List, Optional
from .models import FraudCategory, RiskLevel, FraudScenario


class FraudPatternLibrary:
    """Library of pre-built fraud detection patterns."""

    def __init__(self):
        self.patterns = self._initialize_patterns()

    def _initialize_patterns(self) -> Dict[str, List[Dict]]:
        """Initialize fraud pattern templates."""
        return {
            "transaction_anomaly": self._get_transaction_patterns(),
            "data_quality_risk": self._get_data_quality_patterns(),
            "temporal_anomaly": self._get_temporal_patterns(),
            "relationship_integrity": self._get_relationship_patterns(),
            "schema_vulnerability": self._get_schema_patterns(),
            "statistical_anomaly": self._get_statistical_patterns()
        }

    def _get_transaction_patterns(self) -> List[Dict]:
        """Transaction fraud patterns."""
        return [
            {
                "id": "duplicate_transactions",
                "title": "Duplicate Transactions",
                "description": "Multiple identical transactions that may indicate fraud or system errors",
                "risk_level": RiskLevel.HIGH,
                "detection_difficulty": "easy",
                "sql_template": """
                    SELECT {amount_col}, {timestamp_col}, COUNT(*) as occurrences
                    FROM {table_name}
                    WHERE {timestamp_col} >= NOW() - INTERVAL '24 hours'
                    GROUP BY {amount_col}, {timestamp_col}
                    HAVING COUNT(*) > 1
                    ORDER BY occurrences DESC;
                """,
                "required_columns": ["amount", "timestamp"],
                "reasoning": "Duplicate transactions at the exact same time suggest automated fraud or system replay attacks"
            },
            {
                "id": "round_amount_fraud",
                "title": "Round Amount Transactions",
                "description": "Excessive round-number transactions (e.g., $1000, $5000) that may indicate manual fraud",
                "risk_level": RiskLevel.MEDIUM,
                "detection_difficulty": "easy",
                "sql_template": """
                    SELECT {amount_col}, COUNT(*) as frequency
                    FROM {table_name}
                    WHERE {amount_col} % 100 = 0 OR {amount_col} % 1000 = 0
                    GROUP BY {amount_col}
                    HAVING COUNT(*) > 10
                    ORDER BY frequency DESC;
                """,
                "required_columns": ["amount"],
                "reasoning": "Fraudsters often use round numbers; legitimate transactions tend to have irregular amounts"
            },
            {
                "id": "velocity_fraud",
                "title": "High Transaction Velocity",
                "description": "Unusually high frequency of transactions in short time period",
                "risk_level": RiskLevel.HIGH,
                "detection_difficulty": "medium",
                "sql_template": """
                    SELECT {user_col}, COUNT(*) as tx_count,
                           MAX({timestamp_col}) - MIN({timestamp_col}) as time_span
                    FROM {table_name}
                    WHERE {timestamp_col} >= NOW() - INTERVAL '1 hour'
                    GROUP BY {user_col}
                    HAVING COUNT(*) > 10
                    ORDER BY tx_count DESC;
                """,
                "required_columns": ["user_id", "timestamp"],
                "reasoning": "Rapid-fire transactions often indicate compromised accounts or automated fraud"
            },
            {
                "id": "split_transactions",
                "title": "Transaction Structuring (Smurfing)",
                "description": "Multiple transactions just below reporting thresholds to avoid detection",
                "risk_level": RiskLevel.CRITICAL,
                "detection_difficulty": "medium",
                "sql_template": """
                    SELECT {user_col}, DATE({timestamp_col}) as date,
                           COUNT(*) as tx_count, SUM({amount_col}) as total_amount
                    FROM {table_name}
                    WHERE {amount_col} BETWEEN 9000 AND 9999
                       OR {amount_col} BETWEEN 4900 AND 4999
                    GROUP BY {user_col}, DATE({timestamp_col})
                    HAVING COUNT(*) >= 3
                    ORDER BY total_amount DESC;
                """,
                "required_columns": ["user_id", "amount", "timestamp"],
                "reasoning": "Breaking large amounts into smaller chunks is a classic money laundering technique"
            },
            {
                "id": "off_hours_transactions",
                "title": "Off-Hours Transaction Pattern",
                "description": "Transactions occurring during unusual business hours",
                "risk_level": RiskLevel.MEDIUM,
                "detection_difficulty": "easy",
                "sql_template": """
                    SELECT {user_col}, {timestamp_col}, {amount_col}
                    FROM {table_name}
                    WHERE EXTRACT(HOUR FROM {timestamp_col}) BETWEEN 0 AND 5
                       OR EXTRACT(HOUR FROM {timestamp_col}) BETWEEN 22 AND 23
                    ORDER BY {timestamp_col} DESC
                    LIMIT 100;
                """,
                "required_columns": ["user_id", "timestamp", "amount"],
                "reasoning": "Legitimate business transactions rarely occur at night; fraudsters often exploit off-hours"
            }
        ]

    def _get_data_quality_patterns(self) -> List[Dict]:
        """Data quality fraud risk patterns."""
        return [
            {
                "id": "missing_audit_fields",
                "title": "Missing Audit Trail Data",
                "description": "Records without proper audit fields enable untraceable fraud",
                "risk_level": RiskLevel.HIGH,
                "detection_difficulty": "easy",
                "sql_template": """
                    SELECT COUNT(*) as records_without_audit,
                           COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {table_name}) as percentage
                    FROM {table_name}
                    WHERE {created_by_col} IS NULL
                       OR {created_at_col} IS NULL
                       OR {updated_by_col} IS NULL;
                """,
                "required_columns": ["created_by", "created_at"],
                "reasoning": "Missing audit fields make it impossible to trace who made changes, enabling insider fraud"
            },
            {
                "id": "orphaned_records",
                "title": "Orphaned Records",
                "description": "Records with no valid foreign key relationships",
                "risk_level": RiskLevel.MEDIUM,
                "detection_difficulty": "easy",
                "sql_template": """
                    SELECT t1.*
                    FROM {table_name} t1
                    LEFT JOIN {parent_table} t2 ON t1.{fk_col} = t2.{pk_col}
                    WHERE t2.{pk_col} IS NULL
                    LIMIT 100;
                """,
                "required_columns": ["foreign_key"],
                "reasoning": "Orphaned records may indicate deleted parent records to hide fraud or data manipulation"
            },
            {
                "id": "null_critical_fields",
                "title": "Null Values in Critical Fields",
                "description": "Missing data in fields that should never be null",
                "risk_level": RiskLevel.MEDIUM,
                "detection_difficulty": "easy",
                "sql_template": """
                    SELECT COUNT(*) as null_count,
                           '{column_name}' as column_name
                    FROM {table_name}
                    WHERE {column_name} IS NULL;
                """,
                "required_columns": [],
                "reasoning": "Null values in critical fields may indicate data manipulation or incomplete fraud attempts"
            }
        ]

    def _get_temporal_patterns(self) -> List[Dict]:
        """Temporal anomaly patterns."""
        return [
            {
                "id": "backdated_records",
                "title": "Backdated Records",
                "description": "Records with created_at dates in the past relative to system time",
                "risk_level": RiskLevel.HIGH,
                "detection_difficulty": "easy",
                "sql_template": """
                    SELECT *,
                           EXTRACT(EPOCH FROM (NOW() - {created_at_col})) / 86400 as days_backdated
                    FROM {table_name}
                    WHERE {created_at_col} < NOW() - INTERVAL '7 days'
                      AND {inserted_at_col} >= NOW() - INTERVAL '1 day'
                    ORDER BY days_backdated DESC
                    LIMIT 100;
                """,
                "required_columns": ["created_at"],
                "reasoning": "Backdating records can hide fraudulent activity or manipulate financial reporting"
            },
            {
                "id": "future_dated_records",
                "title": "Future-Dated Records",
                "description": "Records dated in the future",
                "risk_level": RiskLevel.MEDIUM,
                "detection_difficulty": "easy",
                "sql_template": """
                    SELECT *
                    FROM {table_name}
                    WHERE {timestamp_col} > NOW()
                    ORDER BY {timestamp_col} DESC
                    LIMIT 100;
                """,
                "required_columns": ["timestamp"],
                "reasoning": "Future dates may indicate system manipulation or fraudulent scheduling"
            },
            {
                "id": "timestamp_tampering",
                "title": "Timestamp Tampering",
                "description": "Updated_at timestamp is earlier than created_at timestamp",
                "risk_level": RiskLevel.HIGH,
                "detection_difficulty": "easy",
                "sql_template": """
                    SELECT *
                    FROM {table_name}
                    WHERE {updated_at_col} < {created_at_col}
                    LIMIT 100;
                """,
                "required_columns": ["created_at", "updated_at"],
                "reasoning": "Impossible timestamp relationships indicate direct database manipulation"
            }
        ]

    def _get_relationship_patterns(self) -> List[Dict]:
        """Relationship integrity patterns."""
        return [
            {
                "id": "circular_references",
                "title": "Circular Reference Chains",
                "description": "Records that reference each other in a circular pattern",
                "risk_level": RiskLevel.MEDIUM,
                "detection_difficulty": "hard",
                "sql_template": """
                    WITH RECURSIVE chain AS (
                        SELECT {id_col}, {ref_col}, 1 as depth,
                               ARRAY[{id_col}] as path
                        FROM {table_name}
                        WHERE {ref_col} IS NOT NULL
                        UNION ALL
                        SELECT t.{id_col}, t.{ref_col}, c.depth + 1,
                               c.path || t.{id_col}
                        FROM {table_name} t
                        JOIN chain c ON t.{id_col} = c.{ref_col}
                        WHERE t.{id_col} = ANY(c.path) AND c.depth < 10
                    )
                    SELECT * FROM chain WHERE depth > 1 LIMIT 100;
                """,
                "required_columns": ["id", "reference_id"],
                "reasoning": "Circular references can hide fraudulent chains of transactions or approvals"
            }
        ]

    def _get_schema_patterns(self) -> List[Dict]:
        """Schema vulnerability patterns."""
        return [
            {
                "id": "no_primary_key",
                "title": "Missing Primary Key",
                "description": "Table lacks a primary key constraint",
                "risk_level": RiskLevel.HIGH,
                "detection_difficulty": "easy",
                "reasoning": "Without primary keys, duplicate records can be inserted to inflate counts or hide fraud"
            },
            {
                "id": "no_foreign_keys",
                "title": "Missing Foreign Key Constraints",
                "description": "Table has reference columns but no foreign key constraints",
                "risk_level": RiskLevel.MEDIUM,
                "detection_difficulty": "easy",
                "reasoning": "Missing FK constraints allow orphaned records and data inconsistencies"
            },
            {
                "id": "overly_permissive_nulls",
                "title": "Critical Fields Allow NULL",
                "description": "Important business fields allow NULL values",
                "risk_level": RiskLevel.MEDIUM,
                "detection_difficulty": "easy",
                "reasoning": "Nullable critical fields enable data manipulation and incomplete fraud attempts"
            }
        ]

    def _get_statistical_patterns(self) -> List[Dict]:
        """Statistical anomaly patterns."""
        return [
            {
                "id": "benfords_law_violation",
                "title": "Benford's Law Violation",
                "description": "First digit distribution doesn't follow natural logarithmic distribution",
                "risk_level": RiskLevel.MEDIUM,
                "detection_difficulty": "medium",
                "sql_template": """
                    SELECT LEFT(CAST({amount_col} AS TEXT), 1) as first_digit,
                           COUNT(*) as frequency,
                           COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage
                    FROM {table_name}
                    WHERE {amount_col} > 0
                    GROUP BY LEFT(CAST({amount_col} AS TEXT), 1)
                    ORDER BY first_digit;
                """,
                "required_columns": ["amount"],
                "reasoning": "Natural data follows Benford's Law; deviations suggest manufactured or manipulated data"
            },
            {
                "id": "outlier_detection",
                "title": "Statistical Outliers",
                "description": "Values that are significantly outside normal distribution",
                "risk_level": RiskLevel.MEDIUM,
                "detection_difficulty": "medium",
                "sql_template": """
                    WITH stats AS (
                        SELECT AVG({amount_col}) as mean,
                               STDDEV({amount_col}) as stddev
                        FROM {table_name}
                    )
                    SELECT t.*,
                           ABS(t.{amount_col} - s.mean) / s.stddev as z_score
                    FROM {table_name} t, stats s
                    WHERE ABS(t.{amount_col} - s.mean) / s.stddev > 3
                    ORDER BY z_score DESC
                    LIMIT 100;
                """,
                "required_columns": ["amount"],
                "reasoning": "Extreme outliers often indicate errors or fraudulent transactions"
            }
        ]

    def get_patterns_by_category(self, category: FraudCategory) -> List[Dict]:
        """Get patterns for a specific category."""
        category_key = category.value if isinstance(category, FraudCategory) else category
        return self.patterns.get(category_key, [])

    def get_all_patterns(self) -> List[Dict]:
        """Get all fraud patterns."""
        all_patterns = []
        for patterns in self.patterns.values():
            all_patterns.extend(patterns)
        return all_patterns

    def get_pattern_by_id(self, pattern_id: str) -> Optional[Dict]:
        """Get a specific pattern by ID."""
        for patterns in self.patterns.values():
            for pattern in patterns:
                if pattern.get("id") == pattern_id:
                    return pattern
        return None

    def match_patterns_to_schema(
        self,
        table_schema: Dict,
        categories: Optional[List[FraudCategory]] = None
    ) -> List[Dict]:
        """Match applicable patterns to a table schema."""
        columns = [col.lower() for col in table_schema.get("columns", [])]
        column_details = table_schema.get("column_details", {})

        applicable_patterns = []

        patterns_to_check = []
        if categories:
            for cat in categories:
                patterns_to_check.extend(self.get_patterns_by_category(cat))
        else:
            patterns_to_check = self.get_all_patterns()

        for pattern in patterns_to_check:
            required_cols = pattern.get("required_columns", [])

            if not required_cols:
                applicable_patterns.append(pattern)
                continue

            has_all_required = all(
                any(req.lower() in col for col in columns)
                for req in required_cols
            )

            if has_all_required:
                pattern_copy = pattern.copy()
                pattern_copy["matched_columns"] = self._match_columns(
                    required_cols, columns, column_details
                )
                applicable_patterns.append(pattern_copy)

        return applicable_patterns

    def _match_columns(
        self,
        required_cols: List[str],
        actual_cols: List[str],
        column_details: Dict
    ) -> Dict[str, str]:
        """Match required column types to actual column names."""
        matched = {}

        for req in required_cols:
            req_lower = req.lower()

            for actual in actual_cols:
                actual_lower = actual.lower()

                if req_lower in actual_lower or actual_lower in req_lower:
                    matched[req] = actual
                    break

                if req_lower == "amount" and any(
                    term in actual_lower for term in ["price", "total", "value", "cost"]
                ):
                    matched[req] = actual
                    break

                if req_lower == "user_id" and any(
                    term in actual_lower for term in ["user", "customer", "account"]
                ):
                    matched[req] = actual
                    break

                if req_lower == "timestamp" and any(
                    term in actual_lower for term in ["date", "time", "created", "updated"]
                ):
                    matched[req] = actual
                    break

        return matched

    def generate_detection_sql(
        self,
        pattern: Dict,
        table_name: str,
        matched_columns: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """Generate executable SQL from pattern template."""
        sql_template = pattern.get("sql_template")
        if not sql_template:
            return None

        matched = matched_columns or pattern.get("matched_columns", {})

        sql = sql_template
        sql = sql.replace("{table_name}", table_name)

        for placeholder, actual_col in matched.items():
            sql = sql.replace(f"{{{placeholder}_col}}", actual_col)

        return sql.strip()


fraud_pattern_library = FraudPatternLibrary()
