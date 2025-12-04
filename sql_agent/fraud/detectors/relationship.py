"""Relationship integrity detection for fraud."""

from typing import Dict, List, Any, Optional
from ..models import FraudScenario, DataQualityIssue, RiskLevel, FraudCategory


class RelationshipIntegrityDetector:
    """Detect relationship integrity issues that enable fraud."""

    async def detect_integrity_issues(
        self,
        table_name: str,
        table_schema: Dict[str, Any],
        db_manager
    ) -> List[DataQualityIssue]:
        """Detect relationship integrity issues."""
        issues = []

        # Check for orphaned records
        orphaned = await self._detect_orphaned_records(table_name, table_schema, db_manager)
        issues.extend(orphaned)

        return issues

    async def _detect_orphaned_records(
        self,
        table_name: str,
        table_schema: Dict[str, Any],
        db_manager
    ) -> List[DataQualityIssue]:
        """Detect orphaned records (FK points to non-existent parent)."""
        issues = []
        foreign_keys = table_schema.get("foreign_keys", [])

        if not foreign_keys:
            return issues

        for fk in foreign_keys[:5]:  # Check first 5 FKs
            fk_column = fk.get("column")
            ref_table = fk.get("references_table")
            ref_column = fk.get("references_column", "id")

            if not fk_column or not ref_table:
                continue

            sql = f"""
                SELECT COUNT(*) as orphaned_count
                FROM {table_name} t
                LEFT JOIN {ref_table} r ON t.{fk_column} = r.{ref_column}
                WHERE t.{fk_column} IS NOT NULL
                  AND r.{ref_column} IS NULL;
            """

            try:
                result = await db_manager.execute_query(sql, timeout=30)
                if result.row_count > 0:
                    orphaned_count = result.data[0].get("orphaned_count", 0)

                    if orphaned_count > 0:
                        issues.append(DataQualityIssue(
                            issue_id=f"{table_name}_{fk_column}_orphaned",
                            issue_type="orphaned_records",
                            severity=RiskLevel.MEDIUM if orphaned_count < 100 else RiskLevel.HIGH,
                            description=f"Found {orphaned_count} orphaned records in '{fk_column}' referencing '{ref_table}'",
                            affected_columns=[fk_column],
                            affected_rows_estimate=orphaned_count,
                            example_sql=sql,
                            impact="Orphaned records may indicate deleted parent records to hide fraud or data manipulation",
                            remediation=f"Clean up orphaned records or restore parent records in {ref_table}"
                        ))
            except Exception:
                continue

        return issues


relationship_integrity_detector = RelationshipIntegrityDetector()
