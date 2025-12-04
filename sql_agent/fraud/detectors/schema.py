"""Schema vulnerability detection."""

from typing import Dict, List, Any
from ..models import SchemaVulnerability, RiskLevel


class SchemaVulnerabilityDetector:
    """Detect schema-level vulnerabilities that enable fraud."""

    def detect_vulnerabilities(
        self,
        table_name: str,
        table_schema: Dict[str, Any]
    ) -> List[SchemaVulnerability]:
        """Detect all schema vulnerabilities."""
        vulnerabilities = []

        # Check for missing primary key
        pk_vuln = self._check_missing_primary_key(table_name, table_schema)
        if pk_vuln:
            vulnerabilities.append(pk_vuln)

        # Check for missing foreign keys
        fk_vuln = self._check_missing_foreign_keys(table_name, table_schema)
        if fk_vuln:
            vulnerabilities.append(fk_vuln)

        # Check for missing audit fields
        audit_vulns = self._check_missing_audit_fields(table_name, table_schema)
        vulnerabilities.extend(audit_vulns)

        # Check for overly permissive nulls
        null_vulns = self._check_permissive_nulls(table_name, table_schema)
        vulnerabilities.extend(null_vulns)

        # Check for missing indexes
        index_vuln = self._check_missing_indexes(table_name, table_schema)
        if index_vuln:
            vulnerabilities.append(index_vuln)

        return vulnerabilities

    def _check_missing_primary_key(
        self,
        table_name: str,
        table_schema: Dict[str, Any]
    ) -> SchemaVulnerability:
        """Check if table has primary key."""
        primary_keys = table_schema.get("primary_keys", [])

        if not primary_keys:
            return SchemaVulnerability(
                vulnerability_id=f"{table_name}_no_pk",
                vulnerability_type="missing_primary_key",
                severity=RiskLevel.HIGH,
                description=f"Table '{table_name}' has no primary key constraint",
                affected_columns=[],
                affected_tables=[table_name],
                remediation="Add a primary key constraint to ensure row uniqueness and prevent duplicate fraud",
                sql_fix=f"ALTER TABLE {table_name} ADD PRIMARY KEY (id);",
                exploitability="High - allows duplicate records to be inserted, enabling fraud inflation"
            )
        return None

    def _check_missing_foreign_keys(
        self,
        table_name: str,
        table_schema: Dict[str, Any]
    ) -> SchemaVulnerability:
        """Check for potential foreign key columns without constraints."""
        columns = table_schema.get("columns", [])
        foreign_keys = table_schema.get("foreign_keys", [])

        # Find columns that look like FKs but aren't constrained
        potential_fks = [col for col in columns if col.lower().endswith("_id")]
        constrained_fks = [fk.get("column") for fk in foreign_keys]

        missing_fks = [col for col in potential_fks if col not in constrained_fks]

        if missing_fks and len(missing_fks) > 2:
            return SchemaVulnerability(
                vulnerability_id=f"{table_name}_missing_fks",
                vulnerability_type="missing_foreign_key_constraints",
                severity=RiskLevel.MEDIUM,
                description=f"Table has {len(missing_fks)} potential foreign key columns without constraints",
                affected_columns=missing_fks[:5],  # Show first 5
                affected_tables=[table_name],
                remediation="Add foreign key constraints to ensure referential integrity",
                sql_fix=f"-- Example: ALTER TABLE {table_name} ADD FOREIGN KEY ({missing_fks[0]}) REFERENCES parent_table(id);",
                exploitability="Medium - allows orphaned records and invalid references"
            )
        return None

    def _check_missing_audit_fields(
        self,
        table_name: str,
        table_schema: Dict[str, Any]
    ) -> List[SchemaVulnerability]:
        """Check for missing audit trail columns."""
        columns = [col.lower() for col in table_schema.get("columns", [])]
        vulnerabilities = []

        audit_fields = {
            "created_at": ["created_at", "created_date", "create_time"],
            "created_by": ["created_by", "created_user", "creator"],
            "updated_at": ["updated_at", "updated_date", "update_time", "modified_at"],
            "updated_by": ["updated_by", "updated_user", "modifier"]
        }

        for field_type, variants in audit_fields.items():
            has_field = any(any(var in col for var in variants) for col in columns)

            if not has_field:
                vulnerabilities.append(SchemaVulnerability(
                    vulnerability_id=f"{table_name}_no_{field_type}",
                    vulnerability_type="missing_audit_field",
                    severity=RiskLevel.HIGH if "created" in field_type else RiskLevel.MEDIUM,
                    description=f"Table missing '{field_type}' audit field",
                    affected_columns=[],
                    affected_tables=[table_name],
                    remediation=f"Add {field_type} column to track data lineage and changes",
                    sql_fix=f"""ALTER TABLE {table_name} ADD COLUMN {field_type} {'TIMESTAMP' if 'at' in field_type else 'VARCHAR(100)'} NOT NULL DEFAULT {'CURRENT_TIMESTAMP' if 'at' in field_type else 'system'};""",
                    exploitability="High - impossible to track who made changes or when, enabling untraceable fraud"
                ))

        return vulnerabilities

    def _check_permissive_nulls(
        self,
        table_name: str,
        table_schema: Dict[str, Any]
    ) -> List[SchemaVulnerability]:
        """Check for critical fields that allow NULL."""
        column_details = table_schema.get("column_details", {})
        vulnerabilities = []

        critical_patterns = ["amount", "price", "total", "user_id", "customer_id", "status"]

        for col_name, col_detail in column_details.items():
            col_lower = col_name.lower()
            is_nullable = col_detail.get("nullable", True)

            is_critical = any(pattern in col_lower for pattern in critical_patterns)

            if is_critical and is_nullable:
                vulnerabilities.append(SchemaVulnerability(
                    vulnerability_id=f"{table_name}_{col_name}_nullable",
                    vulnerability_type="critical_field_allows_null",
                    severity=RiskLevel.MEDIUM,
                    description=f"Critical field '{col_name}' allows NULL values",
                    affected_columns=[col_name],
                    affected_tables=[table_name],
                    remediation=f"Set NOT NULL constraint on {col_name} to prevent incomplete records",
                    sql_fix=f"ALTER TABLE {table_name} ALTER COLUMN {col_name} SET NOT NULL;",
                    exploitability="Medium - allows incomplete fraud attempts and data manipulation"
                ))

        return vulnerabilities[:5]  # Limit to top 5

    def _check_missing_indexes(
        self,
        table_name: str,
        table_schema: Dict[str, Any]
    ) -> SchemaVulnerability:
        """Check for missing indexes on fraud detection columns."""
        columns = [col.lower() for col in table_schema.get("columns", [])]
        indexes = table_schema.get("indexes", [])

        indexed_cols = []
        for idx in indexes:
            indexed_cols.extend([col.lower() for col in idx.get("columns", [])])

        fraud_detection_cols = [
            col for col in columns
            if any(pattern in col for pattern in ["timestamp", "created_at", "user_id", "amount", "status"])
        ]

        missing_indexes = [col for col in fraud_detection_cols if col not in indexed_cols]

        if len(missing_indexes) > 2:
            return SchemaVulnerability(
                vulnerability_id=f"{table_name}_missing_indexes",
                vulnerability_type="missing_fraud_detection_indexes",
                severity=RiskLevel.LOW,
                description=f"Missing indexes on {len(missing_indexes)} fraud detection columns",
                affected_columns=missing_indexes[:5],
                affected_tables=[table_name],
                remediation="Add indexes to improve fraud detection query performance",
                sql_fix=f"CREATE INDEX idx_{table_name}_{missing_indexes[0]} ON {table_name}({missing_indexes[0]});",
                exploitability="Low - doesn't enable fraud but slows detection"
            )
        return None


schema_vulnerability_detector = SchemaVulnerabilityDetector()
