"""
Schema Fingerprinting System

High-performance fingerprinting for database schemas and tables to enable
intelligent caching and incremental updates. Designed for enterprise-scale
databases with 200+ tables and local LLM processing.

Design Principles:
- Fast computation for real-time change detection
- Hierarchical fingerprints (database -> table -> column)
- Stable across non-semantic changes (whitespace, comments)
- Sensitive to structural and data pattern changes

Author: ML Engineering Team
"""

import hashlib
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class FingerprintComponents:
    """Components used to generate a fingerprint."""
    structural: Dict[str, Any]      # Schema structure (tables, columns, types)
    relational: Dict[str, Any]      # Foreign keys, indexes, constraints
    data_patterns: Dict[str, Any]   # Sample data patterns, cardinality
    metadata: Dict[str, Any]        # Row counts, timestamps, statistics


@dataclass
class TableFingerprint:
    """Complete fingerprint for a single table."""
    table_name: str
    fingerprint: str
    components: FingerprintComponents
    generated_at: datetime
    version: str = "1.0"


@dataclass
class SchemaFingerprint:
    """Complete fingerprint for entire database schema."""
    database_name: str
    schema_fingerprint: str
    table_fingerprints: Dict[str, TableFingerprint]
    table_count: int
    generated_at: datetime
    version: str = "1.0"


class SchemaFingerprintGenerator:
    """
    Enterprise-grade schema fingerprinting system.
    
    Optimized for:
    - Large schemas (200+ tables) 
    - Local LLM processing workflows
    - Incremental change detection
    - Minimal false positives
    """
    
    def __init__(self, include_sample_data: bool = True, 
                 sample_data_depth: int = 5):
        """
        Initialize fingerprint generator.
        
        Args:
            include_sample_data: Whether to include data patterns in fingerprint
            sample_data_depth: Number of sample rows to consider for patterns
        """
        self.include_sample_data = include_sample_data
        self.sample_data_depth = sample_data_depth
        self.version = "1.0"
        
        # Exclude volatile metadata that doesn't affect business analysis
        self.excluded_metadata = {
            'last_vacuum', 'last_autovacuum', 'last_analyze', 'last_autoanalyze',
            'dead_tuples', 'extraction_timestamp', 'analyzed_at'
        }
    
    def generate_schema_fingerprint(self, schema_data: Dict[str, Any]) -> SchemaFingerprint:
        """
        Generate complete schema fingerprint.
        
        Args:
            schema_data: Complete schema data from database manager
            
        Returns:
            SchemaFingerprint with hierarchical fingerprints
        """
        try:
            database_name = schema_data.get("database_name", "unknown")
            tables = schema_data.get("tables", [])
            
            # Generate fingerprint for each table
            table_fingerprints = {}
            table_fps_for_schema = []
            
            for table_data in tables:
                table_name = table_data.get("name", "")
                if not table_name:
                    continue
                    
                table_fp = self.generate_table_fingerprint(table_data)
                table_fingerprints[table_name] = table_fp
                table_fps_for_schema.append(table_fp.fingerprint)
            
            # Generate overall schema fingerprint from sorted table fingerprints
            schema_components = {
                "database_name": database_name,
                "table_count": len(tables),
                "table_fingerprints": sorted(table_fps_for_schema),
                "version": self.version
            }
            
            schema_fp = self._hash_components(schema_components)
            
            return SchemaFingerprint(
                database_name=database_name,
                schema_fingerprint=schema_fp,
                table_fingerprints=table_fingerprints,
                table_count=len(tables),
                generated_at=datetime.utcnow(),
                version=self.version
            )
            
        except Exception as e:
            logger.error(f"Failed to generate schema fingerprint: {e}")
            raise
    
    def generate_table_fingerprint(self, table_data: Dict[str, Any]) -> TableFingerprint:
        """
        Generate fingerprint for single table.
        
        Args:
            table_data: Table data including columns, constraints, sample data
            
        Returns:
            TableFingerprint with detailed components
        """
        try:
            table_name = table_data.get("name", "")
            
            # Extract fingerprint components
            structural = self._extract_structural_components(table_data)
            relational = self._extract_relational_components(table_data)
            data_patterns = self._extract_data_pattern_components(table_data)
            metadata = self._extract_metadata_components(table_data)
            
            components = FingerprintComponents(
                structural=structural,
                relational=relational, 
                data_patterns=data_patterns,
                metadata=metadata
            )
            
            # Generate composite fingerprint
            fingerprint = self._generate_composite_fingerprint(components)
            
            return TableFingerprint(
                table_name=table_name,
                fingerprint=fingerprint,
                components=components,
                generated_at=datetime.utcnow(),
                version=self.version
            )
            
        except Exception as e:
            logger.error(f"Failed to generate table fingerprint for {table_name}: {e}")
            raise
    
    def _extract_structural_components(self, table_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structural schema components."""
        columns = table_data.get("columns", [])
        
        # Create stable column signature
        column_signatures = []
        for col in sorted(columns):  # Sort for stability
            if isinstance(col, dict):
                col_sig = {
                    "name": col.get("name", ""),
                    "type": self._normalize_type(col.get("type", "")),
                    "nullable": col.get("nullable", True),
                    "default": bool(col.get("default"))  # Presence, not value
                }
            else:
                col_sig = {"name": str(col), "type": "unknown"}
            column_signatures.append(col_sig)
        
        return {
            "table_name": table_data.get("name", ""),
            "column_count": len(columns),
            "column_signatures": column_signatures
        }
    
    def _extract_relational_components(self, table_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relational components (FKs, constraints, indexes)."""
        primary_keys = table_data.get("primary_keys", [])
        foreign_keys = table_data.get("foreign_keys", [])
        indexes = table_data.get("indexes", [])
        
        # Create stable signatures for relationships
        pk_signature = sorted([pk.get("column", "") for pk in primary_keys])
        
        fk_signatures = []
        for fk in foreign_keys:
            fk_sig = {
                "column": fk.get("column", ""),
                "referenced_table": fk.get("referenced_table", ""),
                "referenced_column": fk.get("referenced_column", "")
            }
            fk_signatures.append(fk_sig)
        fk_signatures.sort(key=lambda x: x["column"])
        
        index_signatures = []
        for idx in indexes:
            idx_sig = {
                "columns": sorted(idx.get("columns", [])),
                "unique": idx.get("unique", False),
                "type": idx.get("type", "btree")
            }
            index_signatures.append(idx_sig)
        index_signatures.sort(key=lambda x: str(x["columns"]))
        
        return {
            "primary_keys": pk_signature,
            "foreign_keys": fk_signatures,
            "indexes": index_signatures,
            "has_relationships": len(fk_signatures) > 0
        }
    
    def _extract_data_pattern_components(self, table_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data pattern components from sample data."""
        if not self.include_sample_data:
            return {}
        
        sample_data = table_data.get("sample_data", {})
        if not sample_data:
            return {}
        
        rows = sample_data.get("rows", [])
        columns = sample_data.get("columns", [])
        
        if not rows or not columns:
            return {}
        
        # Analyze patterns in limited sample for stability
        limited_rows = rows[:self.sample_data_depth]
        
        patterns = {}
        for i, column in enumerate(columns):
            column_values = [
                row[i] for row in limited_rows 
                if len(row) > i and row[i] is not None
            ]
            
            if column_values:
                patterns[column] = {
                    "cardinality_class": self._classify_cardinality(len(set(column_values)), len(column_values)),
                    "data_type_pattern": self._infer_data_type_pattern(column_values),
                    "null_ratio_class": self._classify_null_ratio(len(column_values), len(limited_rows))
                }
        
        return {
            "sample_size": len(limited_rows),
            "column_patterns": patterns,
            "has_data": len(patterns) > 0
        }
    
    def _extract_metadata_components(self, table_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract stable metadata components."""
        statistics = table_data.get("statistics", {})
        
        # Only include stable, business-relevant statistics
        stable_stats = {}
        for key, value in statistics.items():
            if key not in self.excluded_metadata and value is not None:
                # Classify numeric values into ranges for stability
                if isinstance(value, (int, float)) and value >= 0:
                    if key == "live_tuples":
                        stable_stats["size_class"] = self._classify_table_size(value)
                    elif "size" in key.lower():
                        stable_stats["storage_class"] = self._classify_storage_size(value)
        
        return {
            "has_statistics": len(statistics) > 0,
            "stable_statistics": stable_stats
        }
    
    def _generate_composite_fingerprint(self, components: FingerprintComponents) -> str:
        """Generate composite fingerprint from all components."""
        component_dict = asdict(components)
        return self._hash_components(component_dict)
    
    def _hash_components(self, components: Dict[str, Any]) -> str:
        """Generate stable hash from components."""
        # Sort keys for deterministic output
        stable_json = json.dumps(components, sort_keys=True, default=str)
        return hashlib.sha256(stable_json.encode()).hexdigest()[:16]
    
    def _normalize_type(self, type_str: str) -> str:
        """Normalize database types to standard forms."""
        type_lower = type_str.lower().strip()
        
        # Group similar types
        if "varchar" in type_lower or "text" in type_lower or "char" in type_lower:
            return "text"
        elif "int" in type_lower or "serial" in type_lower:
            return "integer"
        elif "float" in type_lower or "double" in type_lower or "decimal" in type_lower:
            return "numeric"
        elif "bool" in type_lower:
            return "boolean"
        elif "date" in type_lower or "time" in type_lower:
            return "temporal"
        elif "json" in type_lower:
            return "json"
        else:
            return "other"
    
    def _classify_cardinality(self, unique_count: int, total_count: int) -> str:
        """Classify cardinality into stable ranges."""
        if total_count == 0:
            return "empty"
        
        ratio = unique_count / total_count
        if ratio >= 0.95:
            return "unique"
        elif ratio >= 0.5:
            return "high"
        elif ratio >= 0.1:
            return "medium"
        else:
            return "low"
    
    def _infer_data_type_pattern(self, values: List[Any]) -> str:
        """Infer data type pattern from sample values."""
        if not values:
            return "empty"
        
        str_values = [str(v) for v in values[:3]]  # Limit for stability
        
        # Simple pattern detection
        if all(v.isdigit() for v in str_values if v):
            return "numeric_id"
        elif all("@" in v for v in str_values if v):
            return "email"
        elif all(len(v) > 20 for v in str_values if v):
            return "long_text"
        elif all(len(v) < 10 for v in str_values if v):
            return "short_text"
        else:
            return "mixed"
    
    def _classify_null_ratio(self, non_null_count: int, total_count: int) -> str:
        """Classify null ratio into stable ranges."""
        if total_count == 0:
            return "empty"
        
        ratio = non_null_count / total_count
        if ratio >= 0.95:
            return "complete"
        elif ratio >= 0.8:
            return "mostly_complete"
        elif ratio >= 0.5:
            return "partial"
        else:
            return "sparse"
    
    def _classify_table_size(self, row_count: int) -> str:
        """Classify table size into stable ranges."""
        if row_count == 0:
            return "empty"
        elif row_count < 1000:
            return "small"
        elif row_count < 100000:
            return "medium"
        elif row_count < 10000000:
            return "large"
        else:
            return "very_large"
    
    def _classify_storage_size(self, size_bytes: int) -> str:
        """Classify storage size into stable ranges."""
        if size_bytes < 1024 * 1024:  # 1MB
            return "small"
        elif size_bytes < 100 * 1024 * 1024:  # 100MB
            return "medium"
        elif size_bytes < 1024 * 1024 * 1024:  # 1GB
            return "large"
        else:
            return "very_large"


class FingerprintComparator:
    """
    High-performance fingerprint comparison for change detection.
    
    Optimized for incremental analysis workflows where only changed
    tables need re-processing.
    """
    
    @staticmethod
    def compare_schema_fingerprints(old_fp: SchemaFingerprint, 
                                   new_fp: SchemaFingerprint) -> Dict[str, Any]:
        """
        Compare two schema fingerprints and identify changes.
        
        Returns:
            Dictionary with change analysis and affected tables
        """
        if old_fp.database_name != new_fp.database_name:
            return {
                "schema_changed": True,
                "change_type": "database_renamed",
                "requires_full_reanalysis": True,
                "changed_tables": list(new_fp.table_fingerprints.keys())
            }
        
        if old_fp.schema_fingerprint == new_fp.schema_fingerprint:
            return {
                "schema_changed": False,
                "change_type": "no_change",
                "requires_full_reanalysis": False,
                "changed_tables": []
            }
        
        # Identify specific table changes
        old_tables = set(old_fp.table_fingerprints.keys())
        new_tables = set(new_fp.table_fingerprints.keys())
        
        added_tables = new_tables - old_tables
        removed_tables = old_tables - new_tables
        common_tables = old_tables & new_tables
        
        changed_tables = []
        for table_name in common_tables:
            old_table_fp = old_fp.table_fingerprints[table_name].fingerprint
            new_table_fp = new_fp.table_fingerprints[table_name].fingerprint
            
            if old_table_fp != new_table_fp:
                changed_tables.append(table_name)
        
        total_changes = len(added_tables) + len(removed_tables) + len(changed_tables)
        change_ratio = total_changes / max(len(new_tables), 1)
        
        return {
            "schema_changed": True,
            "change_type": "incremental" if change_ratio < 0.3 else "major",
            "requires_full_reanalysis": change_ratio >= 0.5,
            "added_tables": list(added_tables),
            "removed_tables": list(removed_tables),
            "changed_tables": changed_tables,
            "total_changes": total_changes,
            "change_ratio": change_ratio
        }
    
    @staticmethod
    def compare_table_fingerprints(old_fp: TableFingerprint, 
                                  new_fp: TableFingerprint) -> Dict[str, Any]:
        """
        Compare two table fingerprints and identify specific changes.
        
        Returns:
            Detailed change analysis for table-level updates
        """
        if old_fp.table_name != new_fp.table_name:
            return {
                "table_changed": True,
                "change_type": "table_renamed",
                "requires_reanalysis": True
            }
        
        if old_fp.fingerprint == new_fp.fingerprint:
            return {
                "table_changed": False,
                "change_type": "no_change", 
                "requires_reanalysis": False
            }
        
        # Analyze component-level changes
        changes = {}
        
        if old_fp.components.structural != new_fp.components.structural:
            changes["structural"] = "Schema structure changed (columns, types)"
        
        if old_fp.components.relational != new_fp.components.relational:
            changes["relational"] = "Relationships changed (keys, indexes)"
        
        if old_fp.components.data_patterns != new_fp.components.data_patterns:
            changes["data_patterns"] = "Data patterns changed"
        
        if old_fp.components.metadata != new_fp.components.metadata:
            changes["metadata"] = "Table metadata changed"
        
        # Determine reanalysis requirements
        requires_reanalysis = bool(changes.get("structural") or changes.get("relational"))
        
        return {
            "table_changed": True,
            "change_type": "modified",
            "requires_reanalysis": requires_reanalysis,
            "component_changes": changes,
            "impact_level": "high" if requires_reanalysis else "low"
        }