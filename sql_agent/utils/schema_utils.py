"""
Schema utilities for data extraction and preparation.

This module provides lightweight utility functions for schema analysis,
data sampling, and pattern detection. These utilities support the
intelligent schema analyzer without containing business logic.

Focus: Data extraction, pattern recognition, and preparation utilities.
NOT: Business intelligence, domain classification, or LLM interactions.

ENHANCED: Added fingerprinting utilities for intelligent cache invalidation.
"""

import re
import statistics
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json

from ..utils.logging import get_logger


logger = get_logger("utils.schema_utils")


# ==================== DATA SAMPLING UTILITIES ====================

async def smart_sample_data(
    table_name: str,
    db_manager,
    sample_size: int = 50,
    strategy: str = "diverse"
) -> Dict[str, Any]:
    """
    Extract representative sample data with different strategies.
    
    Args:
        table_name: Target table name
        db_manager: Database manager instance
        sample_size: Number of rows to sample
        strategy: Sampling strategy ("diverse", "recent", "random")
    
    Returns:
        Enhanced sample data with basic analysis
    """
    try:
        # Get basic sample from database manager
        sample_data = await db_manager.get_sample_data(table_name, sample_size)
        
        if not sample_data or not sample_data.get("rows"):
            return {"rows": [], "columns": [], "metadata": {"strategy": strategy}}
        
        # Enhance with basic analysis
        enhanced_sample = {
            **sample_data,
            "metadata": {
                "strategy": strategy,
                "sample_size": len(sample_data.get("rows", [])),
                "extracted_at": datetime.utcnow().isoformat(),
                "basic_stats": calculate_sample_statistics(sample_data)
            }
        }
        
        return enhanced_sample
        
    except Exception as e:
        logger.error("smart_sample_data_failed", table=table_name, error=str(e))
        return {"rows": [], "columns": [], "metadata": {"error": str(e)}}


def calculate_sample_statistics(sample_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate basic statistics from sample data."""
    rows = sample_data.get("rows", [])
    columns = sample_data.get("columns", [])
    
    if not rows or not columns:
        return {}
    
    stats = {
        "row_count": len(rows),
        "column_count": len(columns),
        "completeness": {},
        "cardinality": {},
        "data_types": {}
    }
    
    # Calculate per-column statistics
    for i, column in enumerate(columns):
        column_values = [row[i] for row in rows if len(row) > i and row[i] is not None]
        
        stats["completeness"][column] = len(column_values) / len(rows) if rows else 0
        stats["cardinality"][column] = len(set(str(v) for v in column_values))
        stats["data_types"][column] = infer_column_data_type(column_values)
    
    return stats


# ==================== FINGERPRINTING UTILITIES ====================

def generate_table_structure_hash(table_data: Dict[str, Any]) -> str:
    """
    Generate stable hash for table structure.
    
    Focuses on structural elements that affect business analysis:
    - Column names and types
    - Primary/foreign key relationships
    - Table constraints
    
    Excludes volatile metadata like vacuum times.
    """
    try:
        structural_components = {
            "table_name": table_data.get("name", ""),
            "columns": _extract_stable_column_signature(table_data.get("columns", [])),
            "primary_keys": _extract_key_signature(table_data.get("primary_keys", [])),
            "foreign_keys": _extract_fk_signature(table_data.get("foreign_keys", [])),
            "indexes": _extract_index_signature(table_data.get("indexes", []))
        }
        
        # Create stable JSON representation
        stable_json = json.dumps(structural_components, sort_keys=True)
        return hashlib.sha256(stable_json.encode()).hexdigest()[:16]
        
    except Exception as e:
        logger.error(f"Failed to generate table structure hash: {e}")
        return "error"


def generate_data_pattern_hash(sample_data: Dict[str, Any], depth: int = 5) -> str:
    """
    Generate hash for data patterns from sample data.
    
    Args:
        sample_data: Sample data from table
        depth: Number of sample rows to include
        
    Returns:
        Stable hash representing data patterns
    """
    try:
        rows = sample_data.get("rows", [])
        columns = sample_data.get("columns", [])
        
        if not rows or not columns:
            return "empty"
        
        # Limit to specified depth for stability
        limited_rows = rows[:depth]
        
        # Extract stable patterns
        patterns = {}
        for i, column in enumerate(columns):
            column_values = [
                row[i] for row in limited_rows 
                if len(row) > i and row[i] is not None
            ]
            
            if column_values:
                patterns[column] = {
                    "cardinality_class": _classify_cardinality(len(set(column_values)), len(column_values)),
                    "type_pattern": _infer_stable_type_pattern(column_values),
                    "length_pattern": _classify_length_pattern(column_values)
                }
        
        pattern_components = {
            "sample_size": len(limited_rows),
            "column_patterns": patterns
        }
        
        stable_json = json.dumps(pattern_components, sort_keys=True)
        return hashlib.sha256(stable_json.encode()).hexdigest()[:16]
        
    except Exception as e:
        logger.error(f"Failed to generate data pattern hash: {e}")
        return "error"


def generate_combined_fingerprint(table_data: Dict[str, Any], 
                                include_data_patterns: bool = True) -> str:
    """
    Generate combined fingerprint for table including structure and data patterns.
    
    Args:
        table_data: Complete table data including sample data
        include_data_patterns: Whether to include sample data patterns
        
    Returns:
        Combined fingerprint string
    """
    try:
        structure_hash = generate_table_structure_hash(table_data)
        
        if include_data_patterns:
            sample_data = table_data.get("sample_data", {})
            data_hash = generate_data_pattern_hash(sample_data)
            combined = f"{structure_hash}:{data_hash}"
        else:
            combined = structure_hash
        
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
        
    except Exception as e:
        logger.error(f"Failed to generate combined fingerprint: {e}")
        return "error"


def _extract_stable_column_signature(columns: List[Any]) -> List[Dict[str, str]]:
    """Extract stable column signature for fingerprinting."""
    signatures = []
    
    for col in sorted(columns, key=lambda x: x.get("name", "") if isinstance(x, dict) else str(x)):
        if isinstance(col, dict):
            sig = {
                "name": col.get("name", ""),
                "type": _normalize_db_type(col.get("type", "")),
                "nullable": str(col.get("nullable", True)),
                "has_default": str(bool(col.get("default")))
            }
        else:
            sig = {"name": str(col), "type": "unknown", "nullable": "true", "has_default": "false"}
        
        signatures.append(sig)
    
    return signatures


def _extract_key_signature(keys: List[Dict[str, Any]]) -> List[str]:
    """Extract stable primary key signature."""
    return sorted([key.get("column", "") for key in keys])


def _extract_fk_signature(foreign_keys: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Extract stable foreign key signature."""
    fk_sigs = []
    
    for fk in foreign_keys:
        sig = {
            "column": fk.get("column", ""),
            "referenced_table": fk.get("referenced_table", ""),
            "referenced_column": fk.get("referenced_column", "")
        }
        fk_sigs.append(sig)
    
    return sorted(fk_sigs, key=lambda x: x["column"])


def _extract_index_signature(indexes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract stable index signature."""
    index_sigs = []
    
    for idx in indexes:
        sig = {
            "columns": sorted(idx.get("columns", [])),
            "unique": str(idx.get("unique", False)),
            "type": idx.get("type", "btree")
        }
        index_sigs.append(sig)
    
    return sorted(index_sigs, key=lambda x: str(x["columns"]))


def _normalize_db_type(type_str: str) -> str:
    """Normalize database type to standard form for fingerprinting."""
    type_lower = type_str.lower().strip()
    
    # Group similar types for stable fingerprints
    if "varchar" in type_lower or "text" in type_lower or "char" in type_lower:
        return "text"
    elif "int" in type_lower or "serial" in type_lower:
        return "integer" 
    elif "float" in type_lower or "double" in type_lower or "decimal" in type_lower or "numeric" in type_lower:
        return "numeric"
    elif "bool" in type_lower:
        return "boolean"
    elif "date" in type_lower or "time" in type_lower:
        return "temporal"
    elif "json" in type_lower:
        return "json"
    else:
        return "other"


def _classify_cardinality(unique_count: int, total_count: int) -> str:
    """Classify cardinality into stable ranges for fingerprinting."""
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


def _infer_stable_type_pattern(values: List[Any]) -> str:
    """Infer stable data type pattern for fingerprinting."""
    if not values:
        return "empty"
    
    # Sample limited values for stability
    sample_values = [str(v) for v in values[:3]]
    
    if all(v.isdigit() for v in sample_values if v):
        return "numeric_id"
    elif all("@" in v for v in sample_values if v):
        return "email"
    elif all(len(v) > 50 for v in sample_values if v):
        return "long_text"
    elif all(len(v) < 10 for v in sample_values if v):
        return "short_code"
    else:
        return "general_text"


def _classify_length_pattern(values: List[Any]) -> str:
    """Classify value length patterns for fingerprinting."""
    if not values:
        return "empty"
    
    lengths = [len(str(v)) for v in values[:5]]  # Limited sample
    
    if len(set(lengths)) == 1:
        return "fixed_length"
    elif max(lengths) - min(lengths) <= 2:
        return "similar_length"
    else:
        return "variable_length"


# ==================== PATTERN DETECTION UTILITIES ====================

def detect_naming_patterns(table_name: str, columns: List[str]) -> Dict[str, Any]:
    """Detect naming conventions and patterns in schema."""
    patterns = {
        "table_naming": analyze_table_naming_pattern(table_name),
        "column_naming": analyze_column_naming_patterns(columns),
        "consistency_score": calculate_naming_consistency(table_name, columns)
    }
    
    return patterns


def analyze_table_naming_pattern(table_name: str) -> Dict[str, Any]:
    """Analyze table naming convention."""
    return {
        "case_style": detect_case_style(table_name),
        "separator_style": detect_separator_style(table_name),
        "length": len(table_name),
        "has_prefix": "_" in table_name and table_name.index("_") < len(table_name) // 2,
        "has_suffix": "_" in table_name and table_name.rindex("_") > len(table_name) // 2,
        "word_count": len(re.split(r'[_\s-]', table_name))
    }


def analyze_column_naming_patterns(columns: List[str]) -> Dict[str, Any]:
    """Analyze column naming patterns across all columns."""
    if not columns:
        return {}
    
    case_styles = Counter(detect_case_style(col) for col in columns)
    separator_styles = Counter(detect_separator_style(col) for col in columns)
    
    return {
        "dominant_case_style": case_styles.most_common(1)[0][0] if case_styles else "unknown",
        "case_consistency": case_styles.most_common(1)[0][1] / len(columns) if case_styles else 0,
        "dominant_separator": separator_styles.most_common(1)[0][0] if separator_styles else "none",
        "separator_consistency": separator_styles.most_common(1)[0][1] / len(columns) if separator_styles else 0,
        "avg_length": statistics.mean(len(col) for col in columns),
        "length_variance": statistics.variance(len(col) for col in columns) if len(columns) > 1 else 0
    }


def detect_case_style(text: str) -> str:
    """Detect case style of text."""
    if not text:
        return "unknown"
    
    if text.islower():
        return "lowercase"
    elif text.isupper():
        return "uppercase"
    elif text[0].islower() and any(c.isupper() for c in text[1:]):
        return "camelCase"
    elif text[0].isupper() and any(c.isupper() for c in text[1:]):
        return "PascalCase"
    else:
        return "mixed"


def detect_separator_style(text: str) -> str:
    """Detect separator style in text."""
    if "_" in text:
        return "snake_case"
    elif "-" in text:
        return "kebab-case"
    else:
        return "none"


def calculate_naming_consistency(table_name: str, columns: List[str]) -> float:
    """Calculate overall naming consistency score."""
    if not columns:
        return 1.0
    
    table_case = detect_case_style(table_name)
    table_sep = detect_separator_style(table_name)
    
    # Check how many columns match table naming style
    matching_case = sum(1 for col in columns if detect_case_style(col) == table_case)
    matching_sep = sum(1 for col in columns if detect_separator_style(col) == table_sep)
    
    case_consistency = matching_case / len(columns)
    sep_consistency = matching_sep / len(columns)
    
    return (case_consistency + sep_consistency) / 2


# ==================== DATA TYPE INFERENCE UTILITIES ====================

def infer_column_data_type(values: List[Any]) -> Dict[str, Any]:
    """Infer enhanced data type information from sample values."""
    if not values:
        return {"base_type": "unknown", "pattern": "none", "confidence": 0.0}
    
    str_values = [str(v) for v in values if v is not None]
    
    # Base type inference
    type_checks = [
        ("integer", lambda v: v.isdigit()),
        ("float", lambda v: is_float(v)),
        ("boolean", lambda v: is_boolean(v)),
        ("date", lambda v: is_date_like(v)),
        ("email", lambda v: is_email_like(v)),
        ("phone", lambda v: is_phone_like(v)),
        ("url", lambda v: is_url_like(v)),
        ("uuid", lambda v: is_uuid_like(v))
    ]
    
    # Calculate confidence for each type
    type_scores = {}
    for type_name, check_func in type_checks:
        matches = sum(1 for v in str_values[:20] if check_func(v))  # Sample first 20
        type_scores[type_name] = matches / min(len(str_values), 20) if str_values else 0
    
    # Determine best type
    best_type = max(type_scores.items(), key=lambda x: x[1])
    
    return {
        "base_type": best_type[0] if best_type[1] > 0.7 else "string",
        "confidence": best_type[1],
        "pattern": detect_value_pattern(str_values),
        "type_scores": type_scores
    }


def detect_value_pattern(values: List[str]) -> str:
    """Detect common patterns in string values."""
    if not values:
        return "none"
    
    # Check for common patterns
    patterns = {
        "constant_length": len(set(len(v) for v in values[:10])) == 1,
        "numeric_suffix": all(v[-1].isdigit() for v in values[:5] if v),
        "prefix_pattern": len(set(v[:2] for v in values[:5] if len(v) >= 2)) == 1,
        "contains_separators": all(any(sep in v for sep in ["-", "_", "."]) for v in values[:5])
    }
    
    # Return dominant pattern
    for pattern_name, is_present in patterns.items():
        if is_present:
            return pattern_name
    
    return "variable"


# ==================== RELATIONSHIP DETECTION UTILITIES ====================

def detect_potential_relationships(
    source_table: Dict[str, Any],
    target_tables: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Detect potential relationships between tables based on naming patterns."""
    relationships = []
    
    source_name = source_table.get("name", "")
    source_columns = source_table.get("columns", [])
    
    for target_table in target_tables:
        target_name = target_table.get("name", "")
        target_columns = target_table.get("columns", [])
        
        # Look for foreign key naming patterns
        potential_rels = find_naming_relationships(
            source_name, source_columns, target_name, target_columns
        )
        
        relationships.extend(potential_rels)
    
    return relationships


def find_naming_relationships(
    source_table: str,
    source_columns: List[str],
    target_table: str,
    target_columns: List[str]
) -> List[Dict[str, Any]]:
    """Find relationships based on naming conventions."""
    relationships = []
    
    # Pattern: source has column named "target_id" or "target_table_id"
    target_variations = [
        target_table.lower(),
        target_table.lower().rstrip("s"),  # Remove plural
        target_table.lower() + "s"         # Add plural
    ]
    
    for source_col in source_columns:
        source_col_lower = source_col.lower()
        
        # Check if column ends with _id
        if source_col_lower.endswith("_id"):
            col_prefix = source_col_lower[:-3]  # Remove "_id"
            
            if col_prefix in target_variations:
                relationships.append({
                    "source_table": source_table,
                    "source_column": source_col,
                    "target_table": target_table,
                    "target_column": "id",  # Assume primary key
                    "confidence": 0.8,
                    "type": "inferred_foreign_key",
                    "pattern": "naming_convention"
                })
    
    return relationships


# ==================== TEXT PROCESSING UTILITIES ====================

def extract_keywords_from_text(text: str, min_length: int = 3) -> List[str]:
    """Extract meaningful keywords from text."""
    if not text:
        return []
    
    # Split on common separators and filter
    words = re.split(r'[_\s\-\.]', text.lower())
    
    # Filter out common stop words and short words
    stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
    
    keywords = [
        word for word in words
        if len(word) >= min_length and word not in stop_words and word.isalpha()
    ]
    
    return list(set(keywords))  # Remove duplicates


def extract_business_keywords(table_name: str, columns: List[str]) -> List[str]:
    """Extract business-relevant keywords from table and column names."""
    all_text = f"{table_name} {' '.join(columns)}"
    
    # Extract basic keywords
    keywords = extract_keywords_from_text(all_text)
    
    # Add table name as primary keyword
    table_keywords = extract_keywords_from_text(table_name)
    
    # Combine and prioritize
    prioritized_keywords = table_keywords + [kw for kw in keywords if kw not in table_keywords]
    
    return prioritized_keywords[:10]  # Return top 10 keywords


# ==================== QUALITY ASSESSMENT UTILITIES ====================

def calculate_basic_quality_score(sample_data: Dict[str, Any]) -> Dict[str, float]:
    """Calculate basic data quality metrics."""
    rows = sample_data.get("rows", [])
    columns = sample_data.get("columns", [])
    
    if not rows or not columns:
        return {"completeness": 0.0, "consistency": 0.0}
    
    # Completeness: percentage of non-null values
    total_cells = len(rows) * len(columns)
    non_null_cells = sum(
        1 for row in rows for cell in row if cell is not None and str(cell).strip()
    )
    completeness = non_null_cells / total_cells if total_cells > 0 else 0.0
    
    # Consistency: check for format consistency in each column
    consistency_scores = []
    for i, column in enumerate(columns):
        column_values = [row[i] for row in rows if len(row) > i and row[i] is not None]
        
        if column_values:
            # Check format consistency (simple heuristic)
            str_values = [str(v) for v in column_values]
            length_variance = statistics.variance([len(v) for v in str_values]) if len(str_values) > 1 else 0
            
            # Lower variance = higher consistency
            consistency_score = 1.0 / (1.0 + length_variance / 10)
            consistency_scores.append(consistency_score)
    
    consistency = statistics.mean(consistency_scores) if consistency_scores else 0.0
    
    return {
        "completeness": completeness,
        "consistency": consistency,
        "overall": (completeness + consistency) / 2
    }


# ==================== HELPER FUNCTIONS ====================

def is_float(value: str) -> bool:
    """Check if string represents a float."""
    try:
        float(value)
        return '.' in value  # Distinguish from integer
    except (ValueError, TypeError):
        return False


def is_boolean(value: str) -> bool:
    """Check if string represents a boolean."""
    bool_values = {'true', 'false', 't', 'f', '1', '0', 'yes', 'no', 'y', 'n'}
    return str(value).lower() in bool_values


def is_date_like(value: str) -> bool:
    """Check if string looks like a date."""
    date_patterns = [
        r'^\d{4}-\d{2}-\d{2}$',          # YYYY-MM-DD
        r'^\d{2}/\d{2}/\d{4}$',          # MM/DD/YYYY
        r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$'  # YYYY-MM-DD HH:MM:SS
    ]
    return any(re.match(pattern, str(value)) for pattern in date_patterns)


def is_email_like(value: str) -> bool:
    """Check if string looks like an email."""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, str(value)))


def is_phone_like(value: str) -> bool:
    """Check if string looks like a phone number."""
    # Remove common separators and check if mostly digits
    cleaned = re.sub(r'[\s\-\(\)\+\.]', '', str(value))
    return len(cleaned) >= 10 and sum(c.isdigit() for c in cleaned) / len(cleaned) > 0.7


def is_url_like(value: str) -> bool:
    """Check if string looks like a URL."""
    url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(url_pattern, str(value), re.IGNORECASE))


def is_uuid_like(value: str) -> bool:
    """Check if string looks like a UUID."""
    uuid_pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
    return bool(re.match(uuid_pattern, str(value)))


# ==================== CACHING UTILITIES ====================

def generate_cache_key(*args) -> str:
    """Generate a consistent cache key from arguments."""
    key_data = ":".join(str(arg) for arg in args)
    return hashlib.md5(key_data.encode()).hexdigest()


def normalize_table_name(table_name: str) -> str:
    """Normalize table name for consistent processing."""
    return table_name.lower().strip()


def normalize_column_name(column_name: str) -> str:
    """Normalize column name for consistent processing."""
    return column_name.lower().strip()


# ==================== METADATA PREPARATION ====================

def prepare_table_metadata(table_data: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare clean metadata for analysis."""
    return {
        "name": table_data.get("name", ""),
        "columns": table_data.get("columns", []),
        "column_count": len(table_data.get("columns", [])),
        "has_primary_key": bool(table_data.get("primary_keys")),
        "has_foreign_keys": bool(table_data.get("foreign_keys")),
        "relationship_count": len(table_data.get("foreign_keys", [])),
        "estimated_rows": table_data.get("statistics", {}).get("live_tuples", 0),
        "table_size": table_data.get("statistics", {}).get("total_size_bytes", 0)
    }


def prepare_analysis_context(
    table_data: Dict[str, Any],
    sample_data: Dict[str, Any],
    related_tables: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Prepare comprehensive context for LLM analysis."""
    
    metadata = prepare_table_metadata(table_data)
    quality_metrics = calculate_basic_quality_score(sample_data)
    naming_patterns = detect_naming_patterns(metadata["name"], metadata["columns"])
    business_keywords = extract_business_keywords(metadata["name"], metadata["columns"])
    
    return {
        "table_metadata": metadata,
        "sample_data": sample_data,
        "quality_metrics": quality_metrics,
        "naming_patterns": naming_patterns,
        "business_keywords": business_keywords,
        "related_tables": related_tables or [],
        "context_prepared_at": datetime.utcnow().isoformat()
    }