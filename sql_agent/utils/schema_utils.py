"""
Schema utilities for data extraction and preparation.

This module provides lightweight utility functions for schema analysis,
data sampling, and pattern detection. These utilities support the
intelligent schema analyzer without containing business logic.

Focus: Data extraction, pattern recognition, and preparation utilities.
NOT: Business intelligence, domain classification, or LLM interactions.
"""

import re
import statistics
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from collections import Counter, defaultdict
from datetime import datetime
import hashlib

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