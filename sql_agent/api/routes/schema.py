"""
Schema Routes - Fixed Implementation

This module contains endpoints for database schema information.
Fixed to properly work with the enterprise-grade database manager.
"""

from typing import Dict, Any, List, Optional
import time

from fastapi import APIRouter, HTTPException, Depends, Request, Body

from sql_agent.api.models import (
    SchemaResponse, TableInfo, ColumnInfo, DatabaseInfo
)
from sql_agent.core.database import db_manager
from sql_agent.core.models import Table
from sql_agent.services.ai_service import AIDescriptionService
import structlog

router = APIRouter()
logger = structlog.get_logger(__name__)


def get_database() -> Any:
    """Get the database manager instance."""
    if db_manager is None:
        raise HTTPException(status_code=503, detail="Database manager not initialized")
    return db_manager


@router.get("/", response_model=SchemaResponse)
async def get_schema(
    database_name: Optional[str] = None,
    database_manager = Depends(get_database)
) -> SchemaResponse:
    """
    Get database schema information.
    
    This endpoint returns comprehensive schema information including
    tables, columns, relationships, and metadata.
    """
    try:
        # Get schema information from database manager
        schema_info = await database_manager.get_schema_info()
        
        # Convert to API models - FIX: Iterate over tables array, not .items()
        tables = []
        total_columns = 0
        
        print("Extracting schema information...")
        print(schema_info)
        
        # FIX: schema_info returns {"tables": [...]} not {table_name: table_info}
        for table_info in schema_info.get("tables", []):
            table_name = table_info["name"]
            columns = []
            
            # FIX: Use column_details dict with actual column names as keys
            column_details = table_info.get("column_details", {})
            for column_name in table_info.get("columns", []):
                col_detail = column_details.get(column_name, {})
                
                # Check if it's a primary key
                is_primary_key = any(
                    pk.get("column") == column_name 
                    for pk in table_info.get("primary_keys", [])
                )
                
                # Check for foreign key
                foreign_key_ref = None
                for fk in table_info.get("foreign_keys", []):
                    if fk.get("column") == column_name:
                        foreign_key_ref = f"{fk.get('references_table')}.{fk.get('references_column')}"
                        break
                
                column = ColumnInfo(
                    name=column_name,
                    type=col_detail.get("type", "unknown"),
                    nullable=col_detail.get("nullable", True),
                    primary_key=is_primary_key,
                    foreign_key=foreign_key_ref
                )
                columns.append(column)
                total_columns += 1
            
            table = TableInfo(
                name=table_name,
                columns=columns,
                row_count=table_info.get("statistics", {}).get("live_tuples"),
                description=table_info.get("description")
            )
            tables.append(table)
        
        # Extract relationships from schema
        relationships = []
        for table_info in schema_info.get("tables", []):
            for fk in table_info.get("foreign_keys", []):
                relationships.append({
                    "source_table": table_info["name"],
                    "source_column": fk.get("column"),
                    "target_table": fk.get("references_table"),
                    "target_column": fk.get("references_column"),
                    "type": "foreign_key"
                })
        
        return SchemaResponse(
            database_name=database_name or schema_info.get("database_name", "default"),
            tables=tables,
            relationships=relationships,
            total_tables=len(tables),
            total_columns=total_columns
        )
        
    except Exception as e:
        logger.error("Failed to get schema", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get schema: {str(e)}"
        )


@router.get("/tables", response_model=Dict[str, Any])
async def get_tables(
    database_name: Optional[str] = None,
    database_manager = Depends(get_database)
) -> Dict[str, Any]:
    """
    Get list of tables in the database.
    
    This endpoint returns a list of all tables with basic information.
    """
    try:
        # Get tables from database manager
        schema_info = await database_manager.get_schema_info()
        
        tables = []
        # FIX: Iterate over tables array, not .items()
        for table_info in schema_info.get("tables", []):
            table_name = table_info["name"]
            stats = table_info.get("statistics", {})
            
            tables.append({
                "name": table_name,
                "column_count": len(table_info.get("columns", [])),
                "row_count": stats.get("live_tuples"),
                "description": table_info.get("description"),
                "size_bytes": stats.get("total_size_bytes"),
                "table_type": table_info.get("type", "BASE TABLE")
            })
        
        return {
            "database_name": database_name or schema_info.get("database_name", "default"),
            "tables": tables,
            "total": len(tables),
            "extraction_method": schema_info.get("extraction_method"),
            "last_updated": schema_info.get("extraction_timestamp")
        }
        
    except Exception as e:
        logger.error("Failed to get tables", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get tables: {str(e)}"
        )


@router.get("/tables/{table_name}", response_model=TableInfo)
async def get_table_info(
    table_name: str,
    database_name: Optional[str] = None,
    database_manager = Depends(get_database)
) -> TableInfo:
    """
    Get detailed information about a specific table.
    
    This endpoint returns comprehensive information about a table
    including all columns and their properties.
    """
    try:
        # Get table information from database manager
        schema_info = await database_manager.get_schema_info()
        
        # FIX: Find table in tables array
        table_info = None
        for table in schema_info.get("tables", []):
            if table["name"] == table_name:
                table_info = table
                break
        
        if table_info is None:
            raise HTTPException(
                status_code=404,
                detail=f"Table '{table_name}' not found"
            )
        
        # Convert to API model
        columns = []
        column_details = table_info.get("column_details", {})
        
        for column_name in table_info.get("columns", []):
            col_detail = column_details.get(column_name, {})
            
            # Check if it's a primary key
            is_primary_key = any(
                pk.get("column") == column_name 
                for pk in table_info.get("primary_keys", [])
            )
            
            # Check for foreign key
            foreign_key_ref = None
            for fk in table_info.get("foreign_keys", []):
                if fk.get("column") == column_name:
                    foreign_key_ref = f"{fk.get('references_table')}.{fk.get('references_column')}"
                    break
            
            column = ColumnInfo(
                name=column_name,
                type=col_detail.get("type", "unknown"),
                nullable=col_detail.get("nullable", True),
                primary_key=is_primary_key,
                foreign_key=foreign_key_ref
            )
            columns.append(column)
        
        return TableInfo(
            name=table_name,
            columns=columns,
            row_count=table_info.get("statistics", {}).get("live_tuples"),
            description=table_info.get("description")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get table info", table_name=table_name, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get table info: {str(e)}"
        )


@router.get("/search", response_model=Dict[str, Any])
async def search_schema(
    query: str,
    database_name: Optional[str] = None,
    database_manager = Depends(get_database)
) -> Dict[str, Any]:
    """
    Search schema by keywords.
    
    This endpoint searches for tables and columns that match
    the provided keywords.
    """
    try:
        # Search schema using database manager
        schema_info = await database_manager.get_schema_info()
        
        results = []
        query_lower = query.lower()
        
        # FIX: Iterate over tables array
        for table_info in schema_info.get("tables", []):
            table_name = table_info["name"]
            
            # Search in table names
            if query_lower in table_name.lower():
                results.append({
                    "type": "table",
                    "name": table_name,
                    "match": "table_name",
                    "description": table_info.get("description", "")
                })
            
            # Search in table description
            description = table_info.get("description", "")
            if description and query_lower in description.lower():
                results.append({
                    "type": "table",
                    "name": table_name,
                    "match": "description",
                    "description": description
                })
            
            # Search in column names and details
            column_details = table_info.get("column_details", {})
            for column_name in table_info.get("columns", []):
                if query_lower in column_name.lower():
                    col_detail = column_details.get(column_name, {})
                    results.append({
                        "type": "column",
                        "table": table_name,
                        "name": column_name,
                        "match": "column_name",
                        "data_type": col_detail.get("type", "unknown"),
                        "comment": col_detail.get("comment", "")
                    })
        
        return {
            "query": query,
            "database_name": database_name or schema_info.get("database_name", "default"),
            "results": results,
            "total_matches": len(results)
        }
        
    except Exception as e:
        logger.error("Failed to search schema", query=query, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search schema: {str(e)}"
        )


@router.get("/relationships", response_model=Dict[str, Any])
async def get_relationships(
    database_name: Optional[str] = None,
    database_manager = Depends(get_database)
) -> Dict[str, Any]:
    """
    Get database relationships.
    
    This endpoint returns information about foreign key relationships
    between tables.
    """
    try:
        # Get schema information with relationships
        schema_info = await database_manager.get_schema_info()
        
        relationships = []
        
        # Extract foreign key relationships
        for table_info in schema_info.get("tables", []):
            table_name = table_info["name"]
            
            for fk in table_info.get("foreign_keys", []):
                relationships.append({
                    "type": "foreign_key",
                    "source_table": table_name,
                    "source_column": fk.get("column"),
                    "target_table": fk.get("references_table"),
                    "target_column": fk.get("references_column"),
                    "constraint_name": fk.get("constraint_name")
                })
        
        # Add inferred relationships if available
        inferred_relationships = schema_info.get("inferred_relationships", [])
        for rel in inferred_relationships:
            relationships.append({
                "type": rel.relationship_type,
                "source_table": rel.source_table,
                "source_column": rel.source_column,
                "target_table": rel.target_table,
                "target_column": rel.target_column,
                "confidence": rel.confidence,
                "cardinality": rel.cardinality
            })
        
        return {
            "database_name": database_name or schema_info.get("database_name", "default"),
            "relationships": relationships,
            "total": len(relationships),
            "foreign_key_count": len([r for r in relationships if r["type"] == "foreign_key"]),
            "inferred_count": len([r for r in relationships if r["type"] == "inferred"])
        }
        
    except Exception as e:
        logger.error("Failed to get relationships", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get relationships: {str(e)}"
        )


@router.get("/sample/{table_name}", response_model=Dict[str, Any])
async def get_sample_data(
    table_name: str,
    limit: int = 10,
    database_name: Optional[str] = None,
    database_manager = Depends(get_database)
) -> Dict[str, Any]:
    """
    Get sample data from a table.
    
    This endpoint returns sample data from the specified table
    with the specified limit.
    """
    try:
        # Get sample data from database manager
        sample_data = await database_manager.get_sample_data(table_name, limit)
        
        return {
            "table_name": table_name,
            "database_name": database_name or "default",
            "data": sample_data,
            "limit": limit,
            "total_returned": len(sample_data.get("rows", []))
        }
        
    except Exception as e:
        logger.error("Failed to get sample data", table_name=table_name, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get sample data: {str(e)}"
        )


@router.get("/databases", response_model=List[DatabaseInfo])
async def get_databases(database_manager = Depends(get_database)):
    """
    Get list of available databases.
    """
    from sql_agent.core.config import settings
    try:
        db_status = "connected" if await database_manager.test_connection() else "disconnected"
        db_type = getattr(settings, 'database_type', 'postgresql')
        
        # Extract database name from URL safely
        try:
            db_name = settings.database_url.split("/")[-1].split("?")[0] or "default"
        except:
            db_name = "default"
        
        return [
            DatabaseInfo(
                id="default",
                name=db_name,
                type=db_type,
                status=db_status,
                lastSync=int(time.time())
            )
        ]
    except Exception as e:
        logger.error("Failed to get databases", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get databases: {e}") 


# FIX: Remove duplicate route - keeping the one with response model
@router.get("/databases/{database_id}/tables", response_model=List[Table])
async def list_tables_endpoint(database_id: str, database_manager = Depends(get_database)):
    """
    List all tables for a given database.
    """
    try:
        tables = await database_manager.list_tables(database_id)
        return tables
    except Exception as e:
        logger.error("Failed to list tables", database_id=database_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list tables: {e}") 


# FIX: Updated route path to avoid conflicts
@router.get("/databases/{database_id}/tables_simple")
async def api_get_tables(database_id: str, database_manager = Depends(get_database)):
    """
    Get tables in simple format for frontend compatibility.
    """
    try:
        tables = await database_manager.get_tables()
        return {"tables": tables}
    except Exception as e:
        logger.error("Failed to fetch tables", database_id=database_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch tables: {str(e)}")


@router.get("/databases/{database_id}/tables/{table_name}/schema")
async def api_get_table_schema(database_id: str, table_name: str, database_manager = Depends(get_database)):
    """
    Get table schema in legacy format.
    """
    try:
        schema = await database_manager.get_table_schema(table_name)
        return schema
    except Exception as e:
        logger.error("Failed to fetch table schema", database_id=database_id, table_name=table_name, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch table schema: {str(e)}")


@router.get("/databases/{database_id}/tables/{table_name}/sample")
async def api_get_sample_data(database_id: str, table_name: str, limit: int = 5, database_manager = Depends(get_database)):
    """
    Get sample data for a table.
    """
    try:
        sample_data = await database_manager.get_sample_data(table_name, limit)
        return sample_data
    except Exception as e:
        logger.error("Failed to fetch sample data", database_id=database_id, table_name=table_name, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch sample data: {str(e)}")


@router.post("/databases/{database_id}/tables/{table_name}/description")
async def api_generate_table_description(
    database_id: str, 
    table_name: str, 
    body: dict = Body(None),
    database_manager = Depends(get_database)
):
    """
    Generate AI description for a table.
    """
    try:
        regenerate = body.get("regenerate", False) if body else False
        schema = await database_manager.get_table_schema(table_name)
        sample_data = await database_manager.get_sample_data(table_name, 3)
        
        # Check if AIDescriptionService is available
        try:
            ai_service = AIDescriptionService()
            description = await ai_service.generate_table_description(table_name, schema, sample_data, regenerate)
        except Exception as ai_error:
            logger.warning("AI service unavailable, using fallback description", error=str(ai_error))
            # Fallback to simple description
            description = f"Data table containing {len(schema.get('columns', []))} columns"
        
        return {
            "description": description,
            "generatedAt": int(time.time()),
            "cached": not regenerate
        }
    except Exception as e:
        logger.error("Failed to generate table description", database_id=database_id, table_name=table_name, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate description: {str(e)}")


# Additional endpoints for enhanced schema information

@router.get("/quality", response_model=Dict[str, Any])
async def get_schema_quality(
    database_name: Optional[str] = None,
    database_manager = Depends(get_database)
) -> Dict[str, Any]:
    """
    Get schema quality metrics.
    """
    try:
        quality_metrics = await database_manager.get_quality_metrics(database_name)
        return {
            "database_name": database_name or "default",
            "quality_metrics": quality_metrics,
            "timestamp": int(time.time())
        }
    except Exception as e:
        logger.error("Failed to get quality metrics", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get quality metrics: {str(e)}")


@router.get("/business_domains", response_model=Dict[str, Any])
async def get_business_domains(
    database_name: Optional[str] = None,
    database_manager = Depends(get_database)
) -> Dict[str, Any]:
    """
    Get business domain classification of tables.
    """
    try:
        business_domains = await database_manager.get_business_domains(database_name)
        return {
            "database_name": database_name or "default",
            "business_domains": business_domains,
            "timestamp": int(time.time())
        }
    except Exception as e:
        logger.error("Failed to get business domains", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get business domains: {str(e)}")


@router.get("/performance", response_model=Dict[str, Any])
async def get_performance_insights(
    database_name: Optional[str] = None,
    database_manager = Depends(get_database)
) -> Dict[str, Any]:
    """
    Get performance insights for schema optimization.
    """
    try:
        performance_insights = await database_manager.get_performance_insights(database_name)
        return {
            "database_name": database_name or "default",
            "performance_insights": performance_insights,
            "timestamp": int(time.time())
        }
    except Exception as e:
        logger.error("Failed to get performance insights", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get performance insights: {str(e)}")


@router.post("/refresh_cache")
async def refresh_schema_cache(
    database_name: Optional[str] = None,
    database_manager = Depends(get_database)
) -> Dict[str, Any]:
    """
    Force refresh of schema cache.
    """
    try:
        await database_manager.refresh_schema_cache(database_name)
        return {
            "message": "Schema cache refreshed successfully",
            "database_name": database_name or "default",
            "timestamp": int(time.time())
        }
    except Exception as e:
        logger.error("Failed to refresh cache", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to refresh cache: {str(e)}")


@router.get("/stats", response_model=Dict[str, Any])
async def get_extraction_stats(database_manager = Depends(get_database)) -> Dict[str, Any]:
    """
    Get schema extraction statistics and performance metrics.
    """
    try:
        stats = database_manager.get_extraction_stats()
        return {
            "extraction_stats": stats,
            "timestamp": int(time.time())
        }
    except Exception as e:
        logger.error("Failed to get extraction stats", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get extraction stats: {str(e)}")