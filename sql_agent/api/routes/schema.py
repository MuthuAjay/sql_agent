"""
Schema Routes

This module contains endpoints for database schema information.
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
        
        # Convert to API models
        tables = []
        total_columns = 0
        
        for table_name, table_info in schema_info.items():
            columns = []
            for col_info in table_info.get("columns", []):
                column = ColumnInfo(
                    name=col_info.get("column_name", "unknown"),
                    type=col_info.get("data_type", "unknown"),
                    nullable=col_info.get("is_nullable", "YES") == "YES",
                    primary_key=False,  # Would need additional query to determine
                    foreign_key=None
                )
                columns.append(column)
                total_columns += 1
            
            table = TableInfo(
                name=table_name,
                columns=columns,
                row_count=None,  # Would need additional query
                description=None
            )
            tables.append(table)
        
        return SchemaResponse(
            database_name=database_name or "default",
            tables=tables,
            relationships=[],  # Would need additional query
            total_tables=len(tables),
            total_columns=total_columns
        )
        
    except Exception as e:
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
        for table_name, table_info in schema_info.items():
            tables.append({
                "name": table_name,
                "column_count": len(table_info.get("columns", [])),
                "row_count": None,
                "description": None
            })
        
        return {
            "database_name": database_name or "default",
            "tables": tables,
            "total": len(tables)
        }
        
    except Exception as e:
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
        
        if table_name not in schema_info:
            raise HTTPException(
                status_code=404,
                detail=f"Table '{table_name}' not found"
            )
        
        table_info = schema_info[table_name]
        
        # Convert to API model
        columns = []
        for col_info in table_info.get("columns", []):
            column = ColumnInfo(
                name=col_info.get("column_name", "unknown"),
                type=col_info.get("data_type", "unknown"),
                nullable=col_info.get("is_nullable", "YES") == "YES",
                primary_key=False,
                foreign_key=None
            )
            columns.append(column)
        
        return TableInfo(
            name=table_name,
            columns=columns,
            row_count=None,
            description=None
        )
        
    except Exception as e:
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
        
        for table_name, table_info in schema_info.items():
            # Search in table names
            if query_lower in table_name.lower():
                results.append({
                    "type": "table",
                    "name": table_name,
                    "match": "table_name"
                })
            
            # Search in column names
            for col_info in table_info.get("columns", []):
                col_name = col_info.get("column_name", "")
                if query_lower in col_name.lower():
                    results.append({
                        "type": "column",
                        "table": table_name,
                        "name": col_name,
                        "match": "column_name"
                    })
        
        return {
            "query": query,
            "database_name": database_name or "default",
            "results": results,
            "total_matches": len(results)
        }
        
    except Exception as e:
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
        # For now, return empty relationships
        # Would need additional queries to detect foreign keys
        relationships = []
        
        return {
            "database_name": database_name or "default",
            "relationships": relationships,
            "total": len(relationships)
        }
        
    except Exception as e:
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
            "total_returned": len(sample_data)
        }
        
    except Exception as e:
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
        db_type = settings.database_type
        db_name = settings.database_url.split("/")[-1] or "default"
        return [
            DatabaseInfo(
                id="default",
                name=db_name,
                type=db_type,
                status=db_status,
                lastSync=""
            )
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get databases: {e}") 


@router.get("/databases/{database_id}/tables", response_model=List[Table])
async def list_tables_endpoint(database_id: str, database_manager = Depends(get_database)):
    """
    List all tables for a given database.
    """
    try:
        tables = await database_manager.list_tables(database_id)
        return tables
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tables: {e}") 

@router.get("/databases/{database_id}/tables")
async def api_get_tables(database_id: str):
    try:
        tables = await db_manager.get_tables()
        return {"tables": tables}
    except Exception as e:
        logger.error("Failed to fetch tables", database_id=database_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch tables: {str(e)}")

@router.get("/databases/{database_id}/tables/{table_name}/schema")
async def api_get_table_schema(database_id: str, table_name: str):
    try:
        schema = await db_manager.get_table_schema(table_name)
        return schema
    except Exception as e:
        logger.error("Failed to fetch table schema", database_id=database_id, table_name=table_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch table schema: {str(e)}")

@router.get("/databases/{database_id}/tables/{table_name}/sample")
async def api_get_sample_data(database_id: str, table_name: str, limit: int = 5):
    try:
        sample_data = await db_manager.get_sample_data(table_name, limit)
        return sample_data
    except Exception as e:
        logger.error("Failed to fetch sample data", database_id=database_id, table_name=table_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch sample data: {str(e)}")

@router.post("/databases/{database_id}/tables/{table_name}/description")
async def api_generate_table_description(database_id: str, table_name: str, body: dict = Body(None)):
    try:
        regenerate = body.get("regenerate", False) if body else False
        schema = await db_manager.get_table_schema(table_name)
        sample_data = await db_manager.get_sample_data(table_name, 3)
        ai_service = AIDescriptionService()
        description = await ai_service.generate_table_description(table_name, schema, sample_data, regenerate)
        return {
            "description": description,
            "generatedAt": int(time.time()),
            "cached": not regenerate
        }
    except Exception as e:
        logger.error("Failed to generate table description", database_id=database_id, table_name=table_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to generate description: {str(e)}") 