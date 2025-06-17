"""
Schema Routes

This module contains endpoints for database schema information.
"""

from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Depends, Request

from sql_agent.api.models import (
    SchemaResponse, TableInfo, ColumnInfo
)

router = APIRouter()


def get_database() -> Any:
    """Get the database manager instance."""
    from sql_agent.api.main import database_manager
    if database_manager is None:
        raise HTTPException(status_code=503, detail="Database manager not initialized")
    return database_manager


@router.get("/", response_model=SchemaResponse)
async def get_schema(
    database_name: Optional[str] = None,
    req: Optional[Request] = None,
    database_manager = Depends(get_database)
) -> SchemaResponse:
    """
    Get database schema information.
    
    This endpoint returns comprehensive schema information including
    tables, columns, relationships, and metadata.
    """
    try:
        # Get schema information from database manager
        schema_info = await database_manager.get_schema(database_name=database_name)
        
        # Convert to API models
        tables = []
        total_columns = 0
        
        for table_name, table_info in schema_info.get("tables", {}).items():
            columns = []
            for col_name, col_info in table_info.get("columns", {}).items():
                column = ColumnInfo(
                    name=col_name,
                    type=col_info.get("type", "unknown"),
                    nullable=col_info.get("nullable", True),
                    primary_key=col_info.get("primary_key", False),
                    foreign_key=col_info.get("foreign_key")
                )
                columns.append(column)
                total_columns += 1
            
            table = TableInfo(
                name=table_name,
                columns=columns,
                row_count=table_info.get("row_count"),
                description=table_info.get("description")
            )
            tables.append(table)
        
        return SchemaResponse(
            database_name=database_name or "default",
            tables=tables,
            relationships=schema_info.get("relationships", []),
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
    req: Optional[Request] = None,
    database_manager = Depends(get_database)
) -> Dict[str, Any]:
    """
    Get list of tables in the database.
    
    This endpoint returns a list of all tables with basic information.
    """
    try:
        # Get tables from database manager
        tables_info = await database_manager.get_tables(database_name=database_name)
        
        tables = []
        for table_name, table_info in tables_info.items():
            tables.append({
                "name": table_name,
                "column_count": len(table_info.get("columns", {})),
                "row_count": table_info.get("row_count"),
                "description": table_info.get("description")
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
    req: Optional[Request] = None,
    database_manager = Depends(get_database)
) -> TableInfo:
    """
    Get detailed information about a specific table.
    
    This endpoint returns comprehensive information about a table
    including all columns and their properties.
    """
    try:
        # Get table information from database manager
        table_info = await database_manager.get_table_info(
            table_name=table_name,
            database_name=database_name
        )
        
        # Convert to API model
        columns = []
        for col_name, col_info in table_info.get("columns", {}).items():
            column = ColumnInfo(
                name=col_name,
                type=col_info.get("type", "unknown"),
                nullable=col_info.get("nullable", True),
                primary_key=col_info.get("primary_key", False),
                foreign_key=col_info.get("foreign_key")
            )
            columns.append(column)
        
        return TableInfo(
            name=table_name,
            columns=columns,
            row_count=table_info.get("row_count"),
            description=table_info.get("description")
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
    req: Optional[Request] = None,
    database_manager = Depends(get_database)
) -> Dict[str, Any]:
    """
    Search schema by keywords.
    
    This endpoint searches for tables and columns that match
    the provided keywords.
    """
    try:
        # Search schema using database manager
        search_results = await database_manager.search_schema(
            query=query,
            database_name=database_name
        )
        
        return {
            "query": query,
            "database_name": database_name or "default",
            "results": search_results,
            "total_matches": len(search_results)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search schema: {str(e)}"
        )


@router.get("/relationships", response_model=Dict[str, Any])
async def get_relationships(
    database_name: Optional[str] = None,
    req: Optional[Request] = None,
    database_manager = Depends(get_database)
) -> Dict[str, Any]:
    """
    Get database relationships.
    
    This endpoint returns information about foreign key relationships
    between tables.
    """
    try:
        # Get relationships from database manager
        relationships = await database_manager.get_relationships(database_name=database_name)
        
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
    req: Optional[Request] = None,
    database_manager = Depends(get_database)
) -> Dict[str, Any]:
    """
    Get sample data from a table.
    
    This endpoint returns sample data from the specified table
    to help understand the data structure.
    """
    try:
        # Get sample data from database manager
        sample_data = await database_manager.get_sample_data(
            table_name=table_name,
            limit=limit,
            database_name=database_name
        )
        
        return {
            "table_name": table_name,
            "database_name": database_name or "default",
            "sample_data": sample_data.get("data", []),
            "columns": sample_data.get("columns", []),
            "row_count": len(sample_data.get("data", [])),
            "total_rows": sample_data.get("total_rows", 0)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get sample data: {str(e)}"
        ) 