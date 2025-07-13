"""Database management for SQL Agent."""

from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import create_engine, text, MetaData, Table, Column, inspect
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from .config import settings
from .state import QueryResult
from sql_agent.core.models import Table


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self):
        self._engine: Optional[Engine] = None
        self._async_engine = None
        self._session_factory = None
        self._metadata: Optional[MetaData] = None
    
    async def initialize(self) -> None:
        """Initialize database connection."""
        try:
            # Create async engine
            self._async_engine = create_async_engine(
                settings.database_url,
                pool_size=settings.database_pool_size,
                max_overflow=settings.database_max_overflow,
                echo=settings.debug,
            )
            
            # Create session factory
            self._session_factory = async_sessionmaker(
                self._async_engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            
            # Test connection
            async with self._async_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
                
        except SQLAlchemyError as e:
            raise RuntimeError(f"Failed to initialize database: {e}")
    
    async def close(self) -> None:
        """Close database connections."""
        if self._async_engine:
            await self._async_engine.dispose()
    
    async def execute_query(
        self, 
        sql: str, 
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> QueryResult:
        """Execute a SQL query and return results."""
        start_time = time.time()
        
        try:
            if not self._session_factory:
                raise RuntimeError("Session factory is not initialized.")
            async with self._session_factory() as session:
                # Set timeout if specified
                if timeout:
                    await session.execute(text(f"SET statement_timeout = {timeout * 1000}"))
                
                # Execute query
                result = await session.execute(text(sql), parameters or {})
                
                # Fetch results
                rows = result.mappings().all()
                columns = list(result.keys()) if result.keys() else []
                data = [dict(row) for row in rows]
                return QueryResult(
                    data=data,
                    columns=columns,
                    row_count=len(data),
                    execution_time=time.time() - start_time,
                    sql_query=sql,
                )
        
        except SQLAlchemyError as e:
            return QueryResult(
                data=[],
                columns=[],
                row_count=0,
                execution_time=time.time() - start_time,
                sql_query=sql,
                error=str(e),
            )
    
    async def get_schema_info(self) -> Dict[str, Any]:
        """Get database schema information."""
        try:
            if not self._async_engine:
                raise RuntimeError("Async engine is not initialized.")
            async with self._async_engine.begin() as conn:
                # Get table information
                tables_query = """
                SELECT 
                    table_name,
                    table_type
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
                """
                
                result = await conn.execute(text(tables_query))
                tables = result.mappings().all()
                
                # Get column information for each table
                schema_info = {}
                for table in tables:
                    table_name = table['table_name']
                    columns_query = """
                    SELECT 
                        column_name,
                        data_type,
                        is_nullable,
                        column_default
                    FROM information_schema.columns 
                    WHERE table_name = :table_name
                    ORDER BY ordinal_position
                    """
                    
                    result = await conn.execute(text(columns_query), {"table_name": table_name})
                    columns = result.mappings().all()
                    
                    schema_info[table_name] = {
                        "table_type": table['table_type'],
                        "columns": columns,
                    }
                
                return schema_info
                
        except SQLAlchemyError as e:
            raise RuntimeError(f"Failed to get schema info: {e}")
    
    async def validate_query(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Validate a SQL query without executing it."""
        try:
            # This is a simplified validation - in production, you'd want more sophisticated validation
            sql_lower = sql.lower().strip()
            
            # Check for dangerous operations
            dangerous_keywords = [
                "drop", "delete", "truncate", "alter", "create", "insert", "update"
            ]
            
            for keyword in dangerous_keywords:
                if keyword in sql_lower:
                    return False, f"Query contains potentially dangerous keyword: {keyword}"
            
            # Check for basic SQL syntax (simplified)
            if not sql_lower.startswith("select"):
                return False, "Only SELECT queries are allowed"
            
            return True, None
            
        except Exception as e:
            return False, f"Query validation failed: {e}"
    
    async def get_tables(self) -> List[dict]:
        """Get all tables and views in the database."""
        if not self._async_engine:
            raise RuntimeError("Async engine is not initialized.")
        async with self._async_engine.begin() as conn:
            query = """
            SELECT 
                table_name, 
                table_type, 
                table_schema
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
            """
            result = await conn.execute(text(query))
            tables = []
            for row in result:
                tables.append({
                    "name": row.table_name,
                    "type": row.table_type.lower() if row.table_type else 'table',
                    "schema": row.table_schema,
                    "rowCount": None,
                    "size": None,
                    "description": None,
                    "lastDescriptionUpdate": None
                })
            return tables

    async def get_table_schema(self, table_name: str) -> dict:
        """Get detailed schema for a specific table."""
        if not self._async_engine:
            raise RuntimeError("Async engine is not initialized.")
        async with self._async_engine.begin() as conn:
            columns_query = """
            SELECT 
                column_name, 
                data_type, 
                is_nullable, 
                column_default
            FROM information_schema.columns 
            WHERE table_name = :table_name
            ORDER BY ordinal_position
            """
            result = await conn.execute(text(columns_query), {"table_name": table_name})
            columns = []
            for row in result:
                columns.append({
                    "name": row.column_name,
                    "type": row.data_type,
                    "nullable": row.is_nullable == "YES",
                    "primaryKey": False,  # To be filled below
                    "foreignKey": None,
                    "defaultValue": row.column_default,
                    "constraints": []
                })
            # Mark primary keys
            pk_query = """
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = :table_name::regclass AND i.indisprimary;
            """
            pk_result = await conn.execute(text(pk_query), {"table_name": table_name})
            pk_columns = {row.attname for row in pk_result}
            for col in columns:
                if col["name"] in pk_columns:
                    col["primaryKey"] = True
            # Foreign keys
            fk_query = """
            SELECT
                kcu.column_name,
                ccu.table_name AS referenced_table,
                ccu.column_name AS referenced_column
            FROM 
                information_schema.table_constraints AS tc 
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                  AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name
                  AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = :table_name;
            """
            fk_result = await conn.execute(text(fk_query), {"table_name": table_name})
            foreign_keys = []
            for row in fk_result:
                for col in columns:
                    if col["name"] == row.column_name:
                        col["foreignKey"] = {
                            "referencedTable": row.referenced_table,
                            "referencedColumn": row.referenced_column
                        }
                foreign_keys.append({
                    "columnName": row.column_name,
                    "referencedTable": row.referenced_table,
                    "referencedColumn": row.referenced_column
                })
            # Indexes (simplified)
            indexes = []
            index_query = """
            SELECT
                indexname,
                indexdef
            FROM pg_indexes
            WHERE tablename = :table_name;
            """
            idx_result = await conn.execute(text(index_query), {"table_name": table_name})
            for row in idx_result:
                # Parse indexdef for columns and uniqueness
                idx_cols = []
                unique = "UNIQUE" in row.indexdef
                if "(" in row.indexdef and ")" in row.indexdef:
                    idx_cols = row.indexdef.split("(")[1].split(")")[0].replace('"', '').split(', ')
                indexes.append({
                    "name": row.indexname,
                    "columns": idx_cols,
                    "unique": unique
                })
            return {
                "tableName": table_name,
                "columns": columns,
                "indexes": indexes,
                "foreignKeys": foreign_keys
            }

    async def get_sample_data(self, table_name: str, limit: int = 5) -> dict:
        """Get sample data from a table."""
        if not self._async_engine:
            raise RuntimeError("Async engine is not initialized.")
        async with self._async_engine.begin() as conn:
            sql = f"SELECT * FROM {table_name} LIMIT {limit}"
            result = await conn.execute(text(sql))
            rows = result.fetchall()
            columns = result.keys()
            return {
                "columns": list(columns),
                "rows": [list(row) for row in rows]
            }
    
    async def list_tables(self, database_id: str) -> List[Table]:
        # For now, ignore database_id if only one DB is used
        if not self._async_engine:
            raise RuntimeError("Async engine is not initialized.")
        async with self._async_engine.begin() as conn:
            result = await conn.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            )
            tables = [Table(name=row[0], type='table') for row in result]
            return tables
    
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._async_engine is not None

    async def test_connection(self) -> bool:
        """Test database connection."""
        try:
            if not self._async_engine:
                return False
            async with self._async_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False


# Global database manager instance
db_manager = DatabaseManager()


# Import time at the top to avoid circular imports
import time 