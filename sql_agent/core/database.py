"""Database management for SQL Agent."""

from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import create_engine, text, MetaData, Table, Column, inspect
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from .config import settings
from .state import QueryResult


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
            async with self._session_factory() as session:
                # Set timeout if specified
                if timeout:
                    await session.execute(text(f"SET statement_timeout = {timeout * 1000}"))
                
                # Execute query
                result = await session.execute(text(sql), parameters or {})
                
                # Fetch results
                if result.returns_rows:
                    rows = result.fetchall()
                    columns = list(result.keys()) if result.keys() else []
                    
                    # Convert to list of dicts
                    data = [dict(zip(columns, row)) for row in rows]
                    
                    return QueryResult(
                        data=data,
                        columns=columns,
                        row_count=len(data),
                        execution_time=time.time() - start_time,
                        sql_query=sql,
                    )
                else:
                    return QueryResult(
                        data=[],
                        columns=[],
                        row_count=0,
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
                tables = [dict(row) for row in result.fetchall()]
                
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
                    columns = [dict(row) for row in result.fetchall()]
                    
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
    
    async def get_sample_data(self, table_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get sample data from a table."""
        try:
            sql = f"SELECT * FROM {table_name} LIMIT {limit}"
            result = await self.execute_query(sql)
            
            if result.error:
                raise RuntimeError(result.error)
            
            return result.data
            
        except Exception as e:
            raise RuntimeError(f"Failed to get sample data: {e}")
    
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._async_engine is not None


# Global database manager instance
db_manager = DatabaseManager()


# Import time at the top to avoid circular imports
import time 