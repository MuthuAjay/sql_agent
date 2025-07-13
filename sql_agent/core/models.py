from typing import Optional, Literal
from pydantic import BaseModel

class Table(BaseModel):
    name: str
    type: Literal['table', 'view']
    table_schema: Optional[str] = None  # Renamed from 'schema' to avoid conflict
    rowCount: Optional[int] = None
    size: Optional[str] = None
    description: Optional[str] = None
    lastDescriptionUpdate: Optional[str] = None 