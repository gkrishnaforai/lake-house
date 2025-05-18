from dataclasses import dataclass
from typing import List, Dict, Any
from .column_schema import ColumnSchema


@dataclass
class TableSchema:
    table_name: str           # You can default to "inferred_table"
    columns: List[ColumnSchema]

    def to_json(self) -> Dict[str, Any]:
        """Convert the table schema to a JSON-serializable dictionary.
        
        Returns:
            Dict containing the table name and a list of column definitions.
        """
        return {
            "table_name": self.table_name,
            "columns": [
                {
                    "name": col.name,
                    "type": col.type,
                    "nullable": col.nullable
                }
                for col in self.columns
            ]
        }
