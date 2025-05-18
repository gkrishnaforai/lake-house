from dataclasses import dataclass


@dataclass
class ColumnSchema:
    name: str           # Column name, e.g., "user_id"
    type: str           # SQL type, e.g., "INTEGER", "VARCHAR"
    nullable: bool      # Whether this column can be NULL

    @property
    def data_type(self) -> str:
        """Alias for type field for backward compatibility."""
        return self.type
