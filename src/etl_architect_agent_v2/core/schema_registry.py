"""Schema registry for managing schema versions."""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field
import json
from pathlib import Path


logger = logging.getLogger(__name__)


class SchemaVersion(BaseModel):
    """Schema version information."""
    version: int = Field(..., description="Version number")
    schema: Dict[str, Any] = Field(..., description="Schema definition")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field(..., description="User who created the version")
    description: Optional[str] = Field(None, description="Version description")


class SchemaMetadata(BaseModel):
    """Schema metadata."""
    name: str = Field(..., description="Schema name")
    type: str = Field(..., description="Schema type (e.g., table, view)")
    database: str = Field(..., description="Database name")
    current_version: int = Field(..., description="Current version number")
    versions: List[SchemaVersion] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SchemaRegistry:
    """Registry for managing schema versions."""
    
    def __init__(self, storage_path: str = "schemas"):
        """Initialize schema registry.
        
        Args:
            storage_path: Path to store schema versions
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.schemas: Dict[str, SchemaMetadata] = {}
        self._load_schemas()
    
    def _load_schemas(self):
        """Load all schemas from storage."""
        try:
            for schema_file in self.storage_path.glob("*.json"):
                with open(schema_file, 'r') as f:
                    schema_data = json.load(f)
                    schema = SchemaMetadata(**schema_data)
                    self.schemas[schema.name] = schema
        except Exception as e:
            logger.error(f"Error loading schemas: {str(e)}", exc_info=True)
    
    def _save_schema(self, schema: SchemaMetadata):
        """Save schema to storage."""
        try:
            schema_file = self.storage_path / f"{schema.name}.json"
            with open(schema_file, 'w') as f:
                json.dump(schema.dict(), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving schema: {str(e)}", exc_info=True)
            raise
    
    def register_schema(
        self,
        name: str,
        schema: Dict[str, Any],
        schema_type: str,
        database: str,
        user: str,
        description: Optional[str] = None
    ) -> SchemaMetadata:
        """Register a new schema.
        
        Args:
            name: Schema name
            schema: Schema definition
            schema_type: Type of schema
            database: Database name
            user: User creating the schema
            description: Optional description
            
        Returns:
            Schema metadata
        """
        if name in self.schemas:
            raise ValueError(f"Schema {name} already exists")
        
        version = SchemaVersion(
            version=1,
            schema=schema,
            created_by=user,
            description=description
        )
        
        metadata = SchemaMetadata(
            name=name,
            type=schema_type,
            database=database,
            current_version=1,
            versions=[version]
        )
        
        self.schemas[name] = metadata
        self._save_schema(metadata)
        return metadata
    
    def add_version(
        self,
        name: str,
        schema: Dict[str, Any],
        user: str,
        description: Optional[str] = None
    ) -> SchemaMetadata:
        """Add a new version to an existing schema.
        
        Args:
            name: Schema name
            schema: New schema definition
            user: User creating the version
            description: Optional description
            
        Returns:
            Updated schema metadata
        """
        if name not in self.schemas:
            raise ValueError(f"Schema {name} does not exist")
        
        metadata = self.schemas[name]
        new_version = SchemaVersion(
            version=metadata.current_version + 1,
            schema=schema,
            created_by=user,
            description=description
        )
        
        metadata.versions.append(new_version)
        metadata.current_version = new_version.version
        metadata.updated_at = datetime.utcnow()
        
        self._save_schema(metadata)
        return metadata
    
    def get_schema(self, name: str, version: Optional[int] = None) -> SchemaMetadata:
        """Get schema metadata.
        
        Args:
            name: Schema name
            version: Optional version number
            
        Returns:
            Schema metadata
        """
        if name not in self.schemas:
            raise ValueError(f"Schema {name} does not exist")
        
        metadata = self.schemas[name]
        if version is not None:
            for v in metadata.versions:
                if v.version == version:
                    return metadata
            raise ValueError(f"Version {version} not found for schema {name}")
        
        return metadata
    
    def list_schemas(self) -> List[str]:
        """List all schema names.
        
        Returns:
            List of schema names
        """
        return list(self.schemas.keys())
    
    def delete_schema(self, name: str):
        """Delete a schema.
        
        Args:
            name: Schema name
        """
        if name not in self.schemas:
            raise ValueError(f"Schema {name} does not exist")
        
        schema_file = self.storage_path / f"{name}.json"
        if schema_file.exists():
            schema_file.unlink()
        
        del self.schemas[name] 