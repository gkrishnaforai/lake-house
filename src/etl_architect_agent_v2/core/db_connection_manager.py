"""Database connection manager for handling multiple database types."""

import logging
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import boto3
import mysql.connector
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool


logger = logging.getLogger(__name__)


class DatabaseConfig(BaseModel):
    """Database configuration."""
    type: str = Field(..., description="Database type (mysql, postgres, s3)")
    host: Optional[str] = Field(None, description="Database host")
    port: Optional[int] = Field(None, description="Database port")
    database: str = Field(..., description="Database name")
    username: Optional[str] = Field(None, description="Database username")
    password: Optional[str] = Field(None, description="Database password")
    region: Optional[str] = Field(None, description="AWS region for S3")
    bucket: Optional[str] = Field(None, description="S3 bucket name")
    pool_size: int = Field(default=5, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Max overflow connections")


class DatabaseConnectionManager:
    """Manager for database connections."""
    
    def __init__(self):
        """Initialize connection manager."""
        self.connections: Dict[str, Union[Engine, boto3.client]] = {}
        self.configs: Dict[str, DatabaseConfig] = {}
    
    def add_connection(
        self,
        name: str,
        config: DatabaseConfig
    ) -> None:
        """Add a new database connection.
        
        Args:
            name: Connection name
            config: Database configuration
        """
        if name in self.connections:
            raise ValueError(f"Connection {name} already exists")
        
        if config.type == "mysql":
            engine = create_engine(
                f"mysql+mysqlconnector://{config.username}:{config.password}@"
                f"{config.host}:{config.port}/{config.database}",
                poolclass=QueuePool,
                pool_size=config.pool_size,
                max_overflow=config.max_overflow
            )
            self.connections[name] = engine
            
        elif config.type == "postgres":
            engine = create_engine(
                f"postgresql://{config.username}:{config.password}@"
                f"{config.host}:{config.port}/{config.database}",
                poolclass=QueuePool,
                pool_size=config.pool_size,
                max_overflow=config.max_overflow
            )
            self.connections[name] = engine
            
        elif config.type == "s3":
            client = boto3.client(
                's3',
                region_name=config.region,
                aws_access_key_id=config.username,
                aws_secret_access_key=config.password
            )
            self.connections[name] = client
            
        else:
            raise ValueError(f"Unsupported database type: {config.type}")
        
        self.configs[name] = config
        logger.info(f"Added connection {name} for {config.type}")
    
    def get_connection(
        self,
        name: str
    ) -> Union[Engine, boto3.client]:
        """Get a database connection.
        
        Args:
            name: Connection name
            
        Returns:
            Database connection
        """
        if name not in self.connections:
            raise ValueError(f"Connection {name} does not exist")
        return self.connections[name]
    
    def remove_connection(self, name: str) -> None:
        """Remove a database connection.
        
        Args:
            name: Connection name
        """
        if name not in self.connections:
            raise ValueError(f"Connection {name} does not exist")
        
        if isinstance(self.connections[name], Engine):
            self.connections[name].dispose()
        
        del self.connections[name]
        del self.configs[name]
        logger.info(f"Removed connection {name}")
    
    def list_connections(self) -> Dict[str, str]:
        """List all database connections.
        
        Returns:
            Dictionary of connection names and types
        """
        return {
            name: config.type
            for name, config in self.configs.items()
        }
    
    def get_config(self, name: str) -> DatabaseConfig:
        """Get database configuration.
        
        Args:
            name: Connection name
            
        Returns:
            Database configuration
        """
        if name not in self.configs:
            raise ValueError(f"Connection {name} does not exist")
        return self.configs[name] 