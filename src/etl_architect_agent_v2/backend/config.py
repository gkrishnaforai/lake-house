"""Configuration settings for the application."""

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List
import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

class Settings(BaseSettings):
    """Application settings."""
    
    # AWS Settings
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_SESSION_TOKEN: str
    AWS_REGION: str = "us-east-1"
    AWS_S3_BUCKET: str
    AWS_DEFAULT_REGION: str = "us-east-1"
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False
    CORS_ORIGINS: str = "http://localhost:3000"  # Comma-separated list of origins
    
    # File Upload Settings
    MAX_UPLOAD_SIZE: int = 10485760  # 10MB
    ALLOWED_FILE_TYPES: str = ".csv,.xlsx,.xls"  # Comma-separated list of file types
    
    # Environment Settings
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    
    # LLM Settings
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: str
    
    # Database Settings
    GLUE_DATABASE_NAME: str = "data_lakehouse"
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Get CORS origins as a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
    @property
    def allowed_file_types_list(self) -> List[str]:
        """Get allowed file types as a list."""
        return [ftype.strip() for ftype in self.ALLOWED_FILE_TYPES.split(",")]
    
    class Config:
        """Pydantic config."""
        env_file = str(PROJECT_ROOT / "src" / "aws_architect_agent" / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def create_default_env():
    """Create default .env file if it doesn't exist."""
    env_path = PROJECT_ROOT / "src" / "aws_architect_agent" / ".env"
    if not env_path.exists():
        env_content = """# AWS Configuration
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_SESSION_TOKEN=your-session-token
AWS_REGION=us-east-1
AWS_S3_BUCKET=your-bucket-name
AWS_DEFAULT_REGION=us-east-1
GLUE_DATABASE_NAME=data_lakehouse

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# CORS Configuration
CORS_ORIGINS=http://localhost:3000

# File Upload Configuration
MAX_UPLOAD_SIZE=10485760
ALLOWED_FILE_TYPES=.csv,.xlsx,.xls

# Environment Settings
ENVIRONMENT=development
LOG_LEVEL=INFO

# LLM Settings
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
"""
        env_path.write_text(env_content)
        print("Created default .env file. Please update it with your actual values.") 