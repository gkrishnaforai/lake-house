from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    # OpenAI settings
    openai_api_key: str = Field(
        default=..., validation_alias="OPENAI_API_KEY"
    )
    openai_model: str = Field(
        default="gpt-4-turbo-preview", validation_alias="OPENAI_MODEL"
    )
    openai_temperature: float = Field(
        default=0.7, validation_alias="OPENAI_TEMPERATURE"
    )
    openai_max_tokens: int = Field(
        default=2000, validation_alias="OPENAI_MAX_TOKENS"
    )

    # AWS settings
    aws_access_key_id: Optional[str] = Field(
        default=None, validation_alias="AWS_ACCESS_KEY_ID"
    )
    aws_secret_access_key: Optional[str] = Field(
        default=None, validation_alias="AWS_SECRET_ACCESS_KEY"
    )
    aws_region: str = Field(
        default="us-east-1", validation_alias="AWS_REGION"
    )
    aws_default_region: str = Field(
        default="us-east-1", validation_alias="AWS_DEFAULT_REGION"
    )

    # Application settings
    debug: bool = Field(
        default=False, validation_alias="DEBUG"
    )
    log_level: str = Field(
        default="INFO", validation_alias="LOG_LEVEL"
    )
    max_conversation_history: int = Field(
        default=100, validation_alias="MAX_CONVERSATION_HISTORY"
    )
    environment: str = Field(
        default="development", validation_alias="ENVIRONMENT"
    )

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


# Create global settings instance
settings = Settings() 