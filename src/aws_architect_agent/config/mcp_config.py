from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum

class ModelType(Enum):
    """Enum for different types of models."""
    GPT4 = "gpt-3.5-turbo"
    GPT35 = "gpt-3.5-turbo"
    CLAUDE = "claude-3-opus"
    CLAUDE_SONNET = "claude-3-sonnet"

@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""
    endpoint: str
    api_key: str
    model_type: ModelType
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 1

class MCPConfig:
    """Main configuration class for MCP servers."""
    
    def __init__(self):
        self.servers: Dict[str, MCPServerConfig] = {
            "primary": MCPServerConfig(
                endpoint="https://api.openai.com/v1",
                api_key="",  # Set this from environment
                model_type=ModelType.GPT4,
                temperature=0.7,
                max_tokens=2000
            ),
            "validator": MCPServerConfig(
                endpoint="https://api.openai.com/v1",
                api_key="",  # Set this from environment
                model_type=ModelType.GPT35,
                temperature=0.3,  # Lower temperature for more consistent validation
                max_tokens=1000
            ),
            "reviewer": MCPServerConfig(
                endpoint="https://api.openai.com/v1",
                api_key="",  # Set this from environment
                model_type=ModelType.GPT35,
                temperature=0.5,
                max_tokens=1500
            )
        }
    
    def get_server(self, name: str) -> MCPServerConfig:
        """Get configuration for a specific server."""
        if name not in self.servers:
            raise ValueError(f"Unknown server name: {name}")
        return self.servers[name]
    
    def update_api_keys(self, api_keys: Dict[str, str]) -> None:
        """Update API keys for all servers."""
        for server_name, config in self.servers.items():
            if server_name in api_keys:
                config.api_key = api_keys[server_name]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            name: {
                "endpoint": config.endpoint,
                "model_type": config.model_type.value,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "timeout": config.timeout,
                "retry_attempts": config.retry_attempts,
                "retry_delay": config.retry_delay
            }
            for name, config in self.servers.items()
        }


# Create a singleton instance
mcp_config = MCPConfig()