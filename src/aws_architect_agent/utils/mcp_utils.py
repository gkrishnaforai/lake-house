import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from aws_architect_agent.config.mcp_config import MCPConfig, ModelType

# Load environment variables from .env file in current directory
load_dotenv(dotenv_path="/Users/krishnag/tools/llm/lanng-chain/web-app/ai_project/web-app/aws_architect_agent")

# Get API keys from environment variables
openai_api_key = "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"
anthropic_api_key = "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"


# Debug prints
print(f"Current directory: {os.getcwd()}")
print(f"OPENAI_API_KEY exists: {bool(openai_api_key)}")
print(f"ANTHROPIC_API_KEY exists: {bool(anthropic_api_key)}")

class MCPServerManager:
    """Manager for Model Control Protocol (MCP) servers.
    
    This class manages multiple LLM (Large Language Model) servers with different configurations
    and capabilities. It provides a centralized way to initialize, access, and manage these servers.
    
    Key Features:
    - Singleton pattern implementation for global access
    - Automatic initialization of servers from configuration
    - Support for multiple LLM providers (OpenAI, Anthropic)
    - Error handling and graceful degradation
    - API key management from environment variables
    
    Usage:
        # Get the singleton instance
        server_manager = MCPServerManager()
        
        # Get a specific server
        primary_server = server_manager.get_server("primary")
        
        # Get all configured servers
        all_servers = server_manager.get_all_servers()
    
    Configuration:
    - Servers are configured in MCPConfig
    - API keys are loaded from environment variables
    - Each server can have different model types and parameters
    
    Error Handling:
    - Missing API keys raise ValueError
    - Failed server initialization is logged but doesn't stop other servers
    - Invalid server names raise ValueError when requested
    """
    
    def __init__(self):
        self.servers: Dict[str, ChatOpenAI | ChatAnthropic] = {}
        self._initialize_servers()
    
    def _initialize_servers(self):
        """Initialize MCP servers with API keys from environment."""
        config = MCPConfig()
        
        for server_name, server_config in config.servers.items():
            try:
                # Get API key from environment
                api_key = self._get_api_key(server_config.model_type)
                if not api_key:
                    raise ValueError(
                        f"API key not found for {server_config.model_type}. "
                        f"Please set the appropriate environment variable."
                    )
                
                # Update config with API key
                server_config.api_key = api_key
                
                # Create server instance based on model type
                if server_config.model_type in [
                    ModelType.GPT4, ModelType.GPT35
                ]:
                    self.servers[server_name] = ChatOpenAI(
                        model_name=server_config.model_type.value,
                        temperature=server_config.temperature,
                        max_tokens=server_config.max_tokens,
                        timeout=server_config.timeout,
                        max_retries=server_config.retry_attempts,
                        api_key=server_config.api_key
                    )
                
            except Exception as e:
                print(
                    f"Warning: Failed to initialize {server_name} "
                    f"server: {str(e)}"
                )
                # Continue with other servers even if one fails
    
    def _get_api_key(self, model_type: ModelType) -> Optional[str]:
        """Get API key from environment based on model type."""
        if model_type in [ModelType.GPT4, ModelType.GPT35]:
            return openai_api_key
        elif model_type in [ModelType.CLAUDE, ModelType.CLAUDE_SONNET]:
            return anthropic_api_key
        return None
    
    def get_server(self, server_name: str) -> ChatOpenAI | ChatAnthropic:
        """Get a specific MCP server instance.
        
        Args:
            server_name: Name of the server to retrieve (e.g., "primary", "validator")
            
        Returns:
            The requested server instance
            
        Raises:
            ValueError: If the server name is not found
        """
        if server_name not in self.servers:
            raise ValueError(
                f"Server {server_name} not found. Available servers: "
                f"{list(self.servers.keys())}"
            )
        return self.servers[server_name]
    
    def get_all_servers(self) -> Dict[str, ChatOpenAI | ChatAnthropic]:
        """Get all configured MCP servers.
        
        Returns:
            Dictionary mapping server names to their instances
        """
        return self.servers


# Create a singleton instance
mcp_server_manager = MCPServerManager() 