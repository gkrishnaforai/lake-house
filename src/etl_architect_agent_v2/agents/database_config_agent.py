"""Database Configuration Agent.

This agent helps configure and set up different types of databases.
"""

from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field, SecretStr
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools import BaseTool
import json
import logging
import os
from pathlib import Path
from core.llm.manager import LLMManager


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseCredentials(BaseModel):
    """Secure database credentials model."""
    username: Optional[str] = Field(None, description="Database username")
    password: Optional[SecretStr] = Field(None, description="Database password")
    host: Optional[str] = Field(None, description="Database host")
    port: Optional[int] = Field(None, description="Database port")
    database: str = Field(..., description="Database name")
    
    class Config:
        """Pydantic config."""
        json_encoders = {
            SecretStr: lambda v: v.get_secret_value() if v else None
        }


class DatabaseConfig(BaseModel):
    """Database configuration model."""
    db_type: Literal[
        # Relational Databases
        'postgresql',
        'mysql',
        'mariadb',
        'oracle',
        'sqlserver',
        'sqlite',
        
        # NoSQL Databases
        'mongodb',
        'cassandra',
        'redis',
        'dynamodb',
        'couchdb',
        
        # Cloud Data Warehouses
        'snowflake',
        'bigquery',
        'redshift',
        'databricks',
        
        # Data Lakes/Lakehouses
        'aws_s3_lakehouse',
        'azure_data_lake',
        'gcp_data_lake',
        
        # Time Series Databases
        'influxdb',
        'timescaledb',
        'prometheus',
        
        # Graph Databases
        'neo4j',
        'janusgraph',
        'arangodb'
    ] = Field(..., description="Type of database to configure")
    
    # Credentials
    credentials: Optional[DatabaseCredentials] = Field(
        None,
        description="Database credentials"
    )
    
    # Cloud-specific parameters
    s3_bucket: Optional[str] = Field(None, description="S3 bucket")
    region: Optional[str] = Field(None, description="AWS region")
    project_id: Optional[str] = Field(None, description="GCP project ID")
    subscription_id: Optional[str] = Field(None, description="Azure subscription ID")
    
    # Database-specific parameters
    additional_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional configuration parameters"
    )


class DatabaseSetupResult(BaseModel):
    """Result of database setup."""
    success: bool = Field(..., description="Whether setup was successful")
    message: str = Field(..., description="Result message")
    connection_string: Optional[str] = Field(
        None,
        description="Connection string"
    )
    setup_commands: List[str] = Field(
        default_factory=list,
        description="Commands used for setup"
    )
    validation_queries: List[str] = Field(
        default_factory=list,
        description="Queries to validate setup"
    )
    security_recommendations: List[str] = Field(
        default_factory=list,
        description="Security recommendations"
    )
    performance_tips: List[str] = Field(
        default_factory=list,
        description="Performance optimization tips"
    )


class DatabaseConfigTool(BaseTool):
    """Tool for configuring databases."""
    
    name: str = "configure_database"
    description: str = "Configure and set up a database"
    args_schema: type[BaseModel] = DatabaseConfig
    llm: Any = None
    
    def __init__(self, llm):
        """Initialize the tool with an LLM instance."""
        super().__init__()
        self.llm = llm
    
    def _run(
        self,
        config: DatabaseConfig
    ) -> Dict[str, Any]:
        """Configure database."""
        try:
            logger.debug(f"Configuring database: {config.dict()}")
            
            # Create a prompt for database configuration
            prompt = f"""Configure a {config.db_type} database with these parameters:
            {json.dumps(config.dict(), indent=2)}
            
            Generate:
            1. Setup commands
            2. Connection string
            3. Validation queries
            4. Best practices configuration
            5. Security recommendations
            6. Performance optimization tips
            
            Return in this JSON format:
            {{
                "success": true,
                "message": "Setup instructions",
                "connection_string": "connection string",
                "setup_commands": ["command1", "command2"],
                "validation_queries": ["query1", "query2"],
                "security_recommendations": ["rec1", "rec2"],
                "performance_tips": ["tip1", "tip2"]
            }}"""
            
            # Get response from LLM
            response = self.llm.invoke(prompt)
            
            # Extract content from AIMessage
            if hasattr(response, 'content'):
                response_content = response.content
            else:
                response_content = str(response)
            
            # Parse the response
            try:
                # Try to parse as JSON directly
                result = json.loads(response_content)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from markdown
                import re
                json_match = re.search(
                    r'```(?:json)?\s*(\{[\s\S]*?\})\s*```',
                    response_content
                )
                if json_match:
                    result = json.loads(json_match.group(1))
                else:
                    raise ValueError(
                        "Could not find valid JSON in response"
                    )
            
            return result
            
        except Exception as e:
            logger.error(
                f"Error configuring database: {str(e)}", exc_info=True
            )
            raise ValueError(f"Error configuring database: {str(e)}")


class DatabaseConfigAgent:
    """Agent for configuring databases."""
    
    def __init__(self, llm_manager: LLMManager):
        """Initialize the agent."""
        self.llm = llm_manager.llm
        self.tools = [DatabaseConfigTool(llm=self.llm)]
        
        # Create the agent prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a database configuration expert. Your task is to help 
            users set up and configure their preferred database system. Follow these 
            guidelines:
            1. Validate input parameters
            2. Generate appropriate setup commands
            3. Provide connection strings
            4. Include validation queries
            5. Follow security best practices
            6. Provide performance optimization tips
            7. Return only the raw configuration without markdown formatting"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create the agent
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Create the agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True
        )
    
    async def configure_database(
        self,
        db_type: str,
        database: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        region: Optional[str] = None,
        project_id: Optional[str] = None,
        subscription_id: Optional[str] = None,
        additional_config: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> DatabaseSetupResult:
        """Configure a database."""
        try:
            logger.debug("Starting database configuration")
            
            # Prepare chat history
            messages = []
            if chat_history:
                for msg in chat_history:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    else:
                        messages.append(AIMessage(content=msg["content"]))
            
            # Create credentials
            credentials = DatabaseCredentials(
                username=username,
                password=SecretStr(password) if password else None,
                host=host,
                port=port,
                database=database
            )
            
            # Create configuration
            config = DatabaseConfig(
                db_type=db_type,
                credentials=credentials,
                s3_bucket=s3_bucket,
                region=region,
                project_id=project_id,
                subscription_id=subscription_id,
                additional_config=additional_config or {}
            )
            
            # Call the tool directly
            db_config = self.tools[0]  # Get the DatabaseConfigTool instance
            result = db_config._run(config)
            
            # Convert to DatabaseSetupResult
            setup_result = DatabaseSetupResult(
                success=result.get("success", False),
                message=result.get("message", ""),
                connection_string=result.get("connection_string"),
                setup_commands=result.get("setup_commands", []),
                validation_queries=result.get("validation_queries", []),
                security_recommendations=result.get(
                    "security_recommendations", []
                ),
                performance_tips=result.get("performance_tips", [])
            )
            
            return setup_result
            
        except Exception as e:
            logger.error(f"Error configuring database: {str(e)}", exc_info=True)
            raise ValueError(f"Error configuring database: {str(e)}")
    
    @staticmethod
    def load_credentials_from_env() -> Dict[str, Any]:
        """Load database credentials from environment variables."""
        return {
            'username': os.getenv('DB_USERNAME'),
            'password': os.getenv('DB_PASSWORD'),
            'host': os.getenv('DB_HOST'),
            'port': int(os.getenv('DB_PORT', '0')) or None,
            'database': os.getenv('DB_NAME')
        }
    
    @staticmethod
    def load_credentials_from_file(
        file_path: str = '~/.config/db_credentials.json'
    ) -> Dict[str, Any]:
        """Load database credentials from a JSON file."""
        try:
            with open(Path(file_path).expanduser(), 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading credentials: {str(e)}")
            return {} 