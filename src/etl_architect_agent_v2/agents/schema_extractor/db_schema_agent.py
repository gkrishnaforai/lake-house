"""Database Schema Agent.

This agent converts JSON schema to database schema using LangChain.
"""

from pydantic import BaseModel, Field, field_validator 
from typing import Dict, List, Any, Optional, Union
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools import BaseTool
import json
import logging
from enum import Enum
from core.llm.manager import LLMManager


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TableSchemaModel(BaseModel):
    """Database table schema model."""
    table_name: str = Field(..., description="Name of the table")
    columns: List[Dict[str, Any]] = Field(..., description="List of columns")
    primary_key: List[str] = Field(default_factory=list)
    foreign_keys: List[Dict[str, Any]] = Field(default_factory=list)
    indexes: List[Dict[str, Any]] = Field(default_factory=list)


class DatabaseType(str, Enum):
    """Supported database types."""
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    ORACLE = "oracle"
    SQLSERVER = "sqlserver"
    SNOWFLAKE = "snowflake"
    REDSHIFT = "redshift"
    S3_LAKEHOUSE = "s3_lakehouse"


class SchemaConverterInput(BaseModel):
    json_schema: Dict[str, Any] = Field(
        ...,
        description="JSON schema to convert (must be a valid dictionary)"
    )
    db_type: str = Field(
        ...,
        description="Type of database to convert to (must be one of: mysql, postgresql, oracle, sqlserver, snowflake, redshift, s3_lakehouse)"
    )
    
    @field_validator('db_type')  # Note: field_validator instead of validator
    @classmethod  # Note: Must be a classmethod in v2
    def validate_db_type(cls, v: str) -> str:
        valid_types = ['mysql', 'postgresql', 'oracle', 'sqlserver', 'snowflake', 'redshift', 's3_lakehouse']
        if v.lower() not in valid_types:
            raise ValueError(f"Invalid db_type. Must be one of: {', '.join(valid_types)}")
        return v.lower()


class SchemaConverterTool(BaseTool):
    """Tool for converting JSON schema to database schema."""
    
    name: str = "convert_schema"
    description: str = (
       "Convert JSON schema to database schema.\n"
       "REQUIRED PARAMETERS:\n"
       "1. json_schema: The JSON schema to convert (must be a valid dictionary)\n"
       "2. db_type: Type of database (must be one of: mysql, postgresql, oracle, sqlserver, snowflake, redshift, s3_lakehouse)\n"
       "FORMAT: Provide both parameters in a dictionary: {{'json_schema': <dict>, 'db_type': <str>}}\n"
       "ERROR CASES:\n"
       "- If json_schema is missing: 'Error: json_schema is required'\n"
       "- If db_type is missing: 'Error: db_type is required'\n"
       "- If db_type is invalid: 'Error: Invalid db_type. Must be one of: [valid types]'"
    )
    args_schema: type[BaseModel] = SchemaConverterInput
    llm: Any = None
    
    def __init__(self, llm: Any = None, **kwargs):
        """Initialize the tool with an optional LLM."""
        super().__init__(**kwargs)
        self.llm = llm

    def _run(
        self,
        input_data: SchemaConverterInput
    ) -> List[Dict[str, Any]]:
        """Convert JSON schema to database schema."""
        try:
            if not self.llm:
                raise ValueError("LLM not initialized. Please provide an LLM instance.")
                
            logger.debug("Converting schema")
            logger.debug(f"Input schema: {input_data.json_schema}")
            
            # Create prompt template
            template = """
            Convert the following JSON Schema into a {db_type} database schema.
            Create appropriate tables, columns, relationships, and constraints.
            Consider:
            1. Proper data types for each field (use {db_type} specific types)
            2. Primary and foreign keys
            3. Indexes for performance
            4. Normalization rules
            5. Proper table relationships
            6. {db_type} specific features and best practices
            
            JSON Schema:
            {json_schema}
            
            Generate a complete {db_type} schema with:
            - Table definitions
            - Column definitions with proper types
            - Primary keys
            - Foreign key relationships
            - Indexes
            
            Return the schema as a list of tables, where each table has:
            - table_name: string
            - columns: list of {{name: string, type: string, nullable: boolean}}
            - primary_key: list of column names
            - foreign_keys: list of {{column: string, references: string}}
            - indexes: list of {{name: string, columns: list of strings}}
            
            Return ONLY the raw JSON array without any markdown formatting or 
            additional text.
            """
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a database schema expert."),
                ("human", template)
            ])
            
            # Create chain and run it
            chain = prompt | self.llm
            logger.debug("Invoking LLM for DB schema generation")
            
            result = chain.invoke({
                "json_schema": json.dumps(input_data.json_schema, indent=2),
                "db_type": input_data.db_type
            })
            
            if not result or not result.content:
                raise ValueError("LLM returned empty response")
                
            logger.debug(f"LLM response: {result.content[:200]}...")
            
            # Extract JSON from markdown if present
            json_content = self._extract_json_from_markdown(result.content)
            logger.debug(f"Extracted JSON content: {json_content[:200]}...")
            
            # Parse the result into a list of tables
            tables = json.loads(json_content)
            
            if not tables:
                raise ValueError("No tables were generated from the schema")
                
            logger.debug(f"Generated {len(tables)} tables")
            
            # Validate table structure
            for table in tables:
                if not isinstance(table, dict):
                    raise ValueError(f"Invalid table format: {table}")
                if "table_name" not in table:
                    raise ValueError(f"Table missing name: {table}")
                if "columns" not in table:
                    raise ValueError(f"Table missing columns: {table}")
            
            return tables
            
        except Exception as e:
            logger.error(f"Error converting schema: {str(e)}", exc_info=True)
            raise ValueError(f"Error converting schema: {str(e)}")
    
    def _extract_json_from_markdown(self, content: str) -> str:
        """Extract JSON content from markdown text."""
        # Look for JSON array pattern
        import re
        json_pattern = r'\[\s*\{.*\}\s*\]'
        match = re.search(json_pattern, content, re.DOTALL)
        if match:
            return match.group(0)
        return content

    def _arun(
        self,
        input_data: SchemaConverterInput
    ) -> List[Dict[str, Any]]:
        """Async implementation of _run."""
        return self._run(input_data)


class DatabaseSchemaAgent:
    """Agent for converting JSON schema to database schema."""
    
    def __init__(self, llm_manager: LLMManager):
        """Initialize the agent."""
        self.llm = llm_manager.llm
        self.tools = [SchemaConverterTool(llm=self.llm)]
        
        # Create the agent prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a database schema expert. Your task is to 
            convert JSON schema into a database schema. Follow these guidelines:
            1. Use database-specific data types
            2. Create appropriate tables and relationships
            3. Define primary and foreign keys
            4. Add necessary indexes
            5. Follow database best practices
            6. Return only the raw JSON schema without markdown formatting
            
            IMPORTANT - Tool Usage Rules:
            1. The convert_schema tool REQUIRES two arguments:
            - json_schema: A valid JSON schema dictionary
            - db_type: One of: mysql, postgresql, oracle, sqlserver, snowflake, redshift, s3_lakehouse
            
            2. Input Handling:
            - ALWAYS check if both json_schema and db_type are provided
            - NEVER call the tool with only one argument
            
            3. Example Valid Input:
            {{
                "json_schema": {{"type": "object", "properties": {{...}}}},
                "db_type": "postgresql"
            }}
            
            4. Error Cases:
            - If input is missing json_schema: "Error: json_schema is required"
            - If input is missing db_type: "Error: db_type is required"
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
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
    
    async def convert_schema(
        self,
        json_schema: Dict[str, Any],
        db_type: Union[str, DatabaseType],
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> List[TableSchemaModel]:
        try:
            logger.debug("Starting schema conversion")
            logger.debug(f"Input json_schema: {json.dumps(json_schema, indent=2)}")
            logger.debug(f"Input db_type: {db_type}")
            
            # Convert string to DatabaseType if needed
            if isinstance(db_type, str):
                try:
                    db_type = DatabaseType(db_type.lower())
                    logger.debug(f"Converted db_type to enum: {db_type}")
                except ValueError as e:
                    logger.error(f"Invalid database type: {db_type}")
                    raise ValueError(
                        f"Invalid database type: {db_type}. "
                        f"Must be one of: {', '.join(t.value for t in DatabaseType)}"
                    ) from e
            
            # Prepare chat history
            messages = []
            if chat_history:
                for msg in chat_history:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    else:
                        messages.append(AIMessage(content=msg["content"]))
            
            # Directly use the parameters we have
            tool_input = SchemaConverterInput(
                json_schema=json_schema,
                db_type=db_type.value
            )
            logger.debug(f"Using tool input: {json.dumps(tool_input.dict(), indent=2)}")
            
            # Call the tool directly instead of through the agent
            schema_converter = self.tools[0]  # Get the SchemaConverterTool instance
            result = schema_converter._run(tool_input)  # Use _run instead of _arun
            
            logger.debug(f"Tool execution result: {json.dumps(result, indent=2)}")
            
            # Parse the result - result is already a list of tables
            if not result:
                logger.warning("No tables were generated from the schema")
                return []
            
            tables = [TableSchemaModel(**table) for table in result]
            logger.debug(f"Generated {len(tables)} tables: {json.dumps([table.dict() for table in tables], indent=2)}")
            
            return tables
            
        except Exception as e:
            logger.error(f"Error converting schema: {str(e)}", exc_info=True)
            raise ValueError(f"Error converting schema: {str(e)}") 