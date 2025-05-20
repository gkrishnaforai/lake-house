"""SQL Generator Agent.

This agent generates SQL statements from database schema using LangChain.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools import BaseTool
import json
import logging
from core.llm.manager import LLMManager


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SQLGeneratorInput(BaseModel):
    """Input model for SQL generator tool."""
    db_schema: List[Dict[str, Any]] = Field(
        ...,
        description="Database schema to generate SQL for"
    )
    db_type: str = Field(
        ...,
        description="Type of database (e.g. postgresql, mysql)"
    )
    operation: str = Field(
        ...,
        description="Type of SQL operation (e.g. query, create, insert)"
    )
    is_descriptive_query: bool = Field(
        default=True,
        description="Whether the input is a descriptive query (True) or direct SQL (False)"
    )

    def validate_fields(self) -> None:
        """Validate input fields."""
        if not self.db_schema:
            raise ValueError("Database schema is required")
        if not self.db_type:
            raise ValueError("Database type is required")
        if not self.operation:
            raise ValueError("Operation type is required")
        if self.operation not in ["query", "create", "insert", "update", "delete"]:
            raise ValueError(
                f"Invalid operation type: {self.operation}. "
                "Must be one of: query, create, insert, update, delete"
            )
        
        # Validate schema structure
        for table in self.db_schema:
            if not isinstance(table, dict):
                raise ValueError(
                    "Each table in schema must be a dictionary"
                )
            if "table_name" not in table:
                raise ValueError(
                    "Each table must have a table_name"
                )
            if "columns" not in table:
                raise ValueError(
                    "Each table must have columns"
                )
            if not table["columns"]:
                raise ValueError(
                    "Table columns cannot be empty"
                )


class SQLStatements(BaseModel):
    """SQL statements model."""
    create_tables: List[str] = Field(default_factory=list)
    create_indexes: List[str] = Field(default_factory=list)
    create_constraints: List[str] = Field(default_factory=list)
    insert_data: List[str] = Field(default_factory=list)
    update_data: List[str] = Field(default_factory=list)


class SQLGeneratorTool(BaseTool):
    """Tool for generating SQL statements."""
    
    name: str = "generate_sql"
    description: str = "Generate SQL statements from database schema"
    args_schema: type[BaseModel] = SQLGeneratorInput
    llm: Any = Field(..., description="LLM instance for generating SQL")
    
    def __init__(self, llm):
        """Initialize the tool with an LLM instance."""
        super().__init__(llm=llm)
    
    def _run(
        self,
        input_data: SQLGeneratorInput
    ) -> Dict[str, List[str]]:
        """Generate SQL statements."""
        try:
            # Validate input fields
            input_data.validate_fields()
            
            logger.info("Generating SQL statements, SQL Generator Tool")
            logger.info(f"Input schema: {input_data.db_schema}")
            logger.info(f"Input db_type: {input_data.db_type}")
            logger.info(f"Input operation: {input_data.operation}")
            
            # Create a prompt for SQL generation
            prompt = f"""Generate SQL statements for the following database schema:
            Database Type: {input_data.db_type}
            Operation: {input_data.operation}
            Schema: {json.dumps(input_data.db_schema, indent=2)}
            
            Generate the following SQL statements:
            1. CREATE TABLE statements
            2. CREATE INDEX statements
            3. CREATE CONSTRAINT statements
            4. INSERT statements for sample data
            5. UPDATE statements if needed
            
            Return the statements in this JSON format:
            {{
                "create_tables": ["CREATE TABLE ...", ...],
                "create_indexes": ["CREATE INDEX ...", ...],
                "create_constraints": ["ALTER TABLE ...", ...],
                "insert_data": ["INSERT INTO ...", ...],
                "update_data": ["UPDATE ...", ...]
            }}
            
            Use {input_data.db_type} specific syntax and follow best practices."""
            
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
                sql_statements = json.loads(response_content)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from markdown
                import re
                json_match = re.search(
                    r'```(?:json)?\s*(\{[\s\S]*?\})\s*```',
                    response_content
                )
                if json_match:
                    sql_statements = json.loads(json_match.group(1))
                else:
                    raise ValueError(
                        "Could not find valid JSON in response"
                    )
            
            # Validate the response structure
            required_keys = [
                "create_tables",
                "create_indexes",
                "create_constraints",
                "insert_data",
                "update_data"
            ]
            for key in required_keys:
                if key not in sql_statements:
                    sql_statements[key] = []
                elif not isinstance(sql_statements[key], list):
                    sql_statements[key] = [sql_statements[key]]
            
            return sql_statements
            
        except Exception as e:
            logger.error(
                f"Error generating SQL: {str(e)}", exc_info=True
            )
            raise ValueError(f"Error generating SQL: {str(e)}")
    
    def _arun(
        self,
        input_data: SQLGeneratorInput
    ) -> Dict[str, List[str]]:
        """Async implementation of _run."""
        return self._run(input_data)


class SQLGeneratorAgent:
    """Agent for generating SQL statements."""
    
    def __init__(self, llm_manager: LLMManager):
        """Initialize the agent."""
        self.llm = llm_manager.llm
        self.tools = [SQLGeneratorTool(llm=self.llm)]
        
        # Create the agent prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL expert. Your task is to generate SQL 
            statements from a database schema. Follow these guidelines:
            1. Use database-specific SQL syntax
            2. Include all necessary DDL statements
            3. Add appropriate constraints
            4. Follow SQL best practices
            5. Return only the raw SQL statements without markdown formatting"""),
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
    
    async def generate_sql(
        self,
        db_schema: List[Dict[str, Any]],
        db_type: str,
        operation: str = "query",
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> SQLStatements:
        """Generate SQL statements from database schema."""
        try:
            logger.info("Starting SQL generation")
            logger.info(f"Input db_schema: {json.dumps(db_schema, indent=2)}")
            logger.info(f"Input db_type: {db_type}")
            logger.info(f"Input operation: {operation}")
            
            # Prepare chat history
            messages = []
            if chat_history:
                for msg in chat_history:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    else:
                        messages.append(AIMessage(content=msg["content"]))
            
            # Directly use the parameters we have
            tool_input = SQLGeneratorInput(
                db_schema=db_schema,
                db_type=db_type,
                operation=operation
            )
            logger.info(f"Using tool input: {json.dumps(tool_input.dict(), indent=2)}")
            
            # Call the tool directly instead of through the agent
            sql_generator = self.tools[0]  # Get the SQLGeneratorTool instance
            result = sql_generator._run(tool_input)  # Use _run instead of _arun
            
            logger.debug(f"Tool execution result: {json.dumps(result, indent=2)}")
            
            # Convert dictionary result to SQLStatements object
            statements = SQLStatements(
                create_tables=result.get("create_tables", []),
                create_indexes=result.get("create_indexes", []),
                create_constraints=result.get("create_constraints", []),
                insert_data=result.get("insert_data", []),
                update_data=result.get("update_data", [])
            )
            
            if not any([
                statements.create_tables,
                statements.create_indexes,
                statements.create_constraints,
                statements.insert_data,
                statements.update_data
            ]):
                logger.warning("No SQL statements were generated")
            
            return statements
            
        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}", exc_info=True)
            raise ValueError(f"Error generating SQL: {str(e)}") 