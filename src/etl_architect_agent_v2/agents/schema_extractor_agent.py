from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import JSONLoader
import json
import logging
import re
from core.llm.manager import LLMManager

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JsonSchemaModel(BaseModel):
    """JSON Schema model."""
    schema: str = Field("http://json-schema.org/draft-04/schema#")
    type: str = Field("object")
    properties: Dict[str, Any] = Field(...)
    required: List[str] = Field(default_factory=list)

class TableSchemaModel(BaseModel):
    """Database table schema model."""
    table_name: str = Field(..., description="Name of the table")
    columns: List[Dict[str, Any]] = Field(..., description="List of columns")
    primary_key: List[str] = Field(default_factory=list)
    foreign_keys: List[Dict[str, Any]] = Field(default_factory=list)
    indexes: List[Dict[str, Any]] = Field(default_factory=list)

class DatabaseType(str, Literal["mysql", "postgresql", "oracle", "sqlserver", "snowflake", "redshift", "s3_lakehouse"]):
    """Supported database types."""
    pass

class SchemaExtractorState:
    """State for schema extraction workflow."""
    def __init__(
        self,
        table_name: str,
        file_path: str,
        db_type: DatabaseType,
        llm: Optional[LLMManager] = None
    ):
        self.table_name = table_name
        self.file_path = file_path
        self.db_type = db_type
        self.llm = llm
        self.json_data = None
        self.json_schema = None
        self.db_schema = None
        self.sql_statements = None
        self.errors = []

class FileHandler:
    """Handles file operations for schema extraction."""
    
    @staticmethod
    def load_json_file(file_path: str) -> Dict[str, Any]:
        """Load and validate JSON file."""
        try:
            logger.debug(f"Loading JSON file from: {file_path}")
            # Load JSON file using LangChain's JSONLoader
            loader = JSONLoader(
                file_path=file_path,
                jq_schema='.',
                text_content=False
            )
            data = loader.load()
            
            if not data:
                raise ValueError("No data found in JSON file")
            
            logger.debug(f"Successfully loaded data: {data[0].page_content[:100]}...")
                
            # Convert string content to Python object
            if isinstance(data[0].page_content, str):
                return json.loads(data[0].page_content)
            return data[0].page_content
            
        except Exception as e:
            logger.error(f"Error loading JSON file: {str(e)}", exc_info=True)
            raise ValueError(f"Error loading JSON file: {str(e)}")

class SchemaProcessor:
    """Processes and extracts schema from JSON data."""
    
    @staticmethod
    def _extract_json_from_markdown(text: str) -> str:
        """Extract JSON from markdown-formatted text."""
        # Look for JSON content between ```json and ``` markers
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(json_pattern, text)
        if match:
            return match.group(1).strip()
        return text.strip()
    
    @staticmethod
    def extract_json_schema(
        json_data: Dict[str, Any],
        llm: Optional[LLMManager] = None
    ) -> JsonSchemaModel:
        """Extract JSON schema from data using LangChain."""
        try:
            logger.debug("Starting JSON schema extraction")
            logger.debug(f"Input data: {json.dumps(json_data, indent=2)[:200]}...")
            
            # Create prompt template
            template = """
            Analyze the following JSON data and generate a JSON Schema that 
            describes its structure. Include all nested objects, arrays, and 
            their types. Make sure to include required fields and proper type 
            definitions.
            
            JSON Data:
            {json_data}
            
            Generate a complete JSON Schema that follows the JSON Schema 
            Draft-04 specification. Return ONLY the raw JSON schema without any 
            markdown formatting or additional text.
            """
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a JSON Schema expert."),
                ("human", template)
            ])
            
            # Create chain
            chain = prompt | llm.llm
            
            # Run chain
            logger.debug("Invoking LLM for JSON schema generation")
            result = chain.invoke(
                {"json_data": json.dumps(json_data, indent=2)}
            )
            logger.debug(f"LLM response: {result.content[:200]}...")
            
            # Extract JSON from markdown if present
            json_content = SchemaProcessor._extract_json_from_markdown(
                result.content
            )
            logger.debug(f"Extracted JSON content: {json_content[:200]}...")
            
            schema = JsonSchemaModel.parse_raw(json_content)
            logger.debug(f"Parsed schema: {schema.dict()}")
            return schema
            
        except Exception as e:
            logger.error(f"Error generating JSON schema: {str(e)}", exc_info=True)
            raise ValueError(f"Error generating JSON schema: {str(e)}")

    @staticmethod
    def convert_to_db_schema(
        json_schema: JsonSchemaModel,
        db_type: DatabaseType,
        llm: Optional[LLMManager] = None
    ) -> List[TableSchemaModel]:
        """Convert JSON schema to database schema."""
        try:
            logger.debug("Starting DB schema conversion")
            logger.debug(f"Input schema: {json_schema.dict()}")
            
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
            
            # Create chain
            chain = prompt | llm.llm
            
            # Run chain
            logger.debug("Invoking LLM for DB schema generation")
            result = chain.invoke({
                "json_schema": json.dumps(json_schema.dict(), indent=2),
                "db_type": db_type
            })
            logger.debug(f"LLM response: {result.content[:200]}...")
            
            # Extract JSON from markdown if present
            json_content = SchemaProcessor._extract_json_from_markdown(
                result.content
            )
            logger.debug(f"Extracted JSON content: {json_content[:200]}...")
            
            # Parse the result into a list of TableSchemaModel
            tables_data = json.loads(json_content)
            logger.debug(f"Parsed tables data: {tables_data}")
            
            tables = [TableSchemaModel(**table) for table in tables_data]
            logger.debug(f"Generated {len(tables)} tables")
            return tables
            
        except Exception as e:
            logger.error(
                f"Error converting to DB schema: {str(e)}", exc_info=True
            )
            raise ValueError(f"Error converting to DB schema: {str(e)}")

    @staticmethod
    def generate_sql_statements(
        db_schema: List[TableSchemaModel],
        db_type: DatabaseType,
        operation: Literal["create", "insert", "update"],
        llm: Optional[LLMManager] = None
    ) -> Dict[str, str]:
        """Generate SQL statements for database operations."""
        try:
            logger.debug("Starting SQL generation")
            logger.debug(f"Input schema: {db_schema}")
            
            # Create prompt template
            template = """
            Generate {operation} SQL statements for the following {db_type} 
            database schema. Consider:
            1. {db_type} specific syntax and features
            2. Proper data type handling
            3. Constraints and relationships
            4. Best practices for {db_type}
            
            Database Schema:
            {db_schema}
            
            Generate the following SQL statements:
            1. Table creation statements
            2. Index creation statements
            3. Foreign key constraint statements
            4. Sample data insertion statements (if operation is insert)
            5. Data update statements (if operation is update)
            
            Return the SQL statements as a JSON object with the following keys:
            - create_tables: list of CREATE TABLE statements
            - create_indexes: list of CREATE INDEX statements
            - create_constraints: list of ALTER TABLE statements for constraints
            - insert_data: list of INSERT statements (for insert operation)
            - update_data: list of UPDATE statements (for update operation)
            
            Return ONLY the raw JSON without any markdown formatting or 
            additional text.
            """
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a SQL expert."),
                ("human", template)
            ])
            
            # Create chain
            chain = prompt | llm.llm
            
            # Run chain
            logger.debug("Invoking LLM for SQL generation")
            result = chain.invoke({
                "db_schema": json.dumps([table.dict() for table in db_schema], indent=2),
                "db_type": db_type,
                "operation": operation
            })
            logger.debug(f"LLM response: {result.content[:200]}...")
            
            # Extract JSON from markdown if present
            json_content = SchemaProcessor._extract_json_from_markdown(
                result.content
            )
            logger.debug(f"Extracted JSON content: {json_content[:200]}...")
            
            # Parse the result into a dictionary of SQL statements
            sql_statements = json.loads(json_content)
            logger.debug(f"Generated SQL statements: {sql_statements}")
            return sql_statements
            
        except Exception as e:
            logger.error(
                f"Error generating SQL statements: {str(e)}", exc_info=True
            )
            raise ValueError(f"Error generating SQL statements: {str(e)}")

def extract_schema_node(state: SchemaExtractorState) -> SchemaExtractorState:
    """Node for extracting schema from JSON data."""
    try:
        logger.info(f"Starting schema extraction for {state.table_name}")
        
        # Load JSON data
        state.json_data = FileHandler.load_json_file(state.file_path)
        logger.info(f"Successfully loaded JSON data from {state.file_path}")
        
        # Extract JSON schema
        state.json_schema = SchemaProcessor.extract_json_schema(
            state.json_data,
            state.llm
        )
        logger.info("Successfully generated JSON schema")
        
        # Convert to DB schema
        state.db_schema = SchemaProcessor.convert_to_db_schema(
            state.json_schema,
            state.db_type,
            state.llm
        )
        logger.info("Successfully generated database schema")
        
        # Generate SQL statements
        state.sql_statements = SchemaProcessor.generate_sql_statements(
            state.db_schema,
            state.db_type,
            "create",
            state.llm
        )
        logger.info("Successfully generated SQL statements")
        
    except Exception as e:
        error_msg = f"Schema extraction failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        state.errors.append(error_msg)
        
    return state

def run_schema_extractor(
    table_name: str,
    file_path: str,
    db_type: DatabaseType,
    api_key: str
) -> Dict[str, Any]:
    """Run the schema extraction workflow."""
    try:
        logger.info(f"Initializing schema extraction for {table_name}")
        
        # Initialize LLM
        llm = LLMManager(api_key=api_key)
        logger.debug("LLM initialized")
        
        # Create initial state
        state = SchemaExtractorState(
            table_name=table_name,
            file_path=file_path,
            db_type=db_type,
            llm=llm
        )
        
        # Run workflow
        logger.info("Executing schema extraction workflow")
        state = extract_schema_node(state)
            
        if state.errors:
            raise ValueError(state.errors[0])
            
        return {
            "json_schema": state.json_schema.dict(),
            "db_schema": [schema.dict() for schema in state.db_schema],
            "sql_statements": state.sql_statements
        }
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg)
