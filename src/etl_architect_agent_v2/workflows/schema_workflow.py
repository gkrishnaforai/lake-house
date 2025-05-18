"""Schema Workflow.

This module defines the workflow for schema extraction and conversion
using LangGraph.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
import logging
import os
from etl_architect_agent_v2.agents.schema_extractor import (
    JsonSchemaExtractorAgent,
    DatabaseSchemaAgent,
    SQLGeneratorAgent,
    ExcelSchemaExtractorAgent
)
from core.llm.manager import LLMManager


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorkflowState(BaseModel):
    """State for the schema workflow."""
    file_path: str = Field(..., description="Path to the input file")
    file_type: str = Field(..., description="Type of input file (json or xlsx)")
    table_name: str = Field(..., description="Name of the table")
    db_type: str = Field(..., description="Type of database")
    operation: str = Field(..., description="Type of operation")
    json_schema: Optional[Dict[str, Any]] = Field(None)
    db_schema: Optional[List[Dict[str, Any]]] = Field(None)
    sql_statements: Optional[Dict[str, List[str]]] = Field(None)
    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    last_error: Optional[str] = Field(None)
    errors: List[str] = Field(default_factory=list)

    def validate_state(self, required_keys: List[str]) -> None:
        """Validate that required state keys are present and valid."""
        # Check keys and their values
        missing_keys = []
        for key in required_keys:
            value = getattr(self, key)
            if value is None or (isinstance(value, list) and len(value) == 0):
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(
                f"Missing or empty required state: {', '.join(missing_keys)}"
            )
        
        # Validate values
        if "json_schema" in required_keys:
            if not isinstance(self.json_schema, dict):
                raise ValueError("json_schema must be a dictionary")
        
        if "db_type" in required_keys:
            valid_types = [
                'mysql', 'postgresql', 'oracle', 'sqlserver',
                'snowflake', 'redshift', 's3_lakehouse'
            ]
            if self.db_type not in valid_types:
                raise ValueError(
                    f"Invalid db_type. Must be one of: {', '.join(valid_types)}"
                )
        
        if "file_type" in required_keys:
            valid_types = ['json', 'xlsx']
            if self.file_type not in valid_types:
                raise ValueError(
                    f"Invalid file_type. Must be one of: {', '.join(valid_types)}"
                )


class SchemaWorkflow:
    """Schema workflow manager."""
    
    def __init__(self, llm_manager: LLMManager):
        """Initialize the workflow manager."""
        self.llm_manager = llm_manager
        self.json_agent = JsonSchemaExtractorAgent(llm_manager)
        self.excel_agent = ExcelSchemaExtractorAgent(llm_manager)
        self.db_agent = DatabaseSchemaAgent(llm_manager)
        self.sql_agent = SQLGeneratorAgent(llm_manager)
        self.workflow = self._create_workflow()
    
    def _detect_file_type(self, file_path: str) -> str:
        """Detect the type of file based on extension.
        
        Supports:
        - Excel: .xlsx, .xls, .xlsm, .xlsb, .xltx, .xltm
        - JSON: .json
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        # Excel file extensions
        excel_extensions = {
            '.xlsx': 'xlsx',  # Excel Workbook
            '.xls': 'xlsx',   # Excel 97-2003 Workbook
            '.xlsm': 'xlsx',  # Excel Macro-Enabled Workbook
            '.xlsb': 'xlsx',  # Excel Binary Workbook
            '.xltx': 'xlsx',  # Excel Template
            '.xltm': 'xlsx'   # Excel Macro-Enabled Template
        }
        
        # Check file type
        if ext == '.json':
            return 'json'
        elif ext in excel_extensions:
            return excel_extensions[ext]
        else:
            supported_types = (
                f"Excel: {', '.join(excel_extensions.keys())}, "
                f"JSON: .json"
            )
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported types are: {supported_types}"
            )
    
    def _create_workflow(self) -> StateGraph:
        """Create the schema workflow."""
        # Create the workflow
        workflow = StateGraph(WorkflowState)
        
        # Add nodes with state handling
        async def extract_schema(state: WorkflowState) -> WorkflowState:
            """Extract schema based on file type and update state."""
            try:
                if state.file_type == 'json':
                    schema = await self.json_agent.extract_schema(state.file_path)
                    state.json_schema = schema.model_dump()
                elif state.file_type == 'xlsx':
                    schema = await self.excel_agent.extract_schema(state.file_path)
                    # Convert Excel schema to JSON schema format
                    state.json_schema = self._convert_excel_to_json_schema(schema)
                else:
                    raise ValueError(f"Unsupported file type: {state.file_type}")
                return state
            except Exception as e:
                logger.error(f"Error extracting schema: {str(e)}", exc_info=True)
                state.errors.append(f"Schema extraction failed: {str(e)}")
                raise
        
        async def convert_db_schema(state: WorkflowState) -> WorkflowState:
            """Convert schema and update state."""
            try:
                if not state.json_schema:
                    raise ValueError("No schema available for conversion")
                
                tables = await self.db_agent.convert_schema(
                    json_schema=state.json_schema,
                    db_type=state.db_type
                )
                
                if not tables:
                    raise ValueError("No database tables were generated")
                
                state.db_schema = [table.model_dump() for table in tables]
                return state
                
            except Exception as e:
                logger.error(f"Error converting schema: {str(e)}", exc_info=True)
                state.errors.append(f"Schema conversion failed: {str(e)}")
                raise
        
        async def generate_sql(state: WorkflowState) -> WorkflowState:
            """Generate SQL and update state."""
            statements = await self.sql_agent.generate_sql(
                db_schema=state.db_schema,
                db_type=state.db_type,
                operation=state.operation
            )
            state.sql_statements = statements.model_dump()
            return state
        
        # Add nodes
        workflow.add_node("extract_schema", extract_schema)
        workflow.add_node("convert_db_schema", convert_db_schema)
        workflow.add_node("generate_sql", generate_sql)
        
        # Add edges
        workflow.add_edge("extract_schema", "convert_db_schema")
        workflow.add_edge("convert_db_schema", "generate_sql")
        
        # Set entry point
        workflow.set_entry_point("extract_schema")
        
        return workflow.compile()
    
    def _convert_excel_to_json_schema(self, excel_schema: Any) -> Dict[str, Any]:
        """Convert Excel schema to JSON schema format."""
        try:
            # Create a base JSON schema
            json_schema = {
                "type": "object",
                "properties": {},
                "required": []
            }
            
            # Process each sheet
            for sheet_name, sheet_data in excel_schema.sheets.items():
                # Create a property for each sheet
                sheet_property = {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
                
                # Add columns as properties
                for col_name, col_type in sheet_data["data_types"].items():
                    # Map Excel types to JSON types
                    json_type = self._map_excel_type_to_json(col_type)
                    sheet_property["items"]["properties"][col_name] = {
                        "type": json_type
                    }
                
                # Add the sheet to the main schema
                json_schema["properties"][sheet_name] = sheet_property
            
            return json_schema
            
        except Exception as e:
            logger.error(f"Error converting Excel schema: {str(e)}", exc_info=True)
            raise ValueError(f"Error converting Excel schema: {str(e)}")
    
    def _map_excel_type_to_json(self, excel_type: str) -> str:
        """Map Excel data types to JSON schema types."""
        type_mapping = {
            'int64': 'integer',
            'float64': 'number',
            'bool': 'boolean',
            'datetime64[ns]': 'string',
            'object': 'string'
        }
        return type_mapping.get(str(excel_type).lower(), 'string')
    
    async def run_workflow(
        self,
        file_path: str,
        table_name: str,
        db_type: str,
        operation: str
    ) -> Dict[str, Any]:
        """Run the schema workflow."""
        try:
            logger.debug("Starting schema workflow")
            logger.debug(
                f"Input parameters: file_path={file_path}, "
                f"table_name={table_name}, db_type={db_type}, "
                f"operation={operation}"
            )
            
            # Detect file type
            file_type = self._detect_file_type(file_path)
            
            # Prepare initial state
            initial_state = WorkflowState(
                file_path=file_path,
                file_type=file_type,
                table_name=table_name,
                db_type=db_type,
                operation=operation
            )
            
            # Run workflow
            result_dict = await self.workflow.ainvoke(initial_state)
            logger.debug(f"Workflow completed: {result_dict}")
            
            # Convert result back to WorkflowState
            result = WorkflowState(**result_dict)
            
            # Validate final state
            result.validate_state(
                ["json_schema", "db_schema", "sql_statements"]
            )
            
            return {
                "json_schema": result.json_schema,
                "db_schema": result.db_schema,
                "sql_statements": result.sql_statements,
                "errors": result.errors
            }
            
        except Exception as e:
            logger.error(
                f"Error running workflow: {str(e)}", exc_info=True
            )
            raise ValueError(f"Error running workflow: {str(e)}")


async def run_schema_workflow(
    file_path: str,
    table_name: str,
    db_type: str,
    operation: str,
    api_key: str
) -> Dict[str, Any]:
    """Run the schema workflow."""
    try:
        # Initialize LLM manager and workflow
        llm_manager = LLMManager(api_key)
        workflow = SchemaWorkflow(llm_manager)
        
        # Run workflow
        return await workflow.run_workflow(
            file_path=file_path,
            table_name=table_name,
            db_type=db_type,
            operation=operation
        )
        
    except Exception as e:
        logger.error(
            f"Error running workflow: {str(e)}", exc_info=True
        )
        raise ValueError(f"Error running workflow: {str(e)}")


__all__ = [
    "SchemaWorkflow",
    "WorkflowState",
    "run_schema_workflow"
] 