"""Service for SQL generation using SQLGeneratorAgent."""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import logging
from core.llm.manager import LLMManager
from core.sql.sql_state import (
    SQLGenerationState,
    SQLGenerationStep,
    SQLRequirements,
    SQLGenerationOutput
)
from .exceptions import SQLGenerationError
from .glue_service import GlueService
import json
from datetime import datetime


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class SQLGenerationRequest(BaseModel):
    """Request model for SQL generation."""
    query: str
    schema: Dict[str, Any]
    table_name: Optional[str] = None
    preserve_column_names: bool = True
    user_id: str = "test_user"
    chat_history: Optional[List[Dict[str, str]]] = None
    s3_location: Optional[str] = None


class SQLGenerationResponse(BaseModel):
    """Response model for SQL generation."""
    status: str
    sql_query: Optional[str] = None
    explanation: Optional[str] = None
    confidence: float = 1.0
    tables_used: List[str] = []
    columns_used: List[str] = []
    filters: Dict[str, Any] = {}
    error: Optional[str] = None


class SQLGenerationService:
    """Service for SQL generation using SQLGeneratorAgent."""
    
    def __init__(
        self,
        llm_manager: LLMManager,
        glue_service: Optional[GlueService] = None
    ):
        """Initialize SQL generation service."""
        self.llm_manager = llm_manager
        self.glue_service = glue_service
    
    async def generate_sql(
        self,
        request: SQLGenerationRequest
    ) -> SQLGenerationResponse:
        """Generate SQL query based on user request and schema."""
        try:
            logger.info(f"Starting SQL generation for request: {request.dict()}")
            
            # Check if query is direct SQL or descriptive
            sql_keywords = [
                "SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY",
                "HAVING", "LIMIT", "OFFSET", "UNION", "INSERT", "UPDATE",
                "DELETE", "CREATE", "ALTER", "DROP"
            ]
            query_upper = request.query.upper()
            is_descriptive = not any(
                query_upper.strip().startswith(keyword) or
                f" {keyword} " in f" {query_upper} "
                for keyword in sql_keywords
            )
            
            # Initialize SQL generation state
            state = SQLGenerationState(
                workflow_id=f"sql_gen_{request.user_id}_{request.table_name or 'all'}",
                current_step=SQLGenerationStep.REQUIREMENTS_ANALYSIS,
                metadata={
                    "query": request.query,
                    "schema": request.schema,
                    "table_name": request.table_name,
                    "preserve_column_names": request.preserve_column_names,
                    "user_id": request.user_id,
                    "is_descriptive_query": is_descriptive,
                    "constraints": {
                        "preserve_column_names": request.preserve_column_names,
                        "table_name": request.table_name,
                        "user_id": request.user_id
                    }
                }
            )
            logger.info(f"Initialized state: {state.dict()}")

            # Process through workflow steps
            step_count = 0
            max_steps = 10  # Prevent infinite loops
            while not state.is_done and step_count < max_steps:
                step_count += 1
                logger.info(f"Processing step {step_count}: {state.current_step}")
                
                if state.current_step == SQLGenerationStep.REQUIREMENTS_ANALYSIS:
                    logger.info("Starting requirements analysis")
                    state = await self._analyze_requirements(state)
                    logger.info(f"Completed requirements analysis: {state.dict()}")
                elif state.current_step == SQLGenerationStep.SCHEMA_VALIDATION:
                    logger.info("Starting schema validation")
                    state = await self._validate_schema(state)
                    logger.info(f"Completed schema validation: {state.dict()}")
                elif state.current_step == SQLGenerationStep.QUERY_GENERATION:
                    logger.info("Starting query generation")
                    state = await self._generate_query(state)
                    logger.info(f"Completed query generation: {state.dict()}")
                elif state.current_step == SQLGenerationStep.QUERY_REVIEW:
                    logger.info("Starting query review")
                    state = await self._review_query(state)
                    logger.info(f"Completed query review: {state.dict()}")
                elif state.current_step == SQLGenerationStep.QUERY_CORRECTION:
                    logger.info("Starting query correction")
                    state = await self._correct_query(state)
                    logger.info(f"Completed query correction: {state.dict()}")
                else:
                    error_msg = f"Unknown step: {state.current_step}"
                    logger.error(error_msg)
                    raise SQLGenerationError(error_msg)
                
                if step_count >= max_steps:
                    error_msg = f"Maximum number of steps ({max_steps}) exceeded"
                    logger.error(error_msg)
                    state.set_error(error_msg)
                    break

            # Extract final output
            if state.status == "success":
                logger.info("SQL generation completed successfully")
                output = SQLGenerationOutput.from_json(state.metadata.get("final_query", "{}"))
                return SQLGenerationResponse(
                    status="success",
                    sql_query=output.sql_query,
                    explanation=output.explanation,
                    confidence=output.confidence,
                    tables_used=output.tables_used,
                    columns_used=output.columns_used,
                    filters=output.filters
                )
            else:
                error_msg = state.error or "SQL generation failed"
                logger.error(f"SQL generation failed: {error_msg}")
                return SQLGenerationResponse(
                    status="error",
                    error=error_msg
                )

        except Exception as e:
            error_msg = f"Error in SQL generation: {str(e)}"
            logger.error(error_msg)
            return SQLGenerationResponse(
                status="error",
                error=error_msg
            )

    async def _invoke_llm(self, prompt: str) -> str:
        """Helper method to invoke LLM with consistent message structure."""
        response = await self.llm_manager.ainvoke({
            "messages": [{
                "role": "user",
                "content": prompt
            }]
        })
        return response["content"]

    async def _analyze_requirements(self, state: SQLGenerationState) -> SQLGenerationState:
        """Analyze requirements and create SQLRequirements object."""
        try:
            requirements = SQLRequirements(
                query=state.metadata["query"],
                schema=state.metadata["schema"],
                constraints={
                    "preserve_column_names": state.metadata["preserve_column_names"],
                    "table_name": state.metadata["table_name"]
                }
            )
            
            # Store requirements in state
            state.metadata["requirements"] = requirements.to_json()
            state.update_step(SQLGenerationStep.SCHEMA_VALIDATION)
            return state
        except Exception as e:
            state.set_error(str(e))
            return state

    def _validate_schema_format(self, schema: Dict[str, Any]) -> List[Dict[str, str]]:
        """Validate and normalize schema format locally without LLM.
        
        Returns:
            List of normalized columns with name and type only.
        """
        try:
            # Handle string input first
            if isinstance(schema, str):
                try:
                    schema = json.loads(schema)
                except json.JSONDecodeError:
                    raise SQLGenerationError("Invalid schema JSON string")
            
            # Extract columns from various schema formats
            columns = []
            
            if isinstance(schema, dict):
                if "columns" in schema:
                    columns = schema["columns"]
                elif "schema" in schema:
                    schema_data = schema["schema"]
                    if isinstance(schema_data, dict):
                        if "columns" in schema_data:
                            columns = schema_data["columns"]
                        else:
                            columns = [
                                {"name": str(k), "type": str(v)}
                                for k, v in schema_data.items()
                            ]
                    elif isinstance(schema_data, list):
                        columns = schema_data
                    else:
                        raise SQLGenerationError(
                            "Invalid schema format in 'schema' field"
                        )
                else:
                    columns = [
                        {"name": str(k), "type": str(v)}
                        for k, v in schema.items()
                    ]
            elif isinstance(schema, list):
                columns = schema
            else:
                raise SQLGenerationError(f"Invalid schema format: {type(schema)}")
            
            # Normalize columns to minimal format
            normalized_columns = []
            for col in columns:
                if isinstance(col, dict):
                    if "name" in col and "type" in col:
                        normalized_columns.append({
                            "name": str(col["name"]),
                            "type": str(col["type"])
                        })
                    else:
                        # Try to extract name and type from other formats
                        for key, value in col.items():
                            if isinstance(value, (str, int, float, bool)):
                                normalized_columns.append({
                                    "name": str(key),
                                    "type": str(value)
                                })
                elif isinstance(col, (str, int, float, bool)):
                    normalized_columns.append({
                        "name": str(col),
                        "type": "string"
                    })
            
            if not normalized_columns:
                raise SQLGenerationError("No valid columns found in schema")
                
            return normalized_columns
            
        except Exception as e:
            raise SQLGenerationError(f"Schema validation error: {str(e)}")

    async def _validate_schema(self, state: SQLGenerationState) -> SQLGenerationState:
        """Validate schema against requirements."""
        try:
            requirements = SQLRequirements.from_json(
                state.metadata["requirements"]
            )
            
            # Basic schema validation
            if not requirements.schema:
                raise SQLGenerationError("No schema provided")
                
            schema = requirements.schema
            logger.info(
                f"Starting schema validation with schema: "
                f"{json.dumps(schema, indent=2, cls=DateTimeEncoder)}"
            )
            
            # Handle new schema format with table_name and user_id
            if (
                isinstance(schema, dict) 
                and "table_name" in schema 
                and "user_id" in schema
            ):
                # If glue_service is available, validate table exists and get schema
                if self.glue_service:
                    try:
                        database_name = self._get_database_name(
                            schema["user_id"]
                        )
                        table_name = schema["table_name"]
                        
                        logger.info(
                            f"Validating table {table_name} in database "
                            f"{database_name}"
                        )
                        
                        table_info = await self.glue_service.get_table(
                            database_name,
                            table_name
                        )
                        
                        logger.info(
                            f"Retrieved table info: "
                            f"{json.dumps(table_info, indent=2, cls=DateTimeEncoder)}"
                        )
                        
                        if not table_info:
                            raise SQLGenerationError(
                                f"Table {table_name} not found"
                            )
                        
                        # Extract columns from table info
                        existing_columns = [
                            {
                                "name": col["Name"],
                                "type": col["Type"]
                            }
                            for col in table_info["StorageDescriptor"]["Columns"]
                        ]
                        
                        logger.info(
                            f"Existing columns: "
                            f"{json.dumps(existing_columns, indent=2, cls=DateTimeEncoder)}"
                        )
                        
                        # Handle new columns if present
                        new_columns = []
                        if "new_columns" in schema:
                            new_columns = self._validate_schema_format(
                                schema["new_columns"]
                            )
                            logger.info(
                                f"New columns to add: "
                                f"{json.dumps(new_columns, indent=2, cls=DateTimeEncoder)}"
                            )
                            
                            # Merge existing and new columns
                            all_columns = existing_columns + new_columns
                            
                            # Update schema with all columns
                            requirements.schema = {
                                "schema": {
                                    "columns": all_columns,
                                    "database_name": database_name,
                                    "table_name": table_name,
                                    "new_columns": new_columns
                                }
                            }
                        else:
                            # Update schema with existing columns only
                            requirements.schema = {
                                "schema": {
                                    "columns": existing_columns,
                                    "database_name": database_name,
                                    "table_name": table_name
                                }
                            }
                            
                        state.metadata["requirements"] = requirements.to_json()
                        logger.info(
                            f"Table {table_name} validated and schema updated"
                        )
                    except Exception as e:
                        logger.error(f"Error validating table: {str(e)}")
                        raise SQLGenerationError(str(e))
                else:
                    raise SQLGenerationError(
                        "Glue service not available for schema validation"
                    )
            else:
                # Handle old schema format
                try:
                    # Try to normalize schema format
                    if isinstance(schema, dict):
                        if "schema" in schema:
                            schema = schema["schema"]
                        if "columns" in schema:
                            schema = schema["columns"]
                    elif isinstance(schema, list):
                        # Schema is already a list of columns
                        pass
                    else:
                        # Try to convert schema to list of columns
                        if isinstance(schema, dict):
                            schema = [
                                {"name": str(k), "type": str(v)}
                                for k, v in schema.items()
                            ]
                        else:
                            raise SQLGenerationError(
                                f"Invalid schema format: {type(schema)}"
                            )
                    
                    logger.info(
                        f"Normalized schema format: "
                        f"{json.dumps(schema, indent=2, cls=DateTimeEncoder)}"
                    )
                    
                    if not isinstance(schema, list):
                        raise SQLGenerationError(
                            "Invalid schema format - expected list of columns"
                        )
                        
                    if not schema:
                        raise SQLGenerationError("No columns found in schema")
                        
                    # Validate each column has required fields
                    for col in schema:
                        if not isinstance(col, dict):
                            raise SQLGenerationError(
                                f"Invalid column format: {type(col)}"
                            )
                        if "name" not in col or "type" not in col:
                            raise SQLGenerationError(
                                "Each column must have 'name' and 'type' fields"
                            )
                    
                    # Update schema with validated format
                    requirements.schema = {
                        "schema": {
                            "columns": schema,
                            "database_name": self._get_database_name(
                                state.metadata.get("user_id", "test_user")
                            ),
                            "table_name": state.metadata.get("table_name", "")
                        }
                    }
                    state.metadata["requirements"] = requirements.to_json()
                    logger.info("Schema validated and normalized")
                except Exception as e:
                    logger.error(f"Error normalizing schema: {str(e)}")
                    raise SQLGenerationError(
                        f"Schema validation error: {str(e)}"
                    )
            
            state.update_step(SQLGenerationStep.QUERY_GENERATION)
            return state
            
        except Exception as e:
            logger.error(f"Error validating schema: {str(e)}")
            state.set_error(str(e))
            return state

    async def _generate_query(self, state: SQLGenerationState) -> SQLGenerationState:
        """Generate SQL query based on requirements and schema."""
        try:
            requirements = SQLRequirements.from_json(state.metadata["requirements"])
            
            # Generate SQL using LLM
            prompt = self._create_query_generation_prompt(requirements)
            response = await self._invoke_llm(prompt)
            
            # Parse response into SQLGenerationOutput
            schema_data = requirements.schema.get("schema", {})
            columns = schema_data.get("columns", [])
            new_columns = schema_data.get("new_columns", [])
            
            # If there are new columns, generate ALTER TABLE statements
            if new_columns:
                alter_statements = []
                for col in new_columns:
                    alter_statements.append(
                        f"ALTER TABLE {schema_data['database_name']}."
                        f"{schema_data['table_name']} "
                        f"ADD COLUMN {col['name']} {col['type']};"
                    )
                response = "\n".join(alter_statements) + "\n\n" + response
            
            output = SQLGenerationOutput(
                sql_query=response.strip(),
                explanation=f"Generated SQL query for {requirements.query}",
                confidence=1.0,
                tables_used=[requirements.schema.get("table_name", "")],
                columns_used=[col["name"] for col in columns]
            )
            
            state.metadata["generated_query"] = output.to_json()
            state.update_step(SQLGenerationStep.QUERY_REVIEW)
            return state
        except Exception as e:
            state.set_error(str(e))
            return state

    async def _review_query(self, state: SQLGenerationState) -> SQLGenerationState:
        """Review generated query for correctness and optimization."""
        try:
            output = SQLGenerationOutput.from_json(state.metadata["generated_query"])
            
            # Review query using LLM
            prompt = self._create_query_review_prompt(output)
            review_result = await self._invoke_llm(prompt)
            
            # Store review result
            state.metadata["review_result"] = review_result
            
            # Check if corrections are needed
            if "needs_correction" in review_result.lower():
                state.update_step(SQLGenerationStep.QUERY_CORRECTION)
            else:
                state.metadata["final_query"] = state.metadata["generated_query"]
                state.set_success()
            
            return state
        except Exception as e:
            state.set_error(str(e))
            return state

    async def _correct_query(self, state: SQLGenerationState) -> SQLGenerationState:
        """Correct any issues in the generated query."""
        try:
            output = SQLGenerationOutput.from_json(state.metadata["generated_query"])
            requirements = SQLRequirements.from_json(state.metadata["requirements"])
            
            # Generate corrected query using LLM
            prompt = self._create_query_correction_prompt(output, requirements)
            corrected = await self._invoke_llm(prompt)
            
            # Create new output with corrected query
            corrected_output = SQLGenerationOutput(
                sql_query=corrected.strip(),
                explanation=f"Corrected SQL query for {requirements.query}",
                confidence=1.0,
                tables_used=output.tables_used,
                columns_used=output.columns_used
            )
            
            state.metadata["final_query"] = corrected_output.to_json()
            state.set_success()
            return state
        except Exception as e:
            state.set_error(str(e))
            return state

    def _create_query_generation_prompt(self, requirements: SQLRequirements) -> str:
        """Create minimal prompt for query generation."""
        query = requirements.query
        schema = requirements.schema
        
        # Extract database and table names from schema
        database_name = schema.get("schema", {}).get("database_name", "")
        table_name = schema.get("schema", {}).get("table_name", "")
        
        # Get columns from schema
        columns = schema.get("schema", {}).get("columns", [])
        
        logger.info(f"Database name: {database_name}")
        logger.info(f"Table name: {table_name}")
        logger.info(f"Full schema: {json.dumps(requirements.schema, indent=2, cls=DateTimeEncoder)}")
        
        # Create minimal column description
        column_desc = "\n".join([
            f"- {col['name']} ({col['type']})"
            for col in columns[:10]  # Limit to first 10 columns to reduce tokens
        ])
        
        if len(columns) > 10:
            column_desc += f"\n... and {len(columns) - 10} more columns"
        
        return f"""Generate SQL query for:
Query: {query}
Database: {database_name}
Table: {table_name}
Columns:
{column_desc}

Note: Always use the full table name in the format: {database_name}.{table_name}"""

    def _create_query_review_prompt(self, output: SQLGenerationOutput) -> str:
        """Create minimal prompt for query review."""
        return f"""Review SQL query:
{output.sql_query}"""

    def _create_query_correction_prompt(
        self,
        output: SQLGenerationOutput,
        requirements: SQLRequirements
    ) -> str:
        """Create minimal prompt for query correction."""
        return f"""Correct SQL query:
Query: {requirements.query}
Current: {output.sql_query}"""

    def _get_database_name(self, user_id: str) -> str:
        """Get the database name for a user."""
        return f"user_{user_id}" 