"""Catalog service for managing data catalog and metadata."""

from typing import Dict, Any, Optional, List
from datetime import datetime
import boto3
import json
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
import pandas as pd
import avro.schema
import os
from .glue_service import GlueService
import logging
import asyncio
from io import BytesIO
import pyarrow
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from src.core.sql.sql_state import SQLGenerationOutput
from .exceptions import SQLGenerationError
from pydantic import BaseModel
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQLGenerationChain(BaseModel):
    """Chain for SQL generation using LangChain."""
    llm: ChatOpenAI
    prompt: ChatPromptTemplate
    output_parser: PydanticOutputParser
    preserve_column_names: bool = True  # Default to preserving original column names

    def __init__(self, llm: ChatOpenAI, preserve_column_names: bool = True):
        """Initialize the chain with LLM."""
        # Define the SQL generation prompt template
        system_prompt = (
            "You are a SQL expert. Your task is to generate SQL queries "
            "based on natural language descriptions. Follow these "
            "guidelines:\n"
            "1. Use appropriate JOINs if multiple tables are involved\n"
            "2. Include WHERE clauses for any filters\n"
            "3. Use appropriate aggregation functions if needed\n"
            "4. Include ORDER BY if sorting is needed\n"
            "5. Limit the results to 1000 rows\n"
            "6. Ensure the query is valid SQL syntax\n"
            "7. Use EXACT column names as provided in the schema - do not modify them\n"  # Added emphasis on exact names
            "8. Return your response in the following JSON format:\n"
            "{{\n"
            '    "sql_query": "your SQL query here",\n'
            '    "explanation": "explanation of the query",\n'
            '    "confidence": confidence_score (0-1),\n'
            '    "tables_used": ["list", "of", "tables"],\n'
            '    "columns_used": ["list", "of", "columns"],\n'
            '    "filters": {{"filter_name": "filter_value"}}\n'
            "}}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """Given the following database schema:
            {schema}

            Generate a SQL query for this request:
            {query}

            {format_instructions}""")
        ])

        # Set up the output parser
        output_parser = PydanticOutputParser(
            pydantic_object=SQLGenerationOutput
        )

        # Initialize the parent class with all required fields
        super().__init__(
            llm=llm,
            prompt=prompt,
            output_parser=output_parser,
            preserve_column_names=preserve_column_names
        )

    def _validate_input(self, query: str, schema: Dict[str, Any]) -> None:
        """Validate input parameters."""
        if not query or not query.strip():
            raise SQLGenerationError("Query cannot be empty")
        
        if not schema:
            raise SQLGenerationError("Schema cannot be empty")
        
        if "tables" in schema:
            if not schema["tables"]:
                raise SQLGenerationError(
                    "Schema must contain at least one table"
                )
        elif "table_name" not in schema:
            raise SQLGenerationError(
                "Schema must contain table information"
            )

    async def generate_sql(
        self, query: str, schema: Dict[str, Any]
    ) -> SQLGenerationOutput:
        """Generate SQL query using the chain."""
        try:
            # Validate input
            self._validate_input(query, schema)

            # Format the prompt with schema and query
            formatted_prompt = self.prompt.format_messages(
                schema=schema,
                query=query,
                format_instructions=(
                    self.output_parser.get_format_instructions()
                )
            )

            # Generate response using LLM
            response = await self.llm.ainvoke(formatted_prompt)

            # Parse the response
            result = self.output_parser.parse(response.content)
            return result

        except Exception as e:
            raise SQLGenerationError(
                f"Error generating SQL: {str(e)}"
            )

class CatalogService:
    """Service for managing data catalog and metadata."""
    
    def __init__(
        self,
        bucket: str,
        aws_region: str = 'us-east-1'
    ):
        """Initialize the catalog service.
        
        Args:
            bucket: S3 bucket name for storing data and metadata
            aws_region: AWS region name
        """
        self.bucket = bucket
        self.aws_region = aws_region
        self.base_database_name = os.getenv('GLUE_DATABASE_NAME', 'data_lakehouse')
        logger.info(f"CatalogService initialized with base database: {self.base_database_name}")
        
        # Initialize AWS clients
        self.s3 = boto3.client('s3', region_name=aws_region)
        self.glue = boto3.client('glue', region_name=aws_region)
        self.athena = boto3.client('athena', region_name=aws_region)
        self.glue_service = GlueService(region_name=aws_region)
        self.llm = ChatOpenAI(temperature=0)
        self._setup_agent()

    def _setup_agent(self):
        tools = [
            Tool(
                name="update_schema",
                func=self._update_schema,
                description="Update schema with new changes"
            ),
            Tool(
                name="check_data_quality",
                func=self._check_data_quality,
                description="Check data quality metrics"
            ),
            Tool(
                name="track_schema_evolution",
                func=self._track_schema_evolution,
                description="Track schema changes over time"
            )
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data catalog expert. Your task is to:
            1. Manage schema evolution
            2. Track data quality
            3. Maintain catalog metadata
            Use the provided tools to accomplish these tasks."""),
            MessagesPlaceholder(variable_name="chat_history"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.agent = create_openai_functions_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=tools,
            verbose=True
        )

    def _setup_query_agent(self):
        """Set up an agent specifically for processing descriptive queries."""
        # Define the SQL generation prompt template
        sql_prompt_template = PromptTemplate(
            input_variables=["query", "schema"],
            template="""
            Given the following database schema:
            {schema}

            Generate a SQL query for the following request:
            {query}

            The query should:
            1. Use appropriate JOINs if multiple tables are involved
            2. Include WHERE clauses for any filters
            3. Use appropriate aggregation functions if needed
            4. Include ORDER BY if sorting is needed
            5. Limit the results to 1000 rows

            {format_instructions}
            """
        )

        # Set up the output parser
        output_parser = PydanticOutputParser(
            pydantic_object=SQLGenerationOutput
        )

        # Create the LLM chain
        self.query_chain = LLMChain(
            llm=self.llm,
            prompt=sql_prompt_template,
            output_parser=output_parser
        )

    async def update_catalog(
        self,
        file_info: Dict[str, Any],
        schema: str
    ) -> Dict[str, Any]:
        """
        Update catalog with new file information
        """
        try:
            # Prepare catalog update task
            update_task = f"""
            Update catalog with new file information:
            File: {file_info['file_name']}
            Schema: {schema}
            Track schema evolution and check data quality.
            """

            # Execute update using agent
            result = await self.agent_executor.ainvoke({
                "input": update_task,
                "chat_history": []
            })

            return {
                "status": "success",
                "details": result
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def _update_schema(
        self,
        file_name: str,
        schema: Dict[str, Any],
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update schema in catalog
        """
        try:
            # Create schema entry
            schema_entry = {
                "file_name": file_name,
                "schema": schema,
                "version": version or datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "fields": schema.get("fields", [])
            }

            # Store in S3
            schema_key = f"metadata/schema/{file_name}.json"
            self.s3.put_object(
                Bucket=self.bucket,
                Key=schema_key,
                Body=json.dumps(schema_entry)
            )

            return {
                "status": "success",
                "schema_path": f"s3://{self.bucket}/{schema_key}"
            }

        except Exception as e:
            raise Exception(f"Error updating schema: {str(e)}")

    async def _check_data_quality(
        self,
        file_name: str,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Check data quality metrics
        """
        try:
            null_counts = {k: int(v) for k, v in data.isnull().sum().to_dict().items()}
            duplicate_rows = int(data.duplicated().sum())

            metrics = {
                "row_count": len(data),
                "column_count": len(data.columns),
                "null_counts": null_counts,
                "duplicate_rows": duplicate_rows,
                "column_types": data.dtypes.astype(str).to_dict()
            }
            
            return {
                "status": "success",
                "quality_metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Error checking data quality: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    def _track_schema_evolution(
        self,
        file_name: str,
        new_schema: str
    ) -> Dict[str, Any]:
        """
        Track schema evolution
        """
        try:
            # Get current schema
            schema_key = f"metadata/schema/{file_name}.json"
            try:
                response = self.s3.get_object(
                    Bucket=self.bucket,
                    Key=schema_key
                )
                current_schema = json.loads(response['Body'].read())
            except self.s3.exceptions.NoSuchKey:
                current_schema = None

            # Parse new schema
            new_avro_schema = avro.schema.parse(new_schema)
            new_fields = {
                field.name: str(field.type)
                for field in new_avro_schema.fields
            }

            # Compare schemas
            evolution = {
                "file_name": file_name,
                "timestamp": datetime.utcnow().isoformat(),
                "changes": []
            }

            if current_schema:
                current_fields = {
                    field["name"]: field["type"]
                    for field in current_schema["fields"]
                }

                # Find added fields
                for field_name, field_type in new_fields.items():
                    if field_name not in current_fields:
                        evolution["changes"].append({
                            "type": "added",
                            "field": field_name,
                            "new_type": field_type
                        })

                # Find removed fields
                for field_name, field_type in current_fields.items():
                    if field_name not in new_fields:
                        evolution["changes"].append({
                            "type": "removed",
                            "field": field_name,
                            "old_type": field_type
                        })

                # Find type changes
                for field_name, new_type in new_fields.items():
                    if field_name in current_fields:
                        old_type = current_fields[field_name]
                        if old_type != new_type:
                            evolution["changes"].append({
                                "type": "modified",
                                "field": field_name,
                                "old_type": old_type,
                                "new_type": new_type
                            })

            # Store evolution history
            evolution_key = f"metadata/evolution/{file_name}.json"
            self.s3.put_object(
                Bucket=self.bucket,
                Key=evolution_key,
                Body=json.dumps(evolution)
            )

            return {
                "status": "success",
                "evolution": evolution,
                "evolution_path": f"s3://{self.bucket}/{evolution_key}"
            }

        except Exception as e:
            raise Exception(f"Error tracking schema evolution: {str(e)}")

    async def get_quality_metrics(
        self,
        user_id: str = "test_user"  # Add user_id parameter with default
    ) -> Dict[str, float]:
        """Get overall quality metrics for all tables.
        
        Args:
            user_id: The ID of the user whose tables to analyze
            
        Returns:
            Dict containing overall quality metrics
        """
        try:
            # Get all tables
            tables = await self.list_tables(user_id=user_id)
            
            if not tables:
                return {
                    "completeness": 0.0,
                    "uniqueness": 0.0,
                    "consistency": 0.0,
                    "timeliness": 0.0
                }
            
            # Get quality metrics for each table
            table_metrics = []
            for table in tables:
                try:
                    metrics = await self.get_table_quality(
                        table["Name"],
                        user_id=user_id
                    )
                    table_metrics.append(metrics)
                except Exception as e:
                    logger.warning(f"Error getting metrics for table {table['Name']}: {str(e)}")
                    continue
            
            if not table_metrics:
                return {
                    "completeness": 0.0,
                    "uniqueness": 0.0,
                    "consistency": 0.0,
                    "timeliness": 0.0
                }
            
            # Calculate overall metrics
            overall_metrics = {
                "completeness": sum(m["completeness"] for m in table_metrics) / len(table_metrics),
                "uniqueness": sum(m["uniqueness"] for m in table_metrics) / len(table_metrics),
                "consistency": sum(m["consistency"] for m in table_metrics) / len(table_metrics),
                "timeliness": sum(m["timeliness"] for m in table_metrics) / len(table_metrics)
            }
            
            return overall_metrics
            
        except Exception as e:
            logger.error(f"Error getting quality metrics: {str(e)}")
            raise Exception(f"Error getting quality metrics: {str(e)}")

    async def get_table_quality(
        self,
        table_name: str,
        user_id: str = "test_user"  # Add user_id parameter with default
    ) -> Dict[str, float]:
        """Get quality metrics for a specific table.
        
        Args:
            table_name: Name of the table
            user_id: The ID of the user who owns the table
            
        Returns:
            Dict containing quality metrics
        """
        try:
            # Get table schema
            schema = await self.get_schema(table_name, user_id=user_id)
            
            # Get table data
            query = f"SELECT * FROM {table_name} LIMIT 1000"
            result = await self.execute_query(query, user_id=user_id)
            
            if result["status"] == "error":
                raise Exception(f"Error executing query: {result['message']}")
            
            # Convert results to DataFrame
            df = pd.DataFrame(result["results"])
            
            # Calculate quality metrics
            metrics = {
                "completeness": 1.0 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1]),
                "accuracy": 1.0,  # Placeholder for more complex accuracy checks
                "consistency": 1.0,  # Placeholder for more complex consistency checks
                "timeliness": 1.0  # Placeholder for more complex timeliness checks
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting table quality metrics: {str(e)}")
            raise Exception(f"Error getting table quality metrics: {str(e)}")

    async def list_files(
        self,
        user_id: str = "test_user"  # Add user_id parameter with default
    ) -> List[Dict[str, Any]]:
        """List all files in the catalog.
        
        Args:
            user_id: The ID of the user whose files to list
            
        Returns:
            List of file information dictionaries
        """
        try:
            # List all schema files in S3 for this user
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=f"metadata/schema/{user_id}/"  # Use user-specific prefix
            )

            files = []
            for obj in response.get('Contents', []):
                try:
                    # Get schema file content
                    schema_response = self.s3.get_object(
                        Bucket=self.bucket,
                        Key=obj['Key']
                    )
                    schema_data = json.loads(schema_response['Body'].read())
                    
                    # Create file info
                    file_info = {
                        "file_name": schema_data['file_name'],
                        "table_name": schema_data['table_name'],
                        "user_id": user_id,  # Add user_id to file info
                        "schema": schema_data['columns'],
                        "location": f"s3://{self.bucket}/data/{user_id}/{schema_data['table_name']}",  # Update path
                        "created_at": schema_data['created_at'],
                        "updated_at": schema_data['updated_at']
                    }
                    files.append(file_info)
                except Exception as e:
                    logger.error(f"Error processing schema file {obj['Key']}: {str(e)}")
                    continue
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            raise Exception(f"Error listing files: {str(e)}")

    async def get_catalog(
        self,
        user_id: str = "test_user"  # Add user_id parameter with default
    ) -> Dict[str, Any]:
        """Get the complete catalog for a user.
        
        Args:
            user_id: The ID of the user whose catalog to get
            
        Returns:
            Dict containing:
            - tables: List of tables
            - files: List of files
            - quality_metrics: Overall quality metrics
        """
        try:
            # Get tables
            tables = await self.list_tables(user_id=user_id)
            
            # Get files
            files = await self.list_files(user_id=user_id)
            
            # Get quality metrics
            quality_metrics = await self.get_quality_metrics(user_id=user_id)
            
            return {
                "tables": tables,
                "files": files,
                "quality_metrics": quality_metrics,
                "user_id": user_id  # Add user_id to catalog info
            }
            
        except Exception as e:
            logger.error(f"Error getting catalog: {str(e)}")
            raise Exception(f"Error getting catalog: {str(e)}")

    async def list_tables(
        self,
        user_id: str = "test_user"  # Add user_id parameter with default
    ) -> List[Dict[str, Any]]:
        """List all tables in the catalog.
        
        Args:
            user_id: The ID of the user whose tables to list
            
        Returns:
            List of table information dictionaries
        """
        try:
            # Get tables from Glue
            response = self.glue.get_tables(
                DatabaseName=f"user_{user_id}"  # Use user-specific database
            )
            
            # Convert to our format
            tables = []
            for table in response["TableList"]:
                # Handle datetime objects
                created_at = table.get("CreateTime")
                updated_at = table.get("UpdateTime")
                
                # Convert to ISO format if datetime objects exist
                if created_at and isinstance(created_at, datetime):
                    created_at = created_at.isoformat()
                else:
                    created_at = datetime.utcnow().isoformat()
                    
                if updated_at and isinstance(updated_at, datetime):
                    updated_at = updated_at.isoformat()
                else:
                    updated_at = datetime.utcnow().isoformat()
                
                table_info = {
                    "Name": table.get('Name', 'Unknown Table'),
                    "StorageDescriptor": {
                        "Location": table["StorageDescriptor"]["Location"],
                        "Columns": [
                            {
                                "Name": col["Name"],
                                "Type": col["Type"],
                                "Comment": col.get("Comment", "")
                            }
                            for col in table["StorageDescriptor"]["Columns"]
                        ]
                    },
                    "Description": table.get("Description", ""),
                    "CreateTime": created_at,
                    "UpdateTime": updated_at
                }
                tables.append(table_info)
            
            return tables
            
        except Exception as e:
            logger.error(f"Error listing tables: {str(e)}")
            raise Exception(f"Error listing tables: {str(e)}")

    def _standardize_column_name(self, column_name: str) -> str:
        """
        Standardize column names to follow database naming conventions:
        - Convert to lowercase
        - Replace spaces and special characters with underscores
        - Remove any non-alphanumeric characters except underscores
        - Ensure name starts with a letter
        - Limit length to 63 characters (PostgreSQL limit)
        """
        # Convert to lowercase
        name = column_name.lower()
        
        # Replace spaces and special characters with underscores
        name = re.sub(r'[^a-z0-9_]', '_', name)
        
        # Remove multiple consecutive underscores
        name = re.sub(r'_+', '_', name)
        
        # Remove leading/trailing underscores
        name = name.strip('_')
        
        # Ensure name starts with a letter
        if not name[0].isalpha():
            name = 'col_' + name
        
        # Limit length to 63 characters
        if len(name) > 63:
            name = name[:63]
        
        return name

    async def _process_file_content_to_catalog(
        self,
        content: bytes,
        file_name: str,
        file_format: str,
        table_name: str,
        user_id: str = "test_user"
    ) -> Dict[str, Any]:
        """
        Private helper to process file content (from upload or S3) into catalog.
        Handles DataFrame conversion, Parquet creation, S3 uploads, Schema, and Glue table.
        """
        logger.info(
            f"[_process_file_content_to_catalog] Processing file '{file_name}' "
            f"(format: {file_format}) for table '{table_name}' for user '{user_id}'"
        )
        try:
            # Read file into DataFrame
            if file_format == 'csv':
                df = pd.read_csv(BytesIO(content))
            elif file_format in ['xlsx', 'xls']:
                df = pd.read_excel(BytesIO(content))
            elif file_format == 'json':
                try:
                    df = pd.read_json(BytesIO(content))
                except ValueError as e:
                    logger.warning(
                        f"Direct pd.read_json failed for {file_name}: {e}. "
                        "Trying lines=True."
                    )
                    try:
                        df = pd.read_json(BytesIO(content), lines=True)
                    except Exception as e_lines:
                        msg = (
                            f"Unsupported JSON structure in {file_name}. "
                            f"Both direct read and lines=True failed. Error: {e_lines}"
                        )
                        logger.error(msg)
                        raise ValueError(msg)
            elif file_format == 'parquet':
                df = pd.read_parquet(BytesIO(content))
            else:
                msg = (
                    f"Unsupported file format for DataFrame conversion: {file_format}. "
                    f"Cannot process into DataFrame."
                )
                logger.error(msg)
                raise ValueError(msg)

            # Standardize column names
            df.columns = [self._standardize_column_name(col) for col in df.columns]
            logger.info(f"Standardized column names: {list(df.columns)}")

            # Convert to Parquet
            base_name, _ = os.path.splitext(file_name)
            parquet_file_name = f"{base_name}.parquet"
            parquet_s3_key = f"data/{user_id}/{table_name}/{parquet_file_name}"

            logger.info(f"DataFrame columns before to_parquet: {list(df.columns)}")

            parquet_buffer = BytesIO()
            df.to_parquet(parquet_buffer, index=False)
            parquet_buffer.seek(0)

            self.s3.put_object(
                Bucket=self.bucket,
                Key=parquet_s3_key,
                Body=parquet_buffer.getvalue()
            )
            logger.info(
                f"Uploaded Parquet file to s3://{self.bucket}/{parquet_s3_key}"
            )

            # Create/update Glue database
            user_database_name = self.get_user_database_name(user_id)
            await self.ensure_user_database_exists(user_id)

            # Generate schema for GlueService
            schema_for_glue_service = {
                "fields": [
                    {
                        "name": col,
                        "type": str(df[col].dtype),
                        "description": f"Column {col} from {file_name}"
                    }
                    for col in df.columns
                ]
            }
            logger.info(
                f"Generated schema for GlueService for table {table_name}: "
                f"{json.dumps(schema_for_glue_service, indent=2)}"
            )

            # Create/update Glue table in user-specific database
            table_result = await self.glue_service.create_parquet_table(
                database_name=user_database_name,
                table_name=table_name,
                schema=schema_for_glue_service,
                location=f"s3://{self.bucket}/data/{user_id}/{table_name}/"
            )
            logger.info(f"Glue table creation result: {table_result}")

            # Generate quality metrics
            quality_metrics_result = await self._check_data_quality(file_name, df)

            # Store schema metadata in S3
            schema_metadata_for_s3_cols = [
                {
                    "Name": col,
                    "Type": self._pandas_to_athena_type(str(df[col].dtype)),
                    "Comment": f"Column {col} from {file_name}"
                }
                for col in df.columns
            ]
            schema_metadata_for_s3 = {
                "file_name": file_name,
                "table_name": table_name,
                "user_id": user_id,
                "columns": schema_metadata_for_s3_cols,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            schema_metadata_s3_key = f"metadata/schema/{user_id}/{table_name}.json"
            self.s3.put_object(
                Bucket=self.bucket,
                Key=schema_metadata_s3_key,
                Body=json.dumps(schema_metadata_for_s3, indent=2)
            )
            logger.info(
                f"Uploaded schema metadata to s3://{self.bucket}/{schema_metadata_s3_key}"
            )

            file_info_details = {
                "file_name": file_name,
                "table_name": table_name,
                "user_id": user_id,
                "original_format": file_format,
                "parquet_file_name": parquet_file_name,
                "s3_path_parquet": f"s3://{self.bucket}/{parquet_s3_key}",
                "s3_path_schema_metadata": f"s3://{self.bucket}/{schema_metadata_s3_key}",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "glue_table_info": table_result.get("table_info", {})
            }
            
            logger.info(f"File processing successful for {file_name} -> {table_name}")
            return {
                "status": "success",
                "message": (
                    f"File {file_name} processed, converted to Parquet, and "
                    f"cataloged successfully as table {table_name}"
                ),
                "file_info": file_info_details,
                "schema_used_for_glue": schema_for_glue_service,
                "quality_metrics": quality_metrics_result.get(
                    "quality_metrics", {}
                ),
                "glue_table_details": table_result
            }
        except ValueError as ve:
            logger.error(
                f"ValueError during file content processing for table {table_name}, "
                f"file {file_name}: {str(ve)}"
            )
            raise
        except Exception as e:
            logger.error(
                f"Generic error during file content processing for table {table_name}, "
                f"file {file_name}: {str(e)}", exc_info=True
            )
            # Re-raise with a more generic message to avoid leaking too much detail potentially
            raise Exception(f"Error processing file content for {file_name} into table {table_name}: {str(e)}")

    async def upload_file(
        self,
        file: Any, # FastAPI UploadFile
        table_name: str,
        create_new: bool = False, # create_new might be less relevant if table is always updated/created
        user_id: str = "test_user"
    ) -> Dict[str, Any]:
        """Upload a file (via FastAPI) to S3 and create/update catalog entry. Stores original under originals/{user_id}/{table_name}/{file_name}."""
        file_name = file.filename
        logger.info(
            f"Starting file upload (FastAPI) for user '{user_id}', table '{table_name}', file '{file_name}'"
        )
        try:
            file_format = file_name.split('.')[-1].lower()
            content = await file.read()

            # Upload original file to a user-specific prefix
            original_s3_key = f"originals/{user_id}/{table_name}/{file_name}"
            self.s3.put_object(
                Bucket=self.bucket,
                Key=original_s3_key,
                Body=content # content is already bytes
            )
            logger.info(f"Uploaded original file (from FastAPI upload) to s3://{self.bucket}/{original_s3_key}")

            # Delegate to the common processing logic
            processing_result = await self._process_file_content_to_catalog(
                content=content,
                file_name=file_name,
                file_format=file_format,
                table_name=table_name,
                user_id=user_id  # Pass user_id to _process_file_content_to_catalog
            )
            
            # Enhance the file_info with the original S3 path from this upload flow
            if processing_result.get("status") == "success" and "file_info" in processing_result:
                processing_result["file_info"]["s3_path_original_upload"] = f"s3://{self.bucket}/{original_s3_key}"

            return processing_result

        except ValueError as ve:
            logger.error(
                f"ValueError during FastAPI file upload for table {table_name}, "
                f"file {file_name}: {str(ve)}"
            )
            # Re-raise to be caught by FastAPI error handling or test assertions
            raise
        except Exception as e:
            logger.error(
                f"Generic error during FastAPI file upload for table {table_name}, "
                f"file {file_name}: {str(e)}", exc_info=True
            )
            # Re-raise a more generic exception or specific HTTP-style exception if in endpoint context
            raise Exception(f"Error uploading file via FastAPI: {str(e)}")

    async def process_s3_file(
        self,
        original_s3_uri: str,
        table_name: str,
        user_id: str = "test_user"  # Add user_id parameter with default
    ) -> Dict[str, Any]:
        """
        Process an existing file from S3: convert to Parquet, catalog, create Glue table.
        original_s3_uri: The S3 URI of the original file (e.g., s3://bucket-name/path/to/original.xlsx)
        table_name: The target Glue table name.
        user_id: The ID of the user who owns the table.
        """
        logger.info(
            f"Starting processing of existing S3 file '{original_s3_uri}' for table '{table_name}' for user '{user_id}'"
        )
        try:
            if not original_s3_uri.startswith("s3://"):
                raise ValueError("original_s3_uri must be a valid S3 URI (e.g., s3://bucket/key)")

            # Parse S3 URI
            path_parts = original_s3_uri.replace("s3://", "").split("/", 1)
            if len(path_parts) < 2: # Ensure there's a key part
                 raise ValueError(f"Invalid S3 URI: {original_s3_uri}. Must include bucket and key.")
            
            source_bucket = path_parts[0]
            source_key = path_parts[1]
            
            # For now, assume the source bucket is the same as self.bucket.
            # Could be enhanced to handle cross-bucket if needed, with proper permissions.
            if source_bucket != self.bucket:
                logger.warning(
                    f"Source S3 bucket '{source_bucket}' is different from service bucket '{self.bucket}'. "
                    f"Ensure cross-bucket permissions are configured if this is intended. Proceeding with configured service bucket.")
                # Forcing to use configured bucket, this might need adjustment based on actual requirements
                # For simplicity in this step, we'll assume the file processing and output always uses self.bucket
                # If the intention is to read from a different bucket and write to self.bucket, that's fine.

            file_name = os.path.basename(source_key)
            file_format = file_name.split('.')[-1].lower()

            logger.info(f"Downloading original file from s3://{source_bucket}/{source_key}")
            response = self.s3.get_object(Bucket=source_bucket, Key=source_key)
            content = response['Body'].read()
            logger.info(f"Successfully downloaded {len(content)} bytes from s3://{source_bucket}/{source_key}")

            # Delegate to the common processing logic
            processing_result = await self._process_file_content_to_catalog(
                content=content,
                file_name=file_name, # This is the name of the file from the S3 key
                file_format=file_format,
                table_name=table_name,
                user_id=user_id  # Pass user_id to _process_file_content_to_catalog
            )
            
            # Enhance the file_info with the original S3 path from this processing flow
            if processing_result.get("status") == "success" and "file_info" in processing_result:
                processing_result["file_info"]["s3_path_original_source"] = original_s3_uri

            return processing_result

        except ValueError as ve:
            logger.error(
                f"ValueError during S3 file processing for URI {original_s3_uri}, table {table_name}: {str(ve)}"
            )
            raise
        except self.s3.exceptions.NoSuchKey:
            logger.error(f"Original S3 file not found at {original_s3_uri}")
            raise Exception(f"Original S3 file not found at {original_s3_uri}")
        except Exception as e:
            logger.error(
                f"Generic error processing S3 file {original_s3_uri} for table {table_name}: {str(e)}", exc_info=True
            )
            raise Exception(f"Error processing S3 file {original_s3_uri}: {str(e)}")

    async def execute_query(
        self,
        query: str,
        user_id: str = "test_user",  # Add user_id parameter with default
        output_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a SQL query using Athena.
        
        Args:
            query: SQL query to execute
            user_id: The ID of the user who owns the tables being queried
            output_location: Optional S3 location for query results
            
        Returns:
            Dict containing:
            - status: "success" or "error"
            - results: Query results as list of lists
            - query_execution_id: ID of the query execution
        """
        try:
            if not output_location:
                output_location = f's3://{self.bucket}/athena-results/{user_id}/'  # Add user_id to output path
            
            # Start query execution
            response = self.athena.start_query_execution(
                QueryString=query,
                QueryExecutionContext={
                    'Database': f"user_{user_id}"  # Use user-specific database
                },
                ResultConfiguration={
                    'OutputLocation': output_location
                }
            )
            
            query_execution_id = response['QueryExecutionId']
            
            # Wait for query to complete
            while True:
                query_status = self.athena.get_query_execution(
                    QueryExecutionId=query_execution_id
                )['QueryExecution']['Status']['State']
                
                if query_status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                    break
                    
                await asyncio.sleep(1)
            
            if query_status == 'FAILED':
                error = self.athena.get_query_execution(
                    QueryExecutionId=query_execution_id
                )['QueryExecution']['Status']['StateChangeReason']
                raise Exception(f"Query failed: {error}")
            
            # Get results
            results = []
            paginator = self.athena.get_paginator('get_query_results')
            
            for page in paginator.paginate(QueryExecutionId=query_execution_id):
                for row in page['ResultSet']['Rows'][1:]:  # Skip header row
                    results.append([
                        field.get('VarCharValue', '') for field in row['Data']
                    ])
            
            return {
                "status": "success",
                "results": results,
                "query_execution_id": query_execution_id
            }
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def get_schema(
        self,
        table_name: str,
        user_id: str = "test_user"  # Add user_id parameter with default
    ) -> Dict[str, Any]:
        """Get schema for a table.
        
        Args:
            table_name: Name of the table
            user_id: The ID of the user who owns the table
            
        Returns:
            Dict containing table schema
        """
        try:
            # Get table from Glue
            response = self.glue.get_table(
                DatabaseName=f"user_{user_id}",  # Use user-specific database
                Name=table_name
            )
            
            # Convert Glue schema to our format
            schema = {
                "table_name": table_name,
                "user_id": user_id,  # Add user_id to schema
                "columns": [
                    {
                        "name": col["Name"],
                        "type": col["Type"],
                        "description": col.get("Comment", "")
                    }
                    for col in response["Table"]["StorageDescriptor"]["Columns"]
                ],
                "location": response["Table"]["StorageDescriptor"]["Location"],
                "created_at": response["Table"].get("CreateTime", "").isoformat(),
                "updated_at": response["Table"].get("UpdateTime", "").isoformat()
            }
            
            return schema
            
        except Exception as e:
            logger.error(f"Error getting schema for table {table_name}: {str(e)}")
            raise Exception(f"Error getting schema for table {table_name}: {str(e)}")

    async def update_schema(
        self,
        file_name: str,
        schema: Dict[str, Any],
        user_id: str = "test_user"  # Add user_id parameter with default
    ) -> None:
        """Update schema for a file.
        
        Args:
            file_name: Name of the file
            schema: New schema definition
            user_id: The ID of the user who owns the file
        """
        try:
            # Update schema in S3
            schema_key = f"metadata/schema/{user_id}/{file_name}.json"  # Update path
            self.s3.put_object(
                Bucket=self.bucket,
                Key=schema_key,
                Body=json.dumps(schema, indent=2)
            )
            
            # Update Glue table
            table_input = {
                'Name': schema['table_name'],
                'TableType': 'EXTERNAL_TABLE',
                'Parameters': {
                    'classification': 'parquet',
                    'typeOfData': 'file',
                    'user_id': user_id  # Add user_id to parameters
                },
                'StorageDescriptor': {
                    'Columns': [
                        {
                            'Name': col['name'],
                            'Type': col['type'],
                            'Comment': col['description']
                        }
                        for col in schema['columns']
                    ],
                    'Location': (
                        f"s3://{self.bucket}/data/{user_id}/{schema['table_name']}/"  # Update path
                    ),
                    'InputFormat': (
                        'org.apache.hadoop.hive.ql.io.parquet.'
                        'MapredParquetInputFormat'
                    ),
                    'OutputFormat': (
                        'org.apache.hadoop.hive.ql.io.parquet.'
                        'MapredParquetOutputFormat'
                    ),
                    'SerdeInfo': {
                        'SerializationLibrary': (
                            'org.apache.hadoop.hive.ql.io.parquet.serde.'
                            'ParquetHiveSerDe'
                        ),
                        'Parameters': {
                            'serialization.format': '1'
                        }
                    }
                }
            }
            
            try:
                self.glue.create_table(
                    DatabaseName=f"user_{user_id}",  # Use user-specific database
                    TableInput=table_input
                )
            except self.glue.exceptions.AlreadyExistsException:
                self.glue.update_table(
                    DatabaseName=f"user_{user_id}",  # Use user-specific database
                    TableInput=table_input
                )
            
        except Exception as e:
            logger.error(f"Error updating schema: {str(e)}")
            raise

    def _pandas_to_athena_type(self, dtype_str: str) -> str:
        """Convert pandas dtype string to Athena/Glue compatible type string."""
        # Log input for debugging
        # logger.info(f"[_pandas_to_athena_type] Input pandas_type_str: \'{dtype_str}\'") # Too noisy if called for every column during other ops
    
        # Simplified and more robust checks
        if "bool" in dtype_str:
            athena_type = "boolean"
        elif "int64" in dtype_str:
            athena_type = "bigint"
        elif "int32" in dtype_str:
            athena_type = "int"
        elif "int16" in dtype_str:
            athena_type = "smallint"
        elif "int8" in dtype_str:
            athena_type = "tinyint"
        elif "float64" in dtype_str:
            athena_type = "double"
        elif "float32" in dtype_str:
            athena_type = "float"
        elif "datetime64" in dtype_str: # Pandas datetime
            athena_type = "timestamp"
        elif "object" in dtype_str: # Pandas string/mixed
            athena_type = "string"
        elif "category" in dtype_str:
            athena_type = "string" # Treat categories as strings
        else: # Fallback for unhandled types
            logger.warning(f"Unhandled pandas type: {dtype_str}, defaulting to string for Athena.")
            athena_type = "string"
    
        # logger.info(f"[_pandas_to_athena_type] Output Athena type: \'{athena_type}\' for input \'{dtype_str}\'")
        return athena_type 

    def _pandas_to_pyarrow_type(self, dtype_str: str) -> pyarrow.DataType:
        """Convert pandas dtype string to PyArrow DataType."""
        # logger.info(f"[_pandas_to_pyarrow_type] Input pandas_type_str: '{dtype_str}'")
        if "bool" in dtype_str:
            pa_type = pyarrow.bool_()
        elif "int64" in dtype_str:
            pa_type = pyarrow.int64()
        elif "int32" in dtype_str:
            pa_type = pyarrow.int32()
        elif "int16" in dtype_str:
            pa_type = pyarrow.int16()
        elif "int8" in dtype_str:
            pa_type = pyarrow.int8()
        elif "float64" in dtype_str:
            pa_type = pyarrow.float64()
        elif "float32" in dtype_str:
            pa_type = pyarrow.float32()
        elif "datetime64" in dtype_str: # Pandas datetime
            pa_type = pyarrow.timestamp('ns') # nanosecond precision
        elif "object" in dtype_str: # Pandas string/mixed
            pa_type = pyarrow.string()
        elif "category" in dtype_str:
            pa_type = pyarrow.string() # Treat categories as strings
        else: # Fallback for unhandled types
            logger.warning(f"Unhandled pandas type for PyArrow: {dtype_str}, defaulting to string.")
            pa_type = pyarrow.string()
        # logger.info(f"[_pandas_to_pyarrow_type] Output PyArrow type: '{pa_type}' for input '{dtype_str}'")
        return pa_type 

    async def get_file(
        self,
        s3_path: str,
        user_id: str = "test_user"  # Add user_id parameter with default
    ) -> dict:
        """Get file information from S3.
        
        Args:
            s3_path: S3 path to the file
            user_id: The ID of the user who owns the file
            
        Returns:
            Dict containing file information
        """
        try:
            # Parse S3 path
            if not s3_path.startswith("s3://"):
                raise ValueError("s3_path must be a valid S3 URI (e.g., s3://bucket/key)")
            
            path_parts = s3_path.replace("s3://", "").split("/", 1)
            if len(path_parts) < 2:
                raise ValueError(f"Invalid S3 URI: {s3_path}. Must include bucket and key.")
            
            bucket = path_parts[0]
            key = path_parts[1]
            
            # Get file metadata
            response = self.s3.head_object(Bucket=bucket, Key=key)
            
            # Get schema if available
            schema = None
            try:
                schema_key = (
                    f"metadata/schema/{user_id}/{os.path.basename(key)}.json"
                )  # Update path
                schema_response = self.s3.get_object(Bucket=bucket, Key=schema_key)
                schema = json.loads(schema_response['Body'].read())
            except self.s3.exceptions.NoSuchKey:
                pass
            
            return {
                "name": os.path.basename(key),
                "size": response['ContentLength'],
                "last_modified": response['LastModified'].isoformat(),
                "format": os.path.splitext(key)[1][1:].lower(),
                "location": s3_path,
                "user_id": user_id,  # Add user_id to file info
                "schema": schema
            }
            
        except Exception as e:
            logger.error(f"Error getting file info: {str(e)}")
            raise Exception(f"Error getting file info: {str(e)}")

    async def process_descriptive_query(
        self,
        query: str,
        table_name: Optional[str] = None,
        preserve_column_names: bool = True,
        user_id: str = "test_user"
    ) -> Dict[str, Any]:
        """Process a descriptive query using LangChain."""
        try:
            # Get table schema
            schema = await self.get_schema(table_name, user_id=user_id)
            if not schema:
                return {
                    "status": "error",
                    "message": f"Could not find schema for table {table_name}"
                }

            # Format schema for SQL generation
            formatted_schema = {
                "table_name": schema["table_name"],
                "columns": [
                    {
                        "name": col["name"] if preserve_column_names else col["name"].replace(" ", "_"),
                        "type": col["type"],
                        "description": col.get("description", "")
                    }
                    for col in schema["columns"]
                ]
            }

            # Initialize SQL generation chain
            chain = SQLGenerationChain(
                llm=self.llm,
                preserve_column_names=preserve_column_names
            )

            # Generate SQL
            result = await chain.generate_sql(
                query=query,
                schema=formatted_schema
            )

            if not result:
                return {
                    "status": "error",
                    "message": "Failed to generate SQL query"
                }

            # Execute the generated SQL
            query_result = await self.execute_query(result.sql_query, user_id=user_id)

            return {
                "status": "success",
                "query": result.sql_query,
                "results": query_result.get("results", []),
                "explanation": result.explanation,
                "confidence": result.confidence,
                "tables_used": result.tables_used,
                "columns_used": result.columns_used,
                "filters": result.filters
            }

        except Exception as e:
            logger.error(f"Error processing descriptive query: {str(e)}")
            return {
                "status": "error",
                "message": f"Error processing query: {str(e)}"
            }

    async def create_user_table(
        self,
        user_id: str,
        table_name: str,
        schema: Dict[str, Any],
        original_data_location: str,
        parquet_data_location: str,
        file_format: str = "parquet",
        partition_keys: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        compression: str = "none",
        encryption: str = "none"
    ) -> Dict[str, Any]:
        """
        Create a table in a user-specific database with full metadata organization.
        - Database: user_{user_id}
        - Table: {table_name}
        - Supports Parquet and CSV
        - Partition keys optional
        - Metadata includes:
            - original_data_location: S3 path to raw/original data
            - parquet_data_location: S3 path to processed Parquet data
            - created_at, updated_at timestamps
            - file_format, compression, encryption
            - Any additional metadata (e.g., lineage, version, etc.)
        - StorageDescriptor contains primary data location, serialization, columns, etc.
        """
        # Centralized metadata for Glue table parameters
        table_metadata = {
            "original_data_location": original_data_location,
            "parquet_data_location": parquet_data_location,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "file_format": file_format,
            "compression": compression,
            "encryption": encryption,
        }
        if metadata:
            table_metadata.update(metadata)
        # Use Parquet location as the primary location for Glue
        result = await self.glue_service.create_user_table(
            user_id=user_id,
            table_name=table_name,
            schema=schema,
            location=parquet_data_location,
            file_format=file_format,
            partition_keys=partition_keys
        )
        # Patch in our metadata (GlueService already adds some, but we want to ensure all fields)
        if "table_info" in result:
            result["table_info"].update(table_metadata)
        return result 

    async def get_table(
        self,
        table_name: str,
        user_id: str = "test_user"  # Add user_id parameter with default
    ) -> dict:
        """Get table information from Glue catalog.
        
        Args:
            table_name: Name of the table
            user_id: The ID of the user who owns the table
            
        Returns:
            Dict containing table information
        """
        try:
            # Get table from Glue
            database_name = f"user_{user_id}"  # Use user-specific database
            response = self.glue.get_table(
                DatabaseName=database_name,
                Name=table_name
            )
            
            # Get table schema
            schema = []
            for column in response['Table']['StorageDescriptor']['Columns']:
                schema.append({
                    "name": column['Name'],
                    "type": column['Type']
                })
            
            # Get table location
            location = response['Table']['StorageDescriptor']['Location']
            
            # Get table properties
            properties = response['Table'].get('Parameters', {})
            
            return {
                "name": table_name,
                "database": database_name,
                "location": location,
                "schema": schema,
                "properties": properties,
                "user_id": user_id,  # Add user_id to table info
                "created_at": response['Table'].get('CreateTime', '').isoformat(),
                "last_updated": response['Table'].get('UpdateTime', '').isoformat()
            }
            
        except self.glue.exceptions.EntityNotFoundException:
            raise Exception(f"Table {table_name} not found in database {database_name}")
        except Exception as e:
            logger.error(f"Error getting table info: {str(e)}")
            raise Exception(f"Error getting table info: {str(e)}")

    async def delete_table(
        self,
        table_name: str,
        user_id: str = "test_user"  # Add user_id parameter with default
    ) -> bool:
        """Delete a table from Glue catalog.
        
        Args:
            table_name: Name of the table to delete
            user_id: The ID of the user who owns the table
            
        Returns:
            True if deletion was successful
        """
        try:
            # Delete table from Glue
            database_name = f"user_{user_id}"  # Use user-specific database
            self.glue.delete_table(
                DatabaseName=database_name,
                Name=table_name
            )
            
            # Delete schema metadata from S3
            schema_key = f"metadata/schema/{user_id}/{table_name}.json"  # Update path
            try:
                self.s3.delete_object(
                    Bucket=self.bucket,
                    Key=schema_key
                )
            except self.s3.exceptions.NoSuchKey:
                pass  # Schema file may not exist
                
            return True
            
        except self.glue.exceptions.EntityNotFoundException:
            raise Exception(f"Table {table_name} not found in database {database_name}")
        except Exception as e:
            logger.error(f"Error deleting table: {str(e)}")
            raise Exception(f"Error deleting table: {str(e)}")

    async def delete_file(
        self,
        s3_path: str,
        user_id: str = "test_user"  # Add user_id parameter with default
    ) -> bool:
        """Delete a file from S3.
        
        Args:
            s3_path: S3 path to the file
            user_id: The ID of the user who owns the file
            
        Returns:
            True if deletion was successful
        """
        try:
            # Parse S3 path
            if not s3_path.startswith("s3://"):
                raise ValueError("s3_path must be a valid S3 URI (e.g., s3://bucket/key)")
            
            path_parts = s3_path.replace("s3://", "").split("/", 1)
            if len(path_parts) < 2:
                raise ValueError(f"Invalid S3 URI: {s3_path}. Must include bucket and key.")
            
            bucket = path_parts[0]
            key = path_parts[1]
            
            # Delete file from S3
            self.s3.delete_object(
                Bucket=bucket,
                Key=key
            )
            
            # Delete schema metadata if it exists
            schema_key = f"metadata/schema/{user_id}/{os.path.basename(key)}.json"  # Update path
            try:
                self.s3.delete_object(
                    Bucket=bucket,
                    Key=schema_key
                )
            except self.s3.exceptions.NoSuchKey:
                pass  # Schema file may not exist
                
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}")
            raise Exception(f"Error deleting file: {str(e)}")

    async def get_table_schema(
        self,
        table_name: str,
        user_id: str = "test_user"  # Add user_id parameter with default
    ) -> dict:
        """Get schema for a table from Glue catalog.
        
        Args:
            table_name: Name of the table
            user_id: The ID of the user who owns the table
            
        Returns:
            Dict containing table schema
        """
        try:
            # Get table from Glue
            database_name = f"user_{user_id}"  # Use user-specific database
            response = self.glue.get_table(
                DatabaseName=database_name,
                Name=table_name
            )
            
            # Extract schema
            schema = []
            for column in response['Table']['StorageDescriptor']['Columns']:
                schema.append({
                    "name": column['Name'],
                    "type": column['Type'],
                    "comment": column.get('Comment', '')
                })
            
            return {
                "table_name": table_name,
                "database": database_name,
                "schema": schema,
                "user_id": user_id  # Add user_id to schema info
            }
            
        except self.glue.exceptions.EntityNotFoundException:
            raise Exception(f"Table {table_name} not found in database {database_name}")
        except Exception as e:
            logger.error(f"Error getting table schema: {str(e)}")
            raise Exception(f"Error getting table schema: {str(e)}")

    async def get_table_data(
        self,
        table_name: str,
        limit: int = 1000,
        user_id: str = "test_user"  # Add user_id parameter with default
    ) -> dict:
        """Get data from a table using Athena.
        
        Args:
            table_name: Name of the table
            limit: Maximum number of rows to return
            user_id: The ID of the user who owns the table
            
        Returns:
            Dict containing table data and metadata
        """
        try:
            # Get table schema
            schema = await self.get_table_schema(table_name, user_id)
            
            # Construct query
            database_name = f"user_{user_id}"  # Use user-specific database
            query = f"SELECT * FROM {database_name}.{table_name} LIMIT {limit}"
            
            # Execute query
            query_execution_id = self.athena.start_query_execution(
                QueryString=query,
                QueryExecutionContext={
                    'Database': database_name
                },
                ResultConfiguration={
                    'OutputLocation': f"s3://{self.bucket}/athena-results/"
                }
            )['QueryExecutionId']
            
            # Wait for query to complete
            while True:
                response = self.athena.get_query_execution(
                    QueryExecutionId=query_execution_id
                )
                state = response['QueryExecution']['Status']['State']
                if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                    break
                await asyncio.sleep(1)
            
            if state != 'SUCCEEDED':
                raise Exception(
                    f"Query failed: {response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')}"
                )
            
            # Get results
            results = []
            paginator = self.athena.get_paginator('get_query_results')
            for page in paginator.paginate(QueryExecutionId=query_execution_id):
                for row in page['ResultSet']['Rows'][1:]:  # Skip header row
                    results.append([field.get('VarCharValue', '') for field in row['Data']])
            
            return {
                "table_name": table_name,
                "database": database_name,
                "schema": schema,
                "data": results,
                "user_id": user_id,  # Add user_id to data info
                "row_count": len(results)
            }
            
        except Exception as e:
            logger.error(f"Error getting table data: {str(e)}")
            raise Exception(f"Error getting table data: {str(e)}")

    async def get_table_metadata(
        self,
        table_name: str,
        user_id: str = "test_user"  # Add user_id parameter with default
    ) -> dict:
        """Get metadata for a table from Glue catalog.
        
        Args:
            table_name: Name of the table
            user_id: The ID of the user who owns the table
            
        Returns:
            Dict containing table metadata
        """
        try:
            # Get table from Glue
            database_name = f"user_{user_id}"  # Use user-specific database
            response = self.glue.get_table(
                DatabaseName=database_name,
                Name=table_name
            )
            
            # Extract metadata
            table = response['Table']
            metadata = {
                "name": table_name,
                "database": database_name,
                "location": table['StorageDescriptor']['Location'],
                "input_format": table['StorageDescriptor'].get('InputFormat', ''),
                "output_format": table['StorageDescriptor'].get('OutputFormat', ''),
                "compressed": table['StorageDescriptor'].get('Compressed', False),
                "number_of_buckets": table['StorageDescriptor'].get('NumberOfBuckets', -1),
                "ser_de_info": table['StorageDescriptor'].get('SerdeInfo', {}),
                "bucket_columns": table['StorageDescriptor'].get('BucketColumns', []),
                "sort_columns": table['StorageDescriptor'].get('SortColumns', []),
                "parameters": table.get('Parameters', {}),
                "table_type": table.get('TableType', ''),
                "created_at": table.get('CreateTime', '').isoformat(),
                "last_updated": table.get('UpdateTime', '').isoformat(),
                "user_id": user_id  # Add user_id to metadata
            }
            
            return metadata
            
        except self.glue.exceptions.EntityNotFoundException:
            raise Exception(f"Table {table_name} not found in database {database_name}")
        except Exception as e:
            logger.error(f"Error getting table metadata: {str(e)}")
            raise Exception(f"Error getting table metadata: {str(e)}")

    def get_user_database_name(self, user_id: str) -> str:
        """Get the user-specific database name.
        
        Args:
            user_id: The user ID to get the database name for
            
        Returns:
            The user-specific database name
        """
        return f"user_{user_id}"

    async def ensure_user_database_exists(self, user_id: str) -> None:
        """Ensure that the user-specific database exists.
        
        Args:
            user_id: The user ID to ensure the database exists for
        """
        database_name = self.get_user_database_name(user_id)
        try:
            self.glue.get_database(Name=database_name)
        except self.glue.exceptions.EntityNotFoundException:
            self.glue.create_database(
                DatabaseInput={
                    'Name': database_name,
                    'Description': f'Database for user {user_id}'
                }
            )
            logger.info(f"Created database {database_name} for user {user_id}") 