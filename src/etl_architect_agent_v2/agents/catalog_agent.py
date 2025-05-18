"""Catalog agent for managing data catalog and metadata."""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import json
import logging
from datetime import datetime
import boto3
from pydantic import BaseModel, ConfigDict, Field
import uuid
import os
from etl_architect_agent_v2.core.llm_manager import LLMManager
from etl_architect_agent_v2.core.error_handler import ErrorHandler
from io import BytesIO
from etl_architect_agent_v2.core.schema_generator import SchemaGenerator
from etl_architect_agent_v2.core.schema_validator import SchemaValidator
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool, StructuredTool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import asyncio

from src.etl_architect_agent_v2.backend.services.catalog_service import CatalogService

logger = logging.getLogger(__name__)

class SchemaVersion(BaseModel):
    """Schema version information."""
    version: str
    schema: Dict[str, Any]
    created_at: datetime
    created_by: str
    changes: List[Dict[str, Any]]
    compatibility: str  # backward, forward, full, none

class DataLineage(BaseModel):
    """Data lineage information."""
    source_file: str
    source_schema: Dict[str, Any]
    transformations: List[Dict[str, Any]]
    target_file: str
    target_schema: Dict[str, Any]
    created_at: datetime
    created_by: str

class AuditInfo(BaseModel):
    """Audit information for catalog entries."""
    created_by: str
    created_at: datetime
    updated_by: str
    updated_at: datetime
    file_name: str
    table_name: str
    s3_path: str
    format: str
    conversion_history: List[Dict[str, Any]] = []
    schema_versions: List[SchemaVersion] = []
    lineage: Optional[DataLineage] = None

class CatalogState(BaseModel):
    """State management for catalog operations."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    file_path: str
    file_name: str
    file_type: str
    s3_path: str
    schema: Optional[Dict[str, Any]] = None
    table_name: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    progress: float = 0.0
    current_operation: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}
    audit_info: Optional[AuditInfo] = None
    schema_version: Optional[str] = None
    lineage_info: Optional[Dict[str, Any]] = None

class UpdateSchemaInput(BaseModel):
    """Input schema for update_schema tool."""
    file_name: str
    schema: Dict[str, Any]
    version: Optional[str] = None

class CheckDataQualityInput(BaseModel):
    """Input schema for check_data_quality tool."""
    file_name: str
    file_type: Dict[str, Any]
    data: Dict[str, Any]

class ExtractSchemaInput(BaseModel):
    """Input schema for extract_schema tool."""
    file_name: str
    file_type: Dict[str, Any]
    data: Dict[str, Any]
    table_name: str

class CatalogAgent:
    """Central agent for managing data catalog and metadata."""
    
    def __init__(self, catalog_service: CatalogService):
        """Initialize the catalog agent with AWS clients and LLM manager."""
        self.catalog_service = catalog_service
        self.llm = ChatOpenAI(temperature=0)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self._setup_agent()
        
    def _setup_agent(self):
        """Set up the agent with tools and prompt."""
        tools = [
            Tool(
                name="process_descriptive_query",
                func=self.process_descriptive_query,
                description="Process a descriptive query and convert it to SQL"
            ),
            Tool(
                name="process_sql_query",
                func=self.process_sql_query,
                description="Process a SQL query and execute it"
            )
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data catalog expert. Your task is to:
            1. Convert descriptive queries to SQL
            2. Execute SQL queries
            3. Return results in a structured format
            Use the provided tools to accomplish these tasks."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.agent = create_openai_functions_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=tools,
            verbose=True
        )

    async def process_descriptive_query(
        self,
        query: str,
        table_name: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Process a descriptive query and convert it to SQL.
        
        Args:
            query: The descriptive query in natural language
            table_name: The name of the table to query
            user_id: The ID of the user making the query
            
        Returns:
            Dict containing:
            - status: "success" or "error"
            - query: The generated SQL query
            - results: The query results
        """
        try:
            # Get table schema
            schema = await self.catalog_service.get_schema(table_name, user_id)
            
            # Convert descriptive query to SQL using LLM
            sql_prompt = f"""
            Convert the following descriptive query to SQL.
            The query should be against a table named '{table_name}'.
            Query: {query}
            
            Table Schema (for table '{table_name}'):
            {json.dumps(schema, indent=2)}
            
            Generate a SQL query that answers the question using the table named '{table_name}'.
            """
            
            response = await self.llm.ainvoke(sql_prompt)
            sql_query = response.content.strip()
            logger.info(f"Generated SQL query for table '{table_name}': {sql_query}")
            
            # Execute the SQL query
            result = await self.process_sql_query(sql_query, table_name, user_id)
            
            return {
                "status": "success",
                "query": sql_query,
                "results": result["results"]
            }
            
        except Exception as e:
            logger.error(f"Error processing descriptive query: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def process_sql_query(
        self,
        query: str,
        table_name: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Process and execute a SQL query.
        
        Args:
            query: The SQL query to execute
            table_name: The name of the table to query
            user_id: The ID of the user making the query
            
        Returns:
            Dict containing:
            - status: "success" or "error"
            - results: The query results
        """
        try:
            # Execute query using catalog service
            result = await self.catalog_service.execute_query(query, user_id)
            
            if result["status"] == "error":
                raise Exception(result["message"])
            
            return {
                "status": "success",
                "results": result["results"]
            }
            
        except Exception as e:
            logger.error(f"Error processing SQL query: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def process_file_upload(
        self,
        df: pd.DataFrame,
        file_name: str,
        table_name: str,
        create_new: bool = False
    ) -> Dict[str, Any]:
        """Process a file upload and create/update catalog entry.
        
        Args:
            df: The DataFrame containing the data
            file_name: The name of the uploaded file
            table_name: The name of the table to create/update
            create_new: Whether to create a new table or update existing
            
        Returns:
            Dict containing:
            - status: "success" or "error"
            - schema: The generated schema
            - quality_metrics: Data quality metrics
            - s3_path: The S3 path of the uploaded file
        """
        try:
            # Upload file to S3
            s3_path = await self.catalog_service.upload_file(
                df=df,
                file_name=file_name,
                table_name=table_name
            )
            
            # Generate schema
            schema = {
                "file_name": file_name,
                "table_name": table_name,
                "columns": [
                    {
                        "name": col,
                        "type": str(df[col].dtype),
                        "description": f"Column {col} from {table_name}",
                        "nullable": bool(df[col].isnull().any())
                    }
                    for col in df.columns
                ],
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Upload schema
            await self.catalog_service.update_schema(
                file_name=file_name,
                schema=schema
            )
            
            # Generate quality metrics
            quality_metrics = await self.catalog_service._check_data_quality(
                file_name=file_name,
                data=df
            )
            
            return {
                "status": "success",
                "schema": schema,
                "quality_metrics": quality_metrics["quality_metrics"],
                "s3_path": s3_path
            }
            
        except Exception as e:
            logger.error(f"Error processing file upload: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def process_file(self, state: CatalogState, user_id: str) -> CatalogState:
        """Process a file and generate its catalog entry."""
        try:
            # Initialize audit info
            state.audit_info = AuditInfo(
                created_by=user_id,
                created_at=datetime.utcnow(),
                updated_by=user_id,
                updated_at=datetime.utcnow(),
                file_name=state.file_name,
                table_name=state.table_name or "",
                s3_path=state.s3_path,
                format=state.file_type,
                conversion_history=[],
                schema_versions=[],
                lineage=None
            )
            
            # Check for existing schema
            existing_schema = await self._get_existing_schema(state.table_name)
            if existing_schema:
                # Track schema evolution
                state = await self._track_schema_evolution(state, existing_schema, user_id)
                if state.error:
                    return state
            
            # Generate schema
            state = await self._generate_schema(state)
            if state.error:
                return state
                
            # Store metadata
            state = await self._store_metadata(state, user_id)
            if state.error:
                return state
                
            # Create Glue table
            state = await self._create_glue_table(state)
            if state.error:
                return state
                
            # Generate data quality metrics
            state = await self._generate_quality_metrics(state)
            if state.error:
                return state
                
            # Update catalog index
            state = await self._update_catalog_index(state)
            if state.error:
                return state
                
            return state
            
        except Exception as e:
            state.error = str(e)
            return state

    async def _track_schema_evolution(
        self,
        state: CatalogState,
        existing_schema: Dict[str, Any],
        user_id: str
    ) -> CatalogState:
        """Track schema evolution and validate compatibility."""
        try:
            # Create schema comparison prompt
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are a schema evolution expert. Analyze the changes between two schemas and determine:
                1. What fields were added, removed, or modified
                2. The compatibility level (backward, forward, full, none)
                3. Any potential data migration needs
                Return the analysis as a JSON object."""),
                HumanMessage(content=f"""Compare these schemas:
                Existing Schema: {json.dumps(existing_schema, indent=2)}
                New Schema: {json.dumps(state.schema, indent=2)}
                """)
            ])
            
            # Get schema comparison
            chain = prompt | self.llm_manager.llm | StrOutputParser()
            comparison = json.loads(await chain.ainvoke({}))
            
            # Create new schema version
            version = SchemaVersion(
                version=str(uuid.uuid4()),
                schema=state.schema,
                created_at=datetime.utcnow(),
                created_by=user_id,
                changes=comparison["changes"],
                compatibility=comparison["compatibility"]
            )
            
            # Add to audit info
            state.audit_info.schema_versions.append(version)
            
            # Store schema version
            await self._store_schema_version(state.table_name, version)
            
            # Update state
            state.schema_version = version.version
            state.metadata["schema_evolution"] = comparison
            
            return state
            
        except Exception as e:
            state.error = f"Error tracking schema evolution: {str(e)}"
            return state

    async def _store_schema_version(self, table_name: str, version: SchemaVersion) -> None:
        """Store schema version information."""
        try:
            version_key = f"metadata/schema_versions/{table_name}/{version.version}.json"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=version_key,
                Body=version.json()
            )
        except Exception as e:
            logger.error(f"Error storing schema version: {str(e)}")
            raise

    async def _get_existing_schema(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get existing schema for a table."""
        try:
            schema_key = f"metadata/schema/{table_name}.json"
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=schema_key
            )
            return json.loads(response['Body'].read().decode('utf-8'))
        except self.s3_client.exceptions.NoSuchKey:
            return None
        except Exception as e:
            logger.error(f"Error getting existing schema: {str(e)}")
            return None

    async def _update_catalog_index(self, state: CatalogState) -> CatalogState:
        """Update the central catalog index with the new file entry."""
        try:
            # Get existing index
            index_path = 'metadata/catalog_index.json'
            try:
                response = self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=index_path
                )
                index = json.loads(response['Body'].read().decode('utf-8'))
            except:
                index = {'files': [], 'tables': []}
                
            # Add new entry with enhanced metadata
            entry = {
                'file_name': state.file_name,
                's3_path': state.s3_path,
                'table_name': state.table_name,
                'file_type': state.file_type,
                'schema': state.schema,
                'schema_version': state.schema_version,
                'metadata': state.metadata,
                'lineage': state.lineage_info,
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }
            
            index['files'].append(entry)
            index['tables'].append({
                'name': state.table_name,
                'database': self.database_name,
                'location': state.s3_path,
                'schema': state.schema,
                'schema_version': state.schema_version,
                'lineage': state.lineage_info
            })
            
            # Store updated index
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=index_path,
                Body=json.dumps(index, indent=2)
            )
            
            return state
            
        except Exception as e:
            state.error = f"Error updating catalog index: {str(e)}"
            return state

    async def get_catalog_entry(self, file_name: str) -> Dict[str, Any]:
        """Get a catalog entry for a specific file."""
        try:
            # Get catalog index
            index_path = 'metadata/catalog_index.json'
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=index_path
            )
            index = json.loads(response['Body'].read().decode('utf-8'))
            
            # Find entry
            for entry in index['files']:
                if entry['file_name'] == file_name:
                    return entry
                    
            return None
            
        except Exception as e:
            logger.error(f"Error getting catalog entry: {str(e)}")
            return None
            
    async def list_catalog_entries(self, filter_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all catalog entries, optionally filtered by type."""
        try:
            # Get catalog index
            index_path = 'metadata/catalog_index.json'
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=index_path
            )
            index = json.loads(response['Body'].read().decode('utf-8'))
            
            # Filter if needed
            if filter_type:
                return [entry for entry in index['files'] if entry['file_type'] == filter_type]
            return index['files']
            
        except Exception as e:
            logger.error(f"Error listing catalog entries: {str(e)}")
            return []
            
    async def update_catalog_entry(self, file_name: str, updates: Dict[str, Any]) -> bool:
        """Update a catalog entry with new information."""
        try:
            # Get catalog index
            index_path = 'metadata/catalog_index.json'
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=index_path
            )
            index = json.loads(response['Body'].read().decode('utf-8'))
            
            # Find and update entry
            for entry in index['files']:
                if entry['file_name'] == file_name:
                    entry.update(updates)
                    entry['last_updated'] = datetime.now().isoformat()
                    
                    # Store updated index
                    self.s3_client.put_object(
                        Bucket=self.bucket_name,
                        Key=index_path,
                        Body=json.dumps(index, indent=2)
                    )
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error updating catalog entry: {str(e)}")
            return False
            
    async def delete_catalog_entry(self, file_name: str) -> bool:
        """Delete a catalog entry."""
        try:
            # Get catalog index
            index_path = 'metadata/catalog_index.json'
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=index_path
            )
            index = json.loads(response['Body'].read().decode('utf-8'))
            
            # Find and remove entry
            for i, entry in enumerate(index['files']):
                if entry['file_name'] == file_name:
                    # Remove from files and tables
                    index['files'].pop(i)
                    index['tables'] = [t for t in index['tables'] if t['name'] != entry['table_name']]
                    
                    # Store updated index
                    self.s3_client.put_object(
                        Bucket=self.bucket_name,
                        Key=index_path,
                        Body=json.dumps(index, indent=2)
                    )
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error deleting catalog entry: {str(e)}")
            return False

    async def _generate_schema(self, state: CatalogState) -> CatalogState:
        """Generate schema for the file using LangChain components."""
        try:
            logger.info(f"Reading sample data from S3: {state.s3_path}")
            # Read sample data from S3
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=state.s3_path
            )

            # Read data based on file type
            if state.file_type.lower() in ['csv', 'text/csv']:
                df = pd.read_csv(response['Body'], nrows=100)
            elif state.file_type.lower() in [
                'xlsx', 'xls',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'application/vnd.ms-excel'
            ]:
                file_content = response['Body'].read()
                df = pd.read_excel(BytesIO(file_content), nrows=100)
            else:
                raise ValueError(f"Unsupported file type: {state.file_type}")

            logger.info(f"Successfully read sample data from {state.s3_path}")

            # Generate schema using LangChain components
            state.schema = await self.schema_generator.generate_schema(df)
            if state.error:
                return state
                
            logger.info(f"Successfully generated schema for {state.file_name}")
            
            # Add audit columns to schema
            schema = state.schema
            schema["columns"].extend([
                {
                    "name": "_created_at",
                    "type": "timestamp",
                    "description": "Timestamp when the record was created",
                    "sample_value": datetime.utcnow().isoformat(),
                    "quality_metrics": {
                        "completeness": 1.0,
                        "uniqueness": 1.0,
                        "validity": 1.0
                    }
                },
                {
                    "name": "_updated_at",
                    "type": "timestamp",
                    "description": "Timestamp when the record was last updated",
                    "sample_value": datetime.utcnow().isoformat(),
                    "quality_metrics": {
                        "completeness": 1.0,
                        "uniqueness": 1.0,
                        "validity": 1.0
                    }
                },
                {
                    "name": "_created_by",
                    "type": "string",
                    "description": "User who created the record",
                    "sample_value": "system",
                    "quality_metrics": {
                        "completeness": 1.0,
                        "uniqueness": 1.0,
                        "validity": 1.0
                    }
                },
                {
                    "name": "_updated_by",
                    "type": "string",
                    "description": "User who last updated the record",
                    "sample_value": "system",
                    "quality_metrics": {
                        "completeness": 1.0,
                        "uniqueness": 1.0,
                        "validity": 1.0
                    }
                },
                {
                    "name": "_file_type",
                    "type": "string",
                    "description": "Current file format",
                    "sample_value": state.file_type,
                    "quality_metrics": {
                        "completeness": 1.0,
                        "uniqueness": 1.0,
                        "validity": 1.0
                    }
                }
            ])
            
            state.schema = schema
            return state

        except Exception as e:
            logger.error(f"Error generating schema for {state.file_name}: {str(e)}", exc_info=True)
            state.error = f"Error generating schema: {str(e)}"
            state.status = "failed"
            return state

    async def _create_glue_table(self, state: CatalogState) -> CatalogState:
        """Create a Glue table for the file."""
        try:
            # Generate table name
            table_name = f"raw_{state.file_name.split('.')[0]}_{uuid.uuid4().hex[:8]}"
            state.table_name = table_name
            logger.info(f"Creating Glue table {table_name} for {state.file_name}")

            # Convert schema to Glue format
            columns = []
            for col in state.schema["columns"]:
                columns.append({
                    "Name": col["name"],
                    "Type": col["type"],
                    "Comment": col.get("description", "")
                })

            # Create table in Glue
            self.glue_client.create_table(
                DatabaseName=self.database_name,
                TableInput={
                    "Name": table_name,
                    "TableType": "EXTERNAL_TABLE",
                    "Parameters": {
                        "classification": state.file_type,
                        "typeOfData": "file"
                    },
                    "StorageDescriptor": {
                        "Columns": columns,
                        "Location": f"s3://{self.bucket_name}/{state.s3_path}",
                        "InputFormat": "org.apache.hadoop.mapred.TextInputFormat",
                        "OutputFormat": "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
                        "SerdeInfo": {
                            "SerializationLibrary": "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe",
                            "Parameters": {
                                "field.delim": ","
                            }
                        }
                    }
                }
            )
            logger.info(f"Successfully created Glue table {table_name}")

            return state

        except Exception as e:
            logger.error(f"Error creating Glue table for {state.file_name}: {str(e)}", exc_info=True)
            self.error_handler.handle_error(e, {"state": state.dict()})
            state.error = f"Error creating Glue table: {str(e)}"
            return state

    async def _store_metadata(self, state: CatalogState, user_id: str) -> CatalogState:
        """Store file metadata in S3."""
        try:
            metadata = {
                "file_name": state.file_name,
                "file_type": state.file_type,
                "s3_path": state.s3_path,
                "table_name": state.table_name,
                "schema": state.schema,
                "audit_info": state.audit_info.dict(),
                "status": state.status
            }

            # Store metadata in S3
            metadata_path = f"metadata/{state.table_name}.json"
            logger.info(f"Storing metadata at {metadata_path}")
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=metadata_path,
                Body=json.dumps(metadata, indent=2)
            )
            logger.info(f"Successfully stored metadata for {state.file_name}")

            state.metadata = metadata
            return state

        except Exception as e:
            logger.error(f"Error storing metadata for {state.file_name}: {str(e)}", exc_info=True)
            self.error_handler.handle_error(e, {"state": state.dict()})
            state.error = f"Error storing metadata: {str(e)}"
            return state

    async def _generate_quality_metrics(self, state: CatalogState) -> CatalogState:
        """Generate data quality metrics for the file."""
        try:
            # Implementation of generating data quality metrics
            # This is a placeholder and should be replaced with the actual implementation
            state.audit_info['quality_metrics'] = {
                "completeness": 0.95,
                "uniqueness": 0.90,
                "validity": 0.98
            }
            return state
        except Exception as e:
            logger.error(f"Error generating quality metrics for {state.file_name}: {str(e)}", exc_info=True)
            self.error_handler.handle_error(e, {"state": state.dict()})
            state.error = f"Error generating quality metrics: {str(e)}"
            return state

    async def _handle_error(self, state: CatalogState, error: Exception) -> CatalogState:
        """Handle errors during processing."""
        logger.error(f"Handling error for {state.file_name}: {str(error)}", exc_info=True)
        self.error_handler.handle_error(error, {"state": state.dict()})
        state.error = str(error)
        state.status = "failed"
        return state 