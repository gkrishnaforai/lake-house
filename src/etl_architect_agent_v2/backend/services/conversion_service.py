from typing import Dict, Any, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import boto3
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import avro.schema
from avro.datafile import DataFileWriter
from avro.io import DatumWriter
import os
import tempfile
from datetime import datetime

class ConversionService:
    def __init__(self, aws_region: str = "us-east-1"):
        self.s3_client = boto3.client('s3', region_name=aws_region)
        self.llm = ChatOpenAI(temperature=0)
        self._setup_agent()

    def _setup_agent(self):
        # Define tools for the agent
        tools = [
            Tool(
                name="convert_to_avro",
                func=self._convert_to_avro,
                description="Convert a file to Avro format"
            ),
            Tool(
                name="update_catalog",
                func=self._update_catalog,
                description="Update the data catalog with new file information"
            ),
            Tool(
                name="validate_schema",
                func=self._validate_schema,
                description="Validate the schema of the converted file"
            )
        ]

        # Create the agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data conversion expert. Your task is to:
            1. Convert files to Avro format
            2. Validate the schema
            3. Update the catalog
            Use the provided tools to accomplish these tasks."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.agent = create_openai_functions_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=tools, verbose=True)

    async def convert_file(self, file_path: str, target_format: str = "avro") -> Dict[str, Any]:
        """
        Convert a file to the target format using LangChain agent
        """
        try:
            # Download file from S3
            bucket, key = self._parse_s3_path(file_path)
            temp_file = self._download_from_s3(bucket, key)

            # Get file extension
            file_ext = os.path.splitext(temp_file)[1].lower()

            # Prepare conversion task
            conversion_task = f"""
            Convert the file {temp_file} to {target_format} format.
            The file is currently in {file_ext} format.
            After conversion, validate the schema and update the catalog.
            """

            # Execute conversion using agent
            result = await self.agent_executor.ainvoke({
                "input": conversion_task,
                "chat_history": []
            })

            # Clean up temporary file
            os.remove(temp_file)

            return {
                "status": "success",
                "message": "File converted successfully",
                "details": result
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def _convert_to_avro(self, file_path: str) -> Dict[str, Any]:
        """
        Convert a file to Avro format
        """
        try:
            # Read the file based on its extension
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            # Generate Avro schema
            schema = self._generate_avro_schema(df)

            # Create temporary file for Avro output
            with tempfile.NamedTemporaryFile(suffix='.avro', delete=False) as temp_avro:
                # Write data to Avro file
                with DataFileWriter(open(temp_avro.name, 'wb'), DatumWriter(), schema) as writer:
                    for _, row in df.iterrows():
                        writer.append(row.to_dict())

                # Upload to S3
                bucket, key = self._parse_s3_path(file_path)
                new_key = f"processed/{os.path.basename(file_path)}.avro"
                self.s3_client.upload_file(temp_avro.name, bucket, new_key)

                # Clean up
                os.remove(temp_avro.name)

            return {
                "status": "success",
                "s3_path": f"s3://{bucket}/{new_key}",
                "schema": schema.to_json()
            }

        except Exception as e:
            raise Exception(f"Error converting to Avro: {str(e)}")

    def _generate_avro_schema(self, df: pd.DataFrame) -> avro.schema.Schema:
        """
        Generate Avro schema from pandas DataFrame
        """
        fields = []
        for column in df.columns:
            field_type = self._get_avro_type(df[column].dtype)
            fields.append({
                "name": column,
                "type": field_type
            })

        schema_dict = {
            "type": "record",
            "name": "DataRecord",
            "fields": fields
        }

        return avro.schema.parse(json.dumps(schema_dict))

    def _get_avro_type(self, pandas_type) -> str:
        """
        Map pandas data types to Avro types
        """
        type_mapping = {
            'int64': 'long',
            'int32': 'int',
            'float64': 'double',
            'float32': 'float',
            'bool': 'boolean',
            'datetime64[ns]': 'string',
            'object': 'string'
        }
        return type_mapping.get(str(pandas_type), 'string')

    def _update_catalog(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the data catalog with new file information
        """
        try:
            # Create catalog entry
            catalog_entry = {
                "file_name": os.path.basename(file_info["s3_path"]),
                "s3_path": file_info["s3_path"],
                "format": "avro",
                "schema": file_info["schema"],
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "status": "active"
            }

            # Store in S3
            bucket, key = self._parse_s3_path(file_info["s3_path"])
            catalog_key = f"metadata/catalog/{os.path.basename(file_info['s3_path'])}.json"
            
            self.s3_client.put_object(
                Bucket=bucket,
                Key=catalog_key,
                Body=json.dumps(catalog_entry)
            )

            return {
                "status": "success",
                "catalog_path": f"s3://{bucket}/{catalog_key}"
            }

        except Exception as e:
            raise Exception(f"Error updating catalog: {str(e)}")

    def _validate_schema(self, schema: str) -> Dict[str, Any]:
        """
        Validate the Avro schema
        """
        try:
            # Parse and validate schema
            avro.schema.parse(schema)
            return {
                "status": "success",
                "message": "Schema is valid"
            }
        except Exception as e:
            raise Exception(f"Invalid schema: {str(e)}")

    def _parse_s3_path(self, s3_path: str) -> tuple:
        """
        Parse S3 path into bucket and key
        """
        if not s3_path.startswith('s3://'):
            raise ValueError("Invalid S3 path")
        
        path = s3_path[5:]  # Remove 's3://'
        bucket = path.split('/')[0]
        key = '/'.join(path.split('/')[1:])
        return bucket, key

    def _download_from_s3(self, bucket: str, key: str) -> str:
        """
        Download file from S3 to temporary location
        """
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            self.s3_client.download_file(bucket, key, temp_file.name)
            return temp_file.name 