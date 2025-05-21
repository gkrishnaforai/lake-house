"""Service for managing data transformations."""

from typing import Dict, Any, List, Optional
import boto3
import pandas as pd
import json
import logging
from datetime import datetime
from io import BytesIO
import asyncio
from abc import ABC, abstractmethod
from ..models.transformation import TransformationTemplate
from langchain_openai import ChatOpenAI
import os
from etl_architect_agent_v2.backend.config.transformation_tools import (
    TRANSFORMATION_TOOLS
)
from ..utils.llm_manager import LLMManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformationHandler(ABC):
    """Base class for transformation handlers."""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
    
    @abstractmethod
    async def apply(
        self,
        data: pd.DataFrame,
        tool: Dict[str, Any],
        source_columns: List[str]
    ) -> Dict[str, Any]:
        """Apply the transformation."""
        pass
    
    def _generate_column_names(
        self,
        source_column: str,
        output_columns: Dict[str, Dict[str, str]]
    ) -> Dict[str, str]:
        """Generate column names based on templates."""
        return {
            col_type: config["name_template"].format(
                source_col=source_column
            )
            for col_type, config in output_columns.items()
        }
    
    def _format_prompt(
        self,
        prompt_template: str,
        text: str,
        **kwargs
    ) -> str:
        """Format the prompt template with the given text and parameters."""
        return prompt_template.format(
            text=text,
            **kwargs
        )


class CategorizationHandler(TransformationHandler):
    """Handler for categorization transformations."""
    
    async def apply(
        self,
        data: pd.DataFrame,
        tool: Dict[str, Any],
        source_columns: List[str]
    ) -> Dict[str, Any]:
        """Apply categorization transformation."""
        try:
            results = []
            for source_col in source_columns:
                # Generate column names
                column_names = self._generate_column_names(
                    source_col,
                    tool["output_columns"]
                )
                
                # Process each row
                for idx, row in data.iterrows():
                    text = row[source_col]
                    if pd.isna(text) or (
                        isinstance(text, str) and text.strip() == ''
                    ):
                        results.append({
                            "classification": tool["classification_options"][1],
                            "confidence": 0.0,
                            "reasoning": "No text provided for analysis"
                        })
                        continue
                    
                    # Format and send prompt
                    prompt = self._format_prompt(
                        tool["prompt_template"],
                        text,
                        classification_options=" or ".join(
                            tool["classification_options"]
                        ),
                        classification_factors="\n".join(
                            tool.get("classification_factors", [])
                        )
                    )
                    
                    # Get LLM response
                    response = await self.llm_manager.get_structured_response(
                        prompt,
                        required_fields=[
                            "classification",
                            "confidence",
                            "reasoning"
                        ]
                    )
                    
                    results.append(response)
            
            # Create preview data
            preview_data = []
            for idx, row in data.iterrows():
                preview_row = row.to_dict()
                for source_col in source_columns:
                    column_names = self._generate_column_names(
                        source_col,
                        tool["output_columns"]
                    )
                    # Convert DataFrame index to integer for list access
                    result_idx = int(idx) if isinstance(idx, (int, float)) else 0
                    result = results[result_idx]
                    preview_row.update({
                        column_names["classification"]: result["classification"],
                        column_names["confidence"]: result["confidence"],
                        column_names["reasoning"]: result["reasoning"]
                    })
                preview_data.append(preview_row)
            
            # Generate new column definitions
            new_columns = []
            for source_col in source_columns:
                column_names = self._generate_column_names(
                    source_col,
                    tool["output_columns"]
                )
                for col_type, col_name in column_names.items():
                    new_columns.append({
                        "name": col_name,
                        "type": tool["output_columns"][col_type]["type"]
                    })
            
            return {
                "status": "success",
                "new_columns": new_columns,
                "preview_data": preview_data
            }
            
        except Exception as e:
            error_msg = f"Error in categorization: {str(e)}\nFull error: {repr(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)


class TransformationService:
    """Service for managing data transformations."""

    def __init__(self, bucket: str, aws_region: str = 'us-east-1'):
        """Initialize the transformation service."""
        self.bucket = bucket
        self.aws_region = aws_region
        self.s3 = boto3.client('s3', region_name=aws_region)
        self.glue = boto3.client('glue', region_name=aws_region)
        self.athena = boto3.client('athena', region_name=aws_region)
        self.tools = TRANSFORMATION_TOOLS
        
        # Initialize LLM manager
        self.llm_manager = LLMManager(
            model_name="gpt-4",
            temperature=0
        )
        
        # Initialize transformation handlers
        self.handlers = {
            "categorization": CategorizationHandler(self.llm_manager)
        }

    async def apply_transformation(
        self,
        table_name: str,
        tool_id: str,
        source_columns: List[str],
        user_id: str = "test_user"
    ) -> Dict[str, Any]:
        """Apply a transformation to the data.
        
        Args:
            table_name: Name of the table to transform
            tool_id: ID of the transformation tool to apply
            source_columns: List of source columns to transform
            user_id: User ID for the table
            
        Returns:
            Dict containing transformation results
        """
        try:
            # Get the tool configuration
            tool = await self.get_transformation_tool(tool_id)
            if not tool:
                raise ValueError(f"Transformation tool with ID {tool_id} not found")
            
            logger.info(f"Applying transformation: {tool['name']} to table {table_name}")
            logger.info(f"Source columns: {source_columns}")
            
            # Read table data
            data = await self._read_table_data(table_name, user_id)
            if data.empty:
                raise ValueError(f"No data found in table {table_name}")
            
            # Get the appropriate handler
            handler = self.handlers.get(tool["type"])
            if not handler:
                raise ValueError(
                    f"Unsupported transformation type: {tool['type']}"
                )
            
            # Apply the transformation
            result = await handler.apply(data, tool, source_columns)
            
            # Update the DataFrame with the new columns
            for source_col in source_columns:
                column_names = handler._generate_column_names(
                    source_col,
                    tool["output_columns"]
                )
                for col_type, col_name in column_names.items():
                    # Get the transformed values for this column from all rows
                    transformed_values = [row[col_name] for row in result["preview_data"]]
                    data[col_name] = transformed_values
            
            # Write transformed data back to the table
            await self._write_transformed_data(
                data=data,
                table_name=table_name,
                user_id=user_id,
                new_columns=result.get("new_columns", [])
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying transformation: {str(e)}")
            raise ValueError(f"Error applying transformation: {str(e)}")

    async def get_available_transformations(self) -> List[Dict[str, Any]]:
        """Get list of available transformation types and their configurations."""
        # Convert the tools dictionary to a list of tool configurations
        return [
            {
                "id": tool_id,
                **tool_config
            }
            for tool_id, tool_config in self.tools.items()
        ]

    async def get_transformation_tool(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific transformation tool by ID."""
        return self.tools.get(tool_id)

    async def get_table_columns(self, table_name: str, user_id: str) -> List[Dict[str, str]]:
        """Get available columns from a table."""
        try:
            # Get table schema from Glue
            glue_client = boto3.client('glue', region_name=self.aws_region)
            response = glue_client.get_table(
                DatabaseName=f"user_{user_id}",
                Name=table_name
            )
            
            # Extract column information
            columns = []
            for col in response['Table']['StorageDescriptor']['Columns']:
                columns.append({
                    'name': col['Name'],
                    'type': col['Type'],
                    'description': col.get('Comment', '')
                })
            
            return columns
        except Exception as e:
            logger.error(f"Error getting table columns: {str(e)}")
            raise ValueError(f"Error getting table columns: {str(e)}")

    def _serialize_value(self, value: Any) -> Any:
        """Helper method to serialize values for JSON, handling special types like Timestamp."""
        if pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, (pd.Series, pd.DataFrame)):
            return value.to_dict('records')
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        return value

    async def get_transformation_templates(
        self,
        user_id: str
    ) -> List[TransformationTemplate]:
        """Get list of transformation templates for a user."""
        try:
            response = self.s3.get_object(
                Bucket=self.bucket,
                Key=f"metadata/templates/{user_id}/templates.json"
            )
            templates = json.loads(response['Body'].read())
            return [TransformationTemplate(**template) for template in templates]
        except self.s3.exceptions.NoSuchKey:
            return []

    async def save_transformation_template(
        self,
        template: TransformationTemplate,
        user_id: str
    ) -> Dict[str, Any]:
        """Save a transformation template."""
        try:
            # Get existing templates
            templates = await self.get_transformation_templates(user_id)
            
            # Add new template
            templates.append(template)
            
            # Save updated templates
            self.s3.put_object(
                Bucket=self.bucket,
                Key=f"metadata/templates/{user_id}/templates.json",
                Body=json.dumps([t.dict() for t in templates])
            )
            
            return {
                "status": "success",
                "message": "Template saved successfully"
            }
        except Exception as e:
            logger.error(f"Error saving template: {str(e)}")
            raise Exception(f"Error saving template: {str(e)}")

    async def delete_transformation_template(
        self,
        template_name: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Delete a transformation template."""
        try:
            # Get existing templates
            templates = await self.get_transformation_templates(user_id)
            
            # Remove template
            templates = [t for t in templates if t.name != template_name]
            
            # Save updated templates
            self.s3.put_object(
                Bucket=self.bucket,
                Key=f"metadata/templates/{user_id}/templates.json",
                Body=json.dumps([t.dict() for t in templates])
            )
            
            return {
                "status": "success",
                "message": "Template deleted successfully"
            }
        except Exception as e:
            logger.error(f"Error deleting template: {str(e)}")
            raise Exception(f"Error deleting template: {str(e)}")

    async def _get_table_info(
        self,
        table_name: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get table information from Glue."""
        try:
            response = self.glue.get_table(
                DatabaseName=f"user_{user_id}",
                Name=table_name
            )
            return response['Table']
        except Exception as e:
            logger.error(f"Error getting table info: {str(e)}")
            return None

    async def _read_table_data(self, table_name: str, user_id: str) -> pd.DataFrame:
        """Read data from a table using Athena."""
        try:
            # First try to read directly from S3
            try:
                # Get table info to get the correct S3 path
                table_info = await self._get_table_info(table_name, user_id)
                if not table_info:
                    raise ValueError(f"Table '{table_name}' not found")
                
                # Get the S3 location from table info
                s3_location = table_info.get('StorageDescriptor', {}).get('Location')
                if not s3_location:
                    raise ValueError(f"No S3 location found for table '{table_name}'")
                
                logger.info(f"Found S3 location for table: {s3_location}")
                
                # Check if the file exists and has content
                try:
                    # Extract the folder path from the full S3 URI
                    folder_path = s3_location.replace(f"s3://{self.bucket}/", "")
                    
                    # List files in the folder
                    response = self.s3.list_objects_v2(
                        Bucket=self.bucket,
                        Prefix=folder_path
                    )
                    
                    # Get the first file in the folder
                    if 'Contents' in response and len(response['Contents']) > 0:
                        s3_key = response['Contents'][0]['Key']
                        logger.info(f"Found file in folder: {s3_key}")
                    else:
                        raise ValueError(f"No files found in folder: {folder_path}")
                    
                    # URL encode the key to handle spaces and special characters
                    s3_key = s3_key.replace(" ", "%20")
                    logger.info(f"Checking S3 file: {s3_key}")
                    
                    response = self.s3.head_object(Bucket=self.bucket, Key=s3_key)
                    file_size = response.get('ContentLength', 0)
                    logger.info(f"File size: {file_size} bytes")
                    
                    if file_size == 0:
                        raise ValueError(
                            f"The parquet file for table '{table_name}' is empty (0 bytes). "
                            "Please ensure the file contains data before proceeding with transformations."
                        )
                except self.s3.exceptions.ClientError as e:
                    if e.response['Error']['Code'] == '404':
                        raise ValueError(
                            f"The parquet file for table '{table_name}' does not exist in S3. "
                            "Please ensure the file has been properly uploaded."
                        )
                    raise
                
                # Read the Parquet file
                try:
                    response = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
                    df = pd.read_parquet(BytesIO(response['Body'].read()))
                    
                    # Filter out completely empty rows
                    df = df.dropna(how='all')
                    
                    # Filter out rows where all string columns are empty strings
                    string_columns = df.select_dtypes(include=['object']).columns
                    if len(string_columns) > 0:
                        mask = ~(df[string_columns].apply(lambda x: x.str.strip() == '').all(axis=1))
                        df = df[mask]
                    
                    logger.info(f"Successfully read parquet file with {len(df)} rows after filtering empty rows")
                    
                    if len(df) == 0:
                        raise ValueError(
                            f"No valid data rows found in table '{table_name}'. "
                            "The file exists but contains only empty rows. "
                            "Please ensure the file contains valid data before proceeding with transformations."
                        )
                    return df
                except Exception as e:
                    logger.error(f"Error reading parquet file: {str(e)}")
                    raise ValueError(f"Error reading parquet file: {str(e)}")
            except Exception as e:
                logger.warning(f"Failed to read directly from S3: {str(e)}. Falling back to Athena query.")
            
            # Fall back to Athena query if direct S3 read fails
            query = f"SELECT * FROM {table_name} LIMIT 1000"
            
            query_execution = self.athena.start_query_execution(
                QueryString=query,
                QueryExecutionContext={
                    'Database': f"user_{user_id}"
                },
                ResultConfiguration={
                    'OutputLocation': f"s3://lambda-code-q/athena-results/"
                }
            )
            
            query_execution_id = query_execution['QueryExecutionId']
            logger.info(f"Started query execution with ID: {query_execution_id}")
            
            # Wait for query to complete
            while True:
                query_status = self.athena.get_query_execution(
                    QueryExecutionId=query_execution_id
                )['QueryExecution']['Status']['State']
                
                if query_status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                    break
                    
                await asyncio.sleep(1)
            
            if query_status != 'SUCCEEDED':
                error_reason = self.athena.get_query_execution(
                    QueryExecutionId=query_execution_id
                )['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
                raise ValueError(
                    f"Athena query failed for table '{table_name}': {error_reason}. "
                    "Please check if the table exists and contains data."
                )
            
            # Get query results
            results = self.athena.get_query_results(
                QueryExecutionId=query_execution_id
            )
            
            rows = results['ResultSet']['Rows']
            if len(rows) <= 1:  # Only header row or no data
                raise ValueError(
                    f"No data rows found in table '{table_name}'. "
                    "The table exists but contains no data. "
                    "Please ensure the table has been properly populated with data."
                )
            
            columns = [col['Label'] for col in results['ResultSet']['ResultSetMetadata']['ColumnInfo']]
            data = []
            
            for row in rows[1:]:  # Skip header row
                row_data = [field.get('VarCharValue', '') for field in row['Data']]
                # Only add row if it has at least one non-empty value
                if any(value.strip() for value in row_data):
                    data.append(row_data)
            
            if not data:
                raise ValueError(
                    f"No valid data rows found in table '{table_name}'. "
                    "The table exists but contains only empty rows. "
                    "Please ensure the table contains valid data before proceeding with transformations."
                )
            
            df = pd.DataFrame(data, columns=columns)
            logger.info(f"Successfully read {len(df)} rows from Athena query after filtering empty rows")
            return df
            
        except Exception as e:
            logger.error(f"Error reading table data: {str(e)}")
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Error reading table data: {str(e)}")

    async def _write_transformed_data(
        self,
        data: pd.DataFrame,
        table_name: str,
        user_id: str,
        new_columns: List[Dict[str, Any]]
    ) -> None:
        """Write transformed data back to S3 and update Glue table."""
        try:
            logger.info(f"Writing transformed data for table {table_name}")
            
            # Get current table info
            table_info = await self._get_table_info(table_name, user_id)
            if not table_info:
                raise ValueError(f"Table {table_name} not found")
            
            # Generate temporary S3 key with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_s3_key = table_info['StorageDescriptor']['Location'].replace(f"s3://{self.bucket}/", "")
            temp_s3_key = f"temp/{user_id}/{table_name}_{timestamp}.parquet"
            
            # Log the updated values in new columns
            logger.info("Updated values in new columns:")
            for col in new_columns:
                col_name = col['name']
                if col_name in data.columns:
                    sample_values = data[col_name].head(5).tolist()
                    logger.info(f"Column '{col_name}': {json.dumps(sample_values, indent=2)}")
            
            # Convert to Parquet buffer
            parquet_buffer = BytesIO()
            data.to_parquet(parquet_buffer, index=False)
            parquet_buffer.seek(0)
            
            # Upload to temporary S3 location
            self.s3.put_object(
                Bucket=self.bucket,
                Key=temp_s3_key,
                Body=parquet_buffer.getvalue()
            )
            logger.info(f"Uploaded transformed data to temporary location s3://{self.bucket}/{temp_s3_key}")
            
            # Verify the Parquet file was written correctly
            try:
                # Read back the Parquet file to verify
                response = self.s3.get_object(Bucket=self.bucket, Key=temp_s3_key)
                verify_df = pd.read_parquet(BytesIO(response['Body'].read()))
                
                if len(verify_df.columns) != len(data.columns):
                    raise ValueError(f"Column count mismatch: expected {len(data.columns)}, got {len(verify_df.columns)}")
                
                logger.info("Successfully verified temporary Parquet file")
            except Exception as e:
                logger.error(f"Error verifying temporary Parquet file: {str(e)}")
                # Clean up temporary file
                self.s3.delete_object(Bucket=self.bucket, Key=temp_s3_key)
                raise ValueError(f"Failed to verify temporary Parquet file: {str(e)}")
            
            # Get current columns and add new ones
            current_columns = table_info['StorageDescriptor']['Columns']
            formatted_columns = []
            unique_column_names = set()  # Track unique column names
            
            # Add existing columns
            for col in current_columns:
                if col['Name'] not in unique_column_names:
                    formatted_columns.append({
                        'Name': col['Name'],
                        'Type': col['Type'],
                        'Comment': col.get('Comment', '')
                    })
                    unique_column_names.add(col['Name'])
            
            # Add new columns
            for col in new_columns:
                if col['name'] not in unique_column_names:
                    formatted_columns.append({
                        'Name': col['name'],
                        'Type': col['type'],
                        'Comment': col.get('comment', '')
                    })
                    unique_column_names.add(col['name'])
            
            logger.info(f"Updating Glue table with {len(formatted_columns)} columns")
            logger.info(f"Formatted columns: {json.dumps(formatted_columns, indent=2)}")
            
            # Get the folder path from the base S3 key
            folder_path = os.path.dirname(base_s3_key)
            if not folder_path.endswith('/'):
                folder_path += '/'
            
            # Update the Glue table with new schema and folder location
            self.glue.update_table(
                DatabaseName=self.get_user_database_name(user_id),
                TableInput={
                    'Name': table_name,
                    'StorageDescriptor': {
                        'Columns': formatted_columns,
                        'Location': f"s3://{self.bucket}/{folder_path}",
                        'InputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat',
                        'OutputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat',
                        'SerdeInfo': {
                            'SerializationLibrary': 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'
                        }
                    }
                }
            )
            logger.info(f"Updated Glue table {table_name} with new schema")
            
            # Move the temporary file to the final location
            final_s3_key = f"{folder_path}{table_name}_{timestamp}.parquet"
            self.s3.copy_object(
                Bucket=self.bucket,
                CopySource={'Bucket': self.bucket, 'Key': temp_s3_key},
                Key=final_s3_key
            )
            
            # Clean up temporary file
            self.s3.delete_object(Bucket=self.bucket, Key=temp_s3_key)
            
            # Delete the original Parquet file
            try:
                # List files in the folder to find the original file
                response = self.s3.list_objects_v2(
                    Bucket=self.bucket,
                    Prefix=folder_path
                )
                
                # Delete all Parquet files except the new one
                for obj in response.get('Contents', []):
                    key = obj['Key']
                    if key.endswith('.parquet') and key != final_s3_key:
                        self.s3.delete_object(Bucket=self.bucket, Key=key)
                        logger.info(f"Deleted original Parquet file: {key}")
            except Exception as e:
                logger.warning(f"Error deleting original Parquet file: {str(e)}")
            
            logger.info(f"Successfully moved transformed data to final location: s3://{self.bucket}/{final_s3_key}")
            
        except Exception as e:
            logger.error(f"Error writing transformed data: {str(e)}")
            # Clean up temporary file if it exists
            try:
                self.s3.delete_object(Bucket=self.bucket, Key=temp_s3_key)
            except:
                pass
            raise ValueError(f"Error writing transformed data: {str(e)}")

    def _pandas_to_glue_type(self, dtype: str) -> str:
        """Convert pandas dtype to Glue data type."""
        dtype = str(dtype).lower()
        if 'int' in dtype:
            return 'bigint'
        elif 'float' in dtype:
            return 'double'
        elif 'bool' in dtype:
            return 'boolean'
        elif 'datetime' in dtype:
            return 'timestamp'
        else:
            return 'string'

    async def _update_table_schema(
        self,
        table_name: str,
        new_columns: List[Dict[str, str]],
        user_id: str
    ) -> None:
        """Update table schema in Glue."""
        try:
            # Get current table info
            table = await self._get_table_info(table_name, user_id)
            if not table:
                raise Exception(f"Table {table_name} not found")
            
            # Format new columns with proper capitalization
            formatted_new_columns = [
                {
                    'Name': col['name'],
                    'Type': col['type'],
                    'Comment': col.get('comment', '')
                }
                for col in new_columns
            ]
            
            # Get current columns and create a set of existing column names
            current_columns = table['StorageDescriptor']['Columns']
            existing_column_names = {col['Name'] for col in current_columns}
            
            # Only add columns that don't already exist
            for new_col in formatted_new_columns:
                if new_col['Name'] not in existing_column_names:
                    current_columns.append(new_col)
                    existing_column_names.add(new_col['Name'])
            
            # Update table
            self.glue.update_table(
                DatabaseName=f"user_{user_id}",
                TableInput={
                    'Name': table_name,
                    'StorageDescriptor': {
                        'Columns': current_columns,
                        'Location': table['StorageDescriptor']['Location'],
                        'InputFormat': table['StorageDescriptor']['InputFormat'],
                        'OutputFormat': table['StorageDescriptor']['OutputFormat'],
                        'SerdeInfo': table['StorageDescriptor']['SerdeInfo']
                    }
                }
            )
        except Exception as e:
            logger.error(f"Error updating table schema: {str(e)}")
            raise Exception(f"Error updating table schema: {str(e)}")

    async def _apply_sentiment_analysis(
        self,
        data: pd.DataFrame,
        tool: Dict[str, Any],
        source_columns: List[str]
    ) -> Dict[str, Any]:
        """Apply sentiment analysis transformation using the tool's prompt template."""
        try:
            logger.info("Starting sentiment analysis transformation")
            logger.info(f"Tool configuration: {json.dumps(tool, indent=2)}")
            
            # Initialize LLM
            llm = ChatOpenAI(
                model_name="gpt-4",
                temperature=0
            )
            
            # Create new column names
            new_columns = []
            for aspect in tool["default_config"]["parameters"]["aspects"]:
                column_name = f"{source_columns[0]}_{aspect.lower().replace(' ', '_')}_sentiment"
                new_columns.append({"name": column_name, "type": "double"})

            # Apply sentiment analysis using LLM
            results = []
            # Only get the relevant column data
            text_data = data[source_columns[0]].fillna('')
            
            for idx, text in enumerate(text_data):
                try:
                    if pd.isna(text) or text == '':
                        logger.info(f"Row {idx} has null text, using default values")
                        results.append({aspect: 0.0 for aspect in tool["default_config"]["parameters"]["aspects"]})
                        continue
                        
                    # Format the prompt to request JSON response
                    prompt = f"""Analyze the sentiment of the following text across different aspects.
                    Consider these aspects: {', '.join(tool['default_config']['parameters']['aspects'])}
                    
                    Text: {text}
                    
                    Return your analysis in the following JSON format:
                    {{
                        "overall_sentiment": "positive/negative/neutral",
                        "sentiment_score": <float between 0 and 1>,
                        "aspect_sentiments": {{
                            {', '.join([f'"{aspect}": {{"sentiment": "positive/negative/neutral", "score": <float between 0 and 1>}}' for aspect in tool['default_config']['parameters']['aspects']])}
                        }}
                    }}
                    
                    Return ONLY the JSON object, no other text or explanation."""
                    
                    logger.info(f"Sending prompt to LLM for row {idx}")
                    logger.info(f"Prompt: {prompt}")

                    try:
                        response = await llm.ainvoke(prompt)
                        logger.info(f"Raw LLM response for row {idx}: {response}")
                        logger.info(f"Response type: {type(response)}")
                        logger.info(f"Response content type: {type(response.content)}")
                        logger.info(f"Response content: {response.content}")
                        
                        # Clean the response content to ensure it's valid JSON
                        content = response.content
                        
                        # Remove any markdown code block markers
                        content = content.replace('```json', '').replace('```', '').strip()
                        logger.info(f"Cleaned response content: {content}")
                        
                        # Try to parse the response as JSON
                        result = json.loads(content)
                        logger.info(f"Parsed JSON result: {json.dumps(result, indent=2)}")
                        
                        # Extract aspect scores from the response
                        aspect_scores = {}
                        for aspect in tool["default_config"]["parameters"]["aspects"]:
                            aspect_data = result.get("aspect_sentiments", {}).get(aspect, {})
                            aspect_scores[aspect] = aspect_data.get("score", 0.0)
                        
                        results.append(aspect_scores)
                        logger.info(f"Successfully processed row {idx}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse LLM response as JSON for row {idx}")
                        logger.error(f"Content that failed to parse: {content}")
                        logger.error(f"JSON decode error: {str(e)}")
                        # Fallback: create a default result
                        results.append({aspect: 0.0 for aspect in tool["default_config"]["parameters"]["aspects"]})
                    except Exception as e:
                        logger.error(f"Error processing LLM response for row {idx}: {str(e)}")
                        logger.error(f"Response object: {response}")
                        results.append({aspect: 0.0 for aspect in tool["default_config"]["parameters"]["aspects"]})
                except Exception as e:
                    logger.error(f"Error processing row {idx}: {str(e)}")
                    results.append({aspect: 0.0 for aspect in tool["default_config"]["parameters"]["aspects"]})

            logger.info(f"Processed all rows. Total results: {len(results)}")
            
            # Add results to DataFrame
            for i, aspect in enumerate(tool["default_config"]["parameters"]["aspects"]):
                column_name = f"{source_columns[0]}_{aspect.lower().replace(' ', '_')}_sentiment"
                data[column_name] = [r.get(aspect, 0.0) for r in results]

            return {
                "status": "success",
                "data": data,
                "new_columns": new_columns
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            logger.error(f"Full error details: {str(e.__class__.__name__)}: {str(e)}")
            raise ValueError(f"Error applying sentiment analysis: {str(e)}")

    async def _apply_categorization(
        self,
        data: pd.DataFrame,
        tool: Dict[str, Any],
        source_columns: List[str]
    ) -> Dict[str, Any]:
        """Apply categorization transformation using the tool's prompt template."""
        try:
            logger.info("Starting categorization transformation")
            logger.info(f"Tool configuration: {json.dumps(tool, indent=2)}")
            
            # Initialize LLM
            llm = ChatOpenAI(
                model_name="gpt-4",
                temperature=0
            )
            
            # Create new column names
            new_columns = [
                {"name": f"{source_columns[0]}_company_ai_classification", "type": "string"},
                {"name": f"{source_columns[0]}_company_ai_classification_confidence", "type": "double"},
                {"name": f"{source_columns[0]}_company_ai_classification_reasoning", "type": "string"}
            ]
            
            # Apply categorization using LLM
            results = []
            # Only get the relevant column data
            text_data = data[source_columns[0]].fillna('')
            
            for idx, text in enumerate(text_data):
                try:
                    # Check if text is empty or null using pandas methods
                    if pd.isna(text) or (isinstance(text, str) and text.strip() == ''):
                        logger.info(f"Row {idx} has null text, using default values")
                        results.append({
                            "classification": "Non-AI",
                            "confidence": 0.0,
                            "reasoning": "No text provided for analysis"
                        })
                        continue

                    # Format the prompt to request JSON response
                    prompt = f"""You are an AI company classifier. Analyze the following company description and classify it as an AI company or not.

                    IMPORTANT: Analyze ONLY the text provided below. Each analysis should be unique based on the specific company description.

                    Consider these factors for the specific company:
                    1. Use of AI/ML technologies
                    2. Focus on data science or machine learning
                    3. AI-related products or services
                    4. Mention of artificial intelligence, machine learning, or related terms

                    Company Description to Analyze:
                    {text}

                    Return your analysis in the following JSON format:
                    {{
                        "classification": "AI" or "Non-AI",
                        "confidence": <float between 0 and 1>,
                        "reasoning": "<brief explanation specific to this company's description>"
                    }}

                    IMPORTANT: 
                    - Your reasoning must be specific to the company description provided
                    - Do not reuse or copy reasoning from previous analyses
                    - Return ONLY the JSON object, no other text or explanation"""
                    
                    logger.info(f"Sending prompt to LLM for row {idx}")
                    logger.info(f"Prompt: {prompt}")

                    try:
                        response = await llm.ainvoke(prompt)
                        logger.info(f"Raw LLM response for row {idx}: {response}")
                        logger.info(f"Response type: {type(response)}")
                        logger.info(f"Response content type: {type(response.content)}")
                        logger.info(f"Response content: {response.content}")

                        # Clean the response content to ensure it's valid JSON
                        content = response.content
                        
                        # Remove any markdown code block markers
                        content = content.replace('```json', '').replace('```', '').strip()
                        logger.info(f"Cleaned response content: {content}")

                        # Try to parse the response as JSON
                        result = json.loads(content)
                        logger.info(f"Parsed JSON result: {json.dumps(result, indent=2)}")

                        # Validate the result structure
                        if not all(k in result for k in ["classification", "confidence", "reasoning"]):
                            raise ValueError("Missing required fields in response")

                        results.append(result)
                        logger.info(f"Successfully processed row {idx}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse LLM response as JSON for row {idx}")
                        logger.error(f"Content that failed to parse: {content}")
                        logger.error(f"JSON decode error: {str(e)}")
                        # Fallback: create a default result
                        results.append({
                            "classification": "Non-AI",
                            "confidence": 0.0,
                            "reasoning": "Failed to parse LLM response"
                        })
                    except Exception as e:
                        logger.error(f"Error processing LLM response for row {idx}: {str(e)}")
                        logger.error(f"Response object: {response}")
                        results.append({
                            "classification": "Non-AI",
                            "confidence": 0.0,
                            "reasoning": f"Error: {str(e)}"
                        })
                except Exception as e:
                    logger.error(f"Error processing row {idx}: {str(e)}")
                    results.append({
                        "classification": "Non-AI",
                        "confidence": 0.0,
                        "reasoning": f"Error: {str(e)}"
                    })

            logger.info(f"Processed all rows. Total results: {len(results)}")
            
            # Create a copy of the original DataFrame to avoid modifying it directly
            result_df = data.copy()
            
            # Add results to DataFrame
            result_df[f"{source_columns[0]}_company_ai_classification"] = [r["classification"] for r in results]
            result_df[f"{source_columns[0]}_company_ai_classification_confidence"] = [r["confidence"] for r in results]
            result_df[f"{source_columns[0]}_company_ai_classification_reasoning"] = [r["reasoning"] for r in results]

            # Create preview data - use up to 5 rows or all rows if less than 5
            preview_rows = min(5, len(result_df))
            preview_data = []
            for _, row in result_df.head(preview_rows).iterrows():
                preview_row = {}
                for col in new_columns:
                    col_name = col["name"]
                    preview_row[col_name] = row[col_name]
                preview_data.append(preview_row)

            return {
                "status": "success",
                "data": result_df,
                "new_columns": new_columns,
                "preview_data": preview_data
            }
            
        except Exception as e:
            logger.error(f"Error in categorization: {str(e)}")
            logger.error(f"Full error details: {str(e.__class__.__name__)}: {str(e)}")
            raise ValueError(f"Error applying categorization: {str(e)}")

    async def list_s3_files(
        self,
        prefix: str,
        user_id: str = "test_user"
    ) -> List[Dict[str, Any]]:
        """List files in an S3 folder and get their metadata.
        
        Args:
            prefix: S3 prefix (folder path) to list files from
            user_id: User ID for filtering files
            
        Returns:
            List of dictionaries containing file information
        """
        try:
            # Ensure prefix ends with a slash if not empty
            if prefix and not prefix.endswith('/'):
                prefix = f"{prefix}/"
            
            # Add user_id to prefix if not already included
            if not prefix.startswith(f"{user_id}/"):
                prefix = f"{user_id}/{prefix}"
            
            logger.info(f"Listing files in S3 bucket '{self.bucket}' with prefix '{prefix}'")
            
            # Use paginator to handle large number of files
            paginator = self.s3.get_paginator('list_objects_v2')
            files = []
            
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    key = obj.get('Key')
                    if key and not key.endswith('/'):  # Skip directories
                        try:
                            # Get additional metadata using head_object
                            head_response = self.s3.head_object(
                                Bucket=self.bucket,
                                Key=key
                            )
                            
                            file_info = {
                                'file_name': os.path.basename(key),
                                's3_path': f"s3://{self.bucket}/{key}",
                                'size': obj['Size'],
                                'last_modified': obj['LastModified'].isoformat(),
                                'content_type': head_response.get('ContentType', 'application/octet-stream'),
                                'e_tag': head_response.get('ETag'),
                                'metadata': head_response.get('Metadata', {}),
                                'user_id': user_id
                            }
                            files.append(file_info)
                        except Exception as e:
                            logger.error(f"Error getting metadata for file {key}: {str(e)}")
                            continue
            
            logger.info(f"Found {len(files)} files in prefix '{prefix}'")
            return files
            
        except Exception as e:
            logger.error(f"Error listing files in S3: {str(e)}")
            raise ValueError(f"Error listing files in S3: {str(e)}")

    async def get_file_info(
        self,
        s3_path: str,
        user_id: str = "test_user"
    ) -> Dict[str, Any]:
        """Get detailed information about a specific file in S3.
        
        Args:
            s3_path: Full S3 path to the file (s3://bucket/key)
            user_id: User ID for validation
            
        Returns:
            Dictionary containing file information
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
            
            # Validate bucket
            if bucket != self.bucket:
                raise ValueError(f"Invalid bucket: {bucket}. Expected: {self.bucket}")
            
            # Validate user_id in path
            if not key.startswith(f"{user_id}/"):
                raise ValueError(f"File does not belong to user {user_id}")
            
            # Get file metadata
            response = self.s3.head_object(Bucket=bucket, Key=key)
            
            return {
                'file_name': os.path.basename(key),
                's3_path': s3_path,
                'size': response['ContentLength'],
                'last_modified': response['LastModified'].isoformat(),
                'content_type': response.get('ContentType', 'application/octet-stream'),
                'e_tag': response.get('ETag'),
                'metadata': response.get('Metadata', {}),
                'user_id': user_id
            }
            
        except self.s3.exceptions.NoSuchKey:
            raise ValueError(f"File not found: {s3_path}")
        except Exception as e:
            logger.error(f"Error getting file info: {str(e)}")
            raise ValueError(f"Error getting file info: {str(e)}")

    def get_user_database_name(self, user_id: str) -> str:
        """Get the database name for a specific user."""
        return f"user_{user_id}" 