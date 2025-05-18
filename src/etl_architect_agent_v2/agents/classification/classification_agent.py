"""Classification agent for data transformation and categorization."""

from typing import Dict, Any, Optional, List
import pandas as pd
import json
import logging
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, ConfigDict, field_validator
import uuid
from pathlib import Path
import os
from etl_architect_agent_v2.core.llm_manager import LLMManager
from etl_architect_agent_v2.core.error_handler import ErrorHandler, LLMError

logger = logging.getLogger(__name__)

class ClassificationState(BaseModel):
    """State management for classification process."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    classification_needed: bool = True
    user_instruction: str
    original_data_path: str
    classification_code: Optional[str] = None
    transformed_sample: Optional[pd.DataFrame] = None
    classification_explanation: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    progress: float = 0.0
    current_operation: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0
    retry_attempts: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

    @field_validator('transformed_sample', mode='before')
    @classmethod
    def validate_dataframe(cls, v):
        if isinstance(v, pd.DataFrame):
            return v
        return None

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        if isinstance(data.get('transformed_sample'), pd.DataFrame):
            data['transformed_sample'] = data['transformed_sample'].to_dict(orient='records')
        return data

class ClassificationAgent:
    """Agent for classifying and transforming data based on user instructions."""
    
    def __init__(self, llm_manager: Optional[LLMManager] = None):
        """Initialize the classification agent.
        
        Args:
            llm_manager: Optional LLM manager instance
        """
        self.max_retries = 3
        self.preview_rows = 10
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        self.glue_client = boto3.client(
            'glue',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        self.bucket_name = os.getenv('AWS_S3_BUCKET', 'your-bucket-name')
        self.database_name = os.getenv('GLUE_DATABASE_NAME', 'data_lakehouse')
        self.llm_manager = llm_manager or LLMManager()
        self.error_handler = ErrorHandler(max_retries=self.max_retries)

    async def process(self, state: ClassificationState) -> ClassificationState:
        """Process the classification request."""
        try:
            state.status = "in_progress"
            state.progress = 0.0
            state.current_operation = "Starting classification"

            # 1. Generate classification code
            state = await self._generate_classification_code(state)
            if state.error:
                return state

            state.progress = 30.0
            state.current_operation = "Executing classification"

            # 2. Execute classification
            state = await self._execute_classification(state)
            if state.error:
                return state

            state.progress = 60.0
            state.current_operation = "Storing results"

            # 3. Store results
            state = await self._store_results(state)
            if state.error:
                return state

            state.progress = 100.0
            state.status = "completed"
            state.current_operation = "Classification completed"

            return state

        except Exception as e:
            return await self._handle_error(state, e)

    async def _generate_classification_code(self, state: ClassificationState) -> ClassificationState:
        """Generate classification code using LLM."""
        try:
            # Prepare the prompt for code generation
            prompt = f"""
            Generate Python code to classify and transform data based on the following instruction:
            {state.user_instruction}

            The code should:
            1. Read a pandas DataFrame as input
            2. Apply the classification/transformation logic
            3. Return the transformed DataFrame
            4. Include clear comments explaining the logic

            The code should be a function named 'classify_data' that takes a pandas DataFrame as input.
            """

            # Get code generation from LLM
            response = await self.llm_manager.ainvoke({
                "messages": [("user", prompt)],
                "system_prompt": "You are a data engineering expert. Generate clean, efficient Python code for data classification and transformation."
            })

            if not response or "content" not in response:
                raise LLMError("Failed to generate classification code")

            # Extract and validate the generated code
            code = response["content"]
            if "def classify_data" not in code:
                raise LLMError("Generated code does not contain required function")

            state.classification_code = code
            state.classification_explanation = "Code generated successfully"
            return state

        except Exception as e:
            self.error_handler.handle_error(e, {"state": state.dict()})
            state.error = f"Error generating classification code: {str(e)}"
            return state

    async def _execute_classification(self, state: ClassificationState) -> ClassificationState:
        """Execute the classification code on the data."""
        try:
            # Read data from S3
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=state.original_data_path
            )
            df = pd.read_csv(response['Body'])

            # Create a safe execution environment
            local_vars = {"pd": pd, "df": df}
            
            # Execute the generated code
            exec(state.classification_code, globals(), local_vars)
            
            # Get the transformed data
            if "classify_data" not in local_vars:
                raise ValueError("Classification function not found in executed code")
            
            transformed_df = local_vars["classify_data"](df)
            
            if not isinstance(transformed_df, pd.DataFrame):
                raise ValueError("Classification function did not return a DataFrame")
            
            state.transformed_sample = transformed_df.head(self.preview_rows)
            return state

        except Exception as e:
            self.error_handler.handle_error(e, {"state": state.dict()})
            state.error = f"Error executing classification: {str(e)}"
            return state

    async def _store_results(self, state: ClassificationState) -> ClassificationState:
        """Store classification results in S3 and Glue."""
        try:
            # Generate unique ID for this classification
            classification_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Define S3 paths
            classified_path = f"classified/{classification_id}/classification_{timestamp}"
            data_path = f"{classified_path}/data.csv"
            metadata_path = f"{classified_path}/metadata.json"
            explanation_path = f"{classified_path}/explanation.txt"

            # Upload transformed data
            if state.transformed_sample is not None:
                csv_buffer = state.transformed_sample.to_csv(index=False)
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=data_path,
                    Body=csv_buffer
                )

            # Upload metadata
            metadata = {
                "classification_id": classification_id,
                "timestamp": timestamp,
                "original_data_path": state.original_data_path,
                "user_instruction": state.user_instruction,
                "explanation": state.classification_explanation,
                "status": state.status
            }
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=metadata_path,
                Body=json.dumps(metadata)
            )

            # Upload explanation
            if state.classification_explanation:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=explanation_path,
                    Body=state.classification_explanation
                )

            # Create Glue table
            self._create_glue_table(
                f"classification_{classification_id}",
                state.transformed_sample,
                f"s3://{self.bucket_name}/{data_path}"
            )

            state.metadata = metadata
            return state

        except Exception as e:
            self.error_handler.handle_error(e, {"state": state.dict()})
            state.error = f"Error storing results: {str(e)}"
            return state

    def _create_glue_table(self, table_name: str, df: pd.DataFrame, s3_path: str) -> None:
        """Create a Glue table for the classified data."""
        try:
            # Convert schema to Glue format
            columns = []
            for col in df.columns:
                col_type = str(df[col].dtype)
                if "int" in col_type:
                    glue_type = "bigint"
                elif "float" in col_type:
                    glue_type = "double"
                elif "datetime" in col_type:
                    glue_type = "timestamp"
                else:
                    glue_type = "string"
                
                columns.append({
                    "Name": col,
                    "Type": glue_type
                })

            # Create table in Glue
            self.glue_client.create_table(
                DatabaseName=self.database_name,
                TableInput={
                    "Name": table_name,
                    "StorageDescriptor": {
                        "Columns": columns,
                        "Location": s3_path,
                        "InputFormat": "org.apache.hadoop.mapred.TextInputFormat",
                        "OutputFormat": "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
                        "SerdeInfo": {
                            "SerializationLibrary": "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe",
                            "Parameters": {
                                "field.delim": ","
                            }
                        }
                    },
                    "TableType": "EXTERNAL_TABLE",
                    "Parameters": {
                        "classification": "csv",
                        "typeOfData": "file"
                    }
                }
            )
        except Exception as e:
            self.error_handler.handle_error(e, {
                "table_name": table_name,
                "s3_path": s3_path
            })
            raise

    async def _handle_error(self, state: ClassificationState, error: Exception) -> ClassificationState:
        """Handle errors during classification."""
        self.error_handler.handle_error(error, {"state": state.dict()})
        state.error = str(error)
        state.retry_count += 1
        state.retry_attempts.append({
            "attempt": state.retry_count,
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        })

        if state.retry_count < self.max_retries:
            state.status = "pending"
            state.progress = 0.0
            state.current_operation = f"Retrying after error (attempt {state.retry_count})"
        else:
            state.status = "failed"
            state.current_operation = "Classification failed after maximum retries"

        return state 