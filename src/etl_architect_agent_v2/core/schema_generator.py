"""Schema generator using LangChain components."""

from typing import Dict, Any, List, Optional, Set
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import CallbackManager
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field, validator
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.evaluation import load_evaluator
import logging
import asyncio
from functools import wraps
import json
import tiktoken
from datetime import datetime
import pytz
import hashlib
from enum import Enum

logger = logging.getLogger(__name__)


def async_retry(max_retries=3, delay=1):
    """Simple retry decorator for async functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {delay} seconds..."
                    )
                    await asyncio.sleep(delay)
            return None
        return wrapper
    return decorator


class ColumnChangeType(str, Enum):
    """Types of column changes in schema evolution."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


class SchemaColumn(BaseModel):
    """Schema column definition."""
    name: str = Field(..., description="Column name")
    type: str = Field(..., description="Data type")
    description: str = Field(default="", description="Column description")
    sample_value: str = Field(default="", description="Sample value")
    is_required: bool = Field(default=True, description="Is column required")
    is_unique: bool = Field(default=False, description="Is column unique")
    validation_rules: List[str] = Field(
        default_factory=list,
        description="List of validation rules"
    )
    quality_metrics: Dict[str, float] = Field(
        default_factory=lambda: {
            "completeness": 1.0,
            "uniqueness": 1.0,
            "validity": 1.0
        },
        description="Quality metrics"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(pytz.UTC).isoformat(),
        description="Column creation timestamp"
    )
    modified_at: str = Field(
        default_factory=lambda: datetime.now(pytz.UTC).isoformat(),
        description="Column last modified timestamp"
    )
    version: str = Field(
        default="1.0.0",
        description="Column schema version"
    )
    is_deprecated: bool = Field(
        default=False,
        description="Whether the column is deprecated"
    )
    deprecated_at: Optional[str] = Field(
        default=None,
        description="When the column was deprecated"
    )
    replacement_column: Optional[str] = Field(
        default=None,
        description="Name of the replacement column if deprecated"
    )

    @validator('name')
    def validate_name(cls, v):
        """Validate column name."""
        if not v or not v.strip():
            raise ValueError("Column name cannot be empty")
        if not v[0].isalpha() and v[0] != '_':
            raise ValueError(
                "Column name must start with a letter or underscore"
            )
        return v.strip()


class SchemaMetadata(BaseModel):
    """Schema metadata information."""
    last_modified: str = Field(
        default_factory=lambda: datetime.now(pytz.UTC).isoformat(),
        description="Last modified timestamp in UTC"
    )
    table_name: str = Field(..., description="Name of the table")
    source_file: str = Field(..., description="Source file name")
    row_count: int = Field(..., description="Number of rows in the data")
    is_new_table: bool = Field(
        default=True,
        description="Whether this is a new table or existing table update"
    )
    schema_version: str = Field(
        default="1.0.0",
        description="Schema version"
    )
    file_hash: str = Field(
        default="",
        description="MD5 hash of the source file"
    )
    column_count: int = Field(
        default=0,
        description="Number of columns in the schema"
    )
    primary_keys: List[str] = Field(
        default_factory=list,
        description="List of primary key columns"
    )
    foreign_keys: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of foreign key relationships"
    )
    data_quality_score: float = Field(
        default=1.0,
        description="Overall data quality score"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorizing the table"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(pytz.UTC).isoformat(),
        description="Table creation timestamp"
    )
    owner: str = Field(
        default="system",
        description="Table owner"
    )
    description: str = Field(
        default="",
        description="Table description"
    )
    access_level: str = Field(
        default="private",
        description="Table access level"
    )
    retention_period: Optional[int] = Field(
        default=None,
        description="Data retention period in days"
    )
    update_frequency: str = Field(
        default="on-demand",
        description="How often the table is updated"
    )
    last_update_source: str = Field(
        default="",
        description="Source of the last update"
    )
    schema_changes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of schema changes"
    )

    @validator('table_name')
    def validate_table_name(cls, v):
        """Validate table name."""
        if not v or not v.strip():
            raise ValueError("Table name cannot be empty")
        if not v[0].isalpha() and v[0] != '_':
            raise ValueError(
                "Table name must start with a letter or underscore"
            )
        return v.strip()

    @validator('data_quality_score')
    def validate_quality_score(cls, v):
        """Validate data quality score."""
        if not 0 <= v <= 1:
            raise ValueError("Data quality score must be between 0 and 1")
        return v


class SchemaDefinition(BaseModel):
    """Complete schema definition."""
    columns: List[SchemaColumn] = Field(..., description="List of columns")
    metadata: SchemaMetadata = Field(..., description="Schema metadata")

    def get_column(self, name: str) -> Optional[SchemaColumn]:
        """Get column by name."""
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data against schema."""
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }

        # Check required columns
        for col in self.columns:
            if col.is_required and col.name not in df.columns:
                validation_results["is_valid"] = False
                validation_results["errors"].append(
                    f"Required column '{col.name}' is missing"
                )

        # Check data types
        for col in self.columns:
            if col.name in df.columns:
                expected_type = col.type
                actual_type = str(df[col.name].dtype)
                if expected_type != actual_type:
                    validation_results["warnings"].append(
                        f"Column '{col.name}' has type {actual_type}, "
                        f"expected {expected_type}"
                    )

        return validation_results

    def compare_with_existing(
        self,
        existing_schema: 'SchemaDefinition'
    ) -> Dict[str, Any]:
        """Compare with existing schema to detect changes."""
        changes = {
            "added_columns": [],
            "removed_columns": [],
            "modified_columns": [],
            "unchanged_columns": []
        }

        # Get column sets
        new_cols = {col.name: col for col in self.columns}
        existing_cols = {col.name: col for col in existing_schema.columns}

        # Find added columns
        for name, col in new_cols.items():
            if name not in existing_cols:
                changes["added_columns"].append(col.dict())

        # Find removed columns
        for name, col in existing_cols.items():
            if name not in new_cols:
                changes["removed_columns"].append(col.dict())

        # Find modified and unchanged columns
        for name, new_col in new_cols.items():
            if name in existing_cols:
                existing_col = existing_cols[name]
                if new_col.dict() != existing_col.dict():
                    changes["modified_columns"].append({
                        "name": name,
                        "old": existing_col.dict(),
                        "new": new_col.dict()
                    })
                else:
                    changes["unchanged_columns"].append(new_col.dict())

        return changes

    def merge_with_existing(
        self,
        existing_schema: 'SchemaDefinition'
    ) -> 'SchemaDefinition':
        """Merge with existing schema, preserving history."""
        # Get changes
        changes = self.compare_with_existing(existing_schema)
        
        # Create new schema with merged columns
        merged_columns = []
        
        # Add unchanged columns
        for col in changes["unchanged_columns"]:
            merged_columns.append(SchemaColumn(**col))
        
        # Add modified columns with updated metadata
        for change in changes["modified_columns"]:
            col = SchemaColumn(**change["new"])
            col.modified_at = datetime.now(pytz.UTC).isoformat()
            merged_columns.append(col)
        
        # Add new columns
        for col in changes["added_columns"]:
            merged_columns.append(SchemaColumn(**col))
        
        # Create new metadata
        new_metadata = self.metadata.dict()
        new_metadata["schema_changes"].append({
            "timestamp": datetime.now(pytz.UTC).isoformat(),
            "changes": changes
        })
        
        return SchemaDefinition(
            columns=merged_columns,
            metadata=SchemaMetadata(**new_metadata)
        )


class SchemaGenerator:
    """Generates schema using LangChain components."""
    
    def __init__(self, llm):
        """Initialize the schema generator."""
        self.llm = llm
        self._setup_components()
        self.max_tokens = 8100  # Leave some buffer for response
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
    def _setup_components(self):
        """Setup LangChain components."""
        # Setup memory for caching
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Setup streaming callback
        self.streaming_callback = StreamingStdOutCallbackHandler()
        
        # Setup evaluator
        self.evaluator = load_evaluator(
            "criteria",
            criteria={
                "schema_completeness": (
                    "Is the schema complete with all required fields?"
                ),
                "type_accuracy": (
                    "Are the data types accurate and appropriate?"
                ),
                "description_quality": (
                    "Are the column descriptions clear and informative?"
                )
            }
        )
        
        # Schema enhancement prompt
        self.enhancement_prompt = PromptTemplate.from_template(
            """Enhance this schema with better descriptions and validation rules:
            
            Schema: {schema}
            
            Focus on:
            1. Clear column descriptions
            2. Key validation rules
            3. Critical business rules
            
            Return only the enhanced schema in JSON format.
            """
        )
        
        # Setup chains
        self.enhancement_chain = (
            {"schema": RunnablePassthrough()}
            | self.enhancement_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.encoding.encode(text))
    
    def _truncate_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Truncate schema to fit within token limits."""
        # Convert schema to string and count tokens
        schema_str = json.dumps(schema, indent=2)
        total_tokens = self._count_tokens(schema_str)
        
        if total_tokens <= self.max_tokens:
            return schema
        
        # If too many tokens, start removing less important information
        truncated_schema = {
            "columns": [],
            "metadata": schema.get("metadata", {})
        }
        
        for col in schema["columns"]:
            # Keep essential fields
            truncated_col = {
                "name": col["name"],
                "type": col["type"],
                "description": col.get("description", ""),
                "quality_metrics": {
                    "completeness": col["quality_metrics"]["completeness"],
                    "uniqueness": col["quality_metrics"]["uniqueness"]
                }
            }
            
            # Only add sample value if it's short
            if len(str(col.get("sample_value", ""))) < 50:
                truncated_col["sample_value"] = col.get("sample_value", "")
            
            truncated_schema["columns"].append(truncated_col)
            
            # Check if we're under the limit
            if self._count_tokens(
                json.dumps(truncated_schema, indent=2)
            ) <= self.max_tokens:
                break
        
        return truncated_schema
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _infer_schema_from_df(
        self,
        df: pd.DataFrame,
        table_name: str,
        source_file: str,
        is_new_table: bool = True,
        file_path: Optional[str] = None,
        existing_schema: Optional[SchemaDefinition] = None
    ) -> Dict[str, Any]:
        """Infer initial schema from DataFrame using pandas."""
        # Calculate file hash if file path is provided
        file_hash = (
            self._calculate_file_hash(file_path)
            if file_path
            else ""
        )
        
        # Calculate data quality score
        quality_scores = []
        for column in df.columns:
            completeness = float(1 - df[column].isna().mean())
            uniqueness = float(1 - (df[column].nunique() / len(df)))
            quality_scores.append((completeness + uniqueness) / 2)
        
        data_quality_score = (
            sum(quality_scores) / len(quality_scores)
            if quality_scores
            else 1.0
        )
        
        # Create base metadata
        metadata = SchemaMetadata(
            table_name=table_name,
            source_file=source_file,
            row_count=len(df),
            is_new_table=is_new_table,
            file_hash=file_hash,
            column_count=len(df.columns),
            data_quality_score=data_quality_score
        )
        
        # If existing schema, merge metadata
        if existing_schema:
            metadata = existing_schema.metadata
            metadata.last_modified = datetime.now(pytz.UTC).isoformat()
            metadata.row_count = len(df)
            metadata.file_hash = file_hash
            metadata.data_quality_score = data_quality_score
        
        schema = {
            "columns": [],
            "metadata": metadata.dict()
        }
        
        for column in df.columns:
            # Get basic column info
            col_type = str(df[column].dtype)
            sample_val = (
                str(df[column].iloc[0])
                if not df[column].empty
                else ""
            )
            
            # Calculate quality metrics
            completeness = float(1 - df[column].isna().mean())
            uniqueness = float(1 - (df[column].nunique() / len(df)))
            
            # Check if column exists in existing schema
            existing_col = None
            if existing_schema:
                existing_col = existing_schema.get_column(column)
            
            # Create column info
            col_info = SchemaColumn(
                name=str(column),
                type=col_type,
                sample_value=sample_val,
                is_required=completeness > 0.95,
                is_unique=uniqueness > 0.95,
                quality_metrics={
                    "completeness": completeness,
                    "uniqueness": uniqueness,
                    "validity": 1.0
                }
            )
            
            # If column exists, preserve metadata
            if existing_col:
                col_info.created_at = existing_col.created_at
                col_info.version = existing_col.version
                col_info.is_deprecated = existing_col.is_deprecated
                col_info.deprecated_at = existing_col.deprecated_at
                col_info.replacement_column = existing_col.replacement_column
            
            schema["columns"].append(col_info.dict())
        
        return schema
    
    @async_retry(max_retries=3, delay=1)
    async def generate_schema(
        self, 
        df: pd.DataFrame,
        table_name: str,
        source_file: str,
        is_new_table: bool = True,
        file_path: Optional[str] = None,
        existing_schema: Optional[SchemaDefinition] = None,
        sample_size: int = 5,
        use_streaming: bool = False,
        evaluate: bool = True
    ) -> Dict[str, Any]:
        """Generate schema using LangChain components."""
        try:
            # First, infer schema using pandas
            initial_schema = self._infer_schema_from_df(
                df,
                table_name=table_name,
                source_file=source_file,
                is_new_table=is_new_table,
                file_path=file_path,
                existing_schema=existing_schema
            )
            
            # Prepare a smaller sample for LLM
            sample_data = df.head(sample_size).copy()
            for col in sample_data.columns:
                if pd.api.types.is_datetime64_any_dtype(sample_data[col]):
                    sample_data[col] = sample_data[col].dt.strftime(
                        '%Y-%m-%d %H:%M:%S'
                    )
                else:
                    sample_data[col] = sample_data[col].astype(str)
            
            # Add sample values to schema
            for col in initial_schema["columns"]:
                col_name = col["name"]
                if col_name in sample_data.columns:
                    col["sample_value"] = str(sample_data[col_name].iloc[0])
            
            # Truncate schema if needed
            truncated_schema = self._truncate_schema(initial_schema)
            
            # Generate enhanced schema with streaming if requested
            callbacks = [self.streaming_callback] if use_streaming else None
            callback_mgr = (
                CallbackManager(callbacks)
                if callbacks
                else None
            )
            
            # Only enhance the schema with LLM
            enhanced_schema = await self.enhancement_chain.ainvoke(
                json.dumps(truncated_schema, indent=2),
                config={"callbacks": callback_mgr}
            )
            
            # Parse the JSON response
            enhanced_schema = json.loads(enhanced_schema)
            
            # Ensure metadata is preserved
            enhanced_schema["metadata"] = truncated_schema["metadata"]
            
            # If existing schema, merge changes
            if existing_schema:
                enhanced_schema = SchemaDefinition(
                    **enhanced_schema
                ).merge_with_existing(existing_schema).dict()
            
            # Evaluate schema if requested
            if evaluate:
                evaluation_result = await self.evaluator.aevaluate(
                    prediction=enhanced_schema,
                    input={
                        "sample_data": sample_data.to_dict('records'),
                        "initial_schema": truncated_schema
                    }
                )
                logger.info(
                    f"Schema evaluation results: {evaluation_result}"
                )
            
            return enhanced_schema
            
        except Exception as e:
            logger.error(f"Error generating schema: {str(e)}")
            raise 