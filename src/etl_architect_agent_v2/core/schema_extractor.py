"""Schema extractor for data files."""

from typing import Dict, Any, List
import pandas as pd
import json
import logging
from datetime import datetime
from langchain_core.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.chains import LLMChain
from langchain_community.callbacks.manager import get_openai_callback

logger = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects."""
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return super().default(obj)

class SchemaExtractor:
    """Extracts schema information from data files."""
    
    @staticmethod
    def get_pandas_dtype_mapping() -> Dict[str, str]:
        """Map pandas dtypes to our schema types."""
        return {
            'object': 'string',
            'string': 'string',
            'int64': 'integer',
            'float64': 'float',
            'bool': 'boolean',
            'datetime64[ns]': 'datetime',
            'category': 'string'
        }
        
    @staticmethod
    def extract_schema(df: pd.DataFrame, sample_size: int = 5) -> Dict[str, Any]:
        """Extract schema information from a pandas DataFrame."""
        schema = {"columns": []}
        sample_data = df.head(sample_size)
        
        for column in df.columns:
            # Convert sample value to string, handling datetime
            sample_value = sample_data[column].iloc[0]
            if isinstance(sample_value, (pd.Timestamp, datetime)):
                sample_value = sample_value.isoformat()
            else:
                sample_value = str(sample_value) if not pd.isna(sample_value) else ""
            
            col_info = {
                "name": str(column),
                "type": SchemaExtractor.get_pandas_dtype_mapping().get(
                    str(df[column].dtype), 'string'
                ),
                "description": "",  # Will be filled by LLM
                "sample_value": sample_value,
                "quality_metrics": {
                    "completeness": float(1 - df[column].isna().mean()),
                    "uniqueness": float(1 - (df[column].nunique() / len(df))),
                    "validity": 1.0  # Will be refined by LLM
                }
            }
            schema["columns"].append(col_info)
            
        return schema
        
    @staticmethod
    def create_optimized_prompt(df: pd.DataFrame, sample_size: int = 5) -> str:
        """Create an optimized prompt for schema generation."""
        schema = SchemaExtractor.extract_schema(df, sample_size)
        
        # Convert sample data to JSON-serializable format
        sample_data = df.head(sample_size).copy()
        for col in sample_data.columns:
            if pd.api.types.is_datetime64_any_dtype(sample_data[col]):
                sample_data[col] = sample_data[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                sample_data[col] = sample_data[col].astype(str)
        
        sample_records = sample_data.to_dict('records')
        
        prompt = f"""Please enhance the following schema with better descriptions and validation rules.
The schema was automatically extracted from the data.

Current Schema:
{json.dumps(schema, indent=2, cls=DateTimeEncoder)}

Sample Data (first {sample_size} rows):
{json.dumps(sample_records, indent=2, cls=DateTimeEncoder)}

Please provide:
1. Better column descriptions
2. Validation rules if any
3. Suggested data quality improvements
4. Any business rules you can infer

Return the enhanced schema in the same format.
"""
        return prompt 