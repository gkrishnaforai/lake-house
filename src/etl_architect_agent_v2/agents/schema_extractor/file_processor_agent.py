"""File processor agent for handling multiple file types and validation."""

import logging
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import fastavro
import json
from pathlib import Path
from core.llm.manager import LLMManager
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class FileValidationResult(BaseModel):
    """Result of file validation."""
    is_valid: bool = Field(..., description="Whether the file is valid")
    file_type: str = Field(..., description="Detected file type")
    error_messages: List[str] = Field(default_factory=list, description="Validation error messages")
    sample_data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = Field(None, description="Sample data for schema inference")

class FileProcessorAgent:
    """Agent for processing different file types and validation."""
    
    SUPPORTED_EXTENSIONS = {
        '.csv': 'csv',
        '.xlsx': 'excel',
        '.json': 'json',
        '.parquet': 'parquet',
        '.avro': 'avro'
    }
    
    def __init__(self, llm_manager: LLMManager):
        """Initialize the file processor agent."""
        self.llm_manager = llm_manager
    
    async def validate_file(self, file_path: str) -> FileValidationResult:
        """Validate a file and detect its type."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return FileValidationResult(
                    is_valid=False,
                    file_type='unknown',
                    error_messages=['File does not exist']
                )
            
            # Detect file type
            file_type = self.SUPPORTED_EXTENSIONS.get(file_path.suffix.lower())
            if not file_type:
                return FileValidationResult(
                    is_valid=False,
                    file_type='unknown',
                    error_messages=[f'Unsupported file type: {file_path.suffix}']
                )
            
            # Validate based on file type
            if file_type == 'csv':
                return await self._validate_csv(file_path)
            elif file_type == 'excel':
                return await self._validate_excel(file_path)
            elif file_type == 'json':
                return await self._validate_json(file_path)
            elif file_type == 'parquet':
                return await self._validate_parquet(file_path)
            elif file_type == 'avro':
                return await self._validate_avro(file_path)
            
        except Exception as e:
            logger.error(f"Error validating file: {str(e)}", exc_info=True)
            return FileValidationResult(
                is_valid=False,
                file_type='unknown',
                error_messages=[f'Validation error: {str(e)}']
            )
    
    async def _validate_csv(self, file_path: Path) -> FileValidationResult:
        """Validate CSV file."""
        try:
            # Read first few rows to validate
            df = pd.read_csv(file_path, nrows=5)
            return FileValidationResult(
                is_valid=True,
                file_type='csv',
                sample_data=df.to_dict(orient='records')
            )
        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                file_type='csv',
                error_messages=[f'CSV validation error: {str(e)}']
            )
    
    async def _validate_excel(self, file_path: Path) -> FileValidationResult:
        """Validate Excel file."""
        try:
            # Read first sheet to validate
            df = pd.read_excel(file_path, nrows=5)
            return FileValidationResult(
                is_valid=True,
                file_type='excel',
                sample_data=df.to_dict(orient='records')
            )
        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                file_type='excel',
                error_messages=[f'Excel validation error: {str(e)}']
            )
    
    async def _validate_json(self, file_path: Path) -> FileValidationResult:
        """Validate JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Take first 5 items if it's a list
                sample = data[:5] if isinstance(data, list) else data
                return FileValidationResult(
                    is_valid=True,
                    file_type='json',
                    sample_data=sample
                )
        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                file_type='json',
                error_messages=[f'JSON validation error: {str(e)}']
            )
    
    async def _validate_parquet(self, file_path: Path) -> FileValidationResult:
        """Validate Parquet file."""
        try:
            # Read first few rows to validate
            table = pq.read_table(file_path)
            df = table.to_pandas().head(5)
            return FileValidationResult(
                is_valid=True,
                file_type='parquet',
                sample_data=df.to_dict(orient='records')
            )
        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                file_type='parquet',
                error_messages=[f'Parquet validation error: {str(e)}']
            )
    
    async def _validate_avro(self, file_path: Path) -> FileValidationResult:
        """Validate Avro file."""
        try:
            with open(file_path, 'rb') as f:
                # Read first few records to validate
                reader = fastavro.reader(f)
                sample = [next(reader) for _ in range(5)]
                return FileValidationResult(
                    is_valid=True,
                    file_type='avro',
                    sample_data=sample
                )
        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                file_type='avro',
                error_messages=[f'Avro validation error: {str(e)}']
            ) 