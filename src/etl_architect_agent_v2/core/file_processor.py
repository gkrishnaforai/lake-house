"""File processor for handling multiple file types."""

import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as csv
import pyarrow.json as json
import pyarrow.compute as pc
from fastavro import reader, validate
import openpyxl
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class FileValidationResult(BaseModel):
    """File validation result."""
    is_valid: bool = Field(..., description="Whether file is valid")
    file_type: str = Field(..., description="Detected file type")
    error_message: Optional[str] = Field(None, description="Error message if invalid")
    sample_data: Optional[Dict[str, Any]] = Field(None, description="Sample data")
    schema: Optional[Dict[str, Any]] = Field(None, description="Inferred schema")


class FileProcessor:
    """Processor for multiple file types."""
    
    SUPPORTED_TYPES = {
        "csv": [".csv"],
        "excel": [".xlsx", ".xls"],
        "json": [".json"],
        "parquet": [".parquet"],
        "avro": [".avro"]
    }
    
    def __init__(self):
        """Initialize file processor."""
        self._type_map = {}
        for file_type, extensions in self.SUPPORTED_TYPES.items():
            for ext in extensions:
                self._type_map[ext] = file_type
    
    def detect_file_type(self, file_path: Union[str, Path]) -> str:
        """Detect file type from extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            Detected file type
        """
        ext = Path(file_path).suffix.lower()
        if ext not in self._type_map:
            raise ValueError(f"Unsupported file type: {ext}")
        return self._type_map[ext]
    
    def validate_file(
        self,
        file_path: Union[str, Path],
        file_type: Optional[str] = None
    ) -> FileValidationResult:
        """Validate file and extract sample data.
        
        Args:
            file_path: Path to file
            file_type: Optional file type override
            
        Returns:
            Validation result
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return FileValidationResult(
                is_valid=False,
                file_type="unknown",
                error_message="File does not exist"
            )
        
        if file_type is None:
            try:
                file_type = self.detect_file_type(file_path)
            except ValueError as e:
                return FileValidationResult(
                    is_valid=False,
                    file_type="unknown",
                    error_message=str(e)
                )
        
        try:
            if file_type == "csv":
                return self._validate_csv(file_path)
            elif file_type == "excel":
                return self._validate_excel(file_path)
            elif file_type == "json":
                return self._validate_json(file_path)
            elif file_type == "parquet":
                return self._validate_parquet(file_path)
            elif file_type == "avro":
                return self._validate_avro(file_path)
            else:
                return FileValidationResult(
                    is_valid=False,
                    file_type=file_type,
                    error_message=f"Unsupported file type: {file_type}"
                )
        except Exception as e:
            logger.error(f"Error validating file: {str(e)}", exc_info=True)
            return FileValidationResult(
                is_valid=False,
                file_type=file_type,
                error_message=str(e)
            )
    
    def _validate_csv(self, file_path: Path) -> FileValidationResult:
        """Validate CSV file."""
        try:
            table = csv.read_csv(file_path, read_options=csv.ReadOptions(num_rows=5))
            schema = {
                "columns": [
                    {"name": field.name, "type": str(field.type)}
                    for field in table.schema
                ]
            }
            return FileValidationResult(
                is_valid=True,
                file_type="csv",
                sample_data=self._table_to_dict(table),
                schema=schema
            )
        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                file_type="csv",
                error_message=str(e)
            )
    
    def _validate_excel(self, file_path: Path) -> FileValidationResult:
        """Validate Excel file."""
        try:
            wb = openpyxl.load_workbook(file_path, read_only=True)
            sheet = wb.active
            rows = list(sheet.iter_rows(min_row=1, max_row=6, values_only=True))
            
            if not rows:
                return FileValidationResult(
                    is_valid=False,
                    file_type="excel",
                    error_message="Empty file"
                )
            
            headers = rows[0]
            sample_data = {
                "headers": headers,
                "rows": rows[1:6]
            }
            
            # Infer types from sample data
            schema = {
                "columns": [
                    {
                        "name": header,
                        "type": self._infer_type(rows[1:], i)
                    }
                    for i, header in enumerate(headers)
                ]
            }
            
            return FileValidationResult(
                is_valid=True,
                file_type="excel",
                sample_data=sample_data,
                schema=schema
            )
        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                file_type="excel",
                error_message=str(e)
            )
    
    def _validate_json(self, file_path: Path) -> FileValidationResult:
        """Validate JSON file."""
        try:
            table = json.read_json(file_path, read_options=json.ReadOptions(num_rows=5))
            schema = {
                "columns": [
                    {"name": field.name, "type": str(field.type)}
                    for field in table.schema
                ]
            }
            return FileValidationResult(
                is_valid=True,
                file_type="json",
                sample_data=self._table_to_dict(table),
                schema=schema
            )
        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                file_type="json",
                error_message=str(e)
            )
    
    def _validate_parquet(self, file_path: Path) -> FileValidationResult:
        """Validate Parquet file."""
        try:
            table = pq.read_table(file_path, num_rows=5)
            schema = {
                "columns": [
                    {"name": field.name, "type": str(field.type)}
                    for field in table.schema
                ]
            }
            return FileValidationResult(
                is_valid=True,
                file_type="parquet",
                sample_data=self._table_to_dict(table),
                schema=schema
            )
        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                file_type="parquet",
                error_message=str(e)
            )
    
    def _validate_avro(self, file_path: Path) -> FileValidationResult:
        """Validate Avro file."""
        try:
            with open(file_path, 'rb') as f:
                avro_reader = reader(f)
                schema = avro_reader.writer_schema
                
                # Read first 5 records
                records = []
                for i, record in enumerate(avro_reader):
                    if i >= 5:
                        break
                    records.append(record)
                
                return FileValidationResult(
                    is_valid=True,
                    file_type="avro",
                    sample_data={"records": records},
                    schema={"schema": schema.to_json()}
                )
        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                file_type="avro",
                error_message=str(e)
            )
    
    def _table_to_dict(self, table: pa.Table) -> Dict[str, Any]:
        """Convert PyArrow table to dictionary."""
        return {
            "columns": table.column_names,
            "rows": [
                [str(cell) for cell in row]
                for row in table.to_pylist()
            ]
        }
    
    def _infer_type(self, rows: List[tuple], col_idx: int) -> str:
        """Infer column type from sample data."""
        values = [row[col_idx] for row in rows if row[col_idx] is not None]
        if not values:
            return "string"
        
        # Try to infer numeric type
        try:
            float_values = [float(v) for v in values]
            if all(v.is_integer() for v in float_values):
                return "integer"
            return "float"
        except (ValueError, TypeError):
            pass
        
        # Try to infer boolean type
        if all(isinstance(v, bool) for v in values):
            return "boolean"
        
        # Try to infer date type
        try:
            pd.to_datetime(values)
            return "datetime"
        except (ValueError, TypeError):
            pass
        
        return "string" 