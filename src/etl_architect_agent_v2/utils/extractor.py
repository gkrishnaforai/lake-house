import logging
from typing import Any, List, Dict, Tuple
from etl_architect_agent_v2.model.column_schema import ColumnSchema
from etl_architect_agent_v2.model.table_schema import TableSchema
from etl_architect_agent_v2.utils.type_mapper import (
    map_python_type_to_sql_type
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def infer_python_type(value: Any) -> type:
    """
    Infer the Python type of a value, handling None as str (default).
    """
    try:
        if value is None:
            return str
        return type(value)
    except Exception as e:
        logger.error(f"Error inferring type: {str(e)}")
        return str  # Default to string type on error


class SchemaExtractor:
    """Class responsible for extracting database schema from JSON data."""
    
    def __init__(self, table_name: str = "inferred_table"):
        """
        Initialize the schema extractor.
        
        Args:
            table_name: Name for the inferred table.
        """
        self.table_name = table_name
    
    def extract_from_json(self, json_data: Any) -> TableSchema:
        """
        Extract a table schema from JSON data.
        
        Args:
            json_data: The JSON data (list of dicts or dict).
            
        Returns:
            TableSchema: The inferred table schema.
            
        Raises:
            ValueError: If JSON data is empty or in an invalid format.
        """
        try:
            # Check for empty JSON
            if not json_data:
                error_msg = "JSON data is empty, cannot infer schema."
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # If the data is a dict with a single key that points to a list,
            # use that list as the data source
            if (
                isinstance(json_data, dict)
                and len(json_data) == 1
                and isinstance(next(iter(json_data.values())), list)
            ):
                json_data = next(iter(json_data.values()))
            
            sample = self._get_sample(json_data)
            columns = self._extract_columns(sample, json_data)
            
            logger.info(
                f"Successfully extracted schema with {len(columns)} columns "
                f"for table '{self.table_name}'"
            )
            return TableSchema(table_name=self.table_name, columns=columns)
        except Exception as e:
            logger.error(f"Failed to extract schema: {str(e)}")
            raise
    
    def _get_sample(self, json_data: Any) -> Dict:
        """
        Get a sample dictionary from the JSON data.
        
        Args:
            json_data: The JSON data to sample from.
            
        Returns:
            Dict: A sample dictionary.
            
        Raises:
            ValueError: If JSON data is empty or in an invalid format.
        """
        if isinstance(json_data, list):
            if not json_data:
                error_msg = "JSON data list is empty, cannot infer schema."
                logger.error(error_msg)
                raise ValueError(error_msg)
            return json_data[0]
        elif isinstance(json_data, dict):
            return json_data
        else:
            error_msg = (
                f"JSON data must be a dict or a list of dicts, "
                f"got {type(json_data)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _extract_columns(self, sample: Dict, json_data: Any) -> List[ColumnSchema]:
        """
        Extract column schemas from a sample dictionary.
        
        Args:
            sample: A sample dictionary to extract columns from.
            json_data: The original JSON data for nullability checks.
            
        Returns:
            List[ColumnSchema]: The extracted column schemas.
        """
        columns: List[ColumnSchema] = []
        flattened_fields = self._flatten_dict(sample)
        
        for path, value in flattened_fields:
            column = self._create_column(path, value, json_data)
            columns.append(column)
        return columns
    
    def _flatten_dict(self, d: Dict, parent_key: str = "") -> List[Tuple[str, Any]]:
        """
        Flatten a nested dictionary into a list of (path, value) tuples.
        
        Args:
            d: The dictionary to flatten.
            parent_key: The parent key for nested fields.
            
        Returns:
            List[Tuple[str, Any]]: List of (path, value) tuples.
        """
        items: List[Tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key))
            elif isinstance(v, list):
                # For lists, we'll store them as JSON strings
                items.append((new_key, v))
            else:
                items.append((new_key, v))
        return items
    
    def _create_column(self, key: str, value: Any, json_data: Any) -> ColumnSchema:
        """
        Create a column schema for a single field.
        
        Args:
            key: The field name.
            value: The field value.
            json_data: The original JSON data for nullability checks.
            
        Returns:
            ColumnSchema: The created column schema.
        """
        try:
            py_type = infer_python_type(value)
            sql_type = map_python_type_to_sql_type(py_type)
            nullable = self._check_nullability(key, value, json_data)
            return ColumnSchema(name=key, type=sql_type, nullable=nullable)
        except Exception as e:
            logger.warning(
                f"Error processing column '{key}': {str(e)}. "
                f"Using VARCHAR as fallback."
            )
            return ColumnSchema(name=key, type="VARCHAR(255)", nullable=True)
    
    def _check_nullability(self, key: str, value: Any, json_data: Any) -> bool:
        """
        Check if a field can be null.
        
        Args:
            key: The field name.
            value: The field value.
            json_data: The original JSON data.
            
        Returns:
            bool: True if the field can be null, False otherwise.
        """
        if isinstance(json_data, list):
            # For nested fields, we need to traverse the path
            parts = key.split(".")
            return any(
                self._get_nested_value(row, parts) is None for row in json_data
            )
        return value is None
    
    def _get_nested_value(self, d: Dict, path: List[str]) -> Any:
        """
        Get a value from a nested dictionary using a path.
        
        Args:
            d: The dictionary to traverse.
            path: The path to follow.
            
        Returns:
            Any: The value at the path, or None if not found.
        """
        current = d
        for part in path:
            if not isinstance(current, dict):
                return None
            current = current.get(part)
            if current is None:
                return None
        return current


def extract_schema_from_json(
    json_data: Any, table_name: str = "inferred_table"
) -> TableSchema:
    """
    Extracts a table schema from a JSON object (list of dicts or dict).
    
    Args:
        json_data: The JSON data (list of dicts or dict).
        table_name: Name for the inferred table.
    
    Returns:
        TableSchema: The inferred table schema.
        
    Raises:
        ValueError: If JSON data is empty or in an invalid format.
    """
    extractor = SchemaExtractor(table_name)
    return extractor.extract_from_json(json_data)
