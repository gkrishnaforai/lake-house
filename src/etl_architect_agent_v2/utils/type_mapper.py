import datetime
import logging
from typing import Any, Union, Optional
from decimal import Decimal
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def map_python_type_to_sql_type(python_type: type) -> str:
    """
    Map Python types to SQL types with comprehensive error handling.
    
    Args:
        python_type: The Python type to map to SQL
        
    Returns:
        str: The corresponding SQL type
        
    Raises:
        ValueError: If the type cannot be mapped
    """
    try:
        # Basic Python types
        if python_type == int:
            return "INTEGER"
        elif python_type == str:
            return "VARCHAR(255)"  # Default length, can be overridden
        elif python_type == float:
            return "FLOAT"
        elif python_type == bool:
            return "BOOLEAN"
        elif python_type == bytes:
            return "BLOB"
        
        # Date and time types
        elif python_type == datetime.date:
            return "DATE"
        elif python_type == datetime.time:
            return "TIME"
        elif python_type == datetime.datetime:
            return "TIMESTAMP"
        
        # Decimal types
        elif python_type == Decimal:
            return "DECIMAL(10,2)"  # Default precision and scale
        
        # Numpy types
        elif python_type == np.int8:
            return "SMALLINT"
        elif python_type == np.int16:
            return "SMALLINT"
        elif python_type == np.int32:
            return "INTEGER"
        elif python_type == np.int64:
            return "BIGINT"
        elif python_type == np.float32:
            return "FLOAT"
        elif python_type == np.float64:
            return "DOUBLE"
        elif python_type == np.bool_:
            return "BOOLEAN"
        
        # Optional types (from typing)
        elif hasattr(python_type, "__origin__") and python_type.__origin__ == Union:
            # Handle Optional types (Union[T, None])
            if len(python_type.__args__) == 2 and type(None) in python_type.__args__:
                non_none_type = next(t for t in python_type.__args__ if t is not type(None))
                return f"{map_python_type_to_sql_type(non_none_type)} NULL"
            else:
                # For other unions, use the first type
                return map_python_type_to_sql_type(python_type.__args__[0])
        
        # List types (map to JSON)
        elif hasattr(python_type, "__origin__") and python_type.__origin__ == list:
            return "JSON"
        
        # Dictionary types (map to JSON)
        elif hasattr(python_type, "__origin__") and python_type.__origin__ == dict:
            return "JSON"
        
        else:
            error_msg = f"Unsupported Python type: {python_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
    except Exception as e:
        error_msg = f"Error mapping type {python_type}: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def get_sql_type_with_length(python_type: type, length: Optional[int] = None) -> str:
    """
    Get SQL type with optional length specification.
    
    Args:
        python_type: The Python type to map
        length: Optional length for VARCHAR types
        
    Returns:
        str: SQL type with length specification if applicable
    """
    sql_type = map_python_type_to_sql_type(python_type)
    
    if "VARCHAR" in sql_type and length is not None:
        return f"VARCHAR({length})"
    return sql_type

def get_sql_type_with_precision(python_type: type, precision: int = 10, scale: int = 2) -> str:
    """
    Get SQL type with precision and scale for decimal types.
    
    Args:
        python_type: The Python type to map
        precision: Total number of digits
        scale: Number of decimal places
        
    Returns:
        str: SQL type with precision and scale
    """
    sql_type = map_python_type_to_sql_type(python_type)
    
    if "DECIMAL" in sql_type:
        return f"DECIMAL({precision},{scale})"
    return sql_type