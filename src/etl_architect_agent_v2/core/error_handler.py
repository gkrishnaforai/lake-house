"""Error handling for the ETL Architect Agent V2."""

import logging
from typing import Optional, Type, Any, Callable, Dict
from functools import wraps
import time
from datetime import datetime
import traceback
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class AgentError(Exception):
    """Base exception class for agent errors."""
    pass

class LLMError(AgentError):
    """Custom exception for LLM-related errors."""
    pass

class StateError(AgentError):
    """Exception raised for state-related errors."""
    pass

class ValidationError(AgentError):
    """Exception raised for validation errors."""
    pass

class RetryError(AgentError):
    """Exception raised when max retries exceeded."""
    pass

def retry(
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
) -> Callable:
    """Decorator for retrying functions with exponential backoff.
    
    Args:
        exceptions: Tuple of exceptions to catch
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        raise RetryError(
                            f"Max retries ({max_retries}) exceeded. Last error: {str(e)}"
                        ) from e
                    
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {current_delay} seconds..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            raise RetryError("Unexpected error in retry decorator") from last_exception
        return wrapper
    return decorator

class ErrorHandler:
    """Handles errors and provides detailed error information."""
    
    def __init__(self, max_retries: int = 3):
        """Initialize the error handler.
        
        Args:
            max_retries: Maximum number of retry attempts
        """
        self.max_retries = max_retries
        self.error_history: list[dict[str, Any]] = []
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Handle an error with detailed logging.
        
        Args:
            error: The exception that occurred
            context: Optional context information about the error
        """
        error_type = type(error).__name__
        error_msg = str(error)
        stack_trace = traceback.format_exc()
        
        # Extract AWS-specific error information
        aws_error_info = ""
        if isinstance(error, ClientError):
            aws_error_info = self._extract_aws_error_info(error)
        
        # Log the error with context
        logger.error(
            f"Error Type: {error_type}\n"
            f"Error Message: {error_msg}\n"
            f"{aws_error_info}"
            f"Context: {context}\n"
            f"Stack Trace:\n{stack_trace}"
        )
        
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_msg,
            "context": context or {},
            "stack_trace": stack_trace,
            "aws_error_info": aws_error_info
        }
        
        self.error_history.append(error_info)
    
    def _extract_aws_error_info(self, error: ClientError) -> str:
        """Extract detailed information from AWS errors.
        
        Args:
            error: The AWS ClientError exception
            
        Returns:
            str: Formatted AWS error information
        """
        try:
            error_response = error.response
            error_code = error_response.get('Error', {}).get('Code', 'Unknown')
            error_message = error_response.get('Error', {}).get('Message', 'No message')
            request_id = error_response.get('ResponseMetadata', {}).get('RequestId', 'No request ID')
            
            return (
                f"AWS Error Code: {error_code}\n"
                f"AWS Error Message: {error_message}\n"
                f"AWS Request ID: {request_id}\n"
            )
        except Exception as e:
            logger.warning(f"Failed to extract AWS error info: {str(e)}")
            return "Failed to extract AWS error information\n"
    
    def get_error_history(self) -> list[dict[str, Any]]:
        """Get the history of handled errors.
        
        Returns:
            List of error information dictionaries
        """
        return self.error_history.copy()
    
    def clear_error_history(self) -> None:
        """Clear the error history."""
        self.error_history.clear()
    
    def should_retry(self, error: Exception) -> bool:
        """Determine if an error should be retried.
        
        Args:
            error: The exception to check
            
        Returns:
            True if the error should be retried, False otherwise
        """
        return (
            isinstance(error, (LLMError, StateError))
            and len(self.error_history) < self.max_retries
        ) 