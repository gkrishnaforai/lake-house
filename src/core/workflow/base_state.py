from typing import Optional, Dict, Any
from datetime import datetime


class BaseWorkflowState:
    """Base state for all workflows.
    
    This is the foundation for all workflow states, providing common fields
    that are shared across different types of workflows.
    """
    
    def __init__(
        self,
        execution_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None
    ):
        """Initialize base workflow state.
        
        Args:
            execution_id: Unique identifier for this execution
            metadata: Optional additional metadata
            timestamp: Optional timestamp for the state
        """
        self.execution_id = execution_id
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.now().isoformat()


def create_initial_state(
    execution_id: str,
    metadata: Optional[Dict[str, Any]] = None
) -> BaseWorkflowState:
    """Create initial state for any workflow.
    
    Args:
        execution_id: Unique identifier for this execution
        metadata: Optional additional metadata
        
    Returns:
        BaseWorkflowState: Initialized base state
    """
    return BaseWorkflowState(
        execution_id=execution_id,
        metadata=metadata,
        timestamp=datetime.now().isoformat()
    ) 