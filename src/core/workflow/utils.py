from typing import TypeVar, Dict, Any, Optional, List
from datetime import datetime
from .base_state import BaseWorkflowState


T = TypeVar('T', bound=BaseWorkflowState)


def update_state_timestamp(state: T) -> T:
    """Update the timestamp of a state.
    
    Args:
        state: The state to update
        
    Returns:
        Updated state with new timestamp
    """
    state.timestamp = datetime.now().isoformat()
    return state


def add_metadata(state: T, key: str, value: Any) -> T:
    """Add metadata to a state.
    
    Args:
        state: The state to update
        key: Metadata key
        value: Metadata value
        
    Returns:
        Updated state with new metadata
    """
    if state.metadata is None:
        state.metadata = {}
    state.metadata[key] = value
    return state


def get_metadata(state: T, key: str, default: Any = None) -> Any:
    """Get metadata from a state.
    
    Args:
        state: The state to query
        key: Metadata key
        default: Default value if key not found
        
    Returns:
        Metadata value or default
    """
    if state.metadata is None:
        return default
    return state.metadata.get(key, default)


def validate_state_required_fields(state: T, required_fields: List[str]) -> bool:
    """Validate that a state has all required fields.
    
    Args:
        state: The state to validate
        required_fields: List of required field names
        
    Returns:
        True if all required fields are present and non-empty
    """
    return all(
        hasattr(state, field) and getattr(state, field)
        for field in required_fields
    )


def merge_states(base_state: T, update_dict: Dict[str, Any]) -> T:
    """Merge a dictionary of updates into a state.
    
    Args:
        base_state: The base state
        update_dict: Dictionary of updates
        
    Returns:
        Merged state
    """
    for key, value in update_dict.items():
        if hasattr(base_state, key):
            setattr(base_state, key, value)
    return base_state


def create_state_snapshot(state: T) -> Dict[str, Any]:
    """Create a snapshot of a state.
    
    Args:
        state: The state to snapshot
        
    Returns:
        Dictionary containing state snapshot
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "state": state.__dict__
    }


def is_state_complete(state: T, completion_fields: list[str]) -> bool:
    """Check if a state is complete based on specified fields.
    
    Args:
        state: The state to check
        completion_fields: List of fields that must be non-empty
        
    Returns:
        True if all completion fields are non-empty
    """
    return all(
        field in state and state[field] is not None and state[field] != ""
        for field in completion_fields
    )


def get_state_duration(state: T) -> Optional[float]:
    """Get the duration of a state in seconds.
    
    Args:
        state: The state to check
        
    Returns:
        Duration in seconds or None if timestamp is invalid
    """
    if not state.timestamp:
        return None
    
    start_time = datetime.fromisoformat(state.timestamp)
    end_time = datetime.now()
    return (end_time - start_time).total_seconds() 