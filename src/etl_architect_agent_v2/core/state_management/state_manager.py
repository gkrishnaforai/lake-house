"""State management module for workflow state."""

from typing import Dict, Any, Optional


class StateManager:
    """Manages state for workflow execution."""
    
    def __init__(self):
        """Initialize state manager with empty state."""
        self._state: Dict[str, Any] = {}
    
    def get_state(self, key: str) -> Optional[Any]:
        """Get state value for given key.
        
        Args:
            key: State key to retrieve
            
        Returns:
            State value if exists, None otherwise
        """
        return self._state.get(key)
    
    def set_state(self, key: str, value: Any) -> None:
        """Set state value for given key.
        
        Args:
            key: State key to set
            value: Value to store
        """
        self._state[key] = value
    
    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update multiple state values at once.
        
        Args:
            updates: Dictionary of key-value pairs to update
        """
        self._state.update(updates)
    
    def clear_state(self) -> None:
        """Clear all state values."""
        self._state.clear()
    
    def get_all_state(self) -> Dict[str, Any]:
        """Get complete state dictionary.
        
        Returns:
            Copy of complete state dictionary
        """
        return self._state.copy() 