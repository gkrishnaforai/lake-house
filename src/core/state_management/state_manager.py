from typing import Dict, Any, Optional
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class StateManager:
    """Manages the persistence and retrieval of workflow states."""
    
    def __init__(self, storage_path: str = "workflow_states"):
        """Initialize the state manager.
        
        Args:
            storage_path: Path to store workflow states
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._active_states: Dict[str, Any] = {}
    
    async def save_state(self, state: Any) -> None:
        """Save a workflow state.
        
        Args:
            state: The state to save
        """
        try:
            # Save to memory
            self._active_states[state.workflow_id] = state
            
            # Save to disk
            state_path = self.storage_path / f"{state.workflow_id}.json"
            with open(state_path, "w") as f:
                json.dump(state.dict(), f, indent=2)
                
            logger.info(f"Saved state for workflow {state.workflow_id}")
            
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
            raise
    
    async def load_state(self, workflow_id: str) -> Optional[Any]:
        """Load a workflow state.
        
        Args:
            workflow_id: ID of the workflow to load
            
        Returns:
            The loaded state or None if not found
        """
        try:
            # Check memory first
            if workflow_id in self._active_states:
                return self._active_states[workflow_id]
            
            # Load from disk
            state_path = self.storage_path / f"{workflow_id}.json"
            if not state_path.exists():
                return None
                
            with open(state_path, "r") as f:
                state_data = json.load(f)
                
            logger.info(f"Loaded state for workflow {workflow_id}")
            return state_data
            
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            return None
    
    async def delete_state(self, workflow_id: str) -> None:
        """Delete a workflow state.
        
        Args:
            workflow_id: ID of the workflow to delete
        """
        try:
            # Remove from memory
            self._active_states.pop(workflow_id, None)
            
            # Remove from disk
            state_path = self.storage_path / f"{workflow_id}.json"
            if state_path.exists():
                state_path.unlink()
                
            logger.info(f"Deleted state for workflow {workflow_id}")
            
        except Exception as e:
            logger.error(f"Error deleting state: {str(e)}")
            raise
    
    async def get_state(self, workflow_id: str) -> Optional[Any]:
        """Alias for load_state for compatibility with tests."""
        return await self.load_state(workflow_id) 