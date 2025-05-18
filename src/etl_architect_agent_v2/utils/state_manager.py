"""Utility functions for managing LangGraph state."""

from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
from langgraph.graph import StateGraph


class StateManager:
    """Manager for handling LangGraph state transitions."""
    
    @staticmethod
    def create_initial_state(
        input_file: Path,
        file_type: str
    ) -> Dict[str, Any]:
        """Create initial state for the workflow.
        
        Args:
            input_file: Path to input file
            file_type: Type of file (csv, pdf, image)
            
        Returns:
            Initial state dictionary
        """
        return {
            "input_file": str(input_file),
            "file_type": file_type,
            "current_data": None,
            "cleaning_suggestions": [],
            "applied_cleaning": [],
            "conversation_history": [],
            "error": None,
            "output_file": None,
            "schema_analysis": None,
            "user_feedback": None
        }
    
    @staticmethod
    def update_state_safely(
        state: Dict[str, Any],
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update state dictionary safely.
        
        Args:
            state: Current state dictionary
            updates: Dictionary of updates to apply
            
        Returns:
            Updated state dictionary
        """
        new_state = state.copy()
        for key, value in updates.items():
            if key in new_state:
                new_state[key] = value
        return new_state
    
    @staticmethod
    def get_error_state(
        state: Dict[str, Any],
        error: str
    ) -> Dict[str, Any]:
        """Create error state.
        
        Args:
            state: Current state dictionary
            error: Error message
            
        Returns:
            State dictionary with error
        """
        return StateManager.update_state_safely(
            state,
            {"error": error}
        )
    
    @staticmethod
    def validate_state(state: Dict[str, Any]) -> Optional[str]:
        """Validate state dictionary.
        
        Args:
            state: State dictionary to validate
            
        Returns:
            Error message if validation fails, None otherwise
        """
        required_keys = {
            "input_file": str,
            "file_type": str,
            "current_data": (pd.DataFrame, type(None)),
            "cleaning_suggestions": list,
            "applied_cleaning": list,
            "conversation_history": list,
            "error": (str, type(None)),
            "output_file": (str, type(None)),
            "schema_analysis": (dict, type(None)),
            "user_feedback": (str, type(None))
        }
        
        for key, expected_type in required_keys.items():
            if key not in state:
                return f"Missing required key: {key}"
            if not isinstance(state[key], expected_type):
                return f"Invalid type for {key}: {type(state[key])}"
        
        return None 