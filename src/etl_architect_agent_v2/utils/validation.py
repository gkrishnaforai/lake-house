"""Utility functions for data validation and error handling."""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validator for data cleaning operations."""
    
    @staticmethod
    def validate_file_type(file_type: str) -> bool:
        """Validate file type.
        
        Args:
            file_type: Type of file to validate
            
        Returns:
            True if valid, False otherwise
        """
        return file_type in ["csv", "pdf", "image"]
    
    @staticmethod
    def validate_data_frame(
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None
    ) -> Optional[str]:
        """Validate DataFrame.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            Error message if validation fails, None otherwise
        """
        if df is None:
            return "DataFrame is None"
        if df.empty:
            return "DataFrame is empty"
        if required_columns:
            missing = set(required_columns) - set(df.columns)
            if missing:
                return f"Missing required columns: {missing}"
        return None
    
    @staticmethod
    def validate_cleaning_suggestion(
        suggestion: Dict[str, Any]
    ) -> Optional[str]:
        """Validate cleaning suggestion.
        
        Args:
            suggestion: Cleaning suggestion to validate
            
        Returns:
            Error message if validation fails, None otherwise
        """
        required_keys = {
            "strategy": str,
            "description": str,
            "columns": list,
            "parameters": dict,
            "confidence": float
        }
        
        for key, expected_type in required_keys.items():
            if key not in suggestion:
                return f"Missing required key: {key}"
            if not isinstance(suggestion[key], expected_type):
                return f"Invalid type for {key}"
        
        if not (0 <= suggestion["confidence"] <= 1):
            return "Confidence must be between 0 and 1"
        
        return None
    
    @staticmethod
    def validate_cleaning_results(
        original_df: pd.DataFrame,
        cleaned_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Validate cleaning results.
        
        Args:
            original_df: Original DataFrame
            cleaned_df: Cleaned DataFrame
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "is_valid": True,
            "issues": [],
            "metrics": {}
        }
        
        # Check row count
        if len(cleaned_df) != len(original_df):
            results["is_valid"] = False
            results["issues"].append(
                "Row count mismatch after cleaning"
            )
        
        # Check column presence
        if set(cleaned_df.columns) != set(original_df.columns):
            results["is_valid"] = False
            results["issues"].append(
                "Column mismatch after cleaning"
            )
        
        # Calculate metrics
        results["metrics"] = {
            "null_counts": cleaned_df.isnull().sum().to_dict(),
            "unique_counts": cleaned_df.nunique().to_dict()
        }
        
        return results
    
    @staticmethod
    def log_validation_error(
        error_msg: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log validation error with context.
        
        Args:
            error_msg: Error message to log
            context: Additional context for the error
        """
        if context:
            logger.error(
                f"{error_msg} Context: {context}"
            )
        else:
            logger.error(error_msg) 