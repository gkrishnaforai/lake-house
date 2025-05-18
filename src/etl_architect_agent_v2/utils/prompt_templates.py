"""Utility functions for handling prompt templates."""

from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


class PromptTemplateManager:
    """Manager for handling prompt templates with proper JSON escaping."""
    
    @staticmethod
    def create_json_template(fields: Dict[str, Any]) -> str:
        """Create a JSON template with proper escaping.
        
        Args:
            fields: Dictionary of field names and their example values
            
        Returns:
            Properly escaped JSON template string
        """
        json_str = ["{{"]
        for key, value in fields.items():
            if isinstance(value, dict):
                json_str.append(f'  "{key}": {{{value}}},')
            elif isinstance(value, list):
                json_str.append(f'  "{key}": {value},')
            else:
                json_str.append(f'  "{key}": "{value}",')
        json_str.append("}}")
        return "\n".join(json_str)
    
    @staticmethod
    def create_schema_template() -> ChatPromptTemplate:
        """Create template for schema analysis."""
        fields = {
            "data_types": {"column": "type"},
            "missing_patterns": {"column": "pattern"},
            "quality_issues": ["list of issues"],
            "relationships": ["list of relationships"]
        }
        
        prompt = (
            "Analyze the following data schema and provide insights:\n"
            "1. Data types and formats\n"
            "2. Missing value patterns\n"
            "3. Potential data quality issues\n"
            "4. Column relationships\n\n"
            f"Return analysis in JSON format:\n{PromptTemplateManager.create_json_template(fields)}\n\n"
            "Data info:\n{data_info}\n\n"
            "Sample data:\n{data_head}"
        )
        return ChatPromptTemplate.from_template(prompt)
    
    @staticmethod
    def create_cleaning_template() -> ChatPromptTemplate:
        """Create template for cleaning suggestions."""
        suggestion = {
            "strategy": "strategy_name",
            "description": "description",
            "columns": ["affected_columns"],
            "parameters": {"param1": "value1"},
            "confidence": 0.95
        }
        fields = {"suggestions": [suggestion]}
        
        prompt = (
            "Based on schema analysis and feedback, suggest cleaning steps.\n"
            "Consider:\n"
            "1. Data types and formats\n"
            "2. Missing values\n"
            "3. Outliers\n"
            "4. Inconsistent formatting\n"
            "5. Data validation rules\n\n"
            f"Return suggestions in JSON format:\n{PromptTemplateManager.create_json_template(fields)}\n\n"
            "Schema analysis:\n{schema_analysis}\n\n"
            "User feedback:\n{user_feedback}\n\n"
            "Data sample:\n{data_head}"
        )
        return ChatPromptTemplate.from_template(prompt)
    
    @staticmethod
    def create_validation_template() -> ChatPromptTemplate:
        """Create template for validation results."""
        fields = {
            "is_valid": True,
            "remaining_issues": ["list of issues"],
            "suggestions": ["list of suggestions"]
        }
        
        prompt = (
            "Review the cleaning results and validate data quality.\n"
            f"Return validation results in JSON format:\n{PromptTemplateManager.create_json_template(fields)}\n\n"
            "Data sample:\n{data_head}\n\n"
            "Applied cleaning steps:\n{applied_cleaning}"
        )
        return ChatPromptTemplate.from_template(prompt) 