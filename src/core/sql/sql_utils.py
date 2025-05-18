from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from src.core.sql.sql_state import (
    SQLGenerationState,
    SQLRequirements,
    SQLGenerationOutput,
    SQLGenerationError
)
from src.core.llm.manager import LLMManager
import json
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    """Tracks the context of the current conversation."""
    gathered_requirements: Dict[str, Any] = field(default_factory=dict)
    missing_info: List[str] = field(default_factory=list)
    current_focus: str = "query_analysis"  # query_analysis, schema_validation, query_generation
    conversation_history: List[Dict[str, str]] = field(default_factory=list)


class SQLUtils:
    """Manages SQL generation utilities."""

    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.conversation_context = ConversationContext()

    async def analyze_requirements(
        self,
        state: SQLGenerationState
    ) -> SQLRequirements:
        """Analyze SQL generation requirements."""
        try:
            # Create prompt for requirements analysis
            prompt = self._create_requirements_prompt(state)
            
            # Prepare LLM input as a message list
            llm_input = {"messages": [{"role": "user", "content": prompt}]}
            response = await self.llm_manager.ainvoke(llm_input)
            
            # Parse response
            try:
                # Clean the response to ensure valid JSON
                cleaned_response = re.sub(r'```json\n|\n```', '', response["content"])
                requirements_data = json.loads(cleaned_response)
                
                # Create SQLRequirements object
                return SQLRequirements(**requirements_data)
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing requirements response: {str(e)}")
                raise SQLGenerationError("Failed to parse requirements response")
                
        except Exception as e:
            logger.error(f"Error in analyze_requirements: {str(e)}")
            raise SQLGenerationError(f"Error analyzing requirements: {str(e)}")

    async def validate_schema(
        self,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate database schema."""
        try:
            # Create prompt for schema validation
            prompt = self._create_schema_validation_prompt(schema)
            
            # Get LLM response
            response = await self.llm_manager.invoke(prompt, {})
            
            # Parse response
            try:
                # Clean the response to ensure valid JSON
                cleaned_response = re.sub(r'```json\n|\n```', '', response)
                validation_result = json.loads(cleaned_response)
                
                return validation_result
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing validation response: {str(e)}")
                raise SQLGenerationError("Failed to parse validation response")
                
        except Exception as e:
            logger.error(f"Error in validate_schema: {str(e)}")
            raise SQLGenerationError(f"Error validating schema: {str(e)}")

    async def generate_query(
        self,
        requirements: SQLRequirements
    ) -> SQLGenerationOutput:
        """Generate SQL query based on requirements."""
        try:
            # Create prompt for query generation
            prompt = self._create_query_generation_prompt(requirements)
            
            # Get LLM response
            response = await self.llm_manager.invoke(prompt, {})
            
            # Parse response
            try:
                # Clean the response to ensure valid JSON
                cleaned_response = re.sub(r'```json\n|\n```', '', response)
                query_result = json.loads(cleaned_response)
                
                # Create SQLGenerationOutput object
                return SQLGenerationOutput(**query_result)
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing query response: {str(e)}")
                raise SQLGenerationError("Failed to parse query response")
                
        except Exception as e:
            logger.error(f"Error in generate_query: {str(e)}")
            raise SQLGenerationError(f"Error generating query: {str(e)}")

    async def review_query(
        self,
        query_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Review generated SQL query."""
        try:
            # Create prompt for query review
            prompt = self._create_query_review_prompt(query_result)
            
            # Get LLM response
            response = await self.llm_manager.invoke(prompt, {})
            
            # Parse response
            try:
                # Clean the response to ensure valid JSON
                cleaned_response = re.sub(r'```json\n|\n```', '', response)
                review_result = json.loads(cleaned_response)
                
                return review_result
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing review response: {str(e)}")
                raise SQLGenerationError("Failed to parse review response")
                
        except Exception as e:
            logger.error(f"Error in review_query: {str(e)}")
            raise SQLGenerationError(f"Error reviewing query: {str(e)}")

    async def correct_query(
        self,
        query_result: Dict[str, Any],
        review_result: Dict[str, Any]
    ) -> SQLGenerationOutput:
        """Correct SQL query based on review feedback."""
        try:
            # Create prompt for query correction
            prompt = self._create_query_correction_prompt(
                query_result,
                review_result
            )
            
            # Get LLM response
            response = await self.llm_manager.invoke(prompt, {})
            
            # Parse response
            try:
                # Clean the response to ensure valid JSON
                cleaned_response = re.sub(r'```json\n|\n```', '', response)
                corrected_result = json.loads(cleaned_response)
                
                # Create SQLGenerationOutput object
                return SQLGenerationOutput(**corrected_result)
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing correction response: {str(e)}")
                raise SQLGenerationError("Failed to parse correction response")
                
        except Exception as e:
            logger.error(f"Error in correct_query: {str(e)}")
            raise SQLGenerationError(f"Error correcting query: {str(e)}")

    def _create_requirements_prompt(
        self,
        state: SQLGenerationState
    ) -> str:
        """Create prompt for requirements analysis."""
        return f"""
        Analyze the following SQL generation request:
        
        User Request: {state.metadata.get('latest_message', '')}
        
        Previous Conversation:
        {json.dumps(state.metadata.get('conversation_history', []), indent=2)}
        
        Please provide a JSON response with:
        {{
            "query": "the user's query",
            "schema": {{
                "tables": [
                    {{
                        "name": "table_name",
                        "columns": [
                            {{
                                "name": "column_name",
                                "type": "data_type"
                            }}
                        ]
                    }}
                ]
            }},
            "constraints": {{
                "required_tables": ["table_names"],
                "required_columns": ["column_names"],
                "conditions": ["WHERE conditions"]
            }},
            "missing_info": ["questions to clarify"],
            "next_question": "most relevant next question"
        }}
        """

    def _create_schema_validation_prompt(
        self,
        schema: Dict[str, Any]
    ) -> str:
        """Create prompt for schema validation."""
        return f"""
        Validate the following database schema:
        
        {json.dumps(schema, indent=2)}
        
        Please provide a JSON response with:
        {{
            "is_valid": true/false,
            "validation_errors": ["error messages"],
            "suggestions": ["improvement suggestions"],
            "missing_tables": ["required tables"],
            "missing_columns": ["required columns"]
        }}
        """

    def _create_query_generation_prompt(
        self,
        requirements: SQLRequirements
    ) -> str:
        """Create prompt for query generation."""
        return f"""
        Generate a SQL query based on the following requirements:
        
        Query: {requirements.query}
        Schema: {json.dumps(requirements.schema, indent=2)}
        Constraints: {json.dumps(requirements.constraints, indent=2)}
        
        Please provide a JSON response with:
        {{
            "sql_query": "the generated SQL query",
            "confidence": 0.0-1.0,
            "tables_used": ["table names"],
            "columns_used": ["column names"],
            "filters": {{
                "column": "value"
            }},
            "explanation": "explanation of the query"
        }}
        """

    def _create_query_review_prompt(
        self,
        query_result: Dict[str, Any]
    ) -> str:
        """Create prompt for query review."""
        return f"""
        Review the following SQL query:
        
        {json.dumps(query_result, indent=2)}
        
        Please provide a JSON response with:
        {{
            "is_valid": true/false,
            "feedback": "general feedback",
            "suggestions": ["improvement suggestions"],
            "potential_issues": ["potential problems"],
            "optimization_opportunities": ["optimization suggestions"]
        }}
        """

    def _create_query_correction_prompt(
        self,
        query_result: Dict[str, Any],
        review_result: Dict[str, Any]
    ) -> str:
        """Create prompt for query correction."""
        return f"""
        Correct the following SQL query based on the review feedback:
        
        Original Query:
        {json.dumps(query_result, indent=2)}
        
        Review Feedback:
        {json.dumps(review_result, indent=2)}
        
        Please provide a JSON response with:
        {{
            "sql_query": "the corrected SQL query",
            "confidence": 0.0-1.0,
            "tables_used": ["table names"],
            "columns_used": ["column names"],
            "filters": {{
                "column": "value"
            }},
            "explanation": "explanation of the corrections"
        }}
        """ 