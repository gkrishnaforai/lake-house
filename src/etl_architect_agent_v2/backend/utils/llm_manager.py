"""Manager for handling LLM operations."""

import json
import logging
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMManager:
    """Manager for handling LLM operations."""

    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0
    ):
        """Initialize the LLM manager.
        
        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature setting for the model
        """
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )

    async def get_structured_response(
        self,
        prompt: str,
        required_fields: List[str],
        expected_format: str = "json"
    ) -> Dict[str, Any]:
        """Get a structured response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            required_fields: List of fields that must be present in response
            expected_format: Expected format of the response (default: json)
            
        Returns:
            Dict containing the structured response
            
        Raises:
            ValueError: If response is invalid or missing required fields
        """
        try:
            response = await self.llm.ainvoke(prompt)
            if expected_format == "json":
                result = json.loads(response.content)
            else:
                result = response.content
                
            self._validate_response(result, required_fields)
            return result
            
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            raise ValueError(f"Error calling LLM: {str(e)}")

    def _validate_response(
        self,
        response: Dict[str, Any],
        required_fields: List[str]
    ) -> None:
        """Validate the LLM response has all required fields.
        
        Args:
            response: The response to validate
            required_fields: List of fields that must be present
            
        Raises:
            ValueError: If any required fields are missing
        """
        missing_fields = [
            field for field in required_fields if field not in response
        ]
        if missing_fields:
            raise ValueError(
                f"LLM response missing required fields: {missing_fields}"
            ) 