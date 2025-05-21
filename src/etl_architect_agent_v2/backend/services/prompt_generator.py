"""Prompt generator for transformation tools."""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class PromptGenerator:
    """Generates prompts for transformation tools."""
    
    def __init__(self, tool_config: Dict[str, Any]):
        """Initialize with tool configuration."""
        self.tool_config = tool_config
        
    def generate_prompt(self, text: str) -> str:
        """Generate prompt from template with proper formatting."""
        try:
            return self.tool_config["prompt_template"].format(
                text=text,
                classification_options=" or ".join(self.tool_config["classification_options"])
            )
        except Exception as e:
            logger.error(f"Error generating prompt: {str(e)}")
            raise ValueError(f"Error generating prompt: {str(e)}") 