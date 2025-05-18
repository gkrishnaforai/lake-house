"""Example agent implementation for the ETL Architect Agent V2."""

from typing import Dict, Any, Optional
from ..core.state_manager import AgentState, AgentRole
from ..core.llm_manager import LLMManager
from ..core.prompt_manager import PromptManager
from ..core.error_handler import ErrorHandler
import logging

logger = logging.getLogger(__name__)

class ExampleAgent:
    """Example agent implementation showing how to create a specialized agent."""
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        prompt_manager: Optional[PromptManager] = None,
        error_handler: Optional[ErrorHandler] = None
    ):
        """Initialize the example agent.
        
        Args:
            llm_manager: LLM manager instance
            prompt_manager: Prompt manager instance
            error_handler: Error handler instance
        """
        self.llm_manager = llm_manager or LLMManager()
        self.prompt_manager = prompt_manager or PromptManager()
        self.error_handler = error_handler or ErrorHandler()
    
    async def process(self, state: AgentState) -> AgentState:
        """Process the current state and return updated state.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        try:
            # Example processing logic
            logger.info("Processing state in ExampleAgent")
            
            # Update state as needed
            state.current_agent = AgentRole.REQUIREMENTS
            
            return state
            
        except Exception as e:
            self.error_handler.handle_error(e)
            raise 