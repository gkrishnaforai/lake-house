from typing import Dict, Any, Optional
from aws_architect_agent.graph.state.state_manager import StateManager
from aws_architect_agent.core.llm_workflow_manager import LLMWorkflowManager
from aws_architect_agent.models.base import ArchitectureState
from aws_architect_agent.utils.logging import get_logger

logger = get_logger(__name__)


class WorkflowManager:
    """Manages the workflow for architecture design."""

    def __init__(
        self,
        state_manager: StateManager,
        product_manager_model: str = "gpt-3.5-turbo",
        data_architect_model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> None:
        """Initialize the workflow manager.

        Args:
            state_manager: State manager instance
            product_manager_model: Model for Product Manager agent
            data_architect_model: Model for Data Architect agent
            temperature: Temperature for LLM generation
            max_tokens: Maximum tokens for LLM generation
        """
        self.state_manager = state_manager
        self.llm_workflow = LLMWorkflowManager(
            state_manager=state_manager,
            product_manager_model=product_manager_model,
            data_architect_model=data_architect_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _transition_state(self, response: str) -> None:
        """Transition to next state based on response markers.

        Args:
            response: LLM response to check for markers
        """
        current_state = self.state_manager.get_current_state()
        response_lower = response.lower()

        if current_state == ArchitectureState.INITIAL:
            self.state_manager.update_state(
                ArchitectureState.REQUIREMENTS_GATHERING
            )
        elif (
            current_state == ArchitectureState.REQUIREMENTS_GATHERING
            and "requirements_analyzed" in response_lower
        ):
            self.state_manager.update_state(ArchitectureState.DESIGN)
        elif (
            current_state == ArchitectureState.DESIGN
            and "architecture_designed" in response_lower
        ):
            self.state_manager.update_state(ArchitectureState.REVIEW)
        elif (
            current_state == ArchitectureState.REVIEW
            and "architecture_reviewed" in response_lower
        ):
            self.state_manager.update_state(ArchitectureState.REFINEMENT)
        elif (
            current_state == ArchitectureState.REFINEMENT
            and "architecture_refined" in response_lower
        ):
            self.state_manager.update_state(ArchitectureState.FINALIZED)

    def process_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Process a message based on current state.

        Args:
            message: User message
            context: Additional context

        Returns:
            LLM response
        """
        current_state = self.state_manager.get_current_state()
        logger.info(f"Processing message in state: {current_state.value}")

        # Handle initial state
        if current_state == ArchitectureState.INITIAL:
            self.state_manager.update_state(
                ArchitectureState.REQUIREMENTS_GATHERING
            )
            current_state = ArchitectureState.REQUIREMENTS_GATHERING

        # Process message based on current state
        if current_state == ArchitectureState.REQUIREMENTS_GATHERING:
            response = self.llm_workflow.process_requirements_gathering(
                message=message,
                context=context,
            )
        elif current_state == ArchitectureState.DESIGN:
            response = self.llm_workflow.process_design(
                message=message,
                context=context,
            )
        elif current_state == ArchitectureState.REVIEW:
            response = self.llm_workflow.process_review(
                message=message,
                context=context,
            )
        elif current_state == ArchitectureState.REFINEMENT:
            response = self.llm_workflow.process_refinement(
                message=message,
                context=context,
            )
        else:
            response = "Invalid state"

        # Update state based on response
        self._transition_state(response)

        return response 