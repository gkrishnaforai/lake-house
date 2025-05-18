from typing import Dict, List, Optional

from aws_architect_agent.models.base import (
    Architecture,
    ArchitectureState,
    SolutionType,
)
from aws_architect_agent.utils.logging import get_logger

logger = get_logger(__name__)


class StateManager:
    """Manages the state transitions and state data for the agent."""

    def __init__(self) -> None:
        """Initialize the state manager."""
        self.conversation_state = {
            "current_architecture": None,
            "conversation_history": [],
            "current_state": ArchitectureState.INITIAL,
            "requirements": [],
            "feedback": [],
        }

    def update_state(self, new_state: ArchitectureState) -> None:
        """Update the current state.

        Args:
            new_state: New state to transition to
        """
        logger.info(
            f"Transitioning from {self.conversation_state['current_state']} "
            f"to {new_state}"
        )
        self.conversation_state["current_state"] = new_state

    def add_feedback(self, feedback: str) -> None:
        """Add feedback to the state.

        Args:
            feedback: Feedback to add
        """
        self.conversation_state["feedback"].append(feedback)
        logger.info(f"Added feedback: {feedback}")

    def update_requirements(self, requirements: List[str]) -> None:
        """Update requirements in the state.

        Args:
            requirements: Requirements to update
        """
        self.conversation_state["requirements"] = requirements
        logger.info(f"Updated requirements: {requirements}")

    def initialize_architecture(
        self,
        name: str,
        description: str,
        solution_type: SolutionType,
    ) -> None:
        """Initialize a new architecture.

        Args:
            name: Name of the architecture
            description: Description of the architecture
            solution_type: Type of solution
        """
        self.conversation_state["current_architecture"] = Architecture(
            id="arch_1",
            name=name,
            description=description,
            solution_type=solution_type,
        )
        logger.info(
            f"Initialized architecture: {name} ({solution_type})"
        )

    def get_current_state(self) -> ArchitectureState:
        """Get the current state.

        Returns:
            Current state
        """
        return self.conversation_state["current_state"]

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history.

        Returns:
            Conversation history
        """
        return self.conversation_state["conversation_history"]

    def add_to_history(self, role: str, content: str) -> None:
        """Add a message to the conversation history.

        Args:
            role: Role of the message sender
            content: Content of the message
        """
        self.conversation_state["conversation_history"].append(
            {"role": role, "content": content}
        )
        logger.info(f"Added message to history: {role} - {content[:50]}...")

    def get_current_architecture(self) -> Optional[Architecture]:
        """Get the current architecture.

        Returns:
            Current architecture
        """
        return self.conversation_state["current_architecture"] 