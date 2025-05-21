from typing import Dict, Any, Optional, List

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage

from aws_architect_agent.models.base import ArchitectureState
from aws_architect_agent.utils.logging import get_logger

logger = get_logger(__name__)


class LLMStateManager:
    """Manages LLM-based state transitions."""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> None:
        """Initialize the LLM state manager.

        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature for LLM generation
            max_tokens: Maximum tokens for LLM generation
        """
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="user_input",
        )
        self.state_chains = self._setup_state_chains()

    def _setup_state_chains(self) -> Dict[ArchitectureState, LLMChain]:
        """Set up LLM chains for each state.

        Returns:
            Dictionary mapping states to LLM chains
        """
        logger.info("Setting up LLM chains for each state")
        chains = {}

        # Requirements gathering chain
        chains[ArchitectureState.REQUIREMENTS_GATHERING] = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["chat_history", "user_input"],
                template=(
                    "You are an AWS Solutions Architect helping to gather "
                    "requirements for a new architecture.\n\n"
                    "Previous conversation:\n{chat_history}\n\n"
                    "User message: {user_input}\n\n"
                    "Please analyze the requirements and respond with a clear "
                    "list. Add 'requirements_complete' at the end if all "
                    "necessary requirements are gathered."
                ),
            ),
            memory=self.memory,
            verbose=True,
        )

        # Design chain
        chains[ArchitectureState.DESIGN] = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["chat_history", "user_input"],
                template=(
                    "You are an AWS Solutions Architect designing a new "
                    "architecture based on gathered requirements.\n\n"
                    "Previous conversation:\n{chat_history}\n\n"
                    "User message: {user_input}\n\n"
                    "Please design the architecture and explain the "
                    "components. Add both 'architecture designed' and "
                    "'design_complete' at the end if the design is ready."
                ),
            ),
            memory=self.memory,
            verbose=True,
        )

        # Review chain
        chains[ArchitectureState.REVIEW] = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["chat_history", "user_input"],
                template=(
                    "You are an AWS Solutions Architect reviewing a proposed "
                    "architecture.\n\n"
                    "Previous conversation:\n{chat_history}\n\n"
                    "User message: {user_input}\n\n"
                    "Please review the architecture against AWS best "
                    "practices. Add both 'review complete' and "
                    "'review_complete' at the end if the review is done."
                ),
            ),
            memory=self.memory,
            verbose=True,
        )

        # Refinement chain
        chains[ArchitectureState.REFINEMENT] = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["chat_history", "user_input"],
                template=(
                    "You are an AWS Solutions Architect refining an "
                    "architecture based on review feedback.\n\n"
                    "Previous conversation:\n{chat_history}\n\n"
                    "User message: {user_input}\n\n"
                    "Please refine the architecture and explain the "
                    "changes. Add both 'refinement complete' and "
                    "'refinement_complete' at the end if refinement is done."
                ),
            ),
            memory=self.memory,
            verbose=True,
        )

        logger.info("LLM chains setup complete")
        return chains

    def process_message(
        self,
        state: ArchitectureState,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Process a message in the current state.

        Args:
            state: Current architecture state
            message: User message to process
            context: Additional context for the state

        Returns:
            LLM response
        """
        logger.info(f"Processing message in state {state}")
        chain = self.state_chains.get(state)
        if not chain:
            logger.error(f"No chain found for state {state}")
            raise ValueError(f"No chain found for state {state}")

        input_data = {
            "user_input": message,
            "chat_history": self.memory.chat_memory.messages,
        }

        response = chain.run(**input_data)
        logger.debug(f"Generated response: {response[:50]}...")
        return response

    def get_conversation_history(self) -> List[BaseMessage]:
        """Get the conversation history.

        Returns:
            List of conversation messages
        """
        return self.memory.chat_memory.messages

    def clear_memory(self) -> None:
        """Clear the conversation memory."""
        self.memory.clear() 