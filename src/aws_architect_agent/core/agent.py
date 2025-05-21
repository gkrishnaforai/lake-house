from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langgraph.graph import StateGraph

from aws_architect_agent.models.base import (
    Architecture,
    ArchitectureState,
    ConversationState,
    SolutionType,
)


class AWSArchitectAgent:
    """Core agent class for AWS solution architecture design."""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        """Initialize the AWS Architect Agent.

        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature for model generation
            max_tokens: Maximum tokens for model generation
        """
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Initialize chains
        self.requirements_chain = self._create_requirements_chain()
        self.design_chain = self._create_design_chain()
        self.review_chain = self._create_review_chain()
        self.refine_chain = self._create_refine_chain()
        
        self.conversation_state = ConversationState()
        self._setup_workflow()

    def _create_requirements_chain(self) -> LLMChain:
        """Create the requirements gathering chain."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are an AWS Solution Architect helping to gather "
                "requirements for a new project. Ask relevant questions to "
                "understand the project needs."
            ),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        return LLMChain(llm=self.llm, prompt=prompt)

    def _create_design_chain(self) -> LLMChain:
        """Create the architecture design chain."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are an AWS Solution Architect. Design an "
                "architecture based on the gathered requirements. Consider "
                "AWS best practices and the Well-Architected Framework."
            ),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        return LLMChain(llm=self.llm, prompt=prompt)

    def _create_review_chain(self) -> LLMChain:
        """Create the architecture review chain."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "Review the proposed architecture against AWS best "
                "practices and the Well-Architected Framework. Identify any "
                "potential issues or improvements."
            ),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        return LLMChain(llm=self.llm, prompt=prompt)

    def _create_refine_chain(self) -> LLMChain:
        """Create the architecture refinement chain."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "Refine the architecture based on the feedback and "
                "reviews. Make necessary improvements while maintaining "
                "alignment with requirements."
            ),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        return LLMChain(llm=self.llm, prompt=prompt)

    def _setup_workflow(self) -> None:
        """Set up the LangGraph workflow for the agent."""
        workflow = StateGraph(ConversationState)

        # Define nodes
        workflow.add_node("requirements_gathering", self._gather_requirements)
        workflow.add_node("design_architecture", self._design_architecture)
        workflow.add_node("review_architecture", self._review_architecture)
        workflow.add_node("refine_architecture", self._refine_architecture)

        # Define edges
        workflow.add_edge("requirements_gathering", "design_architecture")
        workflow.add_edge("design_architecture", "review_architecture")
        workflow.add_edge("review_architecture", "refine_architecture")
        workflow.add_edge("refine_architecture", "design_architecture")

        # Set entry point
        workflow.set_entry_point("requirements_gathering")

        self.workflow = workflow.compile()

    def _gather_requirements(
        self, state: ConversationState
    ) -> ConversationState:
        """Gather requirements from the user.

        Args:
            state: Current conversation state

        Returns:
            Updated conversation state
        """
        response = self.requirements_chain.run(
            input=state.conversation_history[-1]["content"]
        )
        state.conversation_history.append(
            {"role": "assistant", "content": response}
        )

        if "requirements_complete" in response.lower():
            state.current_state = ArchitectureState.DESIGN

        return state

    def _design_architecture(
        self, state: ConversationState
    ) -> ConversationState:
        """Design the architecture based on gathered requirements.

        Args:
            state: Current conversation state

        Returns:
            Updated conversation state
        """
        if not state.current_architecture:
            state.current_architecture = Architecture(
                id="arch_1",
                name="Initial Design",
                description="First iteration of the architecture",
                solution_type=SolutionType.ETL,  # Default to ETL for now
            )

        response = self.design_chain.run(
            input=str(state.requirements)
        )
        state.conversation_history.append(
            {"role": "assistant", "content": response}
        )
        state.current_state = ArchitectureState.REVIEW

        return state

    def _review_architecture(
        self, state: ConversationState
    ) -> ConversationState:
        """Review the proposed architecture.

        Args:
            state: Current conversation state

        Returns:
            Updated conversation state
        """
        response = self.review_chain.run(
            input=str(state.current_architecture)
        )
        state.conversation_history.append(
            {"role": "assistant", "content": response}
        )
        state.current_state = ArchitectureState.REFINEMENT

        return state

    def _refine_architecture(
        self, state: ConversationState
    ) -> ConversationState:
        """Refine the architecture based on feedback.

        Args:
            state: Current conversation state

        Returns:
            Updated conversation state
        """
        response = self.refine_chain.run(
            input=str(state.feedback)
        )
        state.conversation_history.append(
            {"role": "assistant", "content": response}
        )

        if "refinement_complete" in response.lower():
            state.current_state = ArchitectureState.FINALIZED
        else:
            state.current_state = ArchitectureState.DESIGN

        return state

    def process_message(self, message: str) -> str:
        """Process a user message and return the agent's response.

        Args:
            message: User's message

        Returns:
            Agent's response
        """
        self.conversation_state.conversation_history.append(
            {"role": "user", "content": message}
        )

        # Update the workflow state
        self.conversation_state = self.workflow.invoke(self.conversation_state)

        # Return the last assistant message
        return self.conversation_state.conversation_history[-1]["content"] 