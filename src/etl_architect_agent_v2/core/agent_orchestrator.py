"""Agent orchestration for the ETL Architect Agent V2."""

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from .state_manager import (
    AgentState,
    AgentRole,
    Message,
    MessageRole,
    ConversationContext,
    RequirementsState,
    ArchitectureState,
    ValidationState,
    ImplementationState,
    MemoryState,
    ToolState
)
from .error_handler import ErrorHandler, AgentError
from .llm_manager import LLMManager
from .prompt_manager import PromptManager
import logging

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Coordinates the workflow between different specialized agents."""
    
    def __init__(
        self,
        llm_manager: LLMManager,
        prompt_manager: PromptManager,
        error_handler: ErrorHandler
    ):
        """Initialize the agent orchestrator.
        
        Args:
            llm_manager: LLM manager instance
            prompt_manager: Prompt manager instance
            error_handler: Error handler instance
        """
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager
        self.error_handler = error_handler
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the agent workflow graph.
        
        Returns:
            Compiled LangGraph
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node(
            "analyze_requirements",
            self._analyze_requirements
        )
        workflow.add_node(
            "generate_architecture",
            self._generate_architecture
        )
        workflow.add_node(
            "validate_architecture",
            self._validate_architecture
        )
        workflow.add_node(
            "implement_architecture",
            self._implement_architecture
        )
        workflow.add_node(
            "handle_error",
            self._handle_error
        )
        
        # Add edges
        workflow.add_edge("analyze_requirements", "generate_architecture")
        workflow.add_edge("generate_architecture", "validate_architecture")
        workflow.add_edge("validate_architecture", "implement_architecture")
        workflow.add_edge("implement_architecture", END)
        workflow.add_edge("handle_error", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_requirements")
        
        return workflow.compile()
    
    async def _analyze_requirements(self, state: AgentState) -> AgentState:
        """Analyze requirements using LLM.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        try:
            # Get the latest message
            latest_message = state.conversation.messages[-1]
            
            # Generate response
            response = await self.llm_manager.generate_response(
                messages=[
                    HumanMessage(
                        content=latest_message.content
                    )
                ],
                tools=[],
                tool_choice="auto"
            )
            
            # Update state
            state.conversation.messages.append(
                Message(
                    role=MessageRole.AGENT,
                    content=response.content
                )
            )
            
            return state
            
        except Exception as e:
            return await self._handle_error(state, e)
    
    async def _generate_architecture(self, state: AgentState) -> AgentState:
        """Generate architecture based on requirements.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        try:
            # Get the latest message
            latest_message = state.conversation.messages[-1]
            
            # Generate response
            response = await self.llm_manager.generate_response(
                messages=[
                    HumanMessage(
                        content=latest_message.content
                    )
                ],
                tools=[],
                tool_choice="auto"
            )
            
            # Update state
            state.conversation.messages.append(
                Message(
                    role=MessageRole.AGENT,
                    content=response.content
                )
            )
            
            return state
            
        except Exception as e:
            return await self._handle_error(state, e)
    
    async def _validate_architecture(self, state: AgentState) -> AgentState:
        """Validate the generated architecture.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        try:
            # Get the latest message
            latest_message = state.conversation.messages[-1]
            
            # Generate response
            response = await self.llm_manager.generate_response(
                messages=[
                    HumanMessage(
                        content=latest_message.content
                    )
                ],
                tools=[],
                tool_choice="auto"
            )
            
            # Update state
            state.conversation.messages.append(
                Message(
                    role=MessageRole.AGENT,
                    content=response.content
                )
            )
            
            return state
            
        except Exception as e:
            return await self._handle_error(state, e)
    
    async def _implement_architecture(self, state: AgentState) -> AgentState:
        """Implement the validated architecture.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        try:
            # Get the latest message
            latest_message = state.conversation.messages[-1]
            
            # Generate response
            response = await self.llm_manager.generate_response(
                messages=[
                    HumanMessage(
                        content=latest_message.content
                    )
                ],
                tools=[],
                tool_choice="auto"
            )
            
            # Update state
            state.conversation.messages.append(
                Message(
                    role=MessageRole.AGENT,
                    content=response.content
                )
            )
            
            return state
            
        except Exception as e:
            return await self._handle_error(state, e)
    
    async def _handle_error(self, state: AgentState, error: Exception) -> AgentState:
        """Handle errors in the workflow.
        
        Args:
            state: Current agent state
            error: Exception object
            
        Returns:
            Updated agent state
        """
        error_message = str(error)
        state.conversation.messages.append(
            Message(
                role=MessageRole.AGENT,
                content=f"Error: {error_message}"
            )
        )
        return state
    
    async def process_message(self, message: str) -> AgentState:
        """Process a new message through the workflow.
        
        Args:
            message: New message to process
            
        Returns:
            Final agent state
        """
        try:
            # Initialize state
            state = AgentState(
                conversation=ConversationContext(
                    messages=[
                        Message(
                            role=MessageRole.USER,
                            content=message
                        )
                    ]
                ),
                requirements=RequirementsState(),
                architecture=ArchitectureState(),
                validation=ValidationState(),
                implementation=ImplementationState(),
                memory=MemoryState(),
                tools=ToolState(),
                current_agent=AgentRole.REQUIREMENTS
            )
            
            # Run the graph
            final_state = await self.graph.arun(state)
            return final_state
            
        except Exception as e:
            self.error_handler.handle_error(e)
            raise AgentError(
                f"Error processing message: {e}"
            ) from e 