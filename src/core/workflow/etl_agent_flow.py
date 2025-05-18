from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from langgraph.graph import Graph, StateGraph, END
from src.core.workflow.etl_state import (
    ETLWorkflowState,
    ETLRequirements,
    ETLArchitecture
)
from src.core.llm.manager import LLMManager
from src.core.state_management.state_manager import StateManager
from src.core.workflow.etl_utils import ETLUtils
import logging

logger = logging.getLogger(__name__)

@dataclass
class AgentState:
    """State for the ETL agent flow."""
    current_state: ETLWorkflowState
    requirements: Optional[ETLRequirements] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    next_action: str = "analyze_requirements"
    error: str = ""
    is_done: bool = False

class ETLAgentFlow:
    """Manages the ETL agent workflow using LangGraph."""

    def __init__(self, llm_manager: LLMManager, state_manager: StateManager):
        self.llm_manager = llm_manager
        self.state_manager = state_manager
        self.etl_utils = ETLUtils(llm_manager)
        self.graph = self._build_graph()

    def _build_graph(self) -> Graph:
        """Build the LangGraph for ETL workflow."""
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("analyze_requirements", self._analyze_requirements)
        graph.add_node("generate_architecture", self._generate_architecture)
        graph.add_node("handle_error", self._handle_error)

        # Add conditional edges
        def should_continue_analysis(x: AgentState) -> str:
            if x.error:
                return "handle_error"
            if x.is_done:
                return END
            if x.requirements and not x.requirements.missing_info:
                return "generate_architecture"
            return "analyze_requirements"

        graph.add_conditional_edges(
            "analyze_requirements",
            should_continue_analysis,
            {
                "analyze_requirements": "analyze_requirements",
                "generate_architecture": "generate_architecture",
                "handle_error": "handle_error",
                END: END
            }
        )
        
        # Add edge from generate_architecture to END
        graph.add_edge("generate_architecture", END)
        
        # Add edge from handle_error to analyze_requirements
        graph.add_edge("handle_error", "analyze_requirements")

        # Set entry point
        graph.set_entry_point("analyze_requirements")

        # Compile graph
        return graph.compile()

    async def _analyze_requirements(
        self,
        state: AgentState
    ) -> AgentState:
        """Analyze project requirements."""
        try:
            # Update conversation history with latest message
            if state.current_state.metadata.get("latest_message"):
                state.conversation_history.append({
                    "role": "user",
                    "content": state.current_state.metadata["latest_message"]
                })
            
            # Update state with conversation history
            state.current_state.metadata["conversation_history"] = state.conversation_history
            
            # Analyze requirements
            requirements = await self.etl_utils.analyze_project(
                state.current_state
            )
            
            # Add agent's response to conversation history
            if requirements.next_question:
                state.conversation_history.append({
                    "role": "assistant",
                    "content": requirements.next_question
                })
            
            # Check if we should end the conversation
            is_done = bool(requirements and not requirements.missing_info)
            
            # Update state with requirements
            state.current_state.metadata["requirements"] = requirements.model_dump()
            
            # Create new state with updated requirements
            new_state = AgentState(
                current_state=state.current_state,
                requirements=requirements,
                conversation_history=state.conversation_history,
                next_action="generate_architecture" if is_done else "analyze_requirements",
                error="",
                is_done=is_done
            )
            
            # Save the state
            await self.state_manager.save_state(new_state.current_state)
            
            return new_state
            
        except Exception as e:
            logger.error(f"Error analyzing requirements: {str(e)}")
            return AgentState(
                current_state=state.current_state,
                requirements=state.requirements,
                conversation_history=state.conversation_history,
                next_action="handle_error",
                error=str(e),
                is_done=False
            )

    async def _generate_architecture(
        self,
        state: AgentState
    ) -> AgentState:
        """Generate ETL architecture based on requirements."""
        try:
            if not state.requirements:
                raise ValueError("Requirements not found in state")
            
            architecture = await self.etl_utils.generate_architecture(state.requirements)
            
            # Update state with architecture
            state.current_state.metadata["architecture"] = architecture.model_dump()
            
            return AgentState(
                current_state=state.current_state,
                requirements=state.requirements,
                conversation_history=state.conversation_history,
                next_action=END,
                error="",
                is_done=True
            )
        except Exception as e:
            logger.error(f"Error generating architecture: {str(e)}")
            return AgentState(
                current_state=state.current_state,
                requirements=state.requirements,
                conversation_history=state.conversation_history,
                next_action="handle_error",
                error=str(e),
                is_done=False
            )

    async def _handle_error(
        self,
        state: AgentState
    ) -> AgentState:
        """Handle errors in the workflow."""
        error = state.error
        logger.error(f"Workflow error: {error}")
        
        return AgentState(
            current_state=state.current_state,
            requirements=state.requirements,
            conversation_history=state.conversation_history,
            next_action="analyze_requirements",
            error="",
            is_done=False
        )

    async def run(self, workflow_id: str, user_request: str) -> ETLWorkflowState:
        """Run the ETL workflow."""
        try:
            # Initialize state with user request
            initial_state = AgentState(
                current_state=ETLWorkflowState(
                    workflow_id=workflow_id,
                    metadata={
                        "latest_message": user_request,
                        "conversation_history": [
                            {"role": "user", "content": user_request}
                        ]
                    }
                ),
                requirements=None,
                conversation_history=[
                    {"role": "user", "content": user_request}
                ],
                next_action="analyze_requirements",
                error="",
                is_done=False
            )

            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)
            
            # Handle the final state
            if isinstance(final_state, AgentState):
                # If it's already an AgentState, use it directly
                final_workflow_state = final_state.current_state
            else:
                # Convert dictionary to AgentState
                requirements = None
                if final_state.get("requirements"):
                    try:
                        requirements = ETLRequirements(**final_state["requirements"])
                    except Exception as e:
                        logger.error(f"Error parsing requirements from final state: {str(e)}")
                
                # Create new AgentState
                agent_state = AgentState(
                    current_state=initial_state.current_state,
                    requirements=requirements,
                    conversation_history=final_state.get("conversation_history", []),
                    next_action=final_state.get("next_action", "analyze_requirements"),
                    error=final_state.get("error", ""),
                    is_done=final_state.get("is_done", False)
                )
                final_workflow_state = agent_state.current_state
            
            # Update state with requirements if available
            if final_workflow_state.metadata.get("requirements"):
                try:
                    requirements = ETLRequirements(**final_workflow_state.metadata["requirements"])
                    final_workflow_state.requirements = requirements
                except Exception as e:
                    logger.error(f"Error setting requirements in final state: {str(e)}")
            
            # Save final state
            await self.state_manager.save_state(final_workflow_state)
            
            return final_workflow_state
        except Exception as e:
            logger.error(f"Error running workflow: {str(e)}")
            raise 