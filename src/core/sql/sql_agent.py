from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from langgraph.graph import Graph, StateGraph, END
from src.core.sql.sql_state import (
    SQLGenerationState,
    SQLRequirements,
    SQLGenerationStep
)
from src.core.llm.manager import LLMManager
from src.core.state_management.state_manager import StateManager
from src.core.sql.sql_utils import SQLUtils
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """State for the SQL generation agent flow."""
    current_state: SQLGenerationState
    requirements: Optional[SQLRequirements] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    next_action: str = "analyze_requirements"
    error: str = ""
    is_done: bool = False


class SQLGenerationAgent:
    """Manages the SQL generation workflow using LangGraph."""

    def __init__(self, llm_manager: LLMManager, state_manager: StateManager):
        self.llm_manager = llm_manager
        self.state_manager = state_manager
        self.sql_utils = SQLUtils(llm_manager)
        self.graph = self._build_graph()

    def _build_graph(self) -> Graph:
        """Build the LangGraph for SQL generation workflow."""
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("route_query", self._route_query)
        graph.add_node("analyze_requirements", self._analyze_requirements)
        graph.add_node("validate_schema", self._validate_schema)
        graph.add_node("generate_query", self._generate_query)
        graph.add_node("review_query", self._review_query)
        graph.add_node("correct_query", self._correct_query)
        graph.add_node("handle_error", self._handle_error)

        # Add conditional edges for query routing
        def route_query(x: AgentState) -> str:
            if x.error:
                return "handle_error"
            if x.is_done:
                return END
            if x.current_state.metadata.get("is_descriptive_query", True):
                return "analyze_requirements"
            return "validate_schema"  # Direct SQL path

        graph.add_conditional_edges(
            "route_query",
            route_query,
            {
                "analyze_requirements": "analyze_requirements",
                "validate_schema": "validate_schema",
                "handle_error": "handle_error",
                END: END
            }
        )

        # Add conditional edges for requirements analysis
        def should_continue_analysis(x: AgentState) -> str:
            if x.error:
                return "handle_error"
            if x.is_done:
                return END
            if x.requirements and not x.requirements.missing_info:
                return "validate_schema"
            return "analyze_requirements"

        graph.add_conditional_edges(
            "analyze_requirements",
            should_continue_analysis,
            {
                "analyze_requirements": "analyze_requirements",
                "validate_schema": "validate_schema",
                "handle_error": "handle_error",
                END: END
            }
        )

        # Add edges for the rest of the workflow
        graph.add_edge("validate_schema", "generate_query")
        graph.add_edge("generate_query", "review_query")
        graph.add_edge("review_query", "correct_query")
        graph.add_edge("correct_query", END)

        # Set entry point
        graph.set_entry_point("route_query")

        # Compile graph
        return graph.compile()

    async def _route_query(self, state: AgentState) -> AgentState:
        """Route the query based on type from service."""
        try:
            # The query type should be set in metadata by the service
            is_descriptive = state.current_state.metadata.get("is_descriptive_query", True)
            
            # For direct SQL queries, create requirements directly
            if not is_descriptive:
                requirements = SQLRequirements(
                    query=state.current_state.metadata.get("query", ""),
                    schema=state.current_state.metadata.get("schema", {}),
                    constraints=state.current_state.metadata.get("constraints", {})
                )
                state.requirements = requirements
                state.current_state.metadata["requirements"] = asdict(requirements)
            
            return state
            
        except Exception as e:
            logger.error(f"Error routing query: {str(e)}")
            return AgentState(
                current_state=state.current_state,
                requirements=state.requirements,
                conversation_history=state.conversation_history,
                next_action="handle_error",
                error=str(e),
                is_done=False
            )

    async def _analyze_requirements(
        self,
        state: AgentState
    ) -> AgentState:
        """Analyze SQL generation requirements."""
        try:
            # Update conversation history with latest message
            if state.current_state.metadata.get("latest_message"):
                state.conversation_history.append({
                    "role": "user",
                    "content": state.current_state.metadata["latest_message"]
                })
            
            # Update state with conversation history
            state.current_state.metadata["conversation_history"] = (
                state.conversation_history
            )
            
            # Analyze requirements
            requirements = await self.sql_utils.analyze_requirements(
                state.current_state
            )
            
            # Add agent's response to conversation history
            if requirements.next_question and requirements.missing_info:
                state.conversation_history.append({
                    "role": "assistant",
                    "content": requirements.next_question
                })
            
            # Check if we should end the conversation
            is_done = bool(requirements and not requirements.missing_info)
            
            # Update state with requirements
            state.current_state.metadata["requirements"] = asdict(requirements)
            
            # Create new state with updated requirements
            new_state = AgentState(
                current_state=state.current_state,
                requirements=requirements,
                conversation_history=state.conversation_history,
                next_action="validate_schema" if is_done else "analyze_requirements",
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

    async def _validate_schema(
        self,
        state: AgentState
    ) -> AgentState:
        """Validate the database schema."""
        try:
            if not state.requirements:
                raise ValueError("Requirements not found in state")
            
            # Check if schema validation was already performed
            if "schema_validation" in state.current_state.metadata:
                # If schema is already validated and valid, proceed to query generation
                if state.current_state.metadata["schema_validation"].get("is_valid", False):
                    return AgentState(
                        current_state=state.current_state,
                        requirements=state.requirements,
                        conversation_history=state.conversation_history,
                        next_action="generate_query",
                        error="",
                        is_done=False
                    )
                # If schema validation failed, handle error
                else:
                    validation_errors = state.current_state.metadata["schema_validation"].get("validation_errors", [])
                    error_msg = "Schema validation failed: " + "; ".join(validation_errors)
                    state.current_state.set_error(error_msg)
                    return AgentState(
                        current_state=state.current_state,
                        requirements=state.requirements,
                        conversation_history=state.conversation_history,
                        next_action="handle_error",
                        error=error_msg,
                        is_done=False
                    )
            
            # Perform schema validation
            validation_result = await self.sql_utils.validate_schema(
                state.requirements.schema
            )
            
            # Update state with validation result
            state.current_state.metadata["schema_validation"] = validation_result
            state.current_state.update_step(
                SQLGenerationStep.SCHEMA_VALIDATION
            )
            
            # If validation failed, handle error
            if not validation_result.get("is_valid", False):
                validation_errors = validation_result.get("validation_errors", [])
                error_msg = "Schema validation failed: " + "; ".join(validation_errors)
                state.current_state.set_error(error_msg)
                return AgentState(
                    current_state=state.current_state,
                    requirements=state.requirements,
                    conversation_history=state.conversation_history,
                    next_action="handle_error",
                    error=error_msg,
                    is_done=False
                )
            
            # If validation succeeded, proceed to query generation
            return AgentState(
                current_state=state.current_state,
                requirements=state.requirements,
                conversation_history=state.conversation_history,
                next_action="generate_query",
                error="",
                is_done=False
            )
            
        except Exception as e:
            logger.error(f"Error validating schema: {str(e)}")
            state.current_state.set_error(str(e))
            return AgentState(
                current_state=state.current_state,
                requirements=state.requirements,
                conversation_history=state.conversation_history,
                next_action="handle_error",
                error=str(e),
                is_done=False
            )

    async def _generate_query(
        self,
        state: AgentState
    ) -> AgentState:
        """Generate SQL query based on requirements."""
        try:
            if not state.requirements:
                raise ValueError("Requirements not found in state")
            
            query_result = await self.sql_utils.generate_query(
                state.requirements
            )
            
            # Update state with query result
            state.current_state.metadata["query_result"] = asdict(query_result)
            state.current_state.update_step(
                SQLGenerationStep.QUERY_GENERATION
            )
            
            return AgentState(
                current_state=state.current_state,
                requirements=state.requirements,
                conversation_history=state.conversation_history,
                next_action="review_query",
                error="",
                is_done=False
            )
        except Exception as e:
            logger.error(f"Error generating query: {str(e)}")
            state.current_state.set_error(str(e))
            return AgentState(
                current_state=state.current_state,
                requirements=state.requirements,
                conversation_history=state.conversation_history,
                next_action="handle_error",
                error=str(e),
                is_done=False
            )

    async def _review_query(
        self,
        state: AgentState
    ) -> AgentState:
        """Review generated SQL query."""
        try:
            if not state.current_state.metadata.get("query_result"):
                raise ValueError("Query result not found in state")
            
            review_result = await self.sql_utils.review_query(
                state.current_state.metadata["query_result"]
            )
            
            # Update state with review result
            state.current_state.metadata["review_result"] = review_result
            state.current_state.update_step(
                SQLGenerationStep.QUERY_REVIEW
            )
            
            return AgentState(
                current_state=state.current_state,
                requirements=state.requirements,
                conversation_history=state.conversation_history,
                next_action="correct_query",
                error="",
                is_done=False
            )
        except Exception as e:
            logger.error(f"Error reviewing query: {str(e)}")
            state.current_state.set_error(str(e))
            return AgentState(
                current_state=state.current_state,
                requirements=state.requirements,
                conversation_history=state.conversation_history,
                next_action="handle_error",
                error=str(e),
                is_done=False
            )

    async def _correct_query(
        self,
        state: AgentState
    ) -> AgentState:
        """Correct SQL query based on review feedback."""
        try:
            if not state.current_state.metadata.get("query_result") or \
               not state.current_state.metadata.get("review_result"):
                raise ValueError(
                    "Query result or review result not found in state"
                )
            
            corrected_query = await self.sql_utils.correct_query(
                state.current_state.metadata["query_result"],
                state.current_state.metadata["review_result"]
            )
            
            # Update state with corrected query
            state.current_state.metadata["final_query"] = asdict(corrected_query)
            state.current_state.update_step(
                SQLGenerationStep.COMPLETED
            )
            state.current_state.set_success()
            
            return AgentState(
                current_state=state.current_state,
                requirements=state.requirements,
                conversation_history=state.conversation_history,
                next_action=END,
                error="",
                is_done=True
            )
        except Exception as e:
            logger.error(f"Error correcting query: {str(e)}")
            state.current_state.set_error(str(e))
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
        
        # Update state with error
        state.current_state.set_error(error)
        
        return AgentState(
            current_state=state.current_state,
            requirements=state.requirements,
            conversation_history=state.conversation_history,
            next_action=END,
            error="",
            is_done=True
        )

    async def run(self, workflow_id: str, user_request: str) -> SQLGenerationState:
        """Run the SQL generation workflow."""
        try:
            # Initialize state with user request
            initial_state = AgentState(
                current_state=SQLGenerationState(
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
                        req = final_state["requirements"]
                        if isinstance(req, dict):
                            requirements = SQLRequirements(**req)
                        elif isinstance(req, SQLRequirements):
                            requirements = req
                    except Exception as e:
                        logger.error(
                            f"Error parsing requirements from final state: {str(e)}"
                        )
                
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
                    req = final_workflow_state.metadata["requirements"]
                    if isinstance(req, dict):
                        requirements = SQLRequirements(**req)
                        final_workflow_state.metadata["requirements"] = asdict(requirements)
                    elif isinstance(req, SQLRequirements):
                        final_workflow_state.metadata["requirements"] = asdict(req)
                except Exception as e:
                    logger.error(
                        f"Error setting requirements in final state: {str(e)}"
                    )
            
            # Save final state
            # Set status based on error presence
            if getattr(final_workflow_state, 'error', None):
                final_workflow_state.status = "error"
            else:
                final_workflow_state.status = "success"
            await self.state_manager.save_state(final_workflow_state)
            
            return final_workflow_state
        except Exception as e:
            logger.error(f"Error running workflow: {str(e)}")
            # Create error state
            error_state = SQLGenerationState(
                workflow_id=workflow_id,
                metadata={
                    "latest_message": user_request,
                    "conversation_history": [
                        {"role": "user", "content": user_request}
                    ]
                },
                error=str(e),
                status="error",
                is_done=True
            )
            await self.state_manager.save_state(error_state)
            return error_state 