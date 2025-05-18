from typing import Dict, Any, Optional
from langgraph.graph import Graph
from src.core.workflow.etl_state import ETLWorkflowState, ETLStep
from src.core.workflow.etl_utils import ETLUtils
from src.core.state_management.execution_manager import ExecutionManager

class ETLArchitectAgent:
    def __init__(
        self,
        execution_manager: Optional[ExecutionManager] = None,
        etl_utils: Optional[ETLUtils] = None
    ):
        self.execution_manager = execution_manager or ExecutionManager()
        self.etl_utils = etl_utils or ETLUtils()
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> Graph:
        """Create the LangGraph workflow for ETL architecture generation."""
        workflow = Graph()

        # Add nodes
        workflow.add_node("analyze", self._analyze_project)
        workflow.add_node("generate_architecture", self._generate_architecture)
        workflow.add_node("review_architecture", self._review_architecture)
        workflow.add_node("correct_architecture", self._correct_architecture)
        workflow.add_node("generate_questions", self._generate_questions)
        workflow.add_node("generate_terraform", self._generate_terraform)
        workflow.add_node("review_terraform", self._review_terraform)
        workflow.add_node("correct_terraform", self._correct_terraform)

        # Define edges
        workflow.add_edge("analyze", "generate_architecture")
        workflow.add_edge("generate_architecture", "review_architecture")
        workflow.add_edge("review_architecture", "correct_architecture")
        workflow.add_edge("correct_architecture", "generate_questions")
        workflow.add_edge("generate_questions", "generate_terraform")
        workflow.add_edge("generate_terraform", "review_terraform")
        workflow.add_edge("review_terraform", "correct_terraform")

        return workflow

    async def process_project(self, project_description: str) -> Dict[str, Any]:
        """Process a project description through the ETL workflow."""
        # Initialize state
        state = ETLWorkflowState(
            project_description=project_description,
            current_step=ETLStep.ANALYSIS
        )

        # Create execution
        execution = self.execution_manager.create_execution(
            workflow_id="etl_architecture",
            initial_state=state
        )

        # Execute workflow
        result = await self.workflow.execute(execution.current_state)

        return {
            "architecture": result.get("architecture"),
            "terraform": result.get("terraform"),
            "questions": result.get("questions", []),
            "state": execution.current_state
        }

    async def _analyze_project(self, state: ETLWorkflowState) -> ETLWorkflowState:
        """Analyze project description and extract requirements."""
        requirements = await self.etl_utils.analyze_project(state)
        state.set_requirements(requirements)
        state.update_step(ETLStep.ARCHITECTURE_GENERATION)
        return state

    async def _generate_architecture(
        self, 
        state: ETLWorkflowState
    ) -> ETLWorkflowState:
        """Generate architecture based on requirements."""
        if not state.requirements:
            raise ValueError("Requirements not found in state")
        
        architecture = await self.etl_utils.generate_architecture(
            state.requirements
        )
        state.set_architecture(architecture)
        state.update_step(ETLStep.ARCHITECTURE_REVIEW)
        return state

    async def _review_architecture(
        self, 
        state: ETLWorkflowState
    ) -> ETLWorkflowState:
        """Review architecture and provide feedback."""
        if not state.architecture:
            raise ValueError("Architecture not found in state")
        
        review_result = await self.etl_utils.review_architecture(
            state.architecture
        )
        state.set_review_feedback(review_result)
        state.update_step(ETLStep.ARCHITECTURE_CORRECTION)
        return state

    async def _correct_architecture(
        self, 
        state: ETLWorkflowState
    ) -> ETLWorkflowState:
        """Correct architecture based on review feedback."""
        if not state.architecture or not state.review_feedback:
            raise ValueError("Architecture or review feedback not found in state")
        
        corrected_architecture = await self.etl_utils.correct_architecture(
            state.architecture,
            state.review_feedback
        )
        state.set_architecture(corrected_architecture)
        state.update_step(ETLStep.QUESTION_GENERATION)
        return state

    async def _generate_questions(
        self, 
        state: ETLWorkflowState
    ) -> ETLWorkflowState:
        """Generate follow-up questions based on architecture."""
        if not state.architecture:
            raise ValueError("Architecture not found in state")
        
        questions = await self.etl_utils.generate_questions(state.architecture)
        for question in questions:
            state.add_question(question)
        
        state.update_step(ETLStep.TERRAFORM_GENERATION)
        return state

    async def _generate_terraform(
        self, 
        state: ETLWorkflowState
    ) -> ETLWorkflowState:
        """Generate Terraform code for the architecture."""
        if not state.architecture:
            raise ValueError("Architecture not found in state")
        
        terraform = await self.etl_utils.generate_terraform(state.architecture)
        state.set_terraform(terraform)
        state.update_step(ETLStep.TERRAFORM_REVIEW)
        return state

    async def _review_terraform(
        self, 
        state: ETLWorkflowState
    ) -> ETLWorkflowState:
        """Review Terraform code and provide feedback."""
        if not state.terraform_code:
            raise ValueError("Terraform code not found in state")
        
        review_result = await self.etl_utils.review_terraform(
            state.terraform_code
        )
        state.set_review_feedback(review_result)
        state.update_step(ETLStep.TERRAFORM_CORRECTION)
        return state

    async def _correct_terraform(
        self, 
        state: ETLWorkflowState
    ) -> ETLWorkflowState:
        """Correct Terraform code based on review feedback."""
        if not state.terraform_code or not state.review_feedback:
            raise ValueError(
                "Terraform code or review feedback not found in state"
            )
        
        corrected_terraform = await self.etl_utils.correct_terraform(
            state.terraform_code,
            state.review_feedback
        )
        state.set_terraform(corrected_terraform)
        state.update_step(ETLStep.COMPLETED)
        return state 