import pytest
from typing import List

from src.core.workflow.etl_state import ETLWorkflowState, ETLStep
from src.core.workflow.etl_utils import ETLUtils
from src.core.state_management.execution_manager import ExecutionManager
from src.core.llm.manager import LLMManager


class TestDashboardETL:
    """Test suite for Dashboard ETL functionality."""

    @pytest.fixture
    def initial_user_request(self) -> str:
        """Provide initial user request fixture."""
        return """
        I need to create a dashboard for my sales data.
        I'm not very technical, so I need help with the architecture.
        """

    @pytest.fixture
    def follow_up_responses(self) -> List[str]:
        """Provide follow-up response fixtures."""
        return [
            # Response to data source questions
            """
            I have sales data in Excel files. We get new files every day.
            Each file is about 5-10MB in size.
            We also have customer information in CSV format.
            """,
            
            # Response to dashboard requirements
            """
            I want to see:
            - Daily sales trends
            - Product performance
            - Customer segmentation
            The dashboard should update daily.
            """,
            
            # Response to access requirements
            """
            About 10 people in my team need access to the dashboard.
            They should be able to view and filter the data.
            We don't need editing capabilities.
            """
        ]

    @pytest.fixture
    def execution_manager(self) -> ExecutionManager:
        """Provide execution manager fixture."""
        return ExecutionManager()

    @pytest.fixture
    def llm_manager(self) -> LLMManager:
        """Provide LLM manager fixture."""
        return LLMManager()

    @pytest.fixture
    def etl_utils(self) -> ETLUtils:
        """Provide ETL utils fixture."""
        return ETLUtils()

    @pytest.mark.asyncio
    async def test_etl_conversation_flow(self):
        """Test the conversation flow for gathering ETL requirements."""
        # Initialize components
        llm_manager = LLMManager()
        etl_utils = ETLUtils(llm_manager)
        
        # Create initial state with project description
        project_desc = """
            We need to create a dashboard showing sales across regions.
            The data comes from our CRM system and needs to be updated daily.
            We want to show trends over time and compare performance.
            """
        state = ETLWorkflowState()
        state.metadata = {"project_description": project_desc}
        state.current_step = ETLStep.ANALYSIS
        
        # First analysis
        requirements = await etl_utils.analyze_project(state)
        
        # Validate initial requirements
        assert requirements.data_sources, "Should identify CRM as data source"
        assert "daily" in requirements.update_frequency.lower()
        assert "trends" in requirements.visualization_needs
        assert "regions" in requirements.visualization_needs
        
        # Check for missing information
        assert requirements.missing_info, "Should identify missing information"
        assert any(
            "access" in info.lower() for info in requirements.missing_info
        )
        assert any(
            "format" in info.lower() for info in requirements.missing_info
        )
        
        # Generate follow-up questions
        questions = await etl_utils.generate_questions(state)
        
        # Validate questions
        assert len(questions) >= 3, "Should generate at least 3 questions"
        assert any("access" in q.lower() for q in questions)
        assert any("format" in q.lower() for q in questions)
        assert all(
            len(q) < 100 for q in questions
        ), "Questions should be concise"
        
        # Simulate user response
        state.metadata["project_description"] = """
        The CRM data is in CSV format, stored in S3.
        We need read-only access for the dashboard users.
        The data includes sales amount, region, and date fields.
        """
        
        # Second analysis
        requirements = await etl_utils.analyze_project(state)
        
        # Validate updated requirements
        assert "csv" in str(requirements.data_sources).lower()
        assert "s3" in str(requirements.data_sources).lower()
        assert "read-only" in requirements.access_requirements
        assert all(
            field in str(requirements.data_sources).lower()
            for field in ["sales", "region", "date"]
        )
        
        # Check if missing information is reduced
        assert len(requirements.missing_info) < len(state.questions)
        
        # Generate final questions
        questions = await etl_utils.generate_questions(state)
        
        # Validate final questions focus on remaining gaps
        remaining_keywords = ["scale", "retention", "backup"]
        assert all(
            any(keyword in q.lower() for keyword in remaining_keywords)
            for q in questions
        ), "Questions should focus on remaining requirements"
        
        # Verify conversation history
        assert len(etl_utils.conversation_context.conversation_history) >= 4
        assert all(
            msg["role"] in ["user", "assistant"]
            for msg in etl_utils.conversation_context.conversation_history
        )
        
        # Verify focus progression
        expected_focus = ["data_sources", "visualization", "access"]
        assert etl_utils.conversation_context.current_focus in expected_focus

    @pytest.mark.asyncio
    async def test_dashboard_etl_edge_cases(
        self,
        execution_manager: ExecutionManager,
        llm_manager: LLMManager,
        etl_utils: ETLUtils
    ):
        """Test edge cases for Dashboard ETL functionality."""
        # Test with minimal information
        minimal_request = "I need a dashboard."
        state = ETLWorkflowState()
        state.metadata = {"project_description": minimal_request}
        state.current_step = ETLStep.ANALYSIS
        
        execution = execution_manager.create_execution(
            user="test_user",
            environment="test",
            trigger="test",
            metadata={"workflow_id": "minimal_dashboard"}
        )
        execution.states.append({"state": state})
        
        analysis_result = await etl_utils.analyze_project(state)
        assert len(analysis_result.missing_info) > 0
        questions = await etl_utils.generate_questions(analysis_result)
        assert len(questions) > 0
        assert all(isinstance(q, str) for q in questions)

        # Test with conflicting requirements
        conflicting_request = """
        I need a dashboard that updates in real-time.
        But we only get new data files once a month.
        """
        state = ETLWorkflowState()
        state.metadata = {"project_description": conflicting_request}
        state.current_step = ETLStep.ANALYSIS
        
        execution = execution_manager.create_execution(
            user="test_user",
            environment="test",
            trigger="test",
            metadata={"workflow_id": "conflicting_dashboard"}
        )
        execution.states.append({"state": state})
        
        analysis_result = await etl_utils.analyze_project(state)
        assert "conflicts" in analysis_result.issues
        assert any(
            "update frequency" in issue.lower()
            for issue in analysis_result.issues
        )

    @pytest.mark.asyncio
    async def test_dashboard_etl_error_handling(
        self,
        execution_manager: ExecutionManager,
        llm_manager: LLMManager,
        etl_utils: ETLUtils
    ):
        """Test error handling for Dashboard ETL functionality."""
        # Test with invalid data format
        invalid_request = (
            "I need a dashboard for my sales data in PDF format."
        )
        state = ETLWorkflowState()
        state.metadata = {"project_description": invalid_request}
        state.current_step = ETLStep.ANALYSIS
        
        execution = execution_manager.create_execution(
            user="test_user",
            environment="test",
            trigger="test",
            metadata={"workflow_id": "invalid_dashboard"}
        )
        execution.states.append({"state": state})
        
        with pytest.raises(ValueError):
            await etl_utils.analyze_project(state) 