from typing import Dict, Any, Optional, Tuple, List


from tenacity import retry, stop_after_attempt, wait_exponential


from aws_architect_agent.models.base import ArchitectureState
from aws_architect_agent.core.llm_state_manager import LLMStateManager
from aws_architect_agent.core.prompt_manager import PromptManager
from aws_architect_agent.utils.logging import get_logger


logger = get_logger(__name__)


class ProductManagerAgent:
    """Product Manager Agent for requirements gathering and review."""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> None:
        """Initialize the Product Manager agent.

        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature for LLM generation
            max_tokens: Maximum tokens for LLM generation
        """
        self.llm_manager = LLMStateManager(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.prompt_manager = PromptManager(
            model_name=model_name,
            temperature=temperature,
        )
        self.conversation_history = []
        self.follow_up_questions = []

    def gather_requirements(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[str]]:
        """Gather and clarify requirements.

        Args:
            message: User message
            context: Additional context

        Returns:
            Tuple of (response, follow_up_questions)
        """
        # Generate prompt using PromptManager
        prompt = self.prompt_manager.generate_prompt(
            task_type="requirements_analysis",
            requirements=message,
            context=context or {},
            conversation_history="\n".join(self.conversation_history),
            follow_up_questions="\n".join(self.follow_up_questions),
        )

        response = self.llm_manager.process_message(
            state=ArchitectureState.REQUIREMENTS_GATHERING,
            message=prompt,
            context=context,
        )

        # Update conversation history
        self.conversation_history.append(f"User: {message}")
        self.conversation_history.append(f"Agent: {response}")

        # Extract follow-up questions
        questions = []
        if "Follow-up Questions:" in response:
            questions_section = response.split("Follow-up Questions:")[1]
            if "Requirements Document:" in questions_section:
                questions_section = questions_section.split(
                    "Requirements Document:"
                )[0]
            questions = [
                q.strip() for q in questions_section.split("\n")
                if q.strip() and q.strip().startswith("-")
            ]
            self.follow_up_questions.extend(questions)

        return response, questions

    def review_architecture(
        self,
        architecture: str,
        requirements: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[str]]:
        """Review architecture against requirements.

        Args:
            architecture: Proposed architecture
            requirements: Requirements document
            context: Additional context

        Returns:
            Tuple of (review, follow_up_questions)
        """
        # Generate prompt using PromptManager
        prompt = self.prompt_manager.generate_prompt(
            task_type="architecture_review",
            architecture=architecture,
            requirements=requirements,
            context=context or {},
            conversation_history="\n".join(self.conversation_history),
            previous_feedback="\n".join(self.conversation_history),
        )

        response = self.llm_manager.process_message(
            state=ArchitectureState.REVIEW,
            message=prompt,
            context=context,
        )

        # Update conversation history
        self.conversation_history.append(f"Architecture: {architecture}")
        self.conversation_history.append(f"Review: {response}")

        # Extract follow-up questions
        questions = []
        if "Follow-up Questions:" in response:
            questions_section = response.split("Follow-up Questions:")[1]
            if "Assessment:" in questions_section:
                questions_section = questions_section.split("Assessment:")[0]
            questions = [
                q.strip() for q in questions_section.split("\n")
                if q.strip() and q.strip().startswith("-")
            ]
            self.follow_up_questions.extend(questions)

        return response, questions


class DataArchitectAgent:
    """Data Architect Agent for architecture design and refinement."""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> None:
        """Initialize the Data Architect agent.

        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature for LLM generation
            max_tokens: Maximum tokens for LLM generation
        """
        self.llm_manager = LLMStateManager(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.prompt_manager = PromptManager(
            model_name=model_name,
            temperature=temperature,
        )
        self.conversation_history = []
        self.previous_designs = []
        self.previous_refinements = []

    def design_architecture(
        self,
        requirements: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Design architecture based on requirements.

        Args:
            requirements: Requirements document
            context: Additional context

        Returns:
            Architecture design
        """
        # Generate prompt using PromptManager
        prompt = self.prompt_manager.generate_prompt(
            task_type="architecture_design",
            requirements=requirements,
            context=context or {},
            conversation_history="\n".join(self.conversation_history),
            previous_designs="\n".join(self.previous_designs),
        )

        response = self.llm_manager.process_message(
            state=ArchitectureState.DESIGN,
            message=prompt,
            context=context,
        )

        # Update conversation history and previous designs
        self.conversation_history.append(f"Requirements: {requirements}")
        self.conversation_history.append(f"Design: {response}")
        self.previous_designs.append(response)

        return response

    def refine_architecture(
        self,
        architecture: str,
        feedback: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Refine architecture based on feedback.

        Args:
            architecture: Current architecture
            feedback: Review feedback
            context: Additional context

        Returns:
            Refined architecture
        """
        # Generate prompt using PromptManager
        prompt = self.prompt_manager.generate_prompt(
            task_type="architecture_refinement",
            architecture=architecture,
            feedback=feedback,
            context=context or {},
            conversation_history="\n".join(self.conversation_history),
            previous_refinements="\n".join(self.previous_refinements),
        )

        response = self.llm_manager.process_message(
            state=ArchitectureState.REFINEMENT,
            message=prompt,
            context=context,
        )

        # Update conversation history and previous refinements
        self.conversation_history.append(f"Feedback: {feedback}")
        self.conversation_history.append(f"Refined Design: {response}")
        self.previous_refinements.append(response)

        return response


class LLMWorkflowManager:
    """Manages multi-agent workflow state transitions."""

    def __init__(
        self,
        state_manager: Any,
        product_manager_model: str = "gpt-3.5-turbo",
        data_architect_model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        max_attempts: int = 3,
    ) -> None:
        """Initialize the multi-agent workflow manager.

        Args:
            state_manager: State manager instance
            product_manager_model: Model for Product Manager agent
            data_architect_model: Model for Data Architect agent
            temperature: Temperature for LLM generation
            max_tokens: Maximum tokens for LLM generation
            max_attempts: Maximum number of retry attempts
        """
        self.state_manager = state_manager
        self.product_manager = ProductManagerAgent(
            model_name=product_manager_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.data_architect = DataArchitectAgent(
            model_name=data_architect_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.max_attempts = max_attempts
        self.current_attempt = 0
        self.requirements_doc = None
        self.architecture_doc = None

    def _process_with_agents(
        self,
        state: ArchitectureState,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Process a message using both agents.

        Args:
            state: Current architecture state
            message: User message
            context: Additional context

        Returns:
            Processed response
        """
        if state == ArchitectureState.REQUIREMENTS_GATHERING:
            # Product Manager gathers requirements
            response, questions = self.product_manager.gather_requirements(
                message=message,
                context=context,
            )
            self.requirements_doc = response
            return response

        elif state == ArchitectureState.DESIGN:
            # Data Architect designs architecture
            if not self.requirements_doc:
                raise ValueError("Requirements document not available")
            response = self.data_architect.design_architecture(
                requirements=self.requirements_doc,
                context=context,
            )
            self.architecture_doc = response
            return response

        elif state == ArchitectureState.REVIEW:
            # Product Manager reviews architecture
            if not self.architecture_doc or not self.requirements_doc:
                raise ValueError("Architecture or requirements not available")
            response, questions = self.product_manager.review_architecture(
                architecture=self.architecture_doc,
                requirements=self.requirements_doc,
                context=context,
            )
            return response

        elif state == ArchitectureState.REFINEMENT:
            # Data Architect refines architecture
            if not self.architecture_doc:
                raise ValueError("Architecture document not available")
            response = self.data_architect.refine_architecture(
                architecture=self.architecture_doc,
                feedback=message,
                context=context,
            )
            self.architecture_doc = response
            return response

        return ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def process_requirements_gathering(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Process requirements gathering state.

        Args:
            message: User message
            context: Additional context

        Returns:
            LLM response with requirements
        """
        self.current_attempt += 1
        logger.info(
            f"Processing requirements gathering "
            f"(attempt {self.current_attempt})"
        )
        
        try:
            response = self._process_with_agents(
                state=ArchitectureState.REQUIREMENTS_GATHERING,
                message=message,
                context=context,
            )
            
            if "requirements_analyzed" in response.lower():
                self.current_attempt = 0
                return response
            
            raise ValueError("Requirements analysis not complete")
            
        except Exception as e:
            if self.current_attempt >= self.max_attempts:
                logger.error(
                    f"Max attempts ({self.max_attempts}) reached for "
                    "requirements gathering"
                )
                self.current_attempt = 0
                raise
            logger.warning(f"Error in requirements gathering: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def process_design(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Process design state.

        Args:
            message: User message
            context: Additional context

        Returns:
            LLM response with design
        """
        self.current_attempt += 1
        logger.info(f"Processing design (attempt {self.current_attempt})")
        
        try:
            response = self._process_with_agents(
                state=ArchitectureState.DESIGN,
                message=message,
                context=context,
            )
            
            if "architecture_designed" in response.lower():
                self.current_attempt = 0
                return response
            
            raise ValueError("Architecture design not complete")
            
        except Exception as e:
            if self.current_attempt >= self.max_attempts:
                logger.error(
                    f"Max attempts ({self.max_attempts}) reached for design"
                )
                self.current_attempt = 0
                raise
            logger.warning(f"Error in design: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def process_review(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Process review state.

        Args:
            message: User message
            context: Additional context

        Returns:
            LLM response with review
        """
        self.current_attempt += 1
        logger.info(f"Processing review (attempt {self.current_attempt})")
        
        try:
            response = self._process_with_agents(
                state=ArchitectureState.REVIEW,
                message=message,
                context=context,
            )
            
            if "architecture_reviewed" in response.lower():
                self.current_attempt = 0
                return response
            
            raise ValueError("Architecture review not complete")
            
        except Exception as e:
            if self.current_attempt >= self.max_attempts:
                logger.error(
                    f"Max attempts ({self.max_attempts}) reached for review"
                )
                self.current_attempt = 0
                raise
            logger.warning(f"Error in review: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def process_refinement(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Process refinement state.

        Args:
            message: User message
            context: Additional context

        Returns:
            LLM response with refinement
        """
        self.current_attempt += 1
        logger.info(f"Processing refinement (attempt {self.current_attempt})")
        
        try:
            response = self._process_with_agents(
                state=ArchitectureState.REFINEMENT,
                message=message,
                context=context,
            )
            
            if "architecture_refined" in response.lower():
                self.current_attempt = 0
                return response
            
            raise ValueError("Architecture refinement not complete")
            
        except Exception as e:
            if self.current_attempt >= self.max_attempts:
                logger.error(
                    f"Max attempts ({self.max_attempts}) reached for "
                    "refinement"
                )
                self.current_attempt = 0
                raise
            logger.warning(f"Error in refinement: {str(e)}")
            raise