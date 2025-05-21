from typing import Dict, Any, Optional, List
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from aws_architect_agent.utils.logging import get_logger

logger = get_logger(__name__)


class PromptManager:
    """Manages dynamic prompt generation using LangChain templates."""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
    ) -> None:
        """Initialize the prompt manager.

        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature for LLM generation
        """
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
        )
        self._initialize_templates()

    def _initialize_templates(self) -> None:
        """Initialize prompt templates for different tasks."""
        self.templates = {
            "requirements_analysis": PromptTemplate(
                input_variables=[
                    "requirements",
                    "context",
                    "conversation_history",
                    "follow_up_questions",
                ],
                template=(
                    "As a Product Manager, analyze these requirements:\n\n"
                    "Requirements:\n{requirements}\n\n"
                    "Context:\n{context}\n\n"
                    "Previous Conversation:\n{conversation_history}\n\n"
                    "Previous Questions:\n{follow_up_questions}\n\n"
                    "1. Identify missing or unclear requirements\n"
                    "2. List follow-up questions to clarify\n"
                    "3. Create structured requirements document\n\n"
                    "Format your response with:\n"
                    "- Requirements Analysis\n"
                    "- Follow-up Questions\n"
                    "- Requirements Document\n"
                    "End with 'requirements_analyzed' when complete."
                ),
            ),
            "architecture_review": PromptTemplate(
                input_variables=[
                    "architecture",
                    "requirements",
                    "context",
                    "conversation_history",
                    "previous_feedback",
                ],
                template=(
                    "As a Product Manager, review this architecture:\n\n"
                    "Requirements:\n{requirements}\n\n"
                    "Architecture:\n{architecture}\n\n"
                    "Context:\n{context}\n\n"
                    "Previous Conversation:\n{conversation_history}\n\n"
                    "Previous Feedback:\n{previous_feedback}\n\n"
                    "1. Check alignment with requirements\n"
                    "2. Identify gaps and concerns\n"
                    "3. List follow-up questions\n"
                    "4. Provide assessment\n\n"
                    "Format your response with:\n"
                    "- Requirements Alignment\n"
                    "- Gaps and Concerns\n"
                    "- Follow-up Questions\n"
                    "- Assessment\n"
                    "End with 'architecture_reviewed' when complete."
                ),
            ),
            "architecture_design": PromptTemplate(
                input_variables=[
                    "requirements",
                    "context",
                    "conversation_history",
                    "previous_designs",
                ],
                template=(
                    "As a Data Architect, design AWS architecture:\n\n"
                    "Requirements:\n{requirements}\n\n"
                    "Context:\n{context}\n\n"
                    "Previous Conversation:\n{conversation_history}\n\n"
                    "Previous Designs:\n{previous_designs}\n\n"
                    "1. Design architecture components\n"
                    "2. Explain design decisions\n"
                    "3. Include AWS services\n"
                    "4. Consider scalability, security, cost\n\n"
                    "Format your response with:\n"
                    "- Architecture Overview\n"
                    "- Component Details\n"
                    "- Design Decisions\n"
                    "- AWS Services\n"
                    "End with 'architecture_designed' when complete."
                ),
            ),
            "architecture_refinement": PromptTemplate(
                input_variables=[
                    "architecture",
                    "feedback",
                    "context",
                    "conversation_history",
                    "previous_refinements",
                ],
                template=(
                    "As a Data Architect, refine this architecture:\n\n"
                    "Current Architecture:\n{architecture}\n\n"
                    "Feedback:\n{feedback}\n\n"
                    "Context:\n{context}\n\n"
                    "Previous Conversation:\n{conversation_history}\n\n"
                    "Previous Refinements:\n{previous_refinements}\n\n"
                    "1. Address feedback points\n"
                    "2. Make improvements\n"
                    "3. Explain changes\n"
                    "4. Update architecture\n\n"
                    "Format your response with:\n"
                    "- Changes Made\n"
                    "- Updated Architecture\n"
                    "- Explanation\n"
                    "End with 'architecture_refined' when complete."
                ),
            ),
            "follow_up_question": PromptTemplate(
                input_variables=[
                    "current_response",
                    "context",
                    "conversation_history",
                    "previous_questions",
                ],
                template=(
                    "Generate a follow-up question based on:\n\n"
                    "Current Response:\n{current_response}\n\n"
                    "Context:\n{context}\n\n"
                    "Previous Conversation:\n{conversation_history}\n\n"
                    "Previous Questions:\n{previous_questions}\n\n"
                    "Consider:\n"
                    "1. What information is missing?\n"
                    "2. What needs clarification?\n"
                    "3. What assumptions need validation?\n\n"
                    "Generate a clear, specific question that will help "
                    "move the conversation forward."
                ),
            ),
        }

    def generate_prompt(
        self,
        task_type: str,
        **kwargs: Any,
    ) -> str:
        """Generate a prompt for a specific task.

        Args:
            task_type: Type of task to generate prompt for
            **kwargs: Additional arguments for the prompt template

        Returns:
            Generated prompt
        """
        if task_type not in self.templates:
            raise ValueError(f"Unknown task type: {task_type}")

        chain = LLMChain(
            llm=self.llm,
            prompt=self.templates[task_type],
        )

        try:
            response = chain.run(**kwargs)
            logger.info(f"Generated prompt for {task_type}")
            return response
        except Exception as e:
            logger.error(f"Error generating prompt: {str(e)}")
            raise

    def generate_follow_up_question(
        self,
        current_response: str,
        context: Dict[str, Any],
        conversation_history: List[str],
        previous_questions: List[str],
    ) -> str:
        """Generate a follow-up question based on context.

        Args:
            current_response: Current response to analyze
            context: Additional context
            conversation_history: Previous conversation
            previous_questions: Previously asked questions

        Returns:
            Generated follow-up question
        """
        return self.generate_prompt(
            task_type="follow_up_question",
            current_response=current_response,
            context=context,
            conversation_history="\n".join(conversation_history),
            previous_questions="\n".join(previous_questions),
        ) 