from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from .etl_state import ETLState
from src.core.workflow.etl_state import (
    ETLWorkflowState,
    ETLRequirements,
    ETLArchitecture,
    ETLTerraform,
    DataSourceType,
    DataFormat,
    ETLTerraform
)
from src.core.llm.manager import LLMManager
import json
import re
import logging

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages dynamic prompt generation based on conversation context."""
    
    def __init__(self):
        self.role_description = (
            "I'm your AI Software Architect and Product Strategist. I specialize in "
            "understanding your business needs and transforming them into clear technical "
            "requirements across software functionality, data pipeline architecture (ETL), "
            "and infrastructure (DevOps). I'll ask you guided questions to help shape "
            "your idea into a complete, build-ready solution."
        )
        
        self.expertise_breakdown = {
            "product_manager": {
                "emoji": "🧩",
                "responsibilities": [
                    "Understand business goals",
                    "Define features",
                    "Prioritize use cases"
                ]
            },
            "etl_architect": {
                "emoji": "🔄",
                "responsibilities": [
                    "Clarify data sources",
                    "Structure transformation logic",
                    "Define storage format",
                    "Address compliance needs"
                ]
            },
            "devops_engineer": {
                "emoji": "⚙️",
                "responsibilities": [
                    "Determine deployment model",
                    "Plan scalability",
                    "Design CI/CD pipeline",
                    "Specify infrastructure needs"
                ]
            }
        }
    
    def get_initial_prompt(self) -> str:
        """Get the initial conversation starter prompt."""
        return (
            "Hi! I'm here to help you shape your product or data automation idea. "
            "Let's begin with the basics:\n\n"
            "1. What is the **main goal** of your product or data pipeline?\n"
            "2. Who are the **users** of this system (internal team, customers, analysts)?\n"
            "3. What **data sources** do you already have or plan to use?\n"
            "4. What kind of **outputs or insights** do you want to generate?\n"
            "5. How do you want to **access or deploy** this solution?"
        )
    
    def get_analysis_prompt(
        self,
        context: Dict[str, Any],
        history: List[Dict[str, str]],
        message: str
    ) -> str:
        """Generate analysis prompt based on conversation context."""
        # Determine current focus based on conversation
        current_focus = self._determine_focus(context, history)
        
        # Build role-specific context
        role_context = self._build_role_context(current_focus)
        
        # Format conversation history
        formatted_history = self._format_conversation_history(history)
        
        return f"""
        {self.role_description}
        
        Current Focus: {role_context['title']} {role_context['emoji']}
        Responsibilities:
        {role_context['responsibilities']}
        
        Current Understanding:
        {json.dumps(context, indent=2)}
        
        Previous Conversation:
        {formatted_history}
        
        User's Latest Response:
        {message}
        
        Please provide a JSON response with:
        {{
            "data_sources": {{
                "format": "file/database format",
                "location": "storage location",
                "volume": "data volume estimate",
                "update_freq": "update frequency",
                "access": "access method"
            }},
            "processing_needs": [
                "required transformations",
                "data quality checks",
                "error handling requirements"
            ],
            "missing_info": ["remaining questions"],
            "next_question": "most relevant next question",
            "summary": "brief summary of understanding"
        }}
        """
    
    def _determine_focus(
        self,
        context: Dict[str, Any],
        history: List[Dict[str, str]]
    ) -> str:
        """Determine the current focus based on conversation context."""
        if not context.get("data_sources"):
            return "product_manager"
        elif not context.get("processing_needs"):
            return "etl_architect"
        else:
            return "devops_engineer"
    
    def _build_role_context(self, focus: str) -> Dict[str, str]:
        """Build context for the current role."""
        role = self.expertise_breakdown[focus]
        return {
            "title": focus.replace("_", " ").title(),
            "emoji": role["emoji"],
            "responsibilities": "\n".join(
                f"- {resp}" for resp in role["responsibilities"]
            )
        }
    
    def _format_conversation_history(
        self,
        history: List[Dict[str, str]]
    ) -> str:
        """Format conversation history for the prompt."""
        if not history:
            return "No previous conversation."
        
        formatted = []
        for msg in history[-5:]:  # Use last 5 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        
        return "\n\n".join(formatted)


@dataclass
class ConversationContext:
    """Tracks the context of the current conversation."""
    gathered_requirements: Dict[str, Any] = field(default_factory=dict)
    missing_info: List[str] = field(default_factory=list)
    current_focus: str = "data_sources"  # data_sources, visualization, access
    conversation_history: List[Dict[str, str]] = field(default_factory=list)


def validate_etl_state(state: ETLState) -> bool:
    """Validate ETL state fields and their values."""
    required_fields = ["source", "destination", "execution_id"]
    return all(
        hasattr(state, field) and bool(getattr(state, field))
        for field in required_fields
    )


def is_etl_complete(state: ETLState) -> bool:
    """Check if the ETL process is complete."""
    return (
        state.extraction_status == "completed"
        and state.transformation_status == "completed"
        and state.loading_status == "completed"
        and not any([
            state.extraction_errors,
            state.transformation_errors,
            state.loading_errors
        ])
    )


def get_etl_stats(state: ETLState) -> Dict[str, Any]:
    """Get statistics about the ETL process."""
    return {
        "total_records": len(state.extracted_data),
        "transformed_records": len(state.transformed_data),
        "loaded_records": state.loaded_records,
        "status": "completed" if is_etl_complete(state) else "in_progress",
        "has_errors": bool(
            state.extraction_errors
            or state.transformation_errors
            or state.loading_errors
        ),
        "error_count": len(state.extraction_errors)
        + len(state.transformation_errors)
        + len(state.loading_errors)
    }


def add_etl_metadata(
    state: ETLState,
    key: str,
    value: Any
) -> ETLState:
    """Add metadata to the ETL state."""
    if state.metadata is None:
        state.metadata = {}
    state.metadata[key] = value
    return state


class ETLUtils:
    """Manages ETL workflow utilities."""

    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.prompt_manager = PromptManager()

    async def analyze_project(
        self,
        state: ETLWorkflowState
    ) -> ETLRequirements:
        """Analyze project requirements using LLM."""
        try:
            # Get conversation context
            context = state.metadata.get("conversation_history", [])
            latest_message = state.metadata.get("latest_message", "")
            
            logger.info("=== Current Conversation Context ===")
            logger.info(f"Latest Message: {latest_message}")
            logger.info(f"Conversation History Length: {len(context)}")
            logger.info("Last 3 Messages:")
            for msg in context[-3:]:
                logger.info(f"{msg['role']}: {msg['content']}")
            logger.info("=== End Context ===")
            
            # Check if we already have requirements in state
            current_requirements = None
            if "requirements" in state.metadata:
                try:
                    current_requirements = ETLRequirements(**state.metadata["requirements"])
                except Exception as e:
                    logger.error(f"Error parsing existing requirements: {str(e)}")
            
            # Extract volume information from the message if present
            volume_info = None
            if "volume" in latest_message.lower():
                volume_match = re.search(r'(\d+,?\d*)\s*(?:sales\s*)?records?\s*(?:daily|per\s*day)', latest_message)
                peak_match = re.search(r'(?:peak|up\s*to)\s*(\d+,?\d*)', latest_message)
                
                if volume_match:
                    daily_volume = int(volume_match.group(1).replace(',', ''))
                    peak_volume = int(peak_match.group(1).replace(',', '')) if peak_match else daily_volume
                    volume_info = {
                        "daily": daily_volume,
                        "peak": peak_volume
                    }
            
            # Create prompt for analysis using PromptManager
            prompt = self.prompt_manager.get_analysis_prompt(
                context=state.metadata,
                history=context,
                message=latest_message
            )
            
            # Get LLM response
            response = await self.llm_manager.invoke(prompt, {})
            
            # Parse response
            try:
                # Clean the response to ensure valid JSON
                cleaned_response = re.sub(r'```json\n|\n```', '', response)
                requirements_data = json.loads(cleaned_response)
                
                # Create ETLRequirements object
                requirements = ETLRequirements(**requirements_data)
                
                # Update volume information if we extracted it
                if volume_info:
                    requirements.volume = volume_info
                    # Remove volume from missing info if present
                    if "volume" in requirements.missing_info:
                        requirements.missing_info.remove("volume")
                    if "Data volume information is missing" in requirements.missing_info:
                        requirements.missing_info.remove("Data volume information is missing")
                elif current_requirements and current_requirements.volume:
                    requirements.volume = current_requirements.volume
                
                # If we have current requirements, merge any missing fields
                if current_requirements:
                    if not requirements.data_format and current_requirements.data_format:
                        requirements.data_format = current_requirements.data_format
                    if not requirements.data_sources and current_requirements.data_sources:
                        requirements.data_sources = current_requirements.data_sources
                
                # Extract volume from data_sources if present and volume is empty
                if not requirements.volume and "volume" in requirements.data_sources:
                    volume_str = requirements.data_sources["volume"]
                    volume_match = re.search(r'(\d+,?\d*)\s*records?\s*(?:per|a|each)?\s*(day|month|year)', volume_str)
                    if volume_match:
                        amount = int(volume_match.group(1).replace(',', ''))
                        period = volume_match.group(2)
                        if period == "day":
                            requirements.volume = {"daily": amount, "peak": amount * 2}
                        elif period == "month":
                            requirements.volume = {"daily": amount // 30, "peak": (amount // 30) * 2}
                        elif period == "year":
                            requirements.volume = {"daily": amount // 365, "peak": (amount // 365) * 2}
                
                return requirements
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing LLM response: {str(e)}")
                # Return current requirements if available, otherwise create default
                if current_requirements:
                    return current_requirements
                
                return ETLRequirements(
                    data_sources={
                        "type": "sales",
                        "format": "structured",
                        "processing": "batch"
                    },
                    processing_type="batch",
                    data_format="json",
                    volume=volume_info if volume_info else {"daily": 0, "peak": 0},
                    latency_requirements={
                        "batch_processing": "daily",
                        "dashboard_refresh": "hourly"
                    },
                    analytics_needs=[
                        "sales_trends",
                        "customer_segments",
                        "product_performance"
                    ],
                    missing_info=["Please provide more details about your sales data"],
                    next_question="What is the expected volume of sales data?",
                    summary="Initial analysis of sales data dashboard requirements"
                )
                
        except Exception as e:
            logger.error(f"Error in analyze_project: {str(e)}")
            # Return default requirements on error
            return ETLRequirements(
                data_sources={
                    "type": "sales",
                    "format": "structured",
                    "processing": "batch"
                },
                processing_type="batch",
                data_format="json",
                volume=volume_info if volume_info else {"daily": 0, "peak": 0},
                latency_requirements={
                    "batch_processing": "daily",
                    "dashboard_refresh": "hourly"
                },
                analytics_needs=[
                    "sales_trends",
                    "customer_segments",
                    "product_performance"
                ],
                missing_info=["Error occurred during analysis"],
                next_question="Could you please provide more details about your sales data?",
                summary="Error occurred during analysis"
            )
    
    async def generate_architecture(
        self,
        requirements: ETLRequirements
    ) -> ETLArchitecture:
        """Generate ETL architecture based on requirements."""
        try:
            # Create prompt for architecture generation
            prompt = f"""
            Generate an ETL architecture based on the following requirements:
            
            {json.dumps(requirements.model_dump(), indent=2)}
            
            Generate a response with the following structure:
            {{
                "components": {{
                    "data_source": "Sales Database",
                    "etl_pipeline": "Batch Processing Pipeline",
                    "data_warehouse": "Analytics Warehouse",
                    "dashboard": "Sales Analytics Dashboard"
                }},
                "data_flow": [
                    {{
                        "from": "data_source",
                        "to": "etl_pipeline",
                        "type": "batch"
                    }},
                    {{
                        "from": "etl_pipeline",
                        "to": "data_warehouse",
                        "type": "batch"
                    }},
                    {{
                        "from": "data_warehouse",
                        "to": "dashboard",
                        "type": "real-time"
                    }}
                ],
                "services": [
                    "AWS Glue",
                    "Amazon Redshift",
                    "Amazon QuickSight"
                ]
            }}
            """
            
            # Get LLM response
            response = await self.llm_manager.invoke(prompt, {})
            
            # Parse response
            try:
                # Clean the response to ensure valid JSON
                cleaned_response = re.sub(r'```json\n|\n```', '', response)
                architecture_data = json.loads(cleaned_response)
                
                # Create ETLArchitecture object
                return ETLArchitecture(**architecture_data)
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing architecture response: {str(e)}")
                # Return default architecture if parsing fails
                return ETLArchitecture(
                    components={
                        "data_source": "Sales Database",
                        "etl_pipeline": "Batch Processing Pipeline",
                        "data_warehouse": "Analytics Warehouse",
                        "dashboard": "Sales Analytics Dashboard"
                    },
                    data_flow=[
                        {
                            "from": "data_source",
                            "to": "etl_pipeline",
                            "type": "batch"
                        },
                        {
                            "from": "etl_pipeline",
                            "to": "data_warehouse",
                            "type": "batch"
                        },
                        {
                            "from": "data_warehouse",
                            "to": "dashboard",
                            "type": "real-time"
                        }
                    ],
                    services=[
                        "AWS Glue",
                        "Amazon Redshift",
                        "Amazon QuickSight"
                    ]
                )
                
        except Exception as e:
            logger.error(f"Error in generate_architecture: {str(e)}")
            # Return default architecture on error
            return ETLArchitecture(
                components={
                    "data_source": "Sales Database",
                    "etl_pipeline": "Batch Processing Pipeline",
                    "data_warehouse": "Analytics Warehouse",
                    "dashboard": "Sales Analytics Dashboard"
                },
                data_flow=[
                    {
                        "from": "data_source",
                        "to": "etl_pipeline",
                        "type": "batch"
                    },
                    {
                        "from": "etl_pipeline",
                        "to": "data_warehouse",
                        "type": "batch"
                    },
                    {
                        "from": "data_warehouse",
                        "to": "dashboard",
                        "type": "real-time"
                    }
                ],
                services=[
                    "AWS Glue",
                    "Amazon Redshift",
                    "Amazon QuickSight"
                ]
            )

    async def review_architecture(
        self, 
        architecture: ETLArchitecture
    ) -> Dict[str, Any]:
        """Review architecture and provide feedback."""
        prompt = self._create_review_prompt(architecture)
        response = await self.llm_manager.generate(prompt)
        
        return {
            "feedback": response.get("feedback", ""),
            "improvements": response.get("improvements", [])
        }

    async def correct_architecture(
        self, 
        architecture: ETLArchitecture,
        review_result: Dict[str, Any]
    ) -> ETLArchitecture:
        """Correct architecture based on review feedback."""
        prompt = self._create_correction_prompt(architecture, review_result)
        response = await self.llm_manager.generate(prompt)
        
        corrected_architecture = ETLArchitecture(
            components=response.get("components", architecture.components),
            data_flow=response.get("data_flow", architecture.data_flow),
            services=response.get("services", architecture.services),
            improvements_applied=True
        )
        
        return corrected_architecture

    async def generate_questions(
        self,
        state: ETLWorkflowState
    ) -> List[str]:
        """Generate follow-up questions based on current context."""
        prompt = self._create_questions_prompt(self.conversation_context)
        response = self.llm_manager.invoke(
            template=prompt,
            input_data={}
        )
        
        try:
            response_dict = json.loads(response)
            questions = response_dict.get("questions", [])
            for question in questions:
                state.add_question(question)
            return questions
        except json.JSONDecodeError:
            raise ValueError("Failed to parse LLM response as JSON")

    async def generate_terraform(
        self, 
        architecture: ETLArchitecture
    ) -> ETLTerraform:
        """Generate Terraform code for the architecture."""
        prompt = self._create_terraform_prompt(architecture)
        response = await self.llm_manager.generate(prompt)
        
        return ETLTerraform(code=response.get("code", ""))

    async def review_terraform(
        self, 
        terraform: ETLTerraform
    ) -> Dict[str, Any]:
        """Review Terraform code and provide feedback."""
        prompt = self._create_terraform_review_prompt(terraform)
        response = await self.llm_manager.generate(prompt)
        
        return {
            "feedback": response.get("feedback", ""),
            "improvements": response.get("improvements", [])
        }

    async def correct_terraform(
        self, 
        terraform: ETLTerraform,
        review_result: Dict[str, Any]
    ) -> ETLTerraform:
        """Correct Terraform code based on review feedback."""
        prompt = self._create_terraform_correction_prompt(terraform, review_result)
        response = await self.llm_manager.generate(prompt)
        
        return ETLTerraform(
            code=response.get("code", terraform.code),
            improvements_applied=True
        )

    def _create_review_prompt(
        self,
        architecture: ETLArchitecture
    ) -> str:
        comps = architecture.components
        flow = architecture.data_flow
        svcs = architecture.services
        
        return (
            "Review the following architecture:\n"
            f"Components: {comps}\n"
            f"Data Flow: {flow}\n"
            f"Services: {svcs}\n\n"
            "Provide feedback on:\n"
            "1. Technical feasibility\n"
            "2. Cost optimization\n"
            "3. Performance considerations\n"
            "4. Security aspects"
        )

    def _create_correction_prompt(
        self, 
        architecture: ETLArchitecture,
        review_result: Dict[str, Any]
    ) -> str:
        return (
            "Correct the following architecture based on the review feedback:\n"
            f"Original Architecture: {architecture}\n"
            f"Review Feedback: {review_result}\n\n"
            "Provide an improved architecture addressing all feedback points."
        )

    def _create_questions_prompt(
        self, 
        context: ConversationContext
    ) -> str:
        """Create prompt for question generation."""
        focus = context.current_focus
        reqs = context.gathered_requirements
        missing = context.missing_info
        history = self._format_conversation_history(
            context.conversation_history
        )
        
        state = (
            "Current state:\n"
            f"- Focus: {focus}\n"
            "- Requirements:\n"
            f"{reqs}\n"
            "- Missing info:\n"
            f"{missing}\n\n"
        )
        
        return (
            "Generate questions based on context:\n\n"
            f"{state}"
            "Previous conversation:\n"
            f"{history}\n\n"
            "Create 3-5 questions for missing details.\n"
            "Focus on current area.\n\n"
            "Format as JSON with:\n"
            "- questions\n"
            "- focus_area"
        )

    def _format_conversation_history(
        self, 
        history: List[Dict[str, str]]
    ) -> str:
        """Format conversation history."""
        formatted = []
        # Only use last 5 messages
        recent = history[-5:]
        for msg in recent:
            role = msg['role']
            content = msg['content']
            entry = f"{role}:\n{content}"
            formatted.append(entry)
        return "\n".join(formatted)

    def _create_terraform_prompt(
        self,
        architecture: ETLArchitecture
    ) -> str:
        """Generate Terraform prompt."""
        sections = [
            "Generate Terraform code:",
            f"{architecture}\n",
            "Include:",
            "1. AWS setup",
            "2. IAM roles", 
            "3. Network",
            "4. Security",
            "5. Monitor"
        ]
        return "\n".join(sections)

    def _create_terraform_review_prompt(
        self,
        terraform: ETLTerraform
    ) -> str:
        """Create review prompt."""
        sections = [
            "Review code:",
            f"{terraform.code}\n",
            "Check:",
            "1. Best practices",
            "2. Security", 
            "3. Cost",
            "4. Performance"
        ]
        return "\n".join(sections)

    def _create_terraform_correction_prompt(
        self, 
        terraform: ETLTerraform,
        review_result: Dict[str, Any]
    ) -> str:
        """Create correction prompt."""
        sections = [
            "Update code:",
            f"Code:\n{terraform.code}",
            f"Review:\n{review_result}\n",
            "Provide improved code with fixes."
        ]
        return "\n".join(sections) 