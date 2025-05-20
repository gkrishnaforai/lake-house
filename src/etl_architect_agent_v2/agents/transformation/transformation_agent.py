"""Transformation Agent.

This agent applies transformations to data using LangChain and LangGraph.
"""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools import BaseTool
import json
import logging
from core.llm.manager import LLMManager
from src.core.sql.sql_state import TransformationOutput
from config.transformation_tools import TransformationConfig, TransformationInput


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TransformationTool(BaseTool):
    """Tool for applying transformations to data."""
    
    name: str = "apply_transformation"
    description: str = "Apply transformations to data using LangGraph"
    args_schema: type[BaseModel] = TransformationInput
    llm: Any = Field(..., description="LLM instance for transformations")
    
    def __init__(self, llm):
        """Initialize the tool with an LLM instance."""
        super().__init__(llm=llm)
    
    def _run(
        self,
        input_data: TransformationInput
    ) -> Dict[str, Any]:
        """Apply transformations to data."""
        try:
            # Validate input fields
            input_data.validate_fields()
            
            logger.debug("Applying transformations")
            logger.debug(
                f"Input data: {json.dumps(input_data.data[:2], indent=2)}"
            )
            logger.debug(
                f"Transformation type: {input_data.transformation_type}"
            )
            logger.debug(f"Batch size: {input_data.batch_size}")
            
            # Create a prompt for transformation
            aspects = []
            if input_data.metadata and "aspects" in input_data.metadata:
                aspects = input_data.metadata["aspects"]
            aspects_instruction = ""
            if aspects:
                aspects_instruction = (
                    f"\nFor sentiment analysis, always include all aspects from this list in 'aspect_sentiments': {aspects}. "
                    "If an aspect is not mentioned in the review, set its sentiment to 'neutral' and score to 0.5.\n"
                )
            prompt = f"""Apply the following transformation to the data:
            Transformation Type: {input_data.transformation_type}
            Batch Size: {input_data.batch_size}
            Data: {json.dumps(input_data.data, indent=2)}
            
            Configuration:
            {json.dumps(input_data.config.dict() if input_data.config else {}, indent=2)}
            
            {aspects_instruction}
            For each record, apply the transformation and return:
            1. The transformed record with the following structure based on transformation type:
               - For sentiment analysis:
                 {{
                   "overall_sentiment": "positive/negative/neutral",
                   "sentiment_score": <float between 0 and 1>,
                   "aspect_sentiments": {{
                     "<aspect>": {{
                       "sentiment": "positive/negative/neutral",
                       "score": <float between 0 and 1>
                     }}
                   }}
                 }}
               - For categorization:
                 {{
                   "category": "<category name>",
                   "confidence": <float between 0 and 1>,
                   "details": {{
                     "is_ai_company": <boolean>,
                     "reasoning": "<explanation>",
                     "ai_technologies": ["<technology1>", "<technology2>"],
                     "ai_focus_areas": ["<area1>", "<area2>"]
                   }}
                 }}
            2. Any errors encountered
            3. Metadata about the transformation
            
            Return the results in this JSON format:
            {{
                "transformed_data": [
                    {{"original": {{...}}, "transformed": {{...}}, 
                    "metadata": {{...}}}},
                    ...
                ],
                "errors": [
                    {{"record": {{...}}, "error": "error message"}},
                    ...
                ],
                "metadata": {{
                    "transformation_type": "...",
                    "total_records": 0,
                    "successful_transformations": 0,
                    "failed_transformations": 0,
                    "configuration_used": {{...}},
                    "analysis_summary": {{
                        "positive_reviews": 0,
                        "negative_reviews": 0,
                        "neutral_reviews": 0,
                        "ai_companies": 0,
                        "non_ai_companies": 0,
                        "average_confidence": 0.0,
                        "common_ai_technologies": []
                    }}
                }}
            }}"""
            
            # Get response from LLM
            response = self.llm.invoke(prompt)
            
            # Extract content from AIMessage
            if hasattr(response, 'content'):
                response_content = response.content
            else:
                response_content = str(response)
            
            # Parse the response
            try:
                # Try to parse as JSON directly
                transformation_results = json.loads(response_content)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from markdown
                import re
                json_match = re.search(
                    r'```(?:json)?\s*(\{[\s\S]*?\})\s*```',
                    response_content
                )
                if json_match:
                    transformation_results = json.loads(json_match.group(1))
                else:
                    raise ValueError(
                        "Could not find valid JSON in response"
                    )
            
            # Validate the response structure
            required_keys = [
                "transformed_data",
                "errors",
                "metadata"
            ]
            for key in required_keys:
                if key not in transformation_results:
                    transformation_results[key] = (
                        [] if key != "metadata" else {}
                    )
                elif not isinstance(
                    transformation_results[key], (list, dict)
                ):
                    transformation_results[key] = (
                        [] if key != "metadata" else {}
                    )
            
            # Add configuration to metadata
            if input_data.config:
                transformation_results["metadata"]["configuration_used"] = (
                    input_data.config.dict()
                )
            
            # --- POST-PROCESSING: Rename keys for categorization ---
            if input_data.transformation_type == "categorization":
                for item in transformation_results.get("transformed_data", []):
                    transformed = item.get("transformed", {})
                    details = transformed.get("details", {})
                    # Only process if is_ai_company is present
                    if "is_ai_company" in details:
                        prefix = "is_ai_company"
                        # Move confidence and reasoning to prefixed keys
                        if "confidence" in transformed:
                            transformed[f"{prefix}_ai_confidence"] = transformed.pop("confidence")
                        if "reasoning" in details:
                            transformed[f"{prefix}_ai_reasoning"] = details.pop("reasoning")
                        # Optionally, keep is_ai_company as a top-level key
                        transformed[prefix] = details.get("is_ai_company", None)
                        # Remove old keys if needed
                        if "details" in transformed:
                            del transformed["details"]
            
            return transformation_results
            
        except Exception as e:
            logger.error(
                f"Error applying transformation: {str(e)}", exc_info=True
            )
            raise ValueError(f"Error applying transformation: {str(e)}")
    
    def _arun(
        self,
        input_data: TransformationInput
    ) -> Dict[str, Any]:
        """Async implementation of _run."""
        return self._run(input_data)


class TransformationAgent:
    """Agent for applying transformations to data."""
    
    def __init__(self, llm_manager: LLMManager):
        """Initialize the agent."""
        self.llm = llm_manager.llm
        self.tools = [TransformationTool(llm=self.llm)]
        
        # Create the agent prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data transformation expert. Your task is to 
            apply transformations to data using LangGraph. Follow these 
            guidelines:
            1. Process data in batches for efficiency
            2. Handle errors gracefully
            3. Preserve original data
            4. Add metadata about transformations
            5. Return structured results
            6. Apply configuration parameters correctly
            7. Validate input data against configuration"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create the agent
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Create the agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True
        )
    
    async def apply_transformation(
        self,
        data: List[Dict[str, Any]],
        transformation_type: str,
        batch_size: int = 100,
        metadata: Optional[Dict[str, Any]] = None,
        config: Optional[TransformationConfig] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> TransformationOutput:
        """Apply transformations to data."""
        try:
            logger.debug("Starting transformation")
            logger.debug(
                f"Input data: {json.dumps(data[:2], indent=2)}"
            )
            logger.debug(
                f"Transformation type: {transformation_type}"
            )
            logger.debug(f"Batch size: {batch_size}")
            
            # Prepare chat history
            messages = []
            if chat_history:
                for msg in chat_history:
                    if msg["role"] == "user":
                        messages.append(
                            HumanMessage(content=msg["content"])
                        )
                    else:
                        messages.append(
                            AIMessage(content=msg["content"])
                        )
            
            # Directly use the parameters we have
            tool_input = TransformationInput(
                data=data,
                transformation_type=transformation_type,
                batch_size=batch_size,
                metadata=metadata,
                config=config
            )
            logger.debug(
                f"Using tool input: {json.dumps(tool_input.dict(), indent=2)}"
            )
            
            # Call the tool directly instead of through the agent
            transformation_tool = self.tools[0]  # Get the TransformationTool
            result = transformation_tool._run(tool_input)  # Use _run
            
            logger.debug(
                f"Tool execution result: {json.dumps(result, indent=2)}"
            )
            
            # Convert dictionary result to TransformationOutput object
            transformation_output = TransformationOutput(
                transformed_data=result.get("transformed_data", []),
                errors=result.get("errors", []),
                metadata=result.get("metadata", {}),
                explanation=result.get("explanation"),
                confidence=result.get("confidence")
            )
            if not transformation_output.transformed_data:
                logger.warning("No data was transformed")
            return transformation_output
            
        except Exception as e:
            logger.error(
                f"Error applying transformation: {str(e)}", exc_info=True
            )
            raise ValueError(f"Error applying transformation: {str(e)}") 