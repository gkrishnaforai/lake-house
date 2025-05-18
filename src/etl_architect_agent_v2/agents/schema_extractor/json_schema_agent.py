"""JSON Schema Extractor Agent.

This agent extracts JSON schema from JSON data using LangChain.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools import BaseTool
from langchain_community.document_loaders import JSONLoader
import json
import logging
from core.llm.manager import LLMManager


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JsonSchemaModel(BaseModel):
    """JSON Schema model."""
    schema_version: str = Field(
        default="http://json-schema.org/draft-04/schema#",
        alias="schema"
    )
    type: str = Field("object")
    properties: Dict[str, Any] = Field(...)
    required: List[str] = Field(default_factory=list)


class LoadJsonTool(BaseTool):
    """Tool for loading JSON data from a file."""
    
    name: str = "load_json"
    description: str = "Load and parse JSON data from a file"
    
    def _run(self, file_path: str) -> Dict[str, Any]:
        """Load JSON data from file."""
        try:
            logger.debug(f"Loading JSON file from: {file_path}")
            loader = JSONLoader(
                file_path=file_path,
                jq_schema='.',
                text_content=False
            )
            data = loader.load()
            
            if not data:
                raise ValueError("No data found in JSON file")
            
            logger.debug(
                f"Successfully loaded data: {data[0].page_content[:100]}..."
            )
            
            if isinstance(data[0].page_content, str):
                return json.loads(data[0].page_content)
            return data[0].page_content
            
        except Exception as e:
            logger.error(f"Error loading JSON file: {str(e)}", exc_info=True)
            raise ValueError(f"Error loading JSON file: {str(e)}")


class JsonSchemaExtractorAgent:
    """Agent for extracting JSON schema from data."""
    
    def __init__(self, llm_manager: LLMManager):
        """Initialize the agent."""
        self.llm = llm_manager.llm
        self.tools = [LoadJsonTool()]
        
        # Create the agent prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a JSON Schema expert. Your task is to analyze 
            JSON data and generate a complete JSON Schema that describes its 
            structure. Follow these guidelines:
            1. Include all nested objects, arrays, and their types
            2. Specify required fields
            3. Use proper type definitions
            4. Follow JSON Schema Draft-04 specification
            5. Return only the raw JSON schema without markdown formatting"""),
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
    
    async def extract_schema(
        self,
        file_path: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> JsonSchemaModel:
        """Extract JSON schema from data."""
        try:
            logger.debug("Starting JSON schema extraction")
            
            # Prepare chat history
            messages = []
            if chat_history:
                for msg in chat_history:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    else:
                        messages.append(AIMessage(content=msg["content"]))
            
            # Run the agent
            result = await self.agent_executor.ainvoke({
                "input": f"Extract JSON schema from file: {file_path}",
                "chat_history": messages
            })
            
            # Parse the result
            schema = JsonSchemaModel.parse_raw(result["output"])
            logger.debug(f"Generated schema: {schema.dict()}")
            
            return schema
            
        except Exception as e:
            logger.error(
                f"Error extracting JSON schema: {str(e)}", exc_info=True
            )
            raise ValueError(f"Error extracting JSON schema: {str(e)}") 