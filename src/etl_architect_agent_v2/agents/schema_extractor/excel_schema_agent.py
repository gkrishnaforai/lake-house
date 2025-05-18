"""Excel Schema Extractor Agent.

This agent extracts schema from Excel data using LangChain.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools import BaseTool
import pandas as pd
import json
import logging
from core.llm.manager import LLMManager


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExcelSchemaModel(BaseModel):
    """Excel Schema model."""
    schema_version: str = Field(
        default="excel-schema-v1",
        alias="schema"
    )
    sheets: Dict[str, Dict[str, Any]] = Field(...)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LoadExcelTool(BaseTool):
    """Tool for loading Excel data from a file."""
    
    name: str = "load_excel"
    description: str = "Load and parse Excel data from a file"
    
    def _run(self, file_path: str) -> Dict[str, Any]:
        """Load Excel data from file."""
        try:
            logger.debug(f"Loading Excel file from: {file_path}")
            
            # Load all sheets into a dictionary
            excel_data = pd.read_excel(file_path, sheet_name=None)
            
            if not excel_data:
                raise ValueError("No data found in Excel file")
            
            # Convert each sheet to a dictionary
            result = {}
            for sheet_name, df in excel_data.items():
                # Convert DataFrame to dict with proper types
                sheet_data = {
                    "columns": df.columns.tolist(),
                    "data_types": df.dtypes.astype(str).to_dict(),
                    "sample_data": df.head(5).to_dict(orient='records'),
                    "row_count": len(df),
                    "column_count": len(df.columns)
                }
                result[sheet_name] = sheet_data
            
            logger.debug(f"Successfully loaded {len(result)} sheets")
            return result
            
        except Exception as e:
            logger.error(f"Error loading Excel file: {str(e)}", exc_info=True)
            raise ValueError(f"Error loading Excel file: {str(e)}")


class ExcelSchemaExtractorAgent:
    """Agent for extracting schema from Excel data."""
    
    def __init__(self, llm_manager: LLMManager):
        """Initialize the agent."""
        self.llm = llm_manager.llm
        self.tools = [LoadExcelTool()]
        
        # Create the agent prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an Excel Schema expert. Your task is to analyze 
            Excel data and generate a complete schema that describes its structure. 
            Follow these guidelines:
            1. Analyze each sheet's structure and data types
            2. Identify relationships between sheets if any
            3. Document column names, types, and sample values
            4. Note any data validation or constraints
            5. IMPORTANT: You must return ONLY a valid JSON object with this exact structure:
               {{
                 "schema": "excel-schema-v1",
                 "sheets": {{
                   "sheet_name": {{
                     "columns": ["col1", "col2"],
                     "data_types": {{"col1": "type1", "col2": "type2"}},
                     "sample_data": [{{"col1": "val1", "col2": "val2"}}],
                     "row_count": 10,
                     "column_count": 2
                   }}
                 }},
                 "metadata": {{
                   "total_sheets": 1,
                   "total_rows": 10,
                   "total_columns": 2
                 }}
               }}
            Do not include any other text, markdown formatting, or explanations. Return only the JSON object."""),
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
    ) -> ExcelSchemaModel:
        """Extract schema from Excel data."""
        try:
            logger.debug("Starting Excel schema extraction")
            
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
                "input": f"Extract schema from Excel file: {file_path}",
                "chat_history": messages
            })
            
            # Extract JSON from the output
            output = result["output"]
            try:
                # Try to parse as JSON directly
                schema_dict = json.loads(output)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from markdown
                import re
                # Try to find JSON in markdown code blocks
                json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', output)
                if json_match:
                    try:
                        schema_dict = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        # If still fails, try to find any JSON-like structure
                        json_match = re.search(r'(\{[\s\S]*?\})', output)
                        if json_match:
                            try:
                                schema_dict = json.loads(json_match.group(1))
                            except json.JSONDecodeError:
                                raise ValueError("Could not find valid JSON in output")
                        else:
                            raise ValueError("Could not find JSON structure in output")
                else:
                    raise ValueError("Could not find JSON in markdown code blocks")
            
            # Parse the result
            schema = ExcelSchemaModel.parse_obj(schema_dict)
            logger.debug(f"Generated schema: {schema.dict()}")
            
            return schema
            
        except Exception as e:
            logger.error(
                f"Error extracting Excel schema: {str(e)}", exc_info=True
            )
            raise ValueError(f"Error extracting Excel schema: {str(e)}") 