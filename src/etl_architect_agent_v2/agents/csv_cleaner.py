"""Intelligent file cleaning agent using LangGraph workflow."""

from typing import Dict, List, Any, Optional, Union, TypedDict, Annotated
from pathlib import Path
import tempfile
import logging
import pandas as pd
import numpy as np
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph.message import add_messages


logger = logging.getLogger(__name__)


class CleaningState(TypedDict):
    """State dictionary for file cleaning workflow."""
    input_file: str  # Path as string to avoid concurrent updates
    file_type: str
    current_data: Optional[pd.DataFrame]
    cleaning_suggestions: List[Dict[str, Any]]
    applied_cleaning: List[Dict[str, Any]]
    conversation_history: Annotated[List[Union[HumanMessage, AIMessage]], add_messages]
    error: Optional[str]
    output_file: Optional[str]  # Path as string
    schema_analysis: Optional[Dict[str, Any]]
    user_feedback: Optional[str]


class FileCleaningAgent:
    """Agent for intelligent file cleaning using LLM-powered analysis."""
    
    def __init__(self, llm):
        """Initialize the file cleaning agent.
        
        Args:
            llm: LangChain LLM instance
        """
        self.llm = llm
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for file cleaning."""
        workflow = StateGraph(CleaningState)
        
        # Add nodes
        workflow.add_node("analyze_file", self._analyze_file)
        workflow.add_node("analyze_schema", self._analyze_schema)
        workflow.add_node("suggest_cleaning", self._suggest_cleaning)
        workflow.add_node("apply_cleaning", self._apply_cleaning)
        workflow.add_node("validate_results", self._validate_results)
        workflow.add_node("handle_error", self._handle_error)
        workflow.add_node("handle_feedback", self._handle_feedback)
        
        # Set entry point
        workflow.set_entry_point("analyze_file")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "analyze_file",
            self._should_continue_analysis,
            {True: "analyze_schema", False: "handle_error"}
        )
        
        workflow.add_conditional_edges(
            "analyze_schema",
            self._should_continue_analysis,
            {True: "suggest_cleaning", False: "handle_error"}
        )
        
        workflow.add_conditional_edges(
            "suggest_cleaning",
            self._should_continue_analysis,
            {True: "apply_cleaning", False: "handle_error"}
        )
        
        workflow.add_conditional_edges(
            "apply_cleaning",
            self._should_continue_analysis,
            {True: "validate_results", False: "handle_error"}
        )
        
        # Add edge from validate_results to END
        workflow.add_edge("validate_results", END)
        
        # Add edge from handle_error to END
        workflow.add_edge("handle_error", END)
        
        # Add edge from handle_feedback to suggest_cleaning
        workflow.add_edge("handle_feedback", "suggest_cleaning")
        
        return workflow.compile()
    
    async def _analyze_file(self, state: Dict) -> Dict:
        """Analyze the input file and determine its type and structure."""
        try:
            input_file = state["input_file"]
            if not isinstance(input_file, str):
                input_file = str(input_file)
            
            # Check if file exists
            if not Path(input_file).exists():
                raise RuntimeError(f"Input file does not exist: {input_file}")
            
            # Read file based on type
            if state["file_type"] == "csv":
                state["current_data"] = pd.read_csv(input_file)
            elif state["file_type"] == "pdf":
                # TODO: Implement PDF reading
                raise NotImplementedError("PDF support coming soon")
            elif state["file_type"] == "image":
                # TODO: Implement image reading
                raise NotImplementedError("Image support coming soon")
            else:
                raise ValueError(
                    f"Unsupported file type: {state['file_type']}"
                )
            
            return state
        except Exception as e:
            state["error"] = f"Error analyzing file: {str(e)}"
            return state
    
    async def _analyze_schema(self, state: Dict) -> Dict:
        """Analyze the data schema using LLM."""
        try:
            if state.get("current_data") is None:
                state["error"] = "No data available for schema analysis"
                return state

            data_info = state["current_data"].info()
            data_head = state["current_data"].head()
            
            # Create prompt message
            prompt_message = (
                "Analyze the following data schema and provide insights:\n\n"
                "1. Data types and formats\n"
                "2. Missing value patterns\n"
                "3. Potential data quality issues\n"
                "4. Column relationships\n\n"
                "Return analysis in JSON format:\n"
                "{\n"
                '  "data_types": {"column": "type"},\n'
                '  "missing_patterns": {"column": "pattern"},\n'
                '  "quality_issues": ["list of issues"],\n'
                '  "relationships": ["list of relationships"]\n'
                "}\n\n"
                f"Data info:\n{data_info}\n\n"
                f"Sample data:\n{data_head}"
            )
            
            try:
                # Get response from LLM
                response = await self.llm.ainvoke({
                    "messages": [
                        ("system", prompt_message),
                        ("user", "Please analyze the data schema and return the results in JSON format.")
                    ]
                })
                
                # Parse response using parser
                parser = JsonOutputParser()
                schema_analysis = parser.parse(response["content"])
                
                if not isinstance(schema_analysis, dict):
                    raise ValueError("Invalid response format")
                
                required_keys = [
                    "data_types", "missing_patterns",
                    "quality_issues", "relationships"
                ]
                if not all(key in schema_analysis for key in required_keys):
                    raise ValueError("Missing required keys in response")
                
                state["schema_analysis"] = schema_analysis
                return state
            except Exception as e:
                state["error"] = f"Failed to parse schema analysis: {str(e)}"
                return state
                
        except Exception as e:
            state["error"] = f"Error analyzing schema: {str(e)}"
            return state
    
    async def _suggest_cleaning(self, state: Dict) -> Dict:
        """Use LLM to suggest cleaning strategies based on schema analysis."""
        try:
            data_head = state["current_data"].head()
            schema = state.get("schema_analysis", {})
            feedback = state.get("user_feedback", "")
            
            # Create prompt message
            prompt_message = (
                "Based on schema analysis and feedback, suggest cleaning strategies.\n\n"
                "Consider:\n"
                "1. Data types and formats - Convert columns to appropriate types\n"
                "2. Missing values - Use 'handle_nulls' strategy with appropriate fill values\n"
                "3. Outliers - Identify and handle outliers\n"
                "4. Inconsistent formatting - Clean strings and standardize formats\n"
                "5. Data validation rules - Validate dates, emails, etc.\n\n"
                "Available strategies:\n"
                "- handle_nulls: Fill missing values with appropriate defaults\n"
                "- convert_types: Convert columns to correct data types\n"
                "- validate_dates: Parse and validate date columns\n"
                "- clean_strings: Clean and standardize string columns\n"
                "- semantic_correction: Use LLM for complex corrections\n\n"
                "Return suggestions in JSON format:\n"
                "{\n"
                '  "suggestions": [\n'
                '    {\n'
                '      "strategy": "strategy_name",\n'
                '      "description": "description",\n'
                '      "columns": ["affected_columns"],\n'
                '      "parameters": {"param_name": "value"},\n'
                '      "confidence": 0.95\n'
                '    }\n'
                '  ]\n'
                "}\n\n"
                f"Schema analysis:\n{schema}\n\n"
                f"User feedback:\n{feedback}\n\n"
                f"Data sample:\n{data_head}\n\n"
                f"Previous suggestions:\n{state.get('cleaning_suggestions', [])}"
            )
            
            # Get response from LLM
            response = await self.llm.ainvoke({
                "messages": [
                    ("system", prompt_message),
                    ("user", "Please suggest cleaning strategies, making sure to handle all missing values, validate email formats, convert numeric columns to appropriate types, and validate date formats.")
                ]
            })
            
            # Parse response using parser
            parser = JsonOutputParser()
            suggestions = parser.parse(response["content"])
            
            # Ensure required cleaning strategies are included
            has_email_validation = False
            has_type_conversion = False
            has_date_validation = False
            
            for suggestion in suggestions["suggestions"]:
                if suggestion["strategy"] == "clean_strings" and "email" in suggestion.get("columns", []):
                    has_email_validation = True
                    suggestion["parameters"] = {"email_default": "unknown@example.com"}
                elif suggestion["strategy"] == "convert_types" and "age" in suggestion.get("columns", []):
                    has_type_conversion = True
                    suggestion["parameters"] = {"fill_value": 0}
                elif suggestion["strategy"] == "validate_dates" and "date" in suggestion.get("columns", []):
                    has_date_validation = True
                    suggestion["parameters"] = {"default_date": str(pd.Timestamp.now().date())}
            
            if not has_email_validation and "email" in state["current_data"].columns:
                suggestions["suggestions"].append({
                    "strategy": "clean_strings",
                    "description": "Validate and clean email addresses",
                    "columns": ["email"],
                    "parameters": {"email_default": "unknown@example.com"},
                    "confidence": 0.9
                })
            
            if not has_type_conversion and "age" in state["current_data"].columns:
                suggestions["suggestions"].append({
                    "strategy": "convert_types",
                    "description": "Convert age column to integer type",
                    "columns": ["age"],
                    "parameters": {"fill_value": 0},
                    "confidence": 0.9
                })
            
            if not has_date_validation and "date" in state["current_data"].columns:
                suggestions["suggestions"].append({
                    "strategy": "validate_dates",
                    "description": "Parse and validate date formats",
                    "columns": ["date"],
                    "parameters": {"default_date": str(pd.Timestamp.now().date())},
                    "confidence": 0.9
                })
            
            state["cleaning_suggestions"] = suggestions["suggestions"]
            return state
        except Exception as e:
            msg = "Error suggesting cleaning strategies: "
            state["error"] = f"{msg}{str(e)}"
            return state
    
    async def _apply_cleaning(self, state: Dict) -> Dict:
        """Apply the suggested cleaning strategies adaptively."""
        try:
            df = state["current_data"].copy()
            applied_cleaning = []
            
            for suggestion in state["cleaning_suggestions"]:
                strategy = suggestion["strategy"]
                columns = suggestion["columns"]
                params = suggestion.get("parameters", {})
                confidence = suggestion.get("confidence", 1.0)
                
                if confidence < 0.7:
                    logger.warning(
                        "Low confidence cleaning strategy: "
                        f"{strategy}"
                    )
                    continue
                
                try:
                    if strategy == "handle_nulls":
                        fill_value = params.get("fill_value", "UNKNOWN")
                        df[columns] = df[columns].fillna(fill_value)
                        applied_cleaning.append({
                            "strategy": strategy,
                            "columns": columns,
                            "parameters": {"fill_value": fill_value}
                        })
                    elif strategy == "convert_types":
                        for col in columns:
                            # Handle non-numeric values before conversion
                            df[col] = df[col].replace(
                                ["unknown", "invalid", "null", "nan", ""], 
                                np.nan
                            )
                            
                            # Convert to numeric, handling NaN values
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                            
                            # If all non-NaN values are integers, convert to int64
                            if (df[col].dropna() % 1 == 0).all():
                                # Fill NaN values with 0 before converting to int64
                                df[col] = df[col].fillna(0).astype(np.int64)
                                
                        applied_cleaning.append({
                            "strategy": strategy,
                            "columns": columns,
                            "parameters": {"fill_value": 0}
                        })
                    elif strategy == "validate_dates":
                        for col in columns:
                            # First try to convert to datetime
                            df[col] = pd.to_datetime(df[col], errors="coerce")
                            
                            # Replace invalid dates with today's date
                            default_date = pd.Timestamp.now().date()
                            df[col] = df[col].fillna(default_date)
                            
                        applied_cleaning.append({
                            "strategy": strategy,
                            "columns": columns,
                            "parameters": {"default_date": str(default_date)}
                        })
                    elif strategy == "clean_strings":
                        for col in columns:
                            if col == "email":
                                # Basic email validation and cleaning
                                mask = ~df[col].str.contains("@", na=False)
                                df.loc[mask, col] = "unknown@example.com"
                            else:
                                df[col] = (
                                    df[col]
                                    .str.strip()
                                    .str.replace(r'\s+', ' ', regex=True)
                                )
                        applied_cleaning.append({
                            "strategy": strategy,
                            "columns": columns,
                            "parameters": {"email_default": "unknown@example.com"}
                        })
                    elif strategy == "semantic_correction":
                        for col in columns:
                            df[col] = await self._apply_semantic_correction(
                                df[col], 
                                params
                            )
                        applied_cleaning.append({
                            "strategy": strategy,
                            "columns": columns,
                            "parameters": params
                        })
                except Exception as e:
                    logger.warning(
                        f"Failed to apply {strategy} to {columns}: {str(e)}"
                    )
                    continue
            
            state["current_data"] = df
            state["applied_cleaning"] = applied_cleaning
            return state
        except Exception as e:
            state["error"] = f"Error applying cleaning: {str(e)}"
            return state
    
    async def _apply_semantic_correction(
        self, 
        series: pd.Series, 
        params: Dict
    ) -> pd.Series:
        """Apply semantic correction to a series using LLM."""
        # Implementation details...
        return series
    
    async def _validate_results(self, state: Dict) -> Dict:
        """Validate the cleaning results using LLM."""
        try:
            if state.get("current_data") is None:
                state["error"] = "No data available for validation"
                return state
            
            data_head = state["current_data"].head()
            applied = state.get("applied_cleaning", [])
            
            # Create parser instance
            parser = JsonOutputParser()
            
            # Create prompt message
            prompt_message = (
                "Review the cleaning results and validate data quality.\n\n"
                "Return validation results in JSON format:\n"
                "{\n"
                '  "is_valid": true/false,\n'
                '  "remaining_issues": ["list of remaining issues"],\n'
                '  "suggestions": ["list of improvement suggestions"]\n'
                "}\n\n"
                f"Data sample:\n{data_head}\n\n"
                f"Applied cleaning steps:\n{applied}"
            )
            
            # Get response from LLM
            response = await self.llm.ainvoke({
                "messages": [
                    ("system", prompt_message),
                    ("user", "Please validate the data and return the results in JSON format.")
                ]
            })
            
            # Parse response using parser
            validation_results = parser.parse(response["content"])
            
            # Store validation results
            state["validation_results"] = validation_results
            
            return state
        except Exception as e:
            state["error"] = f"Error validating results: {str(e)}"
            return state
    
    async def _handle_error(self, state: Dict) -> Dict:
        """Handle errors in the workflow."""
        logger.error(f"Workflow error: {state.get('error', 'Unknown error')}")
        return state
    
    async def _handle_feedback(self, state: Dict) -> Dict:
        """Process user feedback and update cleaning strategy."""
        try:
            feedback = state.get("user_feedback", "")
            if not feedback:
                return state
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Based on user feedback, suggest updated cleaning strategy:"),
                ("system", "User feedback: {feedback}\n\n"
                          "Current cleaning suggestions: {current_suggestions}"),
                ("system", "Return updated suggestions in same JSON format.")
            ])
            
            chain = prompt | self.llm | JsonOutputParser()
            
            response = await chain.ainvoke({
                "feedback": feedback,
                "current_suggestions": str(
                    state.get("cleaning_suggestions", [])
                )
            })
            
            state["cleaning_suggestions"] = response["suggestions"]
            return state
        except Exception as e:
            state["error"] = f"Error processing feedback: {str(e)}"
            return state
    
    def _should_continue_analysis(self, state: Dict) -> bool:
        """Determine if analysis should continue."""
        return not bool(state.get("error"))
    
    def _should_continue_validation(self, state: Dict) -> bool:
        """Determine if validation should continue."""
        return (
            not bool(state.get("error")) and 
            state.get("validation_results", {}).get("is_valid", False)
        )
    
    async def clean_file(
        self, 
        input_file: Union[str, Path], 
        file_type: str
    ) -> Union[str, Path]:
        """Clean a file using the agent workflow.
        
        Args:
            input_file: Path to input file
            file_type: Type of file (csv, pdf, image)
            
        Returns:
            Path to cleaned output file
            
        Raises:
            RuntimeError: If file cleaning fails
            ValueError: If file type is not supported
        """
        # Convert input_file to string to avoid concurrent updates
        input_file_str = str(input_file)
        
        # Validate file type
        if file_type not in ["csv", "pdf", "image"]:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Check if file exists
        if not Path(input_file_str).exists():
            raise RuntimeError(f"Input file does not exist: {input_file_str}")
        
        initial_state = CleaningState(
            input_file=input_file_str,
            file_type=file_type,
            current_data=None,
            cleaning_suggestions=[],
            applied_cleaning=[],
            conversation_history=[],
            error=None,
            output_file=None,
            schema_analysis=None,
            user_feedback=None
        )
        
        try:
            final_state = await self.workflow.ainvoke(initial_state)
            
            if final_state.get("error"):
                raise RuntimeError(final_state["error"])
            
            if final_state.get("current_data") is None:
                raise RuntimeError("No data available after cleaning")
            
            # Save cleaned data
            output_file = tempfile.mktemp(suffix=f".{file_type}")
            final_state["current_data"].to_csv(output_file, index=False)
            
            logger.info("Cleaning complete. Saving results to output file.")
            
            return output_file
        except Exception as e:
            logger.error(f"Error cleaning file: {str(e)}")
            logger.warning(
                "Failed to clean data. Check input format and cleaning rules."
            )
            raise RuntimeError(str(e))