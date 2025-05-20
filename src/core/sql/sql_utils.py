from typing import Dict, Any, List
from dataclasses import dataclass, field
from src.core.sql.sql_state import (
    SQLGenerationState,
    SQLRequirements,
    SQLGenerationOutput,
    SQLGenerationError
)
from src.core.llm.manager import LLMManager
import json
import re
import logging
from src.core.glue.glue_service import GlueService

logger = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    """Tracks the context of the current conversation."""
    gathered_requirements: Dict[str, Any] = field(default_factory=dict)
    missing_info: List[str] = field(default_factory=list)
    current_focus: str = "query_analysis"
    conversation_history: List[Dict[str, str]] = field(default_factory=list)


class SQLUtils:
    """Manages SQL generation utilities."""

    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.context = ConversationContext()

    async def analyze_requirements(self, state: SQLGenerationState) -> SQLRequirements:
        prompt = self._create_requirements_prompt(state)
        try:
            response = await self.llm_manager.ainvoke({"messages": [{"role": "user", "content": prompt}]})
            cleaned = re.sub(r'```json\n|\n```', '', response["content"])
            return SQLRequirements(**json.loads(cleaned))
        except json.JSONDecodeError as e:
            logger.error("Requirements JSON parse error: %s", e)
            raise SQLGenerationError("Invalid JSON in requirements response")
        except Exception as e:
            logger.error("Analyze requirements error: %s", e)
            raise SQLGenerationError(str(e))

    async def validate_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        table = schema.get("table_name")
        user_id = schema.get("user_id", "test_user")
        if not table:
            return {"is_valid": False, "validation_errors": ["Table name is required"]}

        db = f"user_{user_id}"
        glue = GlueService()

        try:
            try:
                glue.glue_client.get_database(Name=db)
            except glue.glue_client.exceptions.EntityNotFoundException:
                glue.glue_client.create_database(DatabaseInput={"Name": db, "Description": f"DB for {user_id}"})
                logger.info("Database %s created for user %s", db, user_id)

            table_info = await glue.get_table(database_name=db, table_name=table)
            columns = table_info['StorageDescriptor']['Columns']
            return {
                "is_valid": True,
                "table_name": table,
                "database": db,
                "columns": [{
                    "name": col["Name"],
                    "type": col["Type"],
                    "description": col.get("Comment", "")
                } for col in columns],
                "location": table_info['StorageDescriptor']['Location']
            }

        except glue.glue_client.exceptions.EntityNotFoundException:
            return {"is_valid": False, "validation_errors": [f"Table {table} not found in {db}"]}
        except Exception as e:
            logger.error("Schema validation error: %s", e)
            return {"is_valid": False, "validation_errors": [str(e)]}

    async def generate_query(self, requirements: SQLRequirements) -> SQLGenerationOutput:
        prompt = self._create_query_generation_prompt(requirements)
        try:
            response = await self.llm_manager.ainvoke(prompt, {})
            cleaned = re.sub(r'```json\n|\n```', '', response["content"])
            return SQLGenerationOutput(**json.loads(cleaned))
        except json.JSONDecodeError as e:
            logger.error("Query JSON parse error: %s", e)
            raise SQLGenerationError("Invalid JSON in query response")
        except Exception as e:
            logger.error("Generate query error: %s", e)
            raise SQLGenerationError(str(e))

    async def review_query(self, query_result: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._create_query_review_prompt(query_result)
        try:
            response = await self.llm_manager.ainvoke(prompt, {})
            cleaned = re.sub(r'```json\n|\n```', '', response["content"])
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error("Review JSON parse error: %s", e)
            raise SQLGenerationError("Invalid JSON in review response")
        except Exception as e:
            logger.error("Review query error: %s", e)
            raise SQLGenerationError(str(e))

    async def correct_query(self, query_result: Dict[str, Any], review_result: Dict[str, Any]) -> SQLGenerationOutput:
        prompt = self._create_query_correction_prompt(query_result, review_result)
        try:
            response = await self.llm_manager.ainvoke(prompt, {})
            cleaned = re.sub(r'```json\n|\n```', '', response["content"])
            return SQLGenerationOutput(**json.loads(cleaned))
        except json.JSONDecodeError as e:
            logger.error("Correction JSON parse error: %s", e)
            raise SQLGenerationError("Invalid JSON in correction response")
        except Exception as e:
            logger.error("Correct query error: %s", e)
            raise SQLGenerationError(str(e))

    def _create_requirements_prompt(self, state: SQLGenerationState) -> str:
        return f"""
        Analyze the following SQL generation request:
        
        User Request: {state.metadata.get('latest_message', '')}
        
        Previous Conversation:
        {json.dumps(state.metadata.get('conversation_history', []), indent=2)}
        
        Respond in JSON:
        {{
            "query": "user's query",
            "schema": {{"tables": [{{"name": "table", "columns": [{{"name": "col", "type": "type"}}]}}]}},
            "constraints": {{"required_tables": [], "required_columns": [], "conditions": []}},
            "missing_info": [],
            "next_question": "..."
        }}
        """

    def _create_query_generation_prompt(self, requirements: SQLRequirements) -> str:
        db_type = requirements.schema.get("database_type", "athena").lower()
        rules = self._db_specific_rules().get(db_type, self._db_specific_rules()["athena"])
        return f"""
        Generate a {db_type.upper()} SQL query based on:

        Query: {requirements.query}
        Schema: {json.dumps(requirements.schema, indent=2)}
        Constraints: {json.dumps(requirements.constraints, indent=2)}

        {rules}

        JSON Output:
        {{
            "sql_query": "...",
            "confidence": 0.0,
            "tables_used": [],
            "columns_used": [],
            "filters": {{"column": "value"}},
            "explanation": "..."
        }}
        """

    def _create_query_review_prompt(self, query_result: Dict[str, Any]) -> str:
        return f"""
        Review the following SQL query:

        Query: {query_result.get("sql_query")}

        JSON Output:
        {{
            "issues": ["..."],
            "suggestions": ["..."],
            "confidence": 0.0
        }}
        """

    def _create_query_correction_prompt(self, query_result: Dict[str, Any], review_result: Dict[str, Any]) -> str:
        return f"""
        Based on the review feedback, correct the SQL query:

        Original Query: {query_result.get("sql_query")}
        Issues: {json.dumps(review_result.get("issues", []))}
        Suggestions: {json.dumps(review_result.get("suggestions", []))}

        JSON Output:
        {{
            "sql_query": "...",
            "confidence": 0.0,
            "tables_used": [],
            "columns_used": [],
            "filters": {{"column": "value"}},
            "explanation": "..."
        }}
        """

    def _db_specific_rules(self) -> Dict[str, str]:
        return {
            "athena": """You are generating SQL queries for AWS Athena (Presto SQL). Make sure to use LIMIT for restricting rows instead of TOP. Athena SQL Notes: Use LIMIT, ARRAY_AGG, DATE_TRUNC, etc.""",
            "postgresql": """PostgreSQL Notes: Use LIMIT, STRING_AGG, DATE_TRUNC, etc.""",
            "mysql": """MySQL Notes: Use LIMIT, GROUP_CONCAT, DATE_FORMAT, etc.""",
            "sqlserver": """SQL Server Notes: Use TOP, STRING_AGG, DATEADD, etc."""
        }