from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from ..config import get_settings
import boto3
import json
from datetime import datetime
import logging
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel as LangChainBaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/query", tags=["query"])

class QueryResult(LangChainBaseModel):
    query: str = Field(description="The generated SQL query")
    explanation: str = Field(description="Explanation of the query")
    tables: List[str] = Field(description="Tables used in the query")
    columns: List[str] = Field(description="Columns selected in the query")
    filters: Optional[Dict[str, Any]] = Field(
        description="Filters applied in the query"
    )

class QueryRequest(BaseModel):
    description: str
    tables: List[str]
    filters: Optional[Dict[str, Any]] = None
    limit: Optional[int] = 1000

def get_table_schema(s3_client: boto3.client, bucket: str, table_name: str) -> Dict:
    """
    Get schema information for a table
    """
    try:
        response = s3_client.get_object(
            Bucket=bucket,
            Key="metadata/catalog_index.json"
        )
        catalog_index = json.loads(response['Body'].read())
        
        if table_name not in catalog_index["tables"]:
            raise HTTPException(
                status_code=404,
                detail=f"Table not found: {table_name}"
            )
            
        return catalog_index["tables"][table_name]["schema"]
        
    except Exception as e:
        logger.error(f"Error getting table schema: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_query_prompt(description: str, table_schemas: Dict[str, Dict]) -> str:
    """
    Generate a prompt for the LLM to create a SQL query
    """
    schema_info = []
    for table_name, schema in table_schemas.items():
        columns = [
            f"{col['name']} ({col['type']})"
            for col in schema["columns"]
        ]
        schema_info.append(
            f"Table: {table_name}\n"
            f"Columns: {', '.join(columns)}\n"
            f"Partition Keys: {', '.join(schema.get('partition_keys', []))}"
        )
    
    return (
        "Given the following database schema:\n\n"
        f"{chr(10).join(schema_info)}\n\n"
        f"Generate a SQL query for the following request:\n"
        f"{description}\n\n"
        "The query should:\n"
        "1. Use appropriate JOINs if multiple tables are involved\n"
        "2. Include WHERE clauses for any filters\n"
        "3. Use appropriate aggregation functions if needed\n"
        "4. Include ORDER BY if sorting is needed\n"
        "5. Limit the results to 1000 rows\n\n"
        "Return the query in the following format:\n"
        "{\n"
        '    "query": "SELECT ...",\n'
        '    "explanation": "This query...",\n'
        '    "tables": ["table1", "table2"],\n'
        '    "columns": ["col1", "col2"],\n'
        '    "filters": {"col1": "value1"}\n'
        "}"
    )

@router.post("/generate")
async def generate_query(
    request: QueryRequest,
    s3_client: boto3.client = Depends(get_s3_client)
):
    """
    Generate a SQL query from natural language description
    """
    try:
        settings = get_settings()
        
        # Get schemas for all tables
        table_schemas = {}
        for table_name in request.tables:
            table_schemas[table_name] = get_table_schema(
                s3_client,
                settings.AWS_S3_BUCKET,
                table_name
            )
        
        # Initialize LLM
        llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a SQL query generator. Generate SQL queries based on natural language descriptions and table schemas."),
            ("user", generate_query_prompt(request.description, table_schemas))
        ])
        
        # Create output parser
        parser = PydanticOutputParser(pydantic_object=QueryResult)
        
        # Generate query
        chain = prompt | llm | parser
        result = chain.invoke({})
        
        # Add filters from request
        if request.filters:
            result.filters.update(request.filters)
        
        # Add limit
        if request.limit:
            result.query += f"\nLIMIT {request.limit}"
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/execute")
async def execute_query(
    request: QueryRequest,
    s3_client: boto3.client = Depends(get_s3_client)
):
    """
    Execute a generated query
    """
    try:
        # First generate the query
        query_result = await generate_query(request, s3_client)
        
        # TODO: Execute the query using appropriate engine (e.g., Spark, Athena)
        # For now, return the generated query
        return {
            "query": query_result.query,
            "explanation": query_result.explanation,
            "tables": query_result.tables,
            "columns": query_result.columns,
            "filters": query_result.filters,
            "status": "generated",
            "message": "Query execution not implemented yet"
        }
        
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 