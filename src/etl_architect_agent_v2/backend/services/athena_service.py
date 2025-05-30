from typing import Dict, Any, List, Optional
import boto3
import time
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
import json
import pandas as pd
import asyncio
import logging

logger = logging.getLogger(__name__)

class AthenaService:
    def __init__(
        self,
        database: str,
        workgroup: str = "primary",
        output_location: str = "s3://your-bucket/athena-results/",
        aws_region: str = "us-east-1"
    ):
        self.database = database
        self.workgroup = workgroup
        self.output_location = output_location
        self.athena_client = boto3.client('athena', region_name=aws_region)
        self.s3_client = boto3.client('s3', region_name=aws_region)
        self.llm = ChatOpenAI(temperature=0)
        self._setup_agent()

    def _setup_agent(self):
        tools = [
            Tool(
                name="execute_query",
                func=self._execute_query,
                description="Execute a SQL query using Athena"
            ),
            Tool(
                name="get_query_results",
                func=self._get_query_results,
                description="Get results of an executed query"
            ),
            Tool(
                name="optimize_query",
                func=self._optimize_query,
                description="Optimize a SQL query for better performance"
            )
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL query expert. Your task is to:
            1. Execute SQL queries using Athena
            2. Optimize queries for better performance
            3. Handle query results
            Use the provided tools to accomplish these tasks."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.agent = create_openai_functions_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=tools, verbose=True)

    async def execute_query(self, query: str, database_name: Optional[str] = None) -> Dict[str, Any]:
        """Execute a SQL query using Athena."""
        try:
            logger.info(f"Starting query execution: {query}")
            
            # Use provided database name or fall back to instance database
            query_database = database_name or self.database
            
            # Start query execution
            response = self.athena_client.start_query_execution(
                QueryString=query,
                QueryExecutionContext={
                    'Database': query_database
                },
                ResultConfiguration={
                    'OutputLocation': self.output_location
                }
            )
            
            query_execution_id = response['QueryExecutionId']
            logger.info(f"Query execution started with ID: {query_execution_id}")
            
            # Wait for query to complete
            while True:
                query_status = self.athena_client.get_query_execution(
                    QueryExecutionId=query_execution_id
                )['QueryExecution']['Status']['State']
                
                logger.info(f"Query status: {query_status}")
                
                if query_status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                    break
                    
                await asyncio.sleep(1)
            
            if query_status == 'FAILED':
                error = self.athena_client.get_query_execution(
                    QueryExecutionId=query_execution_id
                )['QueryExecution']['Status']['StateChangeReason']
                logger.error(f"Query failed: {error}")
                raise Exception(f"Query failed: {error}")
            
            # Get results
            logger.info("Query succeeded, fetching results")
            results = []
            column_names = []
            
            # Get the first page of results
            response = self.athena_client.get_query_results(
                QueryExecutionId=query_execution_id
            )
            
            # Extract column names from the first row
            if response['ResultSet']['Rows']:
                header_row = response['ResultSet']['Rows'][0]
                column_names = [field.get('VarCharValue', '') for field in header_row['Data']]
                logger.info(f"Column names: {column_names}")
                
                # Process data rows (skip header row)
                for row in response['ResultSet']['Rows'][1:]:
                    row_data = [field.get('VarCharValue', '') for field in row['Data']]
                    results.append(row_data)
            
            # Get remaining pages if any
            paginator = self.athena_client.get_paginator('get_query_results')
            for page in paginator.paginate(QueryExecutionId=query_execution_id):
                # Skip the first page as we've already processed it
                if page.get('NextToken'):
                    for row in page['ResultSet']['Rows']:  # Process all rows in subsequent pages
                        row_data = [field.get('VarCharValue', '') for field in row['Data']]
                        results.append(row_data)
            
            logger.info(f"Total rows fetched: {len(results)}")
            
            return {
                'status': 'success',
                'results': results,
                'columns': column_names,
                'query_execution_id': query_execution_id
            }
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def _execute_query(self, query: str) -> Dict[str, Any]:
        """
        Execute a query in Athena
        """
        try:
            # Start query execution
            response = self.athena_client.start_query_execution(
                QueryString=query,
                QueryExecutionContext={
                    'Database': self.database
                },
                ResultConfiguration={
                    'OutputLocation': self.output_location
                },
                WorkGroup=self.workgroup
            )

            query_execution_id = response['QueryExecutionId']

            # Wait for query to complete
            while True:
                query_status = self.athena_client.get_query_execution(
                    QueryExecutionId=query_execution_id
                )['QueryExecution']['Status']['State']

                if query_status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                    break

                time.sleep(1)

            if query_status == 'FAILED':
                error_message = self.athena_client.get_query_execution(
                    QueryExecutionId=query_execution_id
                )['QueryExecution']['Status']['StateChangeReason']
                raise Exception(f"Query failed: {error_message}")

            return {
                "status": "success",
                "query_execution_id": query_execution_id
            }

        except Exception as e:
            raise Exception(f"Error executing query: {str(e)}")

    def _get_query_results(self, query_execution_id: str) -> Dict[str, Any]:
        """
        Get results of an executed query
        """
        try:
            # Get query results
            results = self.athena_client.get_query_results(
                QueryExecutionId=query_execution_id
            )

            # Process results
            columns = [col['Label'] for col in results['ResultSet']['Rows'][0]['Data']]
            rows = []
            for row in results['ResultSet']['Rows'][1:]:
                rows.append([cell.get('VarCharValue', '') for cell in row['Data']])

            return {
                "status": "success",
                "data": {
                    "columns": columns,
                    "rows": rows
                },
                "metadata": {
                    "row_count": len(rows),
                    "column_count": len(columns)
                }
            }

        except Exception as e:
            raise Exception(f"Error getting query results: {str(e)}")

    def _optimize_query(self, query: str) -> Dict[str, Any]:
        """
        Optimize a SQL query for better performance
        """
        try:
            # Use LLM to optimize query
            optimization_prompt = f"""
            Optimize the following SQL query for better performance:
            {query}
            
            Consider:
            1. Using appropriate WHERE clauses
            2. Selecting only needed columns
            3. Using appropriate JOIN types
            4. Adding appropriate indexes
            """

            response = self.llm.invoke(optimization_prompt)
            optimized_query = response.content

            return {
                "status": "success",
                "original_query": query,
                "optimized_query": optimized_query,
                "explanation": "Query optimized for better performance"
            }

        except Exception as e:
            raise Exception(f"Error optimizing query: {str(e)}") 