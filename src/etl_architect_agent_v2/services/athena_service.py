import boto3
import os
from typing import List, Dict, Any
import asyncio
from datetime import datetime

class AthenaService:
    def __init__(self):
        self.client = boto3.client('athena')
        self.database = os.getenv('GLUE_DATABASE', 'etl_architect_db')
        self.workgroup = os.getenv('ATHENA_WORKGROUP', 'etl_architect_workgroup')
        self.output_location = os.getenv('ATHENA_OUTPUT_LOCATION', 's3://etl-architect-bucket/athena-results/')

    def generate_query(self, natural_language_query: str, table_name: str) -> str:
        """Generate SQL query from natural language query."""
        # This is a simple implementation. In production, you would use an LLM
        # to generate more sophisticated SQL queries.
        query = natural_language_query.lower()
        
        if "show" in query and "all" in query:
            return f"SELECT * FROM {self.database}.{table_name}"
        elif "count" in query:
            return f"SELECT COUNT(*) FROM {self.database}.{table_name}"
        elif "average" in query or "avg" in query:
            # Extract column name from query
            words = query.split()
            for i, word in enumerate(words):
                if word in ["average", "avg"] and i + 1 < len(words):
                    column = words[i + 1]
                    return f"SELECT AVG({column}) FROM {self.database}.{table_name}"
        
        # Default query
        return f"SELECT * FROM {self.database}.{table_name} LIMIT 10"

    async def execute_query(self, query: str, workgroup: str = None) -> List[Dict[str, Any]]:
        """Execute Athena query and return results."""
        workgroup = workgroup or self.workgroup
        
        # Start query execution
        response = self.client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={'Database': self.database},
            ResultConfiguration={'OutputLocation': self.output_location},
            WorkGroup=workgroup
        )
        
        query_execution_id = response['QueryExecutionId']
        
        # Wait for query to complete
        while True:
            query_status = self.client.get_query_execution(
                QueryExecutionId=query_execution_id
            )
            state = query_status['QueryExecution']['Status']['State']
            
            if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                break
                
            await asyncio.sleep(1)
        
        if state == 'FAILED':
            error_message = query_status['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
            raise Exception(f"Query failed: {error_message}")
        
        # Get results
        results = self.client.get_query_results(
            QueryExecutionId=query_execution_id
        )
        
        # Process results
        if 'ResultSet' not in results or 'Rows' not in results['ResultSet']:
            return []
        
        rows = results['ResultSet']['Rows']
        if not rows:
            return []
        
        # Get headers
        headers = [col['VarCharValue'] for col in rows[0]['Data']]
        
        # Process data rows
        data = []
        for row in rows[1:]:  # Skip header row
            row_data = {}
            for i, col in enumerate(row['Data']):
                row_data[headers[i]] = col.get('VarCharValue')
            data.append(row_data)
        
        return data 