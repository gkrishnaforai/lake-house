import asyncio
import os
from dotenv import load_dotenv
from src.core.aws.aws_service_manager import AWSServiceManager
from src.core.agents.data_exploration_agent import DataExplorationAgent


async def main():
    """Example usage of the data lake management system."""
    # Load environment variables
    load_dotenv()
    
    # Initialize AWS service manager
    aws_manager = AWSServiceManager(
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        profile_name=os.getenv("AWS_PROFILE", "default")
    )
    
    # Create data lake
    print("Creating data lake...")
    result = await aws_manager.create_data_lake(
        bucket_name=os.getenv("DEFAULT_BUCKET_NAME", "my-data-lake"),
        database_name=os.getenv("DEFAULT_DATABASE_NAME", "my_database"),
        table_name="sample_table",
        schema=[
            {"Name": "id", "Type": "string"},
            {"Name": "name", "Type": "string"},
            {"Name": "value", "Type": "double"},
            {"Name": "timestamp", "Type": "timestamp"}
        ]
    )
    print(f"Data lake created: {result}")
    
    # Initialize data exploration agent
    print("\nInitializing data exploration agent...")
    agent = DataExplorationAgent(
        agent_id="explorer_1",
        aws_manager=aws_manager
    )
    await agent.initialize()
    
    # Execute sample query
    print("\nExecuting sample query...")
    query_result = await agent.execute_task({
        "type": "query",
        "query": "SELECT * FROM my_database.sample_table LIMIT 5",
        "database": "my_database",
        "output_location": os.getenv(
            "DEFAULT_OUTPUT_LOCATION",
            "s3://my-data-lake/query-results"
        )
    })
    print(f"Query results: {query_result}")
    
    # Get table schema
    print("\nGetting table schema...")
    schema_result = await agent.execute_task({
        "type": "schema",
        "database": "my_database",
        "table": "sample_table"
    })
    print(f"Schema: {schema_result}")
    
    # Check data quality
    print("\nChecking data quality...")
    quality_result = await agent.execute_task({
        "type": "quality",
        "database": "my_database",
        "table": "sample_table",
        "metrics": ["completeness", "accuracy"]
    })
    print(f"Quality metrics: {quality_result}")
    
    # Cleanup
    print("\nCleaning up...")
    await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main()) 