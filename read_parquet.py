import asyncio
from etl_architect_agent_v2.backend.services.transformation_service import TransformationService

async def main():
    # Initialize the service
    service = TransformationService(bucket="lambda-code-q")
    
    # Extract table name from the S3 path
    s3_path = "s3://lambda-code-q/data/test_user/abc123/abc123_20250519_211056.parquet"
    table_name = "abc123"  # Extracted from the path
    
    try:
        # Read the data
        df = await service._read_table_data(table_name, "test_user")
        
        # Get the last 3 columns
        last_3_columns = df.columns[-3:]
        
        # Print column names and their data
        print("\nLast 3 columns and their data:")
        print("-" * 80)
        for col in last_3_columns:
            print(f"\nColumn: {col}")
            print("-" * 40)
            print(df[col].to_string())
            print("-" * 40)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 