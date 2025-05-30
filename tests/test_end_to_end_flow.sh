#!/bin/bash

# Set environment variables
export AWS_ACCESS_KEY_ID="test"
export AWS_SECRET_ACCESS_KEY="test"
export AWS_DEFAULT_REGION="us-east-1"
export S3_BUCKET="etl-architect-bucket"
export GLUE_DATABASE="etl_architect_db"
export ATHENA_WORKGROUP="etl_architect_workgroup"
export AGENT_TABLE="etl_architect_agents"

# Create test data
echo "Creating test data..."
python3 -c "
import pandas as pd
import os

# Create test directory
os.makedirs('test_data', exist_ok=True)

# Create sample data
df = pd.DataFrame({
    'id': range(1, 6),
    'name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
    'age': [25, 30, 35, 28, 32],
    'department': ['IT', 'HR', 'Finance', 'Marketing', 'Sales']
})

# Save to Excel
df.to_excel('test_data/test_data.xlsx', index=False)
print('Test data created successfully')
"

# Start the FastAPI application
echo "Starting FastAPI application..."
uvicorn etl_architect_agent_v2.api.main:app --reload --port 8000 &
FASTAPI_PID=$!

# Wait for the application to start
sleep 5

# Test file upload
echo "Testing file upload..."
curl -X POST "http://localhost:8000/api/upload" \
  -H "X-User-Id: test_user" \
  -F "file=@test_data/test_data.xlsx"

# Test descriptive query
echo "Testing descriptive query..."
curl -X POST "http://localhost:8000/api/catalog/descriptive_query" \
  -H "Content-Type: application/json" \
  -H "X-User-Id: test_user" \
  -d '{
    "query": "Show me all employees in the IT department",
    "table_name": "test_user_test_data"
  }'

# Test chat with agent
echo "Testing chat with agent..."
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -H "X-User-Id: test_user" \
  -d '{
    "message": "Create an ETL project to process employee data",
    "schema": {
      "id": "integer",
      "name": "string",
      "age": "integer",
      "department": "string"
    },
    "sample_data": [
      {
        "id": 1,
        "name": "John",
        "age": 25,
        "department": "IT"
      }
    ]
  }'

# Cleanup
echo "Cleaning up..."
rm -rf test_data
kill $FASTAPI_PID

echo "End-to-end test completed" 