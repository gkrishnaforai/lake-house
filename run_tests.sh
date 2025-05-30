#!/bin/bash

# Install test dependencies
pip install -r requirements-test.txt

# Set AWS environment variables (replace with your values)
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_REGION="us-east-1"
export AWS_S3_BUCKET="your-bucket-name"
export GLUE_DATABASE_NAME="data_lakehouse"

# Run tests
pytest tests/test_api.py -v 