#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_FILE="src/etl_architect_agent_v2/.env"

# First unset all AWS environment variables
echo "Unsetting existing AWS environment variables..."
unset AWS_ACCESS_KEY_ID
unset AWS_SECRET_ACCESS_KEY
unset AWS_SESSION_TOKEN
unset AWS_DEFAULT_REGION
unset AWS_REGION

# Verify they are unset
echo "Verifying AWS environment variables are unset..."
echo "AWS_ACCESS_KEY_ID: $AWS_ACCESS_KEY_ID"
echo "AWS_SECRET_ACCESS_KEY: $AWS_SECRET_ACCESS_KEY"
echo "AWS_SESSION_TOKEN: $AWS_SESSION_TOKEN"

# Load new values from .env file
echo "Loading new values from .env file..."
if [ -f "$ENV_FILE" ]; then
    echo "Found .env file at: $ENV_FILE"
    while IFS='=' read -r key value; do
        # Skip empty lines and comments
        if [[ -n "$key" && ! "$key" =~ ^# ]]; then
            # Remove any quotes from the value
            value=$(echo "$value" | tr -d '"' | tr -d "'")
            # Remove any trailing commas or spaces
            value=$(echo "$value" | sed 's/,$//' | sed 's/ *$//')
            export "$key=$value"
        fi
    done < "$ENV_FILE"
else
    echo "Error: .env file not found at $ENV_FILE!"
    exit 1
fi

# Verify new values are set
echo "Verifying new AWS environment variables are set..."
echo "AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID:0:4}...${AWS_ACCESS_KEY_ID: -4}"
echo "AWS_SECRET_ACCESS_KEY: ********"
echo "AWS_SESSION_TOKEN: ${AWS_SESSION_TOKEN:0:10}..."
echo "AWS_REGION: $AWS_REGION"

echo "AWS environment variables have been reset successfully!" 
