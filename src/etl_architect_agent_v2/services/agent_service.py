from typing import Dict, Any, List
import json
from datetime import datetime
import boto3
import os

class AgentService:
    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table(os.getenv('AGENT_TABLE', 'etl_architect_agents'))
        self.llm_client = boto3.client('bedrock-runtime')  # Using Amazon Bedrock for LLM

    async def process_chat(
        self,
        user_id: str,
        message: str,
        schema: Dict[str, str],
        sample_data: List[Dict[str, Any]]
    ) -> str:
        """Process chat message and generate response using LLM."""
        try:
            # Store chat context in DynamoDB
            chat_id = f"{user_id}_{datetime.utcnow().isoformat()}"
            self.table.put_item(
                Item={
                    'chat_id': chat_id,
                    'user_id': user_id,
                    'message': message,
                    'schema': json.dumps(schema),
                    'sample_data': json.dumps(sample_data),
                    'timestamp': datetime.utcnow().isoformat(),
                    'ttl': int(datetime.utcnow().timestamp()) + 86400  # 24 hours TTL
                }
            )

            # Prepare prompt for LLM
            prompt = self._prepare_prompt(message, schema, sample_data)

            # Generate response using Bedrock
            response = self.llm_client.invoke_model(
                modelId='anthropic.claude-v2',
                body=json.dumps({
                    'prompt': prompt,
                    'max_tokens': 1000,
                    'temperature': 0.7
                })
            )

            # Parse response
            response_body = json.loads(response['body'].read())
            generated_text = response_body['completion']

            # Store response in DynamoDB
            self.table.put_item(
                Item={
                    'chat_id': f"{chat_id}_response",
                    'user_id': user_id,
                    'message': generated_text,
                    'timestamp': datetime.utcnow().isoformat(),
                    'ttl': int(datetime.utcnow().timestamp()) + 86400
                }
            )

            return generated_text

        except Exception as e:
            raise Exception(f"Error processing chat: {str(e)}")

    def _prepare_prompt(
        self,
        message: str,
        schema: Dict[str, str],
        sample_data: List[Dict[str, Any]]
    ) -> str:
        """Prepare prompt for LLM."""
        prompt = f"""You are an ETL architect agent. Help the user with their data processing needs.

User Message: {message}

Schema Information:
{json.dumps(schema, indent=2)}

Sample Data:
{json.dumps(sample_data, indent=2)}

Please provide a helpful response that:
1. Understands the user's requirements
2. Suggests appropriate ETL steps
3. Provides guidance on data processing
4. Recommends best practices

Response:"""
        return prompt

    async def get_chat_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get chat history for a user."""
        try:
            response = self.table.query(
                KeyConditionExpression='user_id = :uid',
                FilterExpression='begins_with(chat_id, :prefix)',
                ExpressionAttributeValues={
                    ':uid': user_id,
                    ':prefix': f"{user_id}_"
                },
                Limit=limit,
                ScanIndexForward=False  # Get most recent first
            )
            return response.get('Items', [])
        except Exception as e:
            raise Exception(f"Error getting chat history: {str(e)}") 