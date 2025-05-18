import boto3
import os
from typing import Optional

from etl_architect_agent_v2.core.llm_manager import LLMManager


class ETLArchitectAgent:
    def __init__(self, llm_manager: Optional[LLMManager] = None):
        """Initialize the ETL architect agent.
        
        Args:
            llm_manager: Optional LLM manager instance
        """
        self.max_retries = 3
        self.preview_rows = 10
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        self.glue_client = boto3.client(
            'glue',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        ) 