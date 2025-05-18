"""Prompt templates for lakehouse automation."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


LAKEHOUSE_SYSTEM_PROMPT = """You are an AI assistant specialized in data lakehouse automation.
Your role is to help users manage and automate their data lakehouse operations, including:
- File discovery and ingestion
- Schema inference and management
- Table creation and updates
- Data loading and validation
- Report generation

You have access to tools that can:
1. Discover and scan files in S3 buckets
2. Infer schemas from CSV and JSON files
3. Create and manage Iceberg tables in AWS Glue
4. Load data into tables
5. Generate summary reports

Always:
- Validate user inputs before processing
- Handle errors gracefully and provide clear error messages
- Maintain data consistency and integrity
- Follow best practices for data lakehouse management
- Keep users informed about operation progress
"""

LAKEHOUSE_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=LAKEHOUSE_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="conversation_history"),
    HumanMessage(content="{input}"),
])

SCHEMA_INFERENCE_PROMPT = """Analyze the following data sample and infer the schema:

Data Sample:
{data_sample}

Please provide:
1. Column names and types
2. Any potential data quality issues
3. Recommendations for schema optimization
"""

TABLE_CREATION_PROMPT = """Create an Iceberg table with the following schema:

Schema:
{schema}

Requirements:
1. Use appropriate data types
2. Handle nullable fields
3. Set up partitioning if needed
4. Configure storage format and compression
"""

DATA_LOADING_PROMPT = """Load data into the Iceberg table with the following configuration:

Table: {table_name}
Source: {source_path}
Format: {format}

Requirements:
1. Validate data before loading
2. Handle schema evolution
3. Ensure data consistency
4. Monitor loading progress
"""

REPORT_GENERATION_PROMPT = """Generate a summary report for the following operations:

Operations:
{operations}

Include:
1. Number of files processed
2. Tables created/updated
3. Data quality metrics
4. Any issues or warnings
5. Recommendations for improvement
""" 