# ETL Architect Agent V2

A powerful ETL architecture design and implementation agent that helps you create and manage data pipelines with AWS Glue Catalog integration.

## Features

- Automated ETL architecture design
- Intelligent pipeline generation
- AWS Glue Catalog integration for centralized metadata management
- Automatic schema inference and table creation
- Data quality monitoring and tracking
- Schema evolution tracking
- State management for ETL workflows
- Error handling and recovery
- Extensible plugin system
- Async/await support
- Comprehensive logging
- Type hints and documentation

## Architecture

The system uses AWS Glue Catalog as the central metadata store, enabling SQL-based data exploration and querying:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Data Sources   │────▶│  S3 Storage     │────▶│  Glue Catalog   │
│  (CSV/JSON/etc) │     │  (Parquet)      │     │  (Metadata)     │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Query Tools    │◀────│  Athena         │◀────│  Redshift       │
│  (SQL)          │     │  (Serverless)   │     │  Spectrum       │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Key Components

1. **Data Ingestion**

   - File upload to S3
   - Automatic conversion to Parquet format
   - Schema inference
   - Data quality checks

2. **Glue Catalog Integration**

   - Automatic database creation
   - Table registration with proper schemas
   - Format-specific configurations (CSV, JSON, Parquet)
   - SerDe information management

3. **Query Capabilities**
   - SQL-based querying through Athena
   - Redshift Spectrum integration
   - EMR Hive/Spark support
   - SageMaker notebook access

## Installation

```bash
pip install etl-architect-agent-v2
```

## Project Structure

```
etl_architect_agent_v2/
├── backend/
│   ├── services/
│   │   ├── catalog_service.py    # Main catalog management
│   │   ├── glue_service.py       # AWS Glue integration
│   │   └── s3_service.py         # S3 operations
│   ├── routes/
│   │   └── catalog_routes.py     # API endpoints
│   └── models/
│       └── catalog.py            # Data models
├── core/
│   ├── state_manager.py
│   ├── prompt_manager.py
│   ├── llm_manager.py
│   ├── agent_orchestrator.py
│   └── error_handler.py
├── agents/
│   └── example_agent.py
├── nodes/
│   ├── data_extraction_node.py
│   ├── data_cleaning_node.py
│   └── data_export_node.py
├── tools/
│   ├── extract.py
│   ├── clean.py
│   └── export.py
├── prompts/
│   └── example_prompt.txt
├── utils/
│   └── logger.py
├── tests/
│   ├── test_agent.py
│   └── test_tools.py
├── context/
│   └── agent_context.py
├── requirements.txt
├── README.md
└── app.py
```

## Usage

```python
from etl_architect_agent_v2 import AgentOrchestrator

# Initialize the agent
agent = AgentOrchestrator()

# Upload a file and create catalog table
response = await agent.upload_file(
    file="data.csv",
    table_name="sales_data"
)

# Query the data using Athena
query = """
SELECT *
FROM etl_catalog.sales_data
WHERE amount > 1000
"""
results = await agent.execute_query(query)
```

## Configuration

The agent can be configured using environment variables:

```bash
# AWS Configuration
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1
export AWS_S3_BUCKET=your-bucket-name

# Agent Configuration
export ETL_AGENT_LOG_LEVEL=INFO
export ETL_AGENT_MAX_RETRIES=3
export OPENAI_API_KEY=your_api_key
```

## Development

1. Clone the repository
2. Create a virtual environment
3. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Run tests:
   ```bash
   pytest
   ```
5. Format code:
   ```bash
   black .
   isort .
   ```

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
