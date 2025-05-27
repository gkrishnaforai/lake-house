# AWS Data Lake Management System

A modular, AI-powered data lake management system built on AWS using LangChain and LangGraph. This system provides automated data exploration, quality checks, and management capabilities for AWS data lakes.

## Features

- **Data Lake Creation**: Automated setup of S3 buckets, Glue databases, and tables
- **Data Exploration**: Query execution and schema exploration using Athena
- **Data Quality**: Automated data quality checks and monitoring
- **Modular Architecture**: Easy integration of new capabilities and cloud services
- **AI-Powered**: Leverages LangChain and LangGraph for intelligent data management

## Architecture

The system is built with a modular, agent-based architecture:

- **Base Agent**: Abstract base class for all agents
- **Data Exploration Agent**: Handles data exploration tasks
- **AWS Service Manager**: Manages AWS service integrations
- **Agent Orchestrator**: Coordinates communication between agents

## Prerequisites

- Python 3.9+
- AWS CLI configured with appropriate credentials
- Required AWS services:
  - S3
  - Glue
  - Athena

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/aws-data-lake-management.git
cd aws-data-lake-management
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure AWS credentials:

```bash
aws configure
```

## Usage

### Creating a Data Lake

```python
from src.core.aws.aws_service_manager import AWSServiceManager

# Initialize AWS service manager
aws_manager = AWSServiceManager(region_name="us-east-1")

# Create data lake
result = await aws_manager.create_data_lake(
    bucket_name="my-data-lake",
    database_name="my_database",
    table_name="my_table",
    schema=[
        {"Name": "id", "Type": "string"},
        {"Name": "name", "Type": "string"},
        {"Name": "value", "Type": "double"}
    ]
)
```

### Data Exploration

```python
from src.core.agents.data_exploration_agent import DataExplorationAgent

# Initialize data exploration agent
agent = DataExplorationAgent(agent_id="explorer_1")
await agent.initialize()

# Execute query
result = await agent.execute_task({
    "type": "query",
    "query": "SELECT * FROM my_database.my_table LIMIT 10",
    "database": "my_database",
    "output_location": "s3://my-data-lake/query-results"
})
```

### Data Quality Checks

```python
# Check data quality
result = await agent.execute_task({
    "type": "quality",
    "database": "my_database",
    "table": "my_table",
    "metrics": ["completeness", "accuracy"]
})
```

## Testing

Run the integration tests:

```bash
pytest tests/integration/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- AWS for providing the cloud infrastructure
- LangChain and LangGraph for the AI framework
- The open-source community for inspiration and support
