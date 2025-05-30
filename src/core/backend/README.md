# AWS Architect Agent Backend

This is the backend service for the AWS Architect Agent, providing APIs for managing AWS Glue, S3, and Athena operations.

## Features

- Catalog Management
  - List and manage database tables
  - Track file metadata
  - Process descriptive queries
- Glue Operations
  - Create databases and tables
- AWS Integration
  - S3 file operations
  - Athena query execution
  - Glue catalog management
- File Upload
  - Upload files to S3
  - Process file metadata
- File Download
  - Download files from S3
  - Process file metadata
- File Search
  - Search files based on metadata
  - Process search queries
- File Processing
  - Process files based on metadata
  - Process file metadata
- File Analysis
  - Perform analysis on files
  - Process analysis queries
- File Export
  - Export files to a specified format
  - Process export queries
- File Import
  - Import files from a specified format
  - Process import queries
- File to aws Lake house
  - Move files from S3 to AWS Lakehouse
  - Process file metadata
  - create glue catalog
  - create table in Glue
  - create partition in Glue
  - create partition in Athena

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:
   Create a `.env` file based on `.env.example` and fill in your AWS credentials and configuration.

4. Start the server:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Catalog Endpoints

- `GET /api/catalog/tables` - List all tables
- `GET /api/catalog` - Get full database catalog
- `GET /api/catalog/files` - List all files
- `POST /api/catalog/descriptive_query` - Process descriptive query

### Glue Endpoints

- `POST /api/glue/database` - Create database
- `POST /api/glue/table` - Create table
- `POST /api/glue/crawler/start` - Start crawler
- `GET /api/glue/crawler/status` - Get crawler status

## Development

### Running Tests

```bash
pytest
```

### Code Style

The project follows PEP 8 style guidelines. Use a linter to check your code:

```bash
flake8
```

## Architecture

The backend is built using:

- FastAPI for the web framework
- Pydantic for data validation
- Boto3 for AWS SDK integration
- Uvicorn as the ASGI server

## Security

- AWS credentials should be managed securely
- CORS is configured to allow specific origins
- Environment variables are used for sensitive configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
