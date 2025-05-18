# Backend API Documentation

## Overview

The backend API provides a comprehensive interface for managing data catalog, file operations, and data quality. It uses AWS Glue as the primary catalog system and maintains synchronization with S3 metadata.

## Base URL

```
http://localhost:8000/api
```

## Authentication

Currently, the API does not require authentication for development purposes. In production, proper authentication should be implemented.

## API Endpoints

### Catalog Management

#### Get Database Catalog

```http
GET /catalog
```

Returns the complete database catalog including tables, schemas, and metadata.

**Response**

```json
{
  "database_name": "string",
  "tables": [
    {
      "name": "string",
      "columns": [
        {
          "name": "string",
          "type": "string",
          "description": "string",
          "quality_metrics": {
            "completeness": "number",
            "accuracy": "number",
            "consistency": "number"
          }
        }
      ],
      "row_count": "number",
      "last_updated": "string",
      "description": "string"
    }
  ],
  "description": "string",
  "metadata": {
    "total_files": "number",
    "total_tables": "number",
    "total_size_mb": "number"
  }
}
```

#### List Tables

```http
GET /catalog/tables
```

Returns a list of all tables in the catalog with their metadata.

**Response**

```json
[
  {
    "name": "string",
    "columns": [
      {
        "name": "string",
        "type": "string",
        "description": "string"
      }
    ],
    "row_count": "number",
    "last_updated": "string",
    "description": "string"
  }
]
```

#### Get Table Details

```http
GET /catalog/tables/{table_name}
```

Returns detailed information about a specific table.

**Response**

```json
{
  "name": "string",
  "columns": [
    {
      "name": "string",
      "type": "string",
      "description": "string",
      "quality_metrics": {
        "completeness": "number",
        "accuracy": "number",
        "consistency": "number"
      }
    }
  ],
  "row_count": "number",
  "last_updated": "string",
  "description": "string",
  "audit_info": {
    "created_by": "string",
    "created_at": "string",
    "updated_by": "string",
    "updated_at": "string",
    "conversion_history": [
      {
        "timestamp": "string",
        "action": "string",
        "details": "string"
      }
    ]
  }
}
```

#### Get Data Quality Metrics

```http
GET /catalog/quality/{table_name}
```

Returns data quality metrics for a specific table.

**Response**

```json
{
  "columns": [
    {
      "name": "string",
      "type": "string",
      "quality_metrics": {
        "completeness": "number",
        "accuracy": "number",
        "consistency": "number"
      }
    }
  ],
  "overall_quality": {
    "completeness": "number",
    "accuracy": "number",
    "consistency": "number"
  }
}
```

#### Get Audit Trail

```http
GET /catalog/audit/{table_name}
```

Returns the audit trail for a specific table.

**Response**

```json
{
  "audit_metadata": {
    "created_by": "string",
    "created_at": "string",
    "updated_by": "string",
    "updated_at": "string"
  },
  "conversion_history": [
    {
      "timestamp": "string",
      "action": "string",
      "details": "string"
    }
  ]
}
```

### File Operations

#### Upload File

```http
POST /upload
```

Uploads a file to S3 and processes it for catalog.

**Request**

- Content-Type: multipart/form-data
- Body: file

**Response**

```json
{
  "status": "string",
  "message": "string",
  "file_name": "string",
  "s3_path": "string",
  "error": "string",
  "details": {
    "error_type": "string",
    "error_details": "string"
  }
}
```

#### Convert File

```http
POST /files/{s3_path}/convert
```

Converts a file to a different format (Parquet/Avro).

**Request**

```json
{
  "target_format": "string",
  "compression": "string"
}
```

**Response**

```json
{
  "status": "string",
  "message": "string",
  "original_path": "string",
  "converted_path": "string"
}
```

### Data Query

#### Query Data

```http
POST /query
```

Query data across tables using natural language description.

**Request**

```json
{
  "tables": ["string"],
  "description": "string",
  "filters": {
    "column": "string",
    "operator": "string",
    "value": "any"
  },
  "limit": "number"
}
```

**Response**

```json
{
  "query": "string",
  "tables": ["string"],
  "description": "string",
  "filters": {
    "column": "string",
    "operator": "string",
    "value": "any"
  },
  "limit": "number"
}
```

## Error Handling

The API uses standard HTTP status codes and returns error details in the response body:

```json
{
  "detail": "Error message"
}
```

Common status codes:

- 200: Success
- 400: Bad Request
- 404: Not Found
- 500: Internal Server Error

## Health Check

```http
GET /health
```

Returns the health status of the API and metadata system.

**Response**

```json
{
  "status": "string",
  "metadata_initialized": "boolean",
  "total_tables": "number",
  "total_files": "number",
  "error": "string"
}
```
