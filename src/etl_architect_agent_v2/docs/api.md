# Catalog API Documentation

The Catalog API serves as the central interface for managing data assets in the lakehouse. All operations are performed through the catalog, which maintains metadata and provides a unified view of the data.

## Base URL

```
http://localhost:8000/api/catalog
```

## Authentication

All endpoints require authentication. Include the API key in the request header:

```
Authorization: Bearer <your-api-key>
```

## Endpoints

### Catalog Overview

#### GET /

Get the complete database catalog overview.

**Response:**

```json
{
  "database_name": "lakehouse",
  "description": "Main data lakehouse catalog",
  "tables": [],
  "last_updated": "2024-03-14T12:00:00Z"
}
```

### Table Operations

#### GET /tables

List all tables in the catalog.

**Response:**

```json
[
  {
    "name": "customers",
    "schema": {
      "columns": [
        {
          "name": "id",
          "type": "string",
          "description": "Customer ID"
        }
      ]
    },
    "location": "s3://bucket/tables/customers",
    "created_at": "2024-03-14T12:00:00Z",
    "updated_at": "2024-03-14T12:00:00Z"
  }
]
```

#### GET /tables/{table_name}

Get detailed information about a specific table.

**Response:**

```json
{
  "name": "customers",
  "schema": {
    "columns": [
      {
        "name": "id",
        "type": "string",
        "description": "Customer ID"
      }
    ]
  },
  "location": "s3://bucket/tables/customers",
  "created_at": "2024-03-14T12:00:00Z",
  "updated_at": "2024-03-14T12:00:00Z"
}
```

#### GET /tables/{table_name}/schema

Get the schema definition for a specific table.

**Response:**

```json
{
  "columns": [
    {
      "name": "id",
      "type": "string",
      "description": "Customer ID"
    }
  ]
}
```

### File Operations

#### GET /files

List all files in the catalog with their metadata.

**Response:**

```json
[
  {
    "name": "customers.csv",
    "size": 1024,
    "format": "csv",
    "location": "s3://bucket/files/customers.csv",
    "last_modified": "2024-03-14T12:00:00Z"
  }
]
```

#### GET /files/{s3_path}

Get metadata for a specific file.

**Response:**

```json
{
  "name": "customers.csv",
  "size": 1024,
  "format": "csv",
  "location": "s3://bucket/files/customers.csv",
  "last_modified": "2024-03-14T12:00:00Z"
}
```

#### GET /files/{s3_path}/details

Get detailed information about a specific file.

**Response:**

```json
{
  "name": "customers.csv",
  "size": 1024,
  "format": "csv",
  "location": "s3://bucket/files/customers.csv",
  "last_modified": "2024-03-14T12:00:00Z",
  "schema": {
    "columns": [
      {
        "name": "id",
        "type": "string"
      }
    ]
  }
}
```

#### GET /files/{s3_path}/preview

Get a preview of the file contents.

**Query Parameters:**

- `rows` (optional): Number of rows to preview (default: 5, max: 100)
- `columns` (optional): List of columns to include

**Response:**

```json
{
  "rows": [
    {
      "id": "1",
      "name": "John Doe"
    }
  ],
  "total_rows": 1000
}
```

#### POST /files/{s3_path}/convert

Convert a file to a different format.

**Request Body:**

```json
{
  "target_format": "parquet",
  "options": {
    "compression": "snappy"
  }
}
```

**Response:**

```json
{
  "new_location": "s3://bucket/files/customers.parquet",
  "conversion_time": "2024-03-14T12:00:00Z"
}
```

#### DELETE /files/{s3_path}

Delete a file and its metadata from the catalog.

**Response:**

```json
{
  "message": "File deleted successfully"
}
```

### Upload Operations

#### POST /upload

Upload a new file to the catalog.

**Form Data:**

- `file`: The file to upload
- `table_name` (optional): Name of the table to load the data into
- `create_new` (optional): Whether to create a new table (default: false)

**Response:**

```json
{
  "file_name": "customers.csv",
  "s3_path": "s3://bucket/files/customers.csv",
  "table_name": "customers",
  "upload_time": "2024-03-14T12:00:00Z"
}
```

### Quality and Audit

#### GET /quality

Get overall data quality metrics.

**Response:**

```json
{
  "completeness": 0.95,
  "accuracy": 0.98,
  "consistency": 0.97,
  "timeliness": 0.99
}
```

#### GET /quality/{table_name}

Get quality metrics for a specific table.

**Response:**

```json
{
  "completeness": 0.95,
  "accuracy": 0.98,
  "consistency": 0.97,
  "timeliness": 0.99
}
```

#### GET /audit/{table_name}

Get audit logs for a specific table.

**Query Parameters:**

- `start_date` (optional): Start date for audit logs
- `end_date` (optional): End date for audit logs

**Response:**

```json
[
  {
    "timestamp": "2024-03-14T12:00:00Z",
    "action": "CREATE_TABLE",
    "user": "admin",
    "details": {
      "schema": {
        "columns": [
          {
            "name": "id",
            "type": "string"
          }
        ]
      }
    }
  }
]
```

### Query Operations

#### POST /query

Execute a SQL query against the catalog.

**Request Body:**

```json
{
  "query": "SELECT * FROM customers LIMIT 10"
}
```

**Response:**

```json
{
  "columns": ["id", "name"],
  "rows": [
    ["1", "John Doe"],
    ["2", "Jane Smith"]
  ],
  "execution_time": "0.5s",
  "row_count": 2
}
```

## Error Responses

All endpoints may return the following error responses:

### 400 Bad Request

```json
{
  "error": "Invalid request parameters",
  "details": "Detailed error message"
}
```

### 401 Unauthorized

```json
{
  "error": "Unauthorized",
  "details": "Invalid or missing API key"
}
```

### 404 Not Found

```json
{
  "error": "Not Found",
  "details": "Resource not found"
}
```

### 500 Internal Server Error

```json
{
  "error": "Internal Server Error",
  "details": "Detailed error message"
}
```

## Rate Limiting

The API implements rate limiting to ensure fair usage. The current limits are:

- 100 requests per minute per API key
- 1000 requests per hour per API key

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 99
X-RateLimit-Reset: 1615728000
```

## Best Practices

1. Always use the catalog as the source of truth for data operations
2. Include appropriate error handling in your applications
3. Cache responses when appropriate to reduce API load
4. Use the preview endpoints to validate data before processing
5. Monitor rate limits to avoid throttling
6. Use the audit logs to track changes to your data
7. Regularly check quality metrics to ensure data integrity
