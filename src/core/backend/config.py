"""Configuration settings for the backend."""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# AWS Configuration
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# S3 Configuration
CATALOG_BUCKET = os.getenv('CATALOG_BUCKET', 'your-catalog-bucket')
ATHENA_RESULTS_BUCKET = os.getenv('ATHENA_RESULTS_BUCKET', 'your-athena-results')

# Glue Configuration
GLUE_DATABASE = os.getenv('GLUE_DATABASE', 'your_database')
GLUE_CRAWLER = os.getenv('GLUE_CRAWLER', 'your_crawler')

# API Configuration
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', '8000'))
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# CORS Configuration
CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
CORS_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
CORS_HEADERS = ['*']

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Service Configuration
SERVICE_CONFIG: Dict[str, Any] = {
    'catalog': {
        'bucket': CATALOG_BUCKET,
        'database': GLUE_DATABASE
    },
    'glue': {
        'database': GLUE_DATABASE,
        'crawler': GLUE_CRAWLER
    },
    'athena': {
        'results_bucket': ATHENA_RESULTS_BUCKET,
        'database': GLUE_DATABASE
    }
} 