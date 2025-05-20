"""Models package for the ETL Architect Agent."""

from .transformation import (
    TransformationTemplate,
    TransformationConfig,
    TransformationResult
)

from .catalog import (
    DatabaseCatalog,
    TableInfo,
    FileMetadata,
    FileInfo,
    FilePreview,
    FileConversionRequest,
    FileConversionResponse,
    UploadResponse,
    UploadProgress,
    ChatRequest,
    ChatResponse,
    ClassificationRequest,
    DescriptiveQueryRequest,
    DescriptiveQueryResponse,
    QualityMetrics,
    AuditLog,
    ColumnInfo
)

__all__ = [
    'TransformationTemplate',
    'TransformationConfig',
    'TransformationResult',
    'DatabaseCatalog',
    'TableInfo',
    'FileMetadata',
    'FileInfo',
    'FilePreview',
    'FileConversionRequest',
    'FileConversionResponse',
    'UploadResponse',
    'UploadProgress',
    'ChatRequest',
    'ChatResponse',
    'ClassificationRequest',
    'DescriptiveQueryRequest',
    'DescriptiveQueryResponse',
    'QualityMetrics',
    'AuditLog',
    'ColumnInfo'
] 