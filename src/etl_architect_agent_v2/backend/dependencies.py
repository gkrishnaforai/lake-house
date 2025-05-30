"""Dependency injection functions for FastAPI routes."""

from .services.catalog_service import CatalogService
from .config import get_settings


# Get settings instance
settings = get_settings()


async def get_catalog_service() -> CatalogService:
    """Get or create catalog service instance."""
    service = CatalogService(
        bucket=settings.AWS_S3_BUCKET,
        aws_region=settings.AWS_REGION
    )
    await service.start()  # Start the service asynchronously
    return service 