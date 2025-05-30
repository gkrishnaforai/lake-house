"""
Route modules for the ETL Architect Agent V2 backend.
"""

from . import catalog, conversion_routes, file_routes, transformation_routes

# Define router configurations
ROUTER_CONFIGS = {
    "catalog": {
        "router": catalog.router,
        "prefix": "/api/catalog",
        "tags": ["catalog"]
    },
    "conversion": {
        "router": conversion_routes.router,
        "prefix": "/api/conversion",
        "tags": ["conversion"]
    },
    "files": {
        "router": file_routes.router,
        "prefix": "/api/files",
        "tags": ["files"]
    },
    "transformation": {
        "router": transformation_routes.router,
        "prefix": "/api/transformation",
        "tags": ["transformation"]
    }
}

def get_router_configs():
    """Get all router configurations."""
    return ROUTER_CONFIGS 