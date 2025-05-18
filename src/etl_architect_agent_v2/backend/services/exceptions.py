class SQLGenerationError(Exception):
    """Custom exception for SQL generation errors."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {} 