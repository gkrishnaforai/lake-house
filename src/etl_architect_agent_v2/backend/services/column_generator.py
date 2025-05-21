"""Column generator for transformation tools."""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ColumnGenerator:
    """Generates column names for transformation tools."""
    
    def __init__(self, tool_config: Dict[str, Any]):
        """Initialize with tool configuration."""
        self.tool_config = tool_config
        
    def generate_column_names(self, source_column: str) -> List[Dict[str, str]]:
        """Generate column names based on tool configuration."""
        try:
            return [
                {
                    "name": config["name_template"].format(source_col=source_column),
                    "type": config["type"]
                }
                for config in self.tool_config["output_columns"].values()
            ]
        except Exception as e:
            logger.error(f"Error generating column names: {str(e)}")
            raise ValueError(f"Error generating column names: {str(e)}") 