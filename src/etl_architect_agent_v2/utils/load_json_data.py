import json
from typing import Union

def load_json_file(file_path: str) -> dict:
    """
    Load JSON data from a file path.

    Args:
        file_path: Path to the JSON file.

    Returns:
        dict: The loaded JSON data.

    Raises:
        ValueError: If there is an error loading the JSON data.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading JSON data from {file_path}: {e}")

