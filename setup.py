"""Setup file for the etl_architect_agent package."""

from setuptools import setup, find_packages

setup(
    name="etl_architect_agent_v2",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "boto3",
        "pydantic",
        "pydantic-settings",
        "pandas",
        "avro-python3"
    ]
) 