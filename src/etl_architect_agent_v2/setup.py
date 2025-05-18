from setuptools import setup, find_packages

setup(
    name="etl-architect-agent-v2",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langgraph>=0.0.10",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A powerful ETL architecture design agent",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/etl-architect-agent-v2",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
) 