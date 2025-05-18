# Contributing to ETL Architect Agent V2

Thank you for your interest in contributing to ETL Architect Agent V2! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and considerate of others.

## How to Contribute

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes
4. Run tests and ensure they pass
5. Submit a pull request

## Development Setup

1. Clone your fork:

   ```bash
   git clone https://github.com/your-username/etl-architect-agent-v2.git
   cd etl-architect-agent-v2
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style

We use the following tools to maintain code quality:

- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

Run these tools before submitting a pull request:

```bash
black .
isort .
flake8
mypy .
```

## Testing

Write tests for new features and ensure all tests pass:

```bash
pytest
```

## Documentation

- Update the README.md for significant changes
- Add docstrings to new functions and classes
- Update type hints where necessary

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the documentation if needed
3. The PR must pass all CI checks
4. You may merge the PR once you have the sign-off of at least one other developer

## Questions?

Feel free to open an issue if you have any questions about contributing to the project.
