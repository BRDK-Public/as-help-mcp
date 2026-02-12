# Contributing to AS Help MCP Server

Thank you for your interest in contributing to the AS Help MCP Server! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Docker (for testing Docker builds)
- B&R Automation Studio Help files (for integration/E2E tests)

### Development Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/BRDK-Public/as-help-mcp.git
   cd as-help-mcp
   ```

2. **Install dependencies:**

   ```bash
   uv sync --extra test --extra dev
   ```

3. **Set up pre-commit hooks:**
   ```bash
   uv run pre-commit install
   ```

## Making Changes

### Branch Naming

Use descriptive branch names:

- `feature/add-new-search-filter`
- `fix/search-query-escaping`
- `docs/update-installation-guide`
- `refactor/simplify-indexer`

### Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:

- `feat(search): add category filtering to search_help`
- `fix(indexer): handle missing HelpID gracefully`
- `docs(readme): clarify Docker volume configuration`

## Pull Request Process

1. **Create a feature branch** from `main`
2. **Make your changes** with appropriate tests
3. **Run the test suite locally:**
   ```bash
   uv run pytest tests/ -v
   ```
4. **Run linting and formatting:**
   ```bash
   uv run ruff check src tests
   uv run ruff format src tests
   ```
5. **Push your branch** and create a Pull Request
6. **Fill out the PR template** completely
7. **Wait for CI checks** to pass
8. **Address review feedback** if any

### PR Requirements

- All CI checks must pass
- Code coverage should not decrease
- At least one maintainer approval required
- No merge conflicts with `main`

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) with line length of 120
- Use type hints for function parameters and returns
- Use docstrings for public functions and classes

### Linting

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check for issues
uv run ruff check src tests

# Auto-fix issues
uv run ruff check --fix src tests

# Format code
uv run ruff format src tests
```

### Type Checking

```bash
uv run mypy src
```

## Testing

### Test Structure

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Tests with real indexer/search
└── e2e/            # Full workflow tests
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run only unit tests (fast)
uv run pytest tests/unit -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Skip slow tests
uv run pytest tests/ -m "not slow"
```

### Writing Tests

- Use `pytest` fixtures from `conftest.py`
- Mock external dependencies (file system, network)
- Aim for >80% code coverage on new code
- Include both success and failure cases

## Reporting Issues

Please use our [GitHub issue tracker](https://github.com/BRDK-Public/as-help-mcp/issues) to report bugs or request features. When you create a new issue, you will be prompted to choose from several templates that ensure all necessary information is provided.

### Bug Reports

When reporting bugs using the provided template, please include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Minimal steps to reproduce
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**:
   - OS and version
   - Python version
   - AS Help version (AS4/AS6)
   - Docker version (if applicable)
6. **Logs**: Relevant error messages or logs

### Feature Requests

When submitting feature requests using the provided template, please include:

1. **Use Case**: Why you need this feature
2. **Proposed Solution**: Your suggested approach
3. **Alternatives**: Other solutions you've considered

## Security

If you discover a security vulnerability, please follow our [Security Policy](SECURITY.md) and report it privately rather than opening a public issue.

## Questions?

If you have questions about contributing, feel free to:

- Open a [Discussion](https://github.com/BRDK-Public/as-help-mcp/discussions)
- Ask in the issue tracker with the `question` label

Thank you for contributing!
