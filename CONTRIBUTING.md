# Contributing to Predictive Analytics for Proactive Measures

Thank you for your interest in contributing. This document explains how to get started.

## Development Setup

1. Clone the repository and create a conda environment:

   ```bash
   git clone https://github.com/MontyCraig/Predictive-Analytics-for-Proactive-Measures.git
   cd Predictive-Analytics-for-Proactive-Measures
   conda create -n predictive-analytics python=3.12
   conda activate predictive-analytics
   ```

2. Install the package in editable mode with development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

## Code Style

This project enforces consistent style via automated tooling:

- **black** (line length 99) for code formatting
- **isort** (black profile) for import sorting
- **flake8** for style linting
- **mypy --strict** with the Pydantic plugin for type checking
- **bandit** for security linting

Run all checks locally:

```bash
black --check src/ tests/
isort --check-only src/ tests/
flake8 src/ tests/
mypy --strict src/
bandit -r src/ -c pyproject.toml
```

## Testing

All contributions must maintain 100% test coverage:

```bash
pytest --cov-fail-under=100
```

- Write unit tests for every new function and method.
- Use `pytest-mock` for external dependencies (API calls, file I/O).
- Use `tmp_path` fixtures for any file system operations.
- Parametrise tests across model types where applicable.

## Pull Request Process

1. Create a feature branch from `main`:

   ```bash
   git checkout -b feat/your-feature main
   ```

2. Make your changes, ensuring all checks pass.

3. Write clear, atomic commits following [Conventional Commits](https://www.conventionalcommits.org/):

   - `feat:` new features
   - `fix:` bug fixes
   - `docs:` documentation changes
   - `style:` formatting (no logic changes)
   - `refactor:` code restructuring (no behaviour changes)
   - `test:` adding or updating tests
   - `chore:` maintenance tasks

4. Open a pull request against `main` and fill in the PR template.

5. Address review feedback promptly.

## Typing Requirements

- All function signatures must be fully typed (parameters and return types).
- Use `TypeAlias` from `predictive_analytics.types` for complex types.
- Use domain-specific exceptions from `predictive_analytics.exceptions`.
- Every module must define `__all__` for explicit exports.

## Reporting Issues

Use the GitHub issue templates for bug reports and feature requests.
Include reproduction steps, expected behaviour, and environment details.
