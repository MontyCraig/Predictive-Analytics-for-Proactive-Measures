# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-25

### Added

- Enterprise package structure under `src/predictive_analytics/`
- Pydantic v2 configuration with `ConfigDict`, `field_validator`, and `model_validator`
- Custom exception hierarchy (`PredictiveAnalyticsError` and subclasses)
- Shared type aliases (`SARIMAOrder`, `MetricsDict`, `DisruptionReport`)
- Typer-based CLI replacing legacy argparse interface
- PEP 561 `py.typed` marker for downstream type checking
- `pyproject.toml` with full tool configuration (black, isort, mypy, pytest, bandit)
- `.editorconfig` for consistent formatting across editors
- Enterprise documentation suite (CONTRIBUTING, SECURITY, CHANGELOG)
- GitHub issue and pull request templates
- Dependabot configuration for pip and GitHub Actions ecosystems

### Changed

- Migrated all Pydantic models from v1 `@validator` to v2 `@field_validator` / `@model_validator`
- Renamed `Config` to `AppConfig` to avoid shadowing Python builtins
- Replaced all `print()` calls with structured `logging` throughout codebase
- Replaced bare `ValueError` / `FileNotFoundError` with domain-specific exceptions
- Updated all dependencies to latest secure versions (scikit-learn 1.6.1, pydantic 2.10.5, requests 2.32.4)
- Typed `SARIMAConfig.order` as `tuple[int, int, int]` (was bare `tuple`)

### Fixed

- Missing `import numpy as np` at module level in preprocessing (caused `NameError` at runtime)
- Deprecated `data.fillna(method="bfill")` replaced with `data.bfill()`
- Shadowed `datetime` import in disruption analyzer

### Removed

- 12 dead stub files (4 Python snippets, 8 placeholder docs)
- Legacy `requirements.txt` (replaced by `pyproject.toml`)
- Root-level flat module layout (replaced by `src/` package)
