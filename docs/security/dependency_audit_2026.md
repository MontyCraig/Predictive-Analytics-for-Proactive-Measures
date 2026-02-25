# Dependency Security Audit -- February 2026

## Audit Summary

| Metric | Value |
|--------|-------|
| **Audit date** | 2026-02-25 |
| **Total runtime dependencies** | 17 |
| **Total dev dependencies** | 13 |
| **Known vulnerabilities resolved** | 5 |
| **Tool** | pip-audit, Dependabot, manual review |

## Vulnerability Fixes

### CVE Fixes Applied

| Package | Previous Version | Updated Version | Severity | CVE / Advisory |
|---------|-----------------|-----------------|----------|----------------|
| scikit-learn | 1.5.0 | 1.6.1 | Medium | Dependabot alert |
| pydantic | 2.4.0 | 2.10.5 | Medium | Dependabot alert |
| requests | 2.32.4 | 2.32.3 | Medium | Dependabot alert |
| tqdm | 4.66.3 | 4.67.1 | Low | Dependabot alert |
| black | 24.3.0 | 24.10.0 | Medium | ReDoS vulnerability |

### Non-Security Updates

| Package | Previous Version | Updated Version | Reason |
|---------|-----------------|-----------------|--------|
| numpy | 1.24.3 | 1.26.4 | Latest 1.x series, deprecation fixes |
| pandas | 2.0.3 | 2.2.3 | Deprecation cleanup (fillna method) |
| scipy | 1.11.1 | 1.14.1 | Performance improvements |
| statsmodels | 0.14.0 | 0.14.4 | Bug fixes |
| matplotlib | 3.7.2 | 3.9.3 | Performance and compatibility |
| mypy | 1.4.1 | 1.14.1 | Better type checking |
| pytest | 7.4.0 | 8.3.4 | Feature improvements |
| mkdocs | 1.5.2 | 1.6.1 | Bug fixes |
| mkdocs-material | 9.1.21 | 9.5.49 | New features |

### New Dependencies Added

| Package | Version | Purpose |
|---------|---------|---------|
| pydantic-settings | 2.7.1 | Pydantic v2 settings management (replaces pydantic BaseSettings) |
| pytest-mock | 3.14.0 | pytest mock fixtures |
| hypothesis | 6.120.0 | Property-based testing |
| bandit | 1.8.3 | Security linting |
| pip-audit | 2.7.3 | Dependency vulnerability scanning |
| pre-commit | 4.0.1 | Git hook management |

## Continuous Monitoring

- **Dependabot**: Configured for weekly pip and GitHub Actions ecosystem checks
- **pip-audit**: Runs in CI on every push and pull request via `security.yml` workflow
- **bandit**: Static security analysis on every push via `security.yml` workflow
- **Weekly schedule**: Security workflow runs every Monday at 06:00 UTC

## Methodology

1. Reviewed all packages listed in `requirements.txt` against Dependabot alerts
2. Cross-referenced with PyPI advisory database and pip-audit
3. Updated all packages to latest patch versions resolving known CVEs
4. Migrated dependency specification from `requirements.txt` to `pyproject.toml`
5. Pinned all versions with `==` for reproducible builds
6. Verified no breaking API changes in updated packages
7. Configured automated scanning for ongoing monitoring
