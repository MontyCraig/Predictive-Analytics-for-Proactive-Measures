# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | Yes       |
| < 1.0   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly:

1. **Do not** open a public GitHub issue for security vulnerabilities.
2. Report via [GitHub private vulnerability reporting](https://github.com/MontyCraig/Predictive-Analytics-for-Proactive-Measures/security/advisories/new).
3. Include a clear description of the vulnerability, steps to reproduce, and potential impact.
4. Allow reasonable time for a fix before public disclosure.

## Security Practices

### Dependency Management

- All dependencies are pinned to exact versions in `pyproject.toml`.
- Dependabot monitors for known vulnerabilities weekly.
- `pip-audit` runs in CI on every push and pull request.
- `bandit` performs static security analysis on all source code.

### API Key Handling

- API keys are loaded exclusively from environment variables or `.env` files.
- Keys are stored as `pydantic.SecretStr` and never logged or serialised in plain text.
- The `.env` file is excluded from version control via `.gitignore`.

### Data Handling

- All file paths are validated before use.
- User-supplied input is validated through Pydantic models with strict mode enabled.
- No dynamic code execution (`eval`, `exec`) is used anywhere in the codebase.
- Model serialisation uses `joblib` with awareness of pickle deserialisation risks.
