# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in the AS Help MCP Server, please report it responsibly.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following methods:

1. **GitHub Security Advisories**: Use the [Security Advisories](https://github.com/BRDK-Public/as-help-mcp/security/advisories) feature to privately report the vulnerability.

2. **Email**: Contact the maintainers directly at the email addresses listed in the [pyproject.toml](pyproject.toml) file.

### What to Include

When reporting a vulnerability, please include:

- **Description**: A clear description of the vulnerability
- **Impact**: The potential impact of the vulnerability
- **Steps to Reproduce**: Detailed steps to reproduce the issue
- **Affected Versions**: Which versions are affected
- **Suggested Fix**: If you have one, your suggested remediation

### Response Timeline

- **Acknowledgment**: We will acknowledge receipt of your report within 48 hours
- **Initial Assessment**: We will provide an initial assessment within 7 days
- **Resolution**: We aim to resolve critical vulnerabilities within 30 days

### What to Expect

- We will keep you informed of our progress
- We will credit you in the security advisory (unless you prefer to remain anonymous)
- We will not take legal action against you for responsibly disclosing the vulnerability

## Security Best Practices for Users

### Docker Deployment

1. **Use read-only mounts** for help files:
   ```bash
   -v "/path/to/help:/data/help:ro"
   ```

2. **Use named volumes** for database persistence (not host mounts)

3. **Pull from official registry** only:
   ```bash
   ghcr.io/brdk-public/as-help-mcp:latest
   ```

4. **Keep Docker updated** with latest security patches

### Local Development

1. **Never commit** `.env` files with real paths
2. **Use virtual environments** to isolate dependencies
3. **Keep dependencies updated**: Run `uv sync` regularly

## Known Security Considerations

### Local File Access

This MCP server reads files from a configured directory (`AS_HELP_ROOT`). The server:

- Only reads files within the configured help directory
- Does not execute any code from help files
- Uses `defusedxml` for safe XML parsing (prevents XXE attacks)
- Does not expose the file system over network

### SQLite Database

The search index database:

- Is stored locally (not networked)
- Contains only indexed text content (no credentials)
- Should be placed in a secure location with appropriate permissions

## Dependencies

We regularly audit dependencies for vulnerabilities:

```bash
uv run pip-audit
uv run bandit -r src/
```

Our CI pipeline includes automated security scanning.
