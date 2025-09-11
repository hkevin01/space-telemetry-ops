# Contributing to Space Telemetry Operations

Thank you for your interest in contributing to the Space Telemetry Operations project! This document provides guidelines for contributing to this mission-critical space systems project.

## Code of Conduct

This project adheres to professional standards expected in aerospace and defense environments. Please be respectful, collaborative, and security-conscious in all interactions.

## Security First

- Never commit sensitive data, credentials, or classified information
- Follow NIST SP 800-53 security guidelines
- Report security vulnerabilities privately through GitHub Security Advisories
- Ensure all contributions maintain security posture

## Development Process

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/space-telemetry-ops.git
   cd space-telemetry-ops
   ```

2. **Set up Development Environment**
   ```bash
   # Copy environment template
   cp .env.example .env

   # Start development services
   docker compose -f infra/docker/docker-compose.yml up -d
   ```

3. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Changes**
   - Follow coding standards (see .vscode/settings.json)
   - Add comprehensive tests
   - Update documentation
   - Ensure security compliance

5. **Test Your Changes**
   ```bash
   # Run all tests
   pytest tests/ -v

   # Run security scan
   bash scripts/run_trivy.sh

   # Generate SBOM
   bash scripts/gen_sbom.sh
   ```

6. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

7. **Create Pull Request**
   - Use the provided PR template
   - Include mission impact assessment
   - Ensure all checks pass

## Coding Standards

### Python
- Use type hints for all functions
- Follow PEP 8 style guidelines
- Use docstrings (Google style)
- Handle all exceptions appropriately
- Include comprehensive error logging

### TypeScript/JavaScript
- Use strict TypeScript mode
- Follow Prettier formatting
- Use ESLint for code quality
- Implement proper error boundaries

### Security Requirements
- Input validation on all user data
- Secure error handling (no information leakage)
- Proper authentication/authorization
- Audit logging for security events

## Testing Guidelines

- Unit tests for all business logic
- Integration tests for API endpoints
- Security tests for authentication/authorization
- Performance tests for critical paths
- Boundary condition testing

## Documentation

- Update README.md if needed
- Document API changes in OpenAPI specs
- Update architecture diagrams
- Include inline code comments for complex logic

## Mission-Critical Considerations

- Ensure changes don't affect system availability
- Consider impact on real-time telemetry processing
- Test graceful failure modes
- Implement proper monitoring and alerting

## Review Process

1. Automated CI/CD checks must pass
2. Security scan must show no critical issues
3. Code review by at least one maintainer
4. Mission impact assessment if applicable
5. Final approval by project lead

## Questions?

For questions about contributing, please:
1. Check existing documentation
2. Search existing issues
3. Create a new issue with the question label
4. For security concerns, use GitHub Security Advisories
