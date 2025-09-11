# Security Policy

## Reporting Security Vulnerabilities

The Space Telemetry Operations project takes security seriously. We appreciate responsible disclosure of security vulnerabilities.

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities through one of the following methods:

1. **GitHub Security Advisories** (Preferred)
   - Navigate to the repository
   - Click on "Security" tab
   - Click "Report a vulnerability"

2. **Private Communication**
   - Contact the project maintainers directly
   - Encrypt sensitive details using PGP if possible

## What to Include

When reporting a security vulnerability, please include:

- **Type of vulnerability** (e.g., XSS, SQL injection, authentication bypass)
- **Location of vulnerability** (specific file/URL if applicable)
- **Step-by-step instructions** to reproduce the issue
- **Proof of concept** or exploit code (if available)
- **Impact assessment** including potential for data exposure or system compromise
- **Suggested mitigation** (if you have recommendations)

## Response Timeline

- **Initial response**: Within 48 hours
- **Vulnerability assessment**: Within 5 business days
- **Fix development**: Depends on severity (Critical: 72 hours, High: 1 week, Medium: 2 weeks)
- **Public disclosure**: After fix is deployed and verified

## Security Standards

This project adheres to:

- **NIST SP 800-53** Security Controls
- **OWASP Top 10** Web Application Security Risks
- **CIS Controls** for infrastructure security
- **STIG** (Security Technical Implementation Guides) where applicable

## Scope

Security testing is welcome on:
- The application code and configurations
- Docker containers and images
- API endpoints and authentication mechanisms
- Infrastructure-as-Code templates

**Out of scope:**
- Social engineering attacks
- Physical security attacks
- Attacks against third-party services
- DoS/DDoS attacks

## Recognition

We appreciate security researchers who help us maintain the security of this mission-critical system. Contributors will be recognized in our security acknowledgments (unless they prefer to remain anonymous).

## Legal

This security policy is designed to be compatible with good faith security research. We will not pursue legal action against researchers who:

- Follow responsible disclosure practices
- Respect the scope limitations
- Do not access or modify data belonging to others
- Do not perform attacks that could degrade system availability

## Questions

For questions about this security policy, please contact the project maintainers through the normal channels outlined above.
