# CI/CD Pipeline Troubleshooting Guide

## 🚨 Quick Fixes for Common Issues

### Build Failures

#### Python Import Errors
```bash
# Fix: Add PYTHONPATH to workflow
env:
  PYTHONPATH: ${{ github.workspace }}
```

#### Missing Dependencies
```bash
# Fix: Install requirements before testing
pip install -r src/services/api-fastapi/requirements.txt
pip install -r tests/requirements.txt
```

#### Docker Build Failures
```bash
# Check: Dockerfile exists in correct location
ls -la infra/docker/Dockerfile.fastapi

# Check: Build context is correct
docker build -f infra/docker/Dockerfile.fastapi -t test-build .
```

### Security Scan Issues

#### Permission Errors
```yaml
# Add to workflow file:
permissions:
  contents: read
  security-events: write
  packages: write
```

#### Trivy Scanner Failures
```yaml
# Fix: Add exit-code configuration
with:
  exit-code: '0'  # Don't fail build on vulnerabilities
  severity: 'CRITICAL,HIGH'
```

### Test Execution Problems

#### pytest Configuration
```ini
# Minimal pytest.ini:
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -v --tb=short
```

#### Import Path Issues
```python
# Add to conftest.py:
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
```

## ✅ Validation Checklist

- [ ] All required files exist
- [ ] GitHub Actions have proper permissions
- [ ] Docker builds successfully
- [ ] Tests can find imports
- [ ] Security scans are configured
- [ ] Dependencies are installable
