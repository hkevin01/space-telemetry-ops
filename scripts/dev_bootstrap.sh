#!/bin/bash

# Development Bootstrap Script
# This script sets up the development environment for Space Telemetry Operations

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
ENV_EXAMPLE="${PROJECT_ROOT}/.env.example"
DOCKER_COMPOSE_FILE="${PROJECT_ROOT}/infra/docker/docker-compose.yml"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    local missing_tools=()

    if ! command_exists docker; then
        missing_tools+=("docker")
    fi

    if ! command_exists docker-compose && ! docker compose version >/dev/null 2>&1; then
        missing_tools+=("docker-compose")
    fi

    if ! command_exists node; then
        missing_tools+=("node")
    fi

    if ! command_exists python3; then
        missing_tools+=("python3")
    fi

    if ! command_exists git; then
        missing_tools+=("git")
    fi

    if [ ${#missing_tools[@]} -gt 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install the missing tools and run this script again."
        exit 1
    fi

    log_success "All prerequisites are installed"
}

# Setup environment file
setup_env_file() {
    log_info "Setting up environment file..."

    if [ ! -f "$ENV_EXAMPLE" ]; then
        log_error ".env.example file not found at $ENV_EXAMPLE"
        exit 1
    fi

    if [ -f "$ENV_FILE" ]; then
        log_warning ".env file already exists"
        read -p "Do you want to overwrite it? (y/N): " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Keeping existing .env file"
            return 0
        fi
    fi

    cp "$ENV_EXAMPLE" "$ENV_FILE"

    # Generate secure random secrets
    if command_exists openssl; then
        SECRET_KEY=$(openssl rand -hex 32)
        JWT_SECRET=$(openssl rand -hex 32)

        # Replace placeholder secrets in .env file
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' "s/FASTAPI_SECRET=change-me/FASTAPI_SECRET=${SECRET_KEY}/" "$ENV_FILE"
            sed -i '' "s/JWT_SECRET=change-me/JWT_SECRET=${JWT_SECRET}/" "$ENV_FILE"
        else
            # Linux
            sed -i "s/FASTAPI_SECRET=change-me/FASTAPI_SECRET=${SECRET_KEY}/" "$ENV_FILE"
            sed -i "s/JWT_SECRET=change-me/JWT_SECRET=${JWT_SECRET}/" "$ENV_FILE"
        fi

        log_success "Generated secure secrets for .env file"
    else
        log_warning "OpenSSL not found. Please manually update secrets in .env file"
    fi

    log_success "Environment file created: $ENV_FILE"
}

# Setup Python environment
setup_python_env() {
    log_info "Setting up Python environment..."

    local api_dir="${PROJECT_ROOT}/src/services/api-fastapi"

    if [ ! -d "$api_dir" ]; then
        log_error "FastAPI directory not found: $api_dir"
        return 1
    fi

    cd "$api_dir"

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install dependencies
    if [ -f "requirements.txt" ]; then
        log_info "Installing Python dependencies..."
        pip install -r requirements.txt
    elif [ -f "pyproject.toml" ] && command_exists poetry; then
        log_info "Installing dependencies with Poetry..."
        poetry install
    else
        log_warning "No requirements.txt or pyproject.toml found"
    fi

    cd "$PROJECT_ROOT"
    log_success "Python environment setup complete"
}

# Setup Node.js environment
setup_node_env() {
    log_info "Setting up Node.js environment..."

    local frontend_dir="${PROJECT_ROOT}/src/app-frontend"
    local ingest_dir="${PROJECT_ROOT}/src/services/ingest-node"

    # Setup frontend dependencies
    if [ -d "$frontend_dir" ]; then
        cd "$frontend_dir"

        if [ -f "package.json" ]; then
            log_info "Installing frontend dependencies..."

            if command_exists pnpm; then
                pnpm install
            elif command_exists yarn; then
                yarn install
            else
                npm install
            fi

            log_success "Frontend dependencies installed"
        fi
    fi

    # Setup ingest service dependencies
    if [ -d "$ingest_dir" ]; then
        cd "$ingest_dir"

        if [ -f "package.json" ]; then
            log_info "Installing ingest service dependencies..."

            if command_exists pnpm; then
                pnpm install
            elif command_exists yarn; then
                yarn install
            else
                npm install
            fi

            log_success "Ingest service dependencies installed"
        fi
    fi

    cd "$PROJECT_ROOT"
}

# Setup Docker environment
setup_docker_env() {
    log_info "Setting up Docker environment..."

    if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
        log_error "Docker Compose file not found: $DOCKER_COMPOSE_FILE"
        return 1
    fi

    # Pull base images
    log_info "Pulling Docker base images..."
    docker pull postgres:15-alpine
    docker pull redis:7-alpine
    docker pull python:3.11-slim
    docker pull node:20-alpine

    # Build custom images
    log_info "Building custom Docker images..."
    docker compose -f "$DOCKER_COMPOSE_FILE" build --no-cache

    log_success "Docker environment setup complete"
}

# Setup development tools
setup_dev_tools() {
    log_info "Setting up development tools..."

    # Create necessary directories
    mkdir -p "${PROJECT_ROOT}/logs"
    mkdir -p "${PROJECT_ROOT}/data/cache"
    mkdir -p "${PROJECT_ROOT}/data/exports"
    mkdir -p "${PROJECT_ROOT}/security"

    # Set up Git hooks (if this is a Git repository)
    if [ -d "${PROJECT_ROOT}/.git" ]; then
        log_info "Setting up Git pre-commit hooks..."

        # Create pre-commit hook
        cat > "${PROJECT_ROOT}/.git/hooks/pre-commit" << 'EOF'
#!/bin/bash
# Pre-commit hook for Space Telemetry Operations

# Run linting and tests
echo "Running pre-commit checks..."

# Python linting
if command -v black >/dev/null 2>&1; then
    echo "Running Black formatter..."
    black --check src/services/api-fastapi/
fi

if command -v isort >/dev/null 2>&1; then
    echo "Running isort..."
    isort --check-only src/services/api-fastapi/
fi

# TypeScript/JavaScript linting
if [ -d "src/app-frontend" ] && [ -f "src/app-frontend/package.json" ]; then
    cd src/app-frontend
    if npm run lint >/dev/null 2>&1; then
        echo "Frontend linting passed"
    else
        echo "Frontend linting failed"
        exit 1
    fi
    cd ..
fi

echo "Pre-commit checks completed"
EOF

        chmod +x "${PROJECT_ROOT}/.git/hooks/pre-commit"
        log_success "Git hooks configured"
    fi

    log_success "Development tools setup complete"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."

    local errors=0

    # Check .env file
    if [ ! -f "$ENV_FILE" ]; then
        log_error ".env file not found"
        ((errors++))
    else
        log_success ".env file exists"
    fi

    # Check Docker
    if docker compose -f "$DOCKER_COMPOSE_FILE" config >/dev/null 2>&1; then
        log_success "Docker Compose configuration is valid"
    else
        log_error "Docker Compose configuration has errors"
        ((errors++))
    fi

    # Check Python environment
    local api_dir="${PROJECT_ROOT}/src/services/api-fastapi"
    if [ -d "${api_dir}/venv" ]; then
        log_success "Python virtual environment exists"
    else
        log_warning "Python virtual environment not found"
    fi

    if [ $errors -eq 0 ]; then
        log_success "Installation verification passed"
        return 0
    else
        log_error "Installation verification failed with $errors errors"
        return 1
    fi
}

# Start development services
start_services() {
    log_info "Starting development services..."

    # Start Docker services
    docker compose -f "$DOCKER_COMPOSE_FILE" up -d

    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 10

    # Check service health
    local services=("postgres" "redis" "minio")
    for service in "${services[@]}"; do
        if docker compose -f "$DOCKER_COMPOSE_FILE" ps "$service" | grep -q "Up"; then
            log_success "$service is running"
        else
            log_error "$service failed to start"
        fi
    done
}

# Print usage information
print_usage() {
    echo "Space Telemetry Operations - Development Bootstrap"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -s, --start    Start services after setup"
    echo "  --skip-docker  Skip Docker setup (for local development)"
    echo "  --skip-node    Skip Node.js setup"
    echo "  --skip-python  Skip Python setup"
    echo ""
    echo "Examples:"
    echo "  $0                 # Full setup without starting services"
    echo "  $0 --start         # Full setup and start services"
    echo "  $0 --skip-docker   # Setup without Docker"
}

# Parse command line arguments
parse_args() {
    SKIP_DOCKER=false
    SKIP_NODE=false
    SKIP_PYTHON=false
    START_SERVICES=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                print_usage
                exit 0
                ;;
            -s|--start)
                START_SERVICES=true
                shift
                ;;
            --skip-docker)
                SKIP_DOCKER=true
                shift
                ;;
            --skip-node)
                SKIP_NODE=true
                shift
                ;;
            --skip-python)
                SKIP_PYTHON=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
}

# Main execution
main() {
    echo "========================================="
    echo "Space Telemetry Operations"
    echo "Development Environment Bootstrap"
    echo "========================================="
    echo ""

    parse_args "$@"

    cd "$PROJECT_ROOT"

    # Run setup steps
    check_prerequisites
    setup_env_file

    if [ "$SKIP_PYTHON" = false ]; then
        setup_python_env
    fi

    if [ "$SKIP_NODE" = false ]; then
        setup_node_env
    fi

    if [ "$SKIP_DOCKER" = false ]; then
        setup_docker_env
    fi

    setup_dev_tools
    verify_installation

    if [ "$START_SERVICES" = true ]; then
        start_services
    fi

    echo ""
    echo "========================================="
    log_success "Development environment setup complete!"
    echo "========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Review and customize .env file if needed"
    echo "  2. Start services: docker compose -f infra/docker/docker-compose.yml up -d"
    echo "  3. Access applications:"
    echo "     - Frontend: http://localhost:5173"
    echo "     - API Docs: http://localhost:8000/docs"
    echo "     - Airflow: http://localhost:8080"
    echo "     - MinIO: http://localhost:9001"
    echo ""
    echo "For more information, see README.md"
}

# Execute main function with all arguments
main "$@"
