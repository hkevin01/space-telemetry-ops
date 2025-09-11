#!/bin/bash

# Container Build and Management Script
# Builds, tags, and manages Docker containers for the Space Telemetry Operations system

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_DIR="${PROJECT_ROOT}/docker"
BUILD_DIR="${PROJECT_ROOT}/build"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Default configuration
REGISTRY=""
TAG="latest"
PUSH=false
BUILD_ALL=false
NO_CACHE=false
PARALLEL=false
PRUNE=false

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

# Check if Docker is available
check_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running or not accessible"
        exit 1
    fi

    log_success "Docker is available"
}

# Check if Docker Compose is available
check_compose() {
    if docker compose version >/dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    elif command -v docker-compose >/dev/null 2>&1; then
        COMPOSE_CMD="docker-compose"
    else
        log_error "Docker Compose is not available"
        exit 1
    fi

    log_success "Docker Compose is available: $COMPOSE_CMD"
}

# Get Git commit hash for tagging
get_git_commit() {
    if git rev-parse --git-dir >/dev/null 2>&1; then
        git rev-parse --short HEAD
    else
        echo "unknown"
    fi
}

# Get project version
get_project_version() {
    if git describe --tags --always >/dev/null 2>&1; then
        git describe --tags --always
    else
        echo "dev"
    fi
}

# Create build directory
setup_build_dir() {
    log_info "Setting up build directory..."

    mkdir -p "$BUILD_DIR"

    # Create build metadata
    cat > "${BUILD_DIR}/build_info.json" << EOF
{
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "git_commit": "$(get_git_commit)",
    "version": "$(get_project_version)",
    "build_id": "${TIMESTAMP}",
    "builder": "$(whoami)@$(hostname)"
}
EOF

    log_success "Build directory ready: $BUILD_DIR"
}

# Build individual service
build_service() {
    local service="$1"
    local dockerfile="$2"
    local context="$3"
    local build_args="${4:-}"

    log_info "Building $service..."

    local image_name="space-telemetry-ops/$service"
    local full_tag="$image_name:$TAG"

    # Add registry prefix if specified
    if [ -n "$REGISTRY" ]; then
        full_tag="$REGISTRY/$full_tag"
    fi

    # Build arguments
    local docker_args=(
        "build"
        "--file" "$dockerfile"
        "--tag" "$full_tag"
        "--tag" "$image_name:$(get_git_commit)"
        "--tag" "$image_name:build-$TIMESTAMP"
    )

    # Add no-cache flag if requested
    if [ "$NO_CACHE" = true ]; then
        docker_args+=("--no-cache")
    fi

    # Add build args
    if [ -n "$build_args" ]; then
        while IFS= read -r arg; do
            docker_args+=("--build-arg" "$arg")
        done <<< "$build_args"
    fi

    # Add build metadata
    docker_args+=(
        "--label" "org.opencontainers.image.created=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
        "--label" "org.opencontainers.image.version=$(get_project_version)"
        "--label" "org.opencontainers.image.revision=$(get_git_commit)"
        "--label" "org.opencontainers.image.title=Space Telemetry Ops - $service"
        "--label" "org.opencontainers.image.description=Space telemetry operations service: $service"
        "--label" "org.opencontainers.image.source=https://github.com/space-telemetry-ops"
        "--label" "space.telemetry.build.timestamp=$TIMESTAMP"
        "--label" "space.telemetry.service.name=$service"
    )

    # Add context
    docker_args+=("$context")

    # Execute build
    if docker "${docker_args[@]}"; then
        log_success "Built $service: $full_tag"

        # Save image information
        echo "$full_tag" >> "${BUILD_DIR}/built_images.txt"

        return 0
    else
        log_error "Failed to build $service"
        return 1
    fi
}

# Build API service
build_api() {
    log_info "Building FastAPI service..."

    local build_args="$(cat << EOF
PYTHON_VERSION=3.11
SERVICE_NAME=api-fastapi
SERVICE_PORT=8000
EOF
)"

    build_service "api" \
        "${DOCKER_DIR}/api.dockerfile" \
        "${PROJECT_ROOT}" \
        "$build_args"
}

# Build ingest service
build_ingest() {
    log_info "Building Node.js ingest service..."

    local build_args="$(cat << EOF
NODE_VERSION=18
SERVICE_NAME=ingest-node
SERVICE_PORT=3001
EOF
)"

    build_service "ingest" \
        "${DOCKER_DIR}/ingest.dockerfile" \
        "${PROJECT_ROOT}" \
        "$build_args"
}

# Build frontend
build_frontend() {
    log_info "Building React frontend..."

    local build_args="$(cat << EOF
NODE_VERSION=18
SERVICE_NAME=app-frontend
SERVICE_PORT=3000
EOF
)"

    build_service "frontend" \
        "${DOCKER_DIR}/frontend.dockerfile" \
        "${PROJECT_ROOT}" \
        "$build_args"
}

# Build ETL service
build_etl() {
    log_info "Building Airflow ETL service..."

    local build_args="$(cat << EOF
AIRFLOW_VERSION=2.7.0
PYTHON_VERSION=3.11
SERVICE_NAME=etl-airflow
EOF
)"

    build_service "etl" \
        "${DOCKER_DIR}/etl.dockerfile" \
        "${PROJECT_ROOT}" \
        "$build_args"
}

# Build nginx proxy
build_proxy() {
    log_info "Building Nginx proxy..."

    local build_args="$(cat << EOF
NGINX_VERSION=1.24-alpine
SERVICE_NAME=nginx-proxy
SERVICE_PORT=80
EOF
)"

    build_service "proxy" \
        "${DOCKER_DIR}/proxy.dockerfile" \
        "${PROJECT_ROOT}" \
        "$build_args"
}

# Build all services in parallel
build_all_parallel() {
    log_info "Building all services in parallel..."

    local pids=()
    local results=()

    # Start all builds in background
    (build_api; echo $? > "${BUILD_DIR}/api_result") &
    pids+=($!)

    (build_ingest; echo $? > "${BUILD_DIR}/ingest_result") &
    pids+=($!)

    (build_frontend; echo $? > "${BUILD_DIR}/frontend_result") &
    pids+=($!)

    (build_etl; echo $? > "${BUILD_DIR}/etl_result") &
    pids+=($!)

    (build_proxy; echo $? > "${BUILD_DIR}/proxy_result") &
    pids+=($!)

    # Wait for all builds to complete
    log_info "Waiting for parallel builds to complete..."

    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    # Check results
    local failed_builds=()

    for service in api ingest frontend etl proxy; do
        if [ -f "${BUILD_DIR}/${service}_result" ]; then
            local result=$(cat "${BUILD_DIR}/${service}_result")
            if [ "$result" -ne 0 ]; then
                failed_builds+=("$service")
            fi
        else
            failed_builds+=("$service")
        fi
    done

    # Report results
    if [ ${#failed_builds[@]} -eq 0 ]; then
        log_success "All services built successfully in parallel"
    else
        log_error "Failed to build: ${failed_builds[*]}"
        return 1
    fi
}

# Build all services sequentially
build_all_sequential() {
    log_info "Building all services sequentially..."

    local services=(build_api build_ingest build_frontend build_etl build_proxy)
    local failed_builds=()

    for service_func in "${services[@]}"; do
        if ! $service_func; then
            failed_builds+=("${service_func#build_}")
        fi
    done

    if [ ${#failed_builds[@]} -eq 0 ]; then
        log_success "All services built successfully"
    else
        log_error "Failed to build: ${failed_builds[*]}"
        return 1
    fi
}

# Push images to registry
push_images() {
    log_info "Pushing images to registry..."

    if [ -z "$REGISTRY" ]; then
        log_error "Registry not specified, cannot push images"
        return 1
    fi

    if [ ! -f "${BUILD_DIR}/built_images.txt" ]; then
        log_error "No built images found"
        return 1
    fi

    local failed_pushes=()

    while IFS= read -r image; do
        log_info "Pushing $image..."

        if docker push "$image"; then
            log_success "Pushed $image"
        else
            log_error "Failed to push $image"
            failed_pushes+=("$image")
        fi
    done < "${BUILD_DIR}/built_images.txt"

    if [ ${#failed_pushes[@]} -eq 0 ]; then
        log_success "All images pushed successfully"
    else
        log_error "Failed to push: ${failed_pushes[*]}"
        return 1
    fi
}

# Generate image manifest
generate_manifest() {
    log_info "Generating image manifest..."

    local manifest_file="${BUILD_DIR}/image_manifest.json"

    cat > "$manifest_file" << EOF
{
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "build_id": "${TIMESTAMP}",
    "version": "$(get_project_version)",
    "git_commit": "$(get_git_commit)",
    "registry": "${REGISTRY:-"local"}",
    "tag": "$TAG",
    "images": [
EOF

    local first=true
    if [ -f "${BUILD_DIR}/built_images.txt" ]; then
        while IFS= read -r image; do
            if [ "$first" = true ]; then
                first=false
            else
                echo "," >> "$manifest_file"
            fi

            # Get image ID and size
            local image_id=$(docker images --format "{{.ID}}" --filter "reference=$image" | head -1)
            local image_size=$(docker images --format "{{.Size}}" --filter "reference=$image" | head -1)

            cat >> "$manifest_file" << EOF
        {
            "name": "$image",
            "id": "$image_id",
            "size": "$image_size",
            "created": "$(docker inspect --format='{{.Created}}' "$image" 2>/dev/null || echo 'unknown')"
        }
EOF
        done < "${BUILD_DIR}/built_images.txt"
    fi

    cat >> "$manifest_file" << EOF
    ]
}
EOF

    log_success "Image manifest generated: $manifest_file"
}

# Prune Docker system
prune_docker() {
    log_info "Pruning Docker system..."

    # Remove dangling images
    docker image prune -f

    # Remove unused containers
    docker container prune -f

    # Remove unused networks
    docker network prune -f

    # Remove unused volumes (with caution)
    log_warning "Pruning unused volumes..."
    docker volume prune -f

    # Remove build cache
    docker builder prune -f

    log_success "Docker system pruned"
}

# Compose operations
compose_build() {
    log_info "Building with Docker Compose..."

    cd "$PROJECT_ROOT"

    local compose_args=("build")

    if [ "$NO_CACHE" = true ]; then
        compose_args+=("--no-cache")
    fi

    if [ "$PARALLEL" = true ]; then
        compose_args+=("--parallel")
    fi

    if $COMPOSE_CMD "${compose_args[@]}"; then
        log_success "Docker Compose build completed"
    else
        log_error "Docker Compose build failed"
        return 1
    fi
}

# Print usage information
print_usage() {
    echo "Container Build Script for Space Telemetry Operations"
    echo ""
    echo "Usage: $0 [OPTIONS] [SERVICES...]"
    echo ""
    echo "Services:"
    echo "  api                Build FastAPI service"
    echo "  ingest            Build Node.js ingest service"
    echo "  frontend          Build React frontend"
    echo "  etl               Build Airflow ETL service"
    echo "  proxy             Build Nginx proxy"
    echo "  all               Build all services"
    echo ""
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -r, --registry    Docker registry (e.g., docker.io/myorg)"
    echo "  -t, --tag         Image tag (default: latest)"
    echo "  --push            Push images to registry after build"
    echo "  --no-cache        Build without using cache"
    echo "  --parallel        Build services in parallel"
    echo "  --prune           Prune Docker system before build"
    echo "  --compose         Use Docker Compose for building"
    echo ""
    echo "Examples:"
    echo "  $0 api                              # Build API service"
    echo "  $0 all --parallel                   # Build all services in parallel"
    echo "  $0 -r docker.io/myorg -t v1.0 all  # Build and tag for registry"
    echo "  $0 --compose                        # Build using Docker Compose"
}

# Parse command line arguments
parse_args() {
    SERVICES=()
    USE_COMPOSE=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                print_usage
                exit 0
                ;;
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            -t|--tag)
                TAG="$2"
                shift 2
                ;;
            --push)
                PUSH=true
                shift
                ;;
            --no-cache)
                NO_CACHE=true
                shift
                ;;
            --parallel)
                PARALLEL=true
                shift
                ;;
            --prune)
                PRUNE=true
                shift
                ;;
            --compose)
                USE_COMPOSE=true
                shift
                ;;
            api|ingest|frontend|etl|proxy|all)
                SERVICES+=("$1")
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done

    # Default to building all services if none specified
    if [ ${#SERVICES[@]} -eq 0 ] && [ "$USE_COMPOSE" = false ]; then
        SERVICES=("all")
    fi
}

# Main execution
main() {
    echo "========================================="
    echo "Space Telemetry Operations"
    echo "Container Build Script"
    echo "========================================="
    echo ""

    parse_args "$@"

    cd "$PROJECT_ROOT"

    check_docker

    if [ "$USE_COMPOSE" = true ]; then
        check_compose
    fi

    if [ "$PRUNE" = true ]; then
        prune_docker
    fi

    setup_build_dir

    # Clean up any previous build results
    rm -f "${BUILD_DIR}"/*_result
    rm -f "${BUILD_DIR}/built_images.txt"

    # Execute builds
    if [ "$USE_COMPOSE" = true ]; then
        compose_build
    else
        local build_success=true

        for service in "${SERVICES[@]}"; do
            case $service in
                api)
                    build_api || build_success=false
                    ;;
                ingest)
                    build_ingest || build_success=false
                    ;;
                frontend)
                    build_frontend || build_success=false
                    ;;
                etl)
                    build_etl || build_success=false
                    ;;
                proxy)
                    build_proxy || build_success=false
                    ;;
                all)
                    if [ "$PARALLEL" = true ]; then
                        build_all_parallel || build_success=false
                    else
                        build_all_sequential || build_success=false
                    fi
                    ;;
                *)
                    log_error "Unknown service: $service"
                    build_success=false
                    ;;
            esac
        done

        if [ "$build_success" = false ]; then
            log_error "Some builds failed"
            exit 1
        fi
    fi

    generate_manifest

    if [ "$PUSH" = true ]; then
        push_images
    fi

    echo ""
    echo "========================================="
    log_success "Container build complete!"
    echo "========================================="
    echo ""
    echo "Build information:"
    cat "${BUILD_DIR}/build_info.json"
    echo ""
    echo "Built images:"
    if [ -f "${BUILD_DIR}/built_images.txt" ]; then
        cat "${BUILD_DIR}/built_images.txt"
    fi
}

# Execute main function with all arguments
main "$@"
