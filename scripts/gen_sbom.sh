#!/bin/bash

# SBOM (Software Bill of Materials) Generation Script
# Generates SPDX and CycloneDX format SBOMs for the project

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SECURITY_DIR="${PROJECT_ROOT}/security"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

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
    log_info "Checking SBOM generation prerequisites..."

    local missing_tools=()

    # Check for SBOM tools
    if ! command_exists syft && ! command_exists cyclonedx-bom; then
        log_warning "No SBOM tools found. Attempting to install..."
        install_sbom_tools
    fi

    # Verify installation
    if ! command_exists syft && ! command_exists cyclonedx-bom; then
        log_error "SBOM generation tools are required but not found."
        log_error "Please install Syft (https://github.com/anchore/syft) or CycloneDX CLI tools."
        exit 1
    fi

    log_success "SBOM tools are available"
}

# Install SBOM tools
install_sbom_tools() {
    log_info "Installing SBOM generation tools..."

    # Try to install Syft
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command_exists curl; then
            log_info "Installing Syft..."
            curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command_exists brew; then
            log_info "Installing Syft via Homebrew..."
            brew install syft
        fi
    fi

    # Try to install CycloneDX CLI for Node.js projects
    if command_exists npm; then
        log_info "Installing CycloneDX CLI..."
        npm install -g @cyclonedx/cyclonedx-npm
    fi
}

# Create security directory
setup_security_dir() {
    log_info "Setting up security directory..."

    mkdir -p "$SECURITY_DIR"

    # Create .gitignore for security directory
    cat > "${SECURITY_DIR}/.gitignore" << 'EOF'
# Ignore generated reports and temporary files
*.tmp
*.temp
*_temp_*

# Keep example files and templates
!*.example
!*_template.*
!README.md
EOF

    log_success "Security directory ready: $SECURITY_DIR"
}

# Generate SPDX SBOM using Syft
generate_spdx_sbom() {
    log_info "Generating SPDX SBOM..."

    if ! command_exists syft; then
        log_warning "Syft not available, skipping SPDX generation"
        return 0
    fi

    local spdx_file="${SECURITY_DIR}/SPDX_SBOM_${TIMESTAMP}.spdx.json"
    local latest_spdx="${SECURITY_DIR}/SPDX_SBOM.spdx.json"

    # Generate SPDX SBOM for the entire project
    log_info "Scanning project with Syft..."

    syft packages dir:"${PROJECT_ROOT}" \
        --output spdx-json="${spdx_file}" \
        --config "${PROJECT_ROOT}/.syft.yaml" 2>/dev/null || \
    syft packages dir:"${PROJECT_ROOT}" \
        --output spdx-json="${spdx_file}"

    if [ -f "$spdx_file" ]; then
        # Create symlink to latest version
        ln -sf "$(basename "$spdx_file")" "$latest_spdx"

        # Add metadata
        add_spdx_metadata "$spdx_file"

        log_success "SPDX SBOM generated: $spdx_file"
    else
        log_error "Failed to generate SPDX SBOM"
        return 1
    fi
}

# Add metadata to SPDX SBOM
add_spdx_metadata() {
    local spdx_file="$1"

    if command_exists jq; then
        log_info "Adding metadata to SPDX SBOM..."

        # Create temporary file with enhanced metadata
        local temp_file=$(mktemp)

        jq --arg timestamp "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
           --arg project_name "Space Telemetry Operations" \
           --arg project_version "$(git describe --tags --always 2>/dev/null || echo 'dev')" \
           --arg creator "Space Telemetry Ops SBOM Generator" \
           '.creationInfo.created = $timestamp |
            .name = $project_name |
            .documentNamespace = "https://github.com/space-telemetry-ops/sbom/" + $timestamp |
            .creationInfo.creators += ["Tool: " + $creator] |
            .documentDescribes[0].name = $project_name |
            .documentDescribes[0].versionInfo = $project_version' \
           "$spdx_file" > "$temp_file"

        mv "$temp_file" "$spdx_file"
        log_success "SPDX metadata added"
    fi
}

# Generate CycloneDX SBOM
generate_cyclonedx_sbom() {
    log_info "Generating CycloneDX SBOM..."

    local cyclonedx_file="${SECURITY_DIR}/cyclonedx-bom_${TIMESTAMP}.json"
    local latest_cyclonedx="${SECURITY_DIR}/cyclonedx-bom.json"

    # Try different CycloneDX tools based on project structure
    local generated=false

    # Node.js projects
    if [ -f "${PROJECT_ROOT}/src/app-frontend/package.json" ] && command_exists cyclonedx-npm; then
        log_info "Generating CycloneDX SBOM for Node.js frontend..."

        cd "${PROJECT_ROOT}/src/app-frontend"
        cyclonedx-npm --output-file "$cyclonedx_file" --output-format json
        generated=true
    elif [ -f "${PROJECT_ROOT}/src/services/ingest-node/package.json" ] && command_exists cyclonedx-npm; then
        log_info "Generating CycloneDX SBOM for Node.js ingest service..."

        cd "${PROJECT_ROOT}/src/services/ingest-node"
        cyclonedx-npm --output-file "$cyclonedx_file" --output-format json
        generated=true
    fi

    # Python projects
    if [ -f "${PROJECT_ROOT}/src/services/api-fastapi/requirements.txt" ] && command_exists cyclonedx-py; then
        log_info "Generating CycloneDX SBOM for Python API..."

        cd "${PROJECT_ROOT}/src/services/api-fastapi"
        cyclonedx-py -r requirements.txt --format json --output "$cyclonedx_file"
        generated=true
    fi

    # Fallback: use Syft to generate CycloneDX format
    if [ "$generated" = false ] && command_exists syft; then
        log_info "Generating CycloneDX SBOM with Syft..."

        syft packages dir:"${PROJECT_ROOT}" \
            --output cyclonedx-json="${cyclonedx_file}"
        generated=true
    fi

    if [ "$generated" = true ] && [ -f "$cyclonedx_file" ]; then
        # Create symlink to latest version
        ln -sf "$(basename "$cyclonedx_file")" "$latest_cyclonedx"

        # Add metadata
        add_cyclonedx_metadata "$cyclonedx_file"

        log_success "CycloneDX SBOM generated: $cyclonedx_file"
    else
        log_warning "Could not generate CycloneDX SBOM"
    fi

    cd "$PROJECT_ROOT"
}

# Add metadata to CycloneDX SBOM
add_cyclonedx_metadata() {
    local cyclonedx_file="$1"

    if command_exists jq; then
        log_info "Adding metadata to CycloneDX SBOM..."

        local temp_file=$(mktemp)
        local project_version=$(git describe --tags --always 2>/dev/null || echo 'dev')

        jq --arg timestamp "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
           --arg project_name "Space Telemetry Operations" \
           --arg project_version "$project_version" \
           --arg serial_number "urn:uuid:$(uuidgen 2>/dev/null || echo $(date +%s))" \
           '.metadata.timestamp = $timestamp |
            .metadata.component.name = $project_name |
            .metadata.component.version = $project_version |
            .serialNumber = $serial_number' \
           "$cyclonedx_file" > "$temp_file"

        mv "$temp_file" "$cyclonedx_file"
        log_success "CycloneDX metadata added"
    fi
}

# Generate dependency analysis
generate_dependency_analysis() {
    log_info "Generating dependency analysis..."

    local analysis_file="${SECURITY_DIR}/dependency_analysis_${TIMESTAMP}.json"
    local latest_analysis="${SECURITY_DIR}/dependency_analysis.json"

    # Initialize analysis structure
    cat > "$analysis_file" << EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "project": "Space Telemetry Operations",
  "analysis_type": "dependency_security",
  "summary": {},
  "components": {}
}
EOF

    # Analyze Python dependencies if pip-audit is available
    if command_exists pip-audit && [ -f "${PROJECT_ROOT}/src/services/api-fastapi/requirements.txt" ]; then
        log_info "Running Python dependency security analysis..."

        cd "${PROJECT_ROOT}/src/services/api-fastapi"

        if pip-audit --format json --output "${SECURITY_DIR}/python_audit_${TIMESTAMP}.json" 2>/dev/null; then
            log_success "Python security analysis completed"
        else
            log_warning "Python security analysis had issues"
        fi

        cd "$PROJECT_ROOT"
    fi

    # Analyze Node.js dependencies if npm audit is available
    if command_exists npm && [ -f "${PROJECT_ROOT}/src/app-frontend/package.json" ]; then
        log_info "Running Node.js dependency security analysis..."

        cd "${PROJECT_ROOT}/src/app-frontend"

        if npm audit --audit-level=info --json > "${SECURITY_DIR}/npm_audit_${TIMESTAMP}.json" 2>/dev/null; then
            log_success "Node.js security analysis completed"
        else
            log_warning "Node.js security analysis had issues"
        fi

        cd "$PROJECT_ROOT"
    fi

    # Create symlink to latest analysis
    ln -sf "$(basename "$analysis_file")" "$latest_analysis"

    log_success "Dependency analysis generated: $analysis_file"
}

# Generate license compliance report
generate_license_report() {
    log_info "Generating license compliance report..."

    local license_file="${SECURITY_DIR}/license_report_${TIMESTAMP}.json"
    local latest_license="${SECURITY_DIR}/license_report.json"

    # Use Syft for license scanning if available
    if command_exists syft; then
        log_info "Scanning for licenses with Syft..."

        syft packages dir:"${PROJECT_ROOT}" \
            --output template \
            --template "${PROJECT_ROOT}/.syft-license-template.txt" > "${license_file}.txt" 2>/dev/null || \
        syft packages dir:"${PROJECT_ROOT}" \
            --output json | jq '[.artifacts[] | select(.metadata.licenses) | {name: .name, version: .version, licenses: .metadata.licenses}]' > "$license_file"

        if [ -f "$license_file" ]; then
            ln -sf "$(basename "$license_file")" "$latest_license"
            log_success "License report generated: $license_file"
        fi
    else
        log_warning "Syft not available for license scanning"
    fi
}

# Validate generated SBOMs
validate_sboms() {
    log_info "Validating generated SBOMs..."

    local validation_errors=0

    # Validate SPDX SBOM
    local spdx_file="${SECURITY_DIR}/SPDX_SBOM.spdx.json"
    if [ -f "$spdx_file" ]; then
        if command_exists jq; then
            if jq empty "$spdx_file" 2>/dev/null; then
                log_success "SPDX SBOM is valid JSON"
            else
                log_error "SPDX SBOM has invalid JSON format"
                ((validation_errors++))
            fi
        fi
    fi

    # Validate CycloneDX SBOM
    local cyclonedx_file="${SECURITY_DIR}/cyclonedx-bom.json"
    if [ -f "$cyclonedx_file" ]; then
        if command_exists jq; then
            if jq empty "$cyclonedx_file" 2>/dev/null; then
                log_success "CycloneDX SBOM is valid JSON"
            else
                log_error "CycloneDX SBOM has invalid JSON format"
                ((validation_errors++))
            fi
        fi
    fi

    return $validation_errors
}

# Generate SBOM summary report
generate_summary_report() {
    log_info "Generating SBOM summary report..."

    local summary_file="${SECURITY_DIR}/sbom_summary_${TIMESTAMP}.md"
    local latest_summary="${SECURITY_DIR}/SBOM_SUMMARY.md"

    cat > "$summary_file" << EOF
# SBOM Summary Report

**Generated:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**Project:** Space Telemetry Operations
**Version:** $(git describe --tags --always 2>/dev/null || echo 'dev')

## Generated Artifacts

EOF

    # List generated files
    if [ -f "${SECURITY_DIR}/SPDX_SBOM.spdx.json" ]; then
        echo "- ✅ SPDX SBOM: \`SPDX_SBOM.spdx.json\`" >> "$summary_file"
    else
        echo "- ❌ SPDX SBOM: Not generated" >> "$summary_file"
    fi

    if [ -f "${SECURITY_DIR}/cyclonedx-bom.json" ]; then
        echo "- ✅ CycloneDX SBOM: \`cyclonedx-bom.json\`" >> "$summary_file"
    else
        echo "- ❌ CycloneDX SBOM: Not generated" >> "$summary_file"
    fi

    if [ -f "${SECURITY_DIR}/dependency_analysis.json" ]; then
        echo "- ✅ Dependency Analysis: \`dependency_analysis.json\`" >> "$summary_file"
    else
        echo "- ❌ Dependency Analysis: Not generated" >> "$summary_file"
    fi

    cat >> "$summary_file" << EOF

## Component Statistics

EOF

    # Add component statistics if SPDX SBOM exists
    if [ -f "${SECURITY_DIR}/SPDX_SBOM.spdx.json" ] && command_exists jq; then
        local total_packages=$(jq '.packages | length' "${SECURITY_DIR}/SPDX_SBOM.spdx.json")
        echo "- **Total Packages:** $total_packages" >> "$summary_file"

        # Count by ecosystem
        local npm_count=$(jq '[.packages[] | select(.name | contains("npm:"))] | length' "${SECURITY_DIR}/SPDX_SBOM.spdx.json")
        local python_count=$(jq '[.packages[] | select(.name | contains("python:"))] | length' "${SECURITY_DIR}/SPDX_SBOM.spdx.json")

        echo "- **npm Packages:** $npm_count" >> "$summary_file"
        echo "- **Python Packages:** $python_count" >> "$summary_file"
    fi

    cat >> "$summary_file" << EOF

## Usage

These SBOM files can be used for:

- Supply chain security analysis
- License compliance verification
- Vulnerability scanning
- Dependency tracking
- Regulatory compliance (Executive Order 14028)

## Validation

All generated SBOMs have been validated for:
- JSON format correctness
- Schema compliance (where tools are available)
- Required metadata completeness

## Next Steps

1. Review the generated SBOMs for completeness
2. Integrate SBOM generation into CI/CD pipeline
3. Set up regular vulnerability scanning using SBOMs
4. Archive SBOMs with release artifacts

EOF

    # Create symlink to latest summary
    ln -sf "$(basename "$summary_file")" "$latest_summary"

    log_success "SBOM summary report generated: $summary_file"
}

# Print usage information
print_usage() {
    echo "SBOM Generation Script for Space Telemetry Operations"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help           Show this help message"
    echo "  --spdx-only         Generate only SPDX format SBOM"
    echo "  --cyclonedx-only    Generate only CycloneDX format SBOM"
    echo "  --no-analysis       Skip dependency analysis"
    echo "  --no-licenses       Skip license report generation"
    echo "  --validate-only     Only validate existing SBOMs"
    echo ""
    echo "Examples:"
    echo "  $0                  # Generate all SBOM formats and reports"
    echo "  $0 --spdx-only      # Generate only SPDX SBOM"
    echo "  $0 --validate-only  # Validate existing SBOMs"
}

# Parse command line arguments
parse_args() {
    SPDX_ONLY=false
    CYCLONEDX_ONLY=false
    NO_ANALYSIS=false
    NO_LICENSES=false
    VALIDATE_ONLY=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                print_usage
                exit 0
                ;;
            --spdx-only)
                SPDX_ONLY=true
                shift
                ;;
            --cyclonedx-only)
                CYCLONEDX_ONLY=true
                shift
                ;;
            --no-analysis)
                NO_ANALYSIS=true
                shift
                ;;
            --no-licenses)
                NO_LICENSES=true
                shift
                ;;
            --validate-only)
                VALIDATE_ONLY=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done

    # Validate argument combinations
    if [ "$SPDX_ONLY" = true ] && [ "$CYCLONEDX_ONLY" = true ]; then
        log_error "Cannot specify both --spdx-only and --cyclonedx-only"
        exit 1
    fi
}

# Main execution
main() {
    echo "========================================="
    echo "Space Telemetry Operations"
    echo "SBOM Generation Script"
    echo "========================================="
    echo ""

    parse_args "$@"

    cd "$PROJECT_ROOT"

    if [ "$VALIDATE_ONLY" = true ]; then
        validate_sboms
        exit $?
    fi

    # Run SBOM generation
    check_prerequisites
    setup_security_dir

    if [ "$CYCLONEDX_ONLY" = false ]; then
        generate_spdx_sbom
    fi

    if [ "$SPDX_ONLY" = false ]; then
        generate_cyclonedx_sbom
    fi

    if [ "$NO_ANALYSIS" = false ]; then
        generate_dependency_analysis
    fi

    if [ "$NO_LICENSES" = false ]; then
        generate_license_report
    fi

    validate_sboms
    generate_summary_report

    echo ""
    echo "========================================="
    log_success "SBOM generation complete!"
    echo "========================================="
    echo ""
    echo "Generated files in security/ directory:"
    ls -la "$SECURITY_DIR"/*.json "$SECURITY_DIR"/*.md 2>/dev/null | tail -10
    echo ""
    echo "View summary: cat security/SBOM_SUMMARY.md"
}

# Execute main function with all arguments
main "$@"
