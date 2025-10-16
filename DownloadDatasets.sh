#!/bin/bash

# ============================================================================
# Mosquitoes Dataset Pipeline
# Description: Executes the complete dataset preparation pipeline
# Author: Your Name
# Date: 2025-10-16
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TOOLS_DIR="Tools"
LOG_FILE="pipeline_execution.log"
START_TIME=$(date +%s)

# Python scripts to execute in order
SCRIPTS=(
    "DownloaderMosquitoesDataset.py"
    "DownloadNoiseEnviroment.py"
    "MosquitoesAudioConverter.py"
    "OrganizeDataset.py"
)

# ============================================================================
# Functions
# ============================================================================

print_header() {
    echo -e "${BLUE}============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 is not installed or not in PATH"
        exit 1
    fi
    print_success "Python3 found: $(python3 --version)"
}

check_tools_directory() {
    if [ ! -d "$TOOLS_DIR" ]; then
        print_error "Tools directory not found: $TOOLS_DIR"
        exit 1
    fi
    print_success "Tools directory found: $TOOLS_DIR"
}

check_script_exists() {
    local script=$1
    if [ ! -f "$TOOLS_DIR/$script" ]; then
        print_error "Script not found: $TOOLS_DIR/$script"
        return 1
    fi
    return 0
}

execute_script() {
    local script=$1
    local script_path="$TOOLS_DIR/$script"
    local script_name="${script%.*}"

    print_header "Executing: $script"
    log_message "Starting execution of $script"

    local script_start=$(date +%s)

    if python3 "$script_path"; then
        local script_end=$(date +%s)
        local script_duration=$((script_end - script_start))
        print_success "$script completed successfully in ${script_duration}s"
        log_message "$script completed successfully in ${script_duration}s"
        echo ""
        return 0
    else
        local exit_code=$?
        print_error "$script failed with exit code $exit_code"
        log_message "$script failed with exit code $exit_code"
        return $exit_code
    fi
}

cleanup_on_error() {
    print_error "Pipeline execution failed!"
    print_info "Check log file for details: $LOG_FILE"
    exit 1
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    # Clear/Create log file
    > "$LOG_FILE"

    print_header "Mosquitoes Dataset Pipeline - Starting"
    log_message "Pipeline execution started"

    # Pre-flight checks
    print_info "Running pre-flight checks..."
    check_python
    check_tools_directory

    # Verify all scripts exist before starting
    print_info "Verifying all required scripts..."
    local all_scripts_exist=true
    for script in "${SCRIPTS[@]}"; do
        if check_script_exists "$script"; then
            print_success "Found: $script"
        else
            all_scripts_exist=false
        fi
    done

    if [ "$all_scripts_exist" = false ]; then
        print_error "One or more scripts are missing. Aborting."
        exit 1
    fi

    echo ""
    print_success "All pre-flight checks passed!"
    echo ""

    # Execute scripts in sequence
    local script_count=0
    local total_scripts=${#SCRIPTS[@]}

    for script in "${SCRIPTS[@]}"; do
        script_count=$((script_count + 1))
        print_info "Progress: [$script_count/$total_scripts]"

        if ! execute_script "$script"; then
            cleanup_on_error
        fi
    done

    # Calculate total execution time
    END_TIME=$(date +%s)
    TOTAL_DURATION=$((END_TIME - START_TIME))
    MINUTES=$((TOTAL_DURATION / 60))
    SECONDS=$((TOTAL_DURATION % 60))

    # Success message
    print_header "Pipeline Completed Successfully!"
    print_success "All $total_scripts scripts executed successfully"
    print_info "Total execution time: ${MINUTES}m ${SECONDS}s"
    print_info "Log file: $LOG_FILE"

    log_message "Pipeline completed successfully in ${MINUTES}m ${SECONDS}s"

    echo ""
    print_success "✓ Dataset preparation pipeline finished!"
}

# Trap errors
trap cleanup_on_error ERR

# Run main function
main

exit 0