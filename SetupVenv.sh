#!/bin/bash

# ============================================================================
# Setup Virtual Environment for Mosquitoes Pipeline
# Description: Creates/Updates the virtual environment with all dependencies
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

VENV_DIR="MosquitoesVenv"
REQUIREMENTS_FILE="requirements.txt"

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

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_header "Setting up Virtual Environment"

# Check if venv exists
if [ -d "$VENV_DIR" ]; then
    print_info "Virtual environment already exists: $VENV_DIR"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
        print_success "Removed"
    fi
fi

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    print_info "Creating virtual environment: $VENV_DIR"
    python3 -m venv "$VENV_DIR" --clear
    print_success "Virtual environment created"
fi

# Activate venv
print_info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip

# Install/upgrade essential packages
print_info "Installing essential packages..."
pip install --upgrade setuptools wheel

# Uninstall problematic packages first
print_info "Removing potentially conflicting packages..."
pip uninstall -y Pillow PIL 2>/dev/null || true

# Install core dependencies
print_info "Installing core dependencies..."
pip install --no-cache-dir requests numpy scipy

# Install Pillow fresh
print_info "Installing Pillow..."
pip install --no-cache-dir Pillow

# Install audio processing libraries
print_info "Installing audio processing libraries..."
pip install --no-cache-dir librosa soundfile audioread

# Install matplotlib
print_info "Installing matplotlib..."
pip install --no-cache-dir matplotlib

# Create/update requirements.txt
print_info "Generating requirements.txt..."
pip freeze > "$REQUIREMENTS_FILE"

# Verify installations
print_header "Verifying Installation"
python3 << 'EOF'
import sys
print(f"Python: {sys.version}")
print(f"Python executable: {sys.executable}")
print()

packages = [
    'requests',
    'numpy',
    'scipy',
    'PIL',
    'librosa',
    'soundfile',
    'matplotlib',
    'pandas'
]

print("Checking installed packages:")
for package in packages:
    try:
        if package == 'PIL':
            import PIL
            print(f"  ✓ {package:15s} - version {PIL.__version__}")
        else:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {package:15s} - version {version}")
    except ImportError as e:
        print(f"  ✗ {package:15s} - NOT INSTALLED")
        sys.exit(1)

print()
print("All packages installed successfully!")
EOF

if [ $? -eq 0 ]; then
    print_success "Virtual environment setup completed successfully!"
    print_info "Virtual environment: $VENV_DIR"
    print_info "Requirements file: $REQUIREMENTS_FILE"
    echo ""
    print_info "To activate manually: source $VENV_DIR/bin/activate"
else
    print_error "Package verification failed!"
    exit 1
fi

deactivate

exit 0