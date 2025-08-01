#!/bin/bash

# YAAML Package Build and Release Script - Modern uv version
set -e

echo "🚀 YAAML Package Build and Release Script (uv edition)"
echo "===================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    print_error "uv is not installed. Please install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Parse command line arguments
BUILD_ONLY=false
SKIP_TESTS=false
UPLOAD_TEST=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --test-pypi)
            UPLOAD_TEST=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --build-only    Only build the package, don't upload"
            echo "  --skip-tests    Skip running tests before build"
            echo "  --test-pypi     Upload to Test PyPI instead of PyPI"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option $1"
            exit 1
            ;;
    esac
done

# Step 1: Clean previous builds
print_status "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/
print_success "Cleaned previous builds"

# Step 2: Sync dependencies and ensure environment is ready
print_status "Setting up environment with uv..."
if ! uv sync --extra dev; then
    print_error "Failed to sync dependencies with uv"
    exit 1
fi
print_success "Environment ready"

# Step 3: Run tests (unless skipped)
if [ "$SKIP_TESTS" = false ]; then
    print_status "Running tests..."
    if ! uv run pytest tests/ -v --cov=yaaml --cov-report=term-missing; then
        print_error "Tests failed. Fix tests before building."
        exit 1
    fi
    print_success "All tests passed"
else
    print_warning "Skipping tests as requested"
fi

# Step 4: Run code quality checks with modern tools
print_status "Running code quality checks..."

print_status "Running pre-commit hooks..."
if ! uv run pre-commit run --all-files; then
    print_error "Code quality issues found. Run 'uv run pre-commit run --all-files' to see details."
    exit 1
fi

print_success "Code quality checks passed"

# Step 5: Build the package with uv
print_status "Building the package with uv..."
if ! uv build; then
    print_error "Package build failed"
    exit 1
fi
print_success "Package built successfully"

# Step 6: Check the built package (uv has built-in validation)
print_status "Validating the built package..."
if [ ! -d "dist" ] || [ -z "$(ls -A dist/)" ]; then
    print_error "No distribution files found in dist/"
    exit 1
fi

# List the built files
print_status "Built files:"
ls -la dist/
print_success "Package validation passed"

# If build-only flag is set, stop here
if [ "$BUILD_ONLY" = true ]; then
    print_success "Build completed successfully. Files in dist/:"
    ls -la dist/
    exit 0
fi

# Step 7: Upload to PyPI using uv
print_status "Uploading to PyPI using uv..."

if [ "$UPLOAD_TEST" = true ]; then
    print_warning "Uploading to Test PyPI..."
    if ! uv publish --index-url https://test.pypi.org/legacy/; then
        print_error "Upload to Test PyPI failed"
        exit 1
    fi
    print_success "Package uploaded to Test PyPI successfully"
    print_status "You can install the test package with:"
    echo "pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ yaaml"
else
    print_warning "Uploading to PyPI..."
    echo "This will upload to the real PyPI. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        if ! uv publish; then
            print_error "Upload to PyPI failed"
            exit 1
        fi
        print_success "Package uploaded to PyPI successfully"
        print_status "You can install the package with:"
        echo "pip install yaaml"
    else
        print_warning "Upload cancelled"
        exit 0
    fi
fi

print_success "🎉 Release process completed successfully!"
