#!/bin/bash

# YAAML Package Build and Release Script
set -e

echo "🚀 YAAML Package Build and Release Script"
echo "========================================"

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

# Step 2: Run tests (unless skipped)
if [ "$SKIP_TESTS" = false ]; then
    print_status "Running tests..."
    if ! pytest tests/ -v --cov=yaaml --cov-report=term-missing; then
        print_error "Tests failed. Fix tests before building."
        exit 1
    fi
    print_success "All tests passed"
else
    print_warning "Skipping tests as requested"
fi

# Step 3: Run code quality checks
print_status "Running code quality checks..."

print_status "Checking code formatting with black..."
if ! black --check yaaml/; then
    print_error "Code formatting issues found. Run 'black yaaml/' to fix."
    exit 1
fi

print_status "Checking import sorting with isort..."
if ! isort --check-only yaaml/; then
    print_error "Import sorting issues found. Run 'isort yaaml/' to fix."
    exit 1
fi

print_status "Running linting with flake8..."
if ! flake8 yaaml/; then
    print_error "Linting issues found. Fix them before building."
    exit 1
fi

print_status "Running type checking with mypy..."
if ! mypy yaaml/ --ignore-missing-imports; then
    print_warning "Type checking issues found. Consider fixing them."
fi

print_success "Code quality checks passed"

# Step 4: Build the package
print_status "Building the package..."
if ! python -m build; then
    print_error "Package build failed"
    exit 1
fi
print_success "Package built successfully"

# Step 5: Check the built package
print_status "Checking the built package..."
if ! python -m twine check dist/*; then
    print_error "Package check failed"
    exit 1
fi
print_success "Package check passed"

# If build-only flag is set, stop here
if [ "$BUILD_ONLY" = true ]; then
    print_success "Build completed successfully. Files in dist/:"
    ls -la dist/
    exit 0
fi

# Step 6: Upload to PyPI
print_status "Uploading to PyPI..."

if [ "$UPLOAD_TEST" = true ]; then
    print_warning "Uploading to Test PyPI..."
    if ! python -m twine upload --repository testpypi dist/*; then
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
        if ! python -m twine upload dist/*; then
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
