#!/bin/bash
# Comprehensive testing script for LLM infrastructure
# Usage: ./run-tests.sh [test-type]
# Test types: unit, integration, load, all

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

TEST_TYPE=${1:-all}

echo "=========================================="
echo "LLM Infrastructure Testing"
echo "Test Type: $TEST_TYPE"
echo "=========================================="
echo ""

cd "$PROJECT_ROOT"

# Function to print section header
print_header() {
    echo ""
    echo -e "${BLUE}=========================================="
    echo "$1"
    echo -e "==========================================${NC}"
    echo ""
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    print_error "pytest is not installed"
    echo "Install with: pip install pytest pytest-asyncio pytest-cov"
    exit 1
fi

# Unit tests
if [ "$TEST_TYPE" = "unit" ] || [ "$TEST_TYPE" = "all" ]; then
    print_header "Running Unit Tests"
    
    if pytest tests/ -v --ignore=tests/llm/test_integration.py --ignore=tests/llm/test_load.py; then
        print_success "Unit tests passed"
    else
        print_error "Unit tests failed"
        exit 1
    fi
fi

# Integration tests
if [ "$TEST_TYPE" = "integration" ] || [ "$TEST_TYPE" = "all" ]; then
    print_header "Running Integration Tests"
    
    # Check if API keys are set
    if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
        print_warning "API keys not set, skipping integration tests"
        print_warning "Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY to run integration tests"
    else
        if pytest tests/llm/test_integration.py -v -s; then
            print_success "Integration tests passed"
        else
            print_error "Integration tests failed"
            exit 1
        fi
    fi
fi

# Load tests
if [ "$TEST_TYPE" = "load" ] || [ "$TEST_TYPE" = "all" ]; then
    print_header "Running Load Tests"
    
    # Check if service is running
    if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_warning "Service not running at http://localhost:8000"
        print_warning "Start service with: ./deployment/scripts/deploy.sh"
        print_warning "Skipping load tests"
    else
        print_header "Programmatic Load Test"
        if python tests/llm/test_load.py; then
            print_success "Programmatic load test passed"
        else
            print_error "Programmatic load test failed"
            exit 1
        fi
        
        print_header "Locust Load Test"
        print_warning "Starting Locust web UI at http://localhost:8089"
        print_warning "Press Ctrl+C to stop"
        locust -f tests/llm/locustfile.py --host=http://localhost:8000 || true
    fi
fi

# Code quality checks
if [ "$TEST_TYPE" = "quality" ] || [ "$TEST_TYPE" = "all" ]; then
    print_header "Running Code Quality Checks"
    
    # Check for TODO comments
    print_header "Checking for TODO comments"
    TODOS=$(grep -rn "TODO" app/llm --include="*.py" | wc -l)
    if [ "$TODOS" -eq 0 ]; then
        print_success "No TODO comments found"
    else
        print_warning "Found $TODOS TODO comments"
        grep -rn "TODO" app/llm --include="*.py" || true
    fi
    
    # Check for print statements
    print_header "Checking for print statements"
    PRINTS=$(grep -rn "print(" app/llm --include="*.py" | grep -v "# print(" | wc -l)
    if [ "$PRINTS" -eq 0 ]; then
        print_success "No print statements found"
    else
        print_warning "Found $PRINTS print statements"
        grep -rn "print(" app/llm --include="*.py" | grep -v "# print(" || true
    fi
    
    # Check for bare except clauses
    print_header "Checking for bare except clauses"
    BARE_EXCEPT=$(grep -rn "except:" app/llm --include="*.py" | grep -v "# except:" | wc -l)
    if [ "$BARE_EXCEPT" -eq 0 ]; then
        print_success "No bare except clauses found"
    else
        print_error "Found $BARE_EXCEPT bare except clauses"
        grep -rn "except:" app/llm --include="*.py" | grep -v "# except:" || true
        exit 1
    fi
fi

# Test coverage
if [ "$TEST_TYPE" = "coverage" ] || [ "$TEST_TYPE" = "all" ]; then
    print_header "Generating Test Coverage Report"
    
    pytest tests/ --cov=app/llm --cov-report=html --cov-report=term \
        --ignore=tests/llm/test_integration.py --ignore=tests/llm/test_load.py
    
    print_success "Coverage report generated at htmlcov/index.html"
fi

echo ""
print_header "Testing Complete!"
print_success "All tests passed successfully"
echo ""

