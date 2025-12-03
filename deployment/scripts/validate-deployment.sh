#!/bin/bash
# Deployment validation script for LLM infrastructure
# Usage: ./validate-deployment.sh [environment]

set -e

ENVIRONMENT=${1:-production}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "LLM Infrastructure Deployment Validation"
echo "Environment: $ENVIRONMENT"
echo "=========================================="
echo ""

# Function to check endpoint
check_endpoint() {
    local url=$1
    local name=$2
    local expected_status=${3:-200}
    
    echo -n "Checking $name... "
    
    if response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>&1); then
        if [ "$response" -eq "$expected_status" ]; then
            echo -e "${GREEN}✓ OK (HTTP $response)${NC}"
            return 0
        else
            echo -e "${RED}✗ FAILED (HTTP $response, expected $expected_status)${NC}"
            return 1
        fi
    else
        echo -e "${RED}✗ FAILED (Connection error)${NC}"
        return 1
    fi
}

# Function to check JSON endpoint
check_json_endpoint() {
    local url=$1
    local name=$2
    local expected_key=$3
    
    echo -n "Checking $name... "
    
    if response=$(curl -s "$url" 2>&1); then
        if echo "$response" | jq -e ".$expected_key" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ OK${NC}"
            return 0
        else
            echo -e "${RED}✗ FAILED (Missing key: $expected_key)${NC}"
            echo "Response: $response"
            return 1
        fi
    else
        echo -e "${RED}✗ FAILED (Connection error)${NC}"
        return 1
    fi
}

# Check basic health
echo -e "${BLUE}=== Basic Health Checks ===${NC}"
check_endpoint "http://localhost:8000/health" "Application health"
check_endpoint "http://localhost:8000/metrics" "Prometheus metrics"
check_json_endpoint "http://localhost:8000/api/llm/health" "LLM health" "status"

# Check Prometheus
echo ""
echo -e "${BLUE}=== Prometheus Checks ===${NC}"
check_endpoint "http://localhost:9090/-/healthy" "Prometheus health"
check_endpoint "http://localhost:9090/api/v1/targets" "Prometheus targets"

# Check Grafana
echo ""
echo -e "${BLUE}=== Grafana Checks ===${NC}"
check_endpoint "http://localhost:3000/api/health" "Grafana health"

# Check database
echo ""
echo -e "${BLUE}=== Database Checks ===${NC}"
if command -v psql &> /dev/null; then
    echo -n "Checking PostgreSQL connection... "
    if psql -h localhost -U user -d social_media_radar -c "SELECT 1" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ OK${NC}"
    else
        echo -e "${RED}✗ FAILED${NC}"
    fi
else
    echo -e "${YELLOW}⚠ psql not installed, skipping${NC}"
fi

# Check Redis
echo ""
echo -e "${BLUE}=== Redis Checks ===${NC}"
if command -v redis-cli &> /dev/null; then
    echo -n "Checking Redis connection... "
    if redis-cli ping > /dev/null 2>&1; then
        echo -e "${GREEN}✓ OK${NC}"
    else
        echo -e "${RED}✗ FAILED${NC}"
    fi
else
    echo -e "${YELLOW}⚠ redis-cli not installed, skipping${NC}"
fi

# Check LLM endpoints
echo ""
echo -e "${BLUE}=== LLM Endpoint Checks ===${NC}"

# Test simple generation
echo -n "Testing /api/llm/generate... "
if response=$(curl -s -X POST "http://localhost:8000/api/llm/generate" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello", "max_tokens": 5}' 2>&1); then
    if echo "$response" | jq -e ".content" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ OK${NC}"
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo "Response: $response"
    fi
else
    echo -e "${RED}✗ FAILED (Connection error)${NC}"
fi

# Test chat endpoint
echo -n "Testing /api/llm/chat... "
if response=$(curl -s -X POST "http://localhost:8000/api/llm/chat" \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5}' 2>&1); then
    if echo "$response" | jq -e ".content" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ OK${NC}"
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo "Response: $response"
    fi
else
    echo -e "${RED}✗ FAILED (Connection error)${NC}"
fi

# Test stats endpoint
echo -n "Testing /api/llm/stats... "
if response=$(curl -s "http://localhost:8000/api/llm/stats" 2>&1); then
    if echo "$response" | jq -e ".total_requests" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ OK${NC}"
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo "Response: $response"
    fi
else
    echo -e "${RED}✗ FAILED (Connection error)${NC}"
fi

# Check metrics
echo ""
echo -e "${BLUE}=== Metrics Checks ===${NC}"
echo -n "Checking LLM metrics... "
if metrics=$(curl -s "http://localhost:8000/metrics" 2>&1); then
    if echo "$metrics" | grep -q "llm_requests_total"; then
        echo -e "${GREEN}✓ OK${NC}"
    else
        echo -e "${RED}✗ FAILED (Missing llm_requests_total metric)${NC}"
    fi
else
    echo -e "${RED}✗ FAILED (Connection error)${NC}"
fi

# Summary
echo ""
echo "=========================================="
echo -e "${GREEN}Deployment Validation Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Check Grafana dashboards: http://localhost:3000"
echo "  2. Check Prometheus targets: http://localhost:9090/targets"
echo "  3. Run load tests: ./deployment/scripts/run-tests.sh load"
echo ""

