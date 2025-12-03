#!/bin/bash
# Comprehensive implementation validation script
# Validates all critical components are implemented

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "LLM Infrastructure Implementation Validation"
echo "=========================================="
echo ""

cd "$PROJECT_ROOT"

ERRORS=0
WARNINGS=0

# Function to check file exists
check_file() {
    local file=$1
    local description=$2
    
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $description: $file"
    else
        echo -e "${RED}✗${NC} $description: $file (MISSING)"
        ((ERRORS++))
    fi
}

# Function to check for TODOs
check_todos() {
    local dir=$1
    local description=$2
    
    local count=$(grep -rn "TODO\|FIXME\|XXX\|HACK" "$dir" --include="*.py" 2>/dev/null | wc -l)
    
    if [ "$count" -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $description: No TODOs found"
    else
        echo -e "${YELLOW}⚠${NC} $description: Found $count TODOs"
        ((WARNINGS++))
    fi
}

# Function to check for print statements
check_prints() {
    local dir=$1
    local description=$2
    
    local count=$(grep -rn "print(" "$dir" --include="*.py" | grep -v "# print(" | wc -l)
    
    if [ "$count" -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $description: No print statements"
    else
        echo -e "${RED}✗${NC} $description: Found $count print statements"
        ((ERRORS++))
    fi
}

# Function to check for bare except
check_bare_except() {
    local dir=$1
    local description=$2
    
    local count=$(grep -rn "except:" "$dir" --include="*.py" | grep -v "# except:" | wc -l)
    
    if [ "$count" -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $description: No bare except clauses"
    else
        echo -e "${RED}✗${NC} $description: Found $count bare except clauses"
        ((ERRORS++))
    fi
}

echo -e "${BLUE}=== Core LLM Infrastructure ===${NC}"
check_file "app/llm/base_client.py" "Enhanced base client"
check_file "app/llm/router.py" "Intelligent router"
check_file "app/llm/config.py" "Configuration"
check_file "app/llm/models.py" "Data models"
check_file "app/llm/exceptions.py" "Exception handling"
check_file "app/llm/retry.py" "Retry logic"
check_file "app/llm/circuit_breaker.py" "Circuit breaker"
check_file "app/llm/rate_limiter.py" "Rate limiter"
check_file "app/llm/monitoring.py" "Monitoring"
check_file "app/llm/cache.py" "Caching layer"
check_file "app/llm/token_counter.py" "Token counter"

echo ""
echo -e "${BLUE}=== Provider Implementations ===${NC}"
check_file "app/llm/providers/openai_provider.py" "OpenAI provider"
check_file "app/llm/providers/anthropic_provider.py" "Anthropic provider"
check_file "app/llm/providers/vllm_provider.py" "vLLM provider"

echo ""
echo -e "${BLUE}=== Training Infrastructure ===${NC}"
check_file "app/llm/training/data_pipeline.py" "Data pipeline"
check_file "app/llm/training/lora_trainer.py" "LoRA trainer"
check_file "app/llm/training/evaluator.py" "Evaluator"

echo ""
echo -e "${BLUE}=== Testing Infrastructure ===${NC}"
check_file "tests/llm/conftest.py" "Test fixtures"
check_file "tests/llm/test_integration.py" "Integration tests"
check_file "tests/llm/test_load.py" "Load tests"
check_file "tests/llm/locustfile.py" "Locust tests"

echo ""
echo -e "${BLUE}=== Deployment Infrastructure ===${NC}"
check_file "deployment/docker/Dockerfile.llm" "Dockerfile"
check_file "deployment/docker/docker-compose.llm.yml" "Docker Compose"
check_file "deployment/kubernetes/llm-deployment.yaml" "K8s Deployment"
check_file "deployment/kubernetes/hpa.yaml" "K8s HPA"
check_file "deployment/prometheus/prometheus.yml" "Prometheus config"
check_file "deployment/prometheus/alerts.yml" "Alert rules"
check_file "deployment/prometheus/alertmanager.yml" "Alertmanager config"
check_file "deployment/grafana/datasources/prometheus.yml" "Grafana datasource"
check_file "deployment/grafana/dashboards/llm-overview.json" "Grafana dashboard"

echo ""
echo -e "${BLUE}=== Deployment Scripts ===${NC}"
check_file "deployment/scripts/deploy.sh" "Docker deployment"
check_file "deployment/scripts/deploy-k8s.sh" "K8s deployment"
check_file "deployment/scripts/run-tests.sh" "Test runner"
check_file "deployment/scripts/validate-deployment.sh" "Deployment validator"

echo ""
echo -e "${BLUE}=== API Endpoints ===${NC}"
check_file "app/api/routes/llm.py" "LLM API routes"

echo ""
echo -e "${BLUE}=== Documentation ===${NC}"
check_file "deployment/PRODUCTION_READINESS.md" "Production readiness"
check_file "deployment/IMPLEMENTATION_SUMMARY.md" "Implementation summary"
check_file "deployment/FINAL_REPORT.md" "Final report"
check_file "docs/LLM_DEPLOYMENT_GUIDE.md" "Deployment guide"

echo ""
echo -e "${BLUE}=== Code Quality Checks ===${NC}"
check_todos "app/llm" "LLM infrastructure TODOs"
check_prints "app/llm" "LLM infrastructure print statements"
check_bare_except "app/llm" "LLM infrastructure bare except"

echo ""
echo "=========================================="
if [ "$ERRORS" -eq 0 ]; then
    echo -e "${GREEN}✓ All validations passed!${NC}"
    echo -e "Warnings: $WARNINGS"
    exit 0
else
    echo -e "${RED}✗ Validation failed with $ERRORS errors${NC}"
    echo -e "Warnings: $WARNINGS"
    exit 1
fi

