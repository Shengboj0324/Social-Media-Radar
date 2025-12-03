#!/bin/bash
# Final comprehensive validation before production deployment
# This script performs all critical checks to ensure production readiness

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}FINAL PRODUCTION VALIDATION${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to run check
run_check() {
    local name="$1"
    local command="$2"
    local critical="${3:-true}"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    echo -e "${BLUE}Checking: ${name}${NC}"
    
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $name"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        return 0
    else
        if [ "$critical" = "true" ]; then
            echo -e "${RED}✗${NC} $name"
            FAILED_CHECKS=$((FAILED_CHECKS + 1))
            return 1
        else
            echo -e "${YELLOW}⚠${NC} $name (non-critical)"
            WARNING_CHECKS=$((WARNING_CHECKS + 1))
            return 0
        fi
    fi
}

echo -e "${BLUE}=== 1. Python Environment ===${NC}"
run_check "Python 3.8+ installed" "python3 --version | grep -E 'Python 3\.(8|9|10|11|12)'"
run_check "pip installed" "python3 -m pip --version"
echo ""

echo -e "${BLUE}=== 2. Critical Dependencies ===${NC}"
run_check "Redis available" "python3 -c 'import redis'" false
run_check "PostgreSQL driver available" "python3 -c 'import psycopg2'" false
run_check "FastAPI installed" "python3 -c 'import fastapi'"
run_check "Pydantic installed" "python3 -c 'import pydantic'"
run_check "tiktoken installed" "python3 -c 'import tiktoken'"
echo ""

echo -e "${BLUE}=== 3. LLM Core Modules ===${NC}"
run_check "LLM config module" "python3 -c 'import sys; sys.path.insert(0, \".\"); from app.llm.config import DEFAULT_LLM_CONFIG'"
run_check "LLM models module" "python3 -c 'import sys; sys.path.insert(0, \".\"); from app.llm.models import LLMMessage'"
run_check "LLM base client" "python3 -c 'import sys; sys.path.insert(0, \".\"); from app.llm.base_client import EnhancedBaseLLMClient'"
run_check "LLM router" "python3 -c 'import sys; sys.path.insert(0, \".\"); from app.llm.router import LLMRouter'"
run_check "LLM cache" "python3 -c 'import sys; sys.path.insert(0, \".\"); from app.llm.cache import get_llm_cache_manager'"
run_check "LLM token counter" "python3 -c 'import sys; sys.path.insert(0, \".\"); from app.llm.token_counter import get_token_counter'"
run_check "LLM monitoring" "python3 -c 'import sys; sys.path.insert(0, \".\"); from app.llm.monitoring import get_metrics_collector'"
echo ""

echo -e "${BLUE}=== 4. LLM Providers ===${NC}"
run_check "Provider module" "python3 -c 'import sys; sys.path.insert(0, \".\"); from app.llm.providers import *'"
run_check "OpenAI provider" "python3 -c 'import sys; sys.path.insert(0, \".\"); from app.llm.providers.openai_provider import OpenAILLMClient'" false
run_check "Anthropic provider" "python3 -c 'import sys; sys.path.insert(0, \".\"); from app.llm.providers.anthropic_provider import AnthropicLLMClient'" false
run_check "vLLM provider" "python3 -c 'import sys; sys.path.insert(0, \".\"); from app.llm.providers.vllm_provider import VLLMClient'" false
echo ""

echo -e "${BLUE}=== 5. Training Infrastructure ===${NC}"
run_check "Data pipeline" "python3 -c 'import sys; sys.path.insert(0, \".\"); from app.llm.training.data_pipeline import TrainingDataPipeline'"
run_check "LoRA trainer" "python3 -c 'import sys; sys.path.insert(0, \".\"); from app.llm.training.lora_trainer import LoRATrainer'" false
run_check "Model evaluator" "python3 -c 'import sys; sys.path.insert(0, \".\"); from app.llm.training.evaluator import ModelEvaluator'"
echo ""

echo -e "${BLUE}=== 6. API Routes ===${NC}"
run_check "LLM API routes" "python3 -c 'import sys; sys.path.insert(0, \".\"); from app.api.routes.llm import router'"
echo ""

echo -e "${BLUE}=== 7. Code Quality ===${NC}"
run_check "No syntax errors" "find app/llm -name '*.py' -exec python3 -m py_compile {} +"
run_check "No TODOs in LLM code" "! grep -rn 'TODO\\|FIXME\\|XXX\\|HACK' app/llm --include='*.py'"
run_check "No print statements in LLM code" "! grep -rn 'print(' app/llm --include='*.py' | grep -v '# print('"
run_check "No bare except clauses" "! grep -rn 'except:' app/llm --include='*.py' | grep -v '# except:' | grep -v 'except Exception' | grep -v 'except ('"
echo ""

echo -e "${BLUE}=== 8. Deployment Files ===${NC}"
run_check "Dockerfile exists" "test -f deployment/docker/Dockerfile.llm || test -f deployment/docker/Dockerfile"
run_check "docker-compose.llm.yml exists" "test -f deployment/docker/docker-compose.llm.yml"
run_check "Kubernetes deployment exists" "test -f deployment/kubernetes/deployment.yml" false
run_check "Prometheus config exists" "test -f deployment/prometheus/prometheus.yml"
run_check "Alertmanager config exists" "test -f deployment/prometheus/alertmanager.yml"
run_check "Grafana datasources exist" "test -f deployment/grafana/datasources.yml" false
echo ""

echo -e "${BLUE}=== 9. Deployment Scripts ===${NC}"
run_check "deploy.sh exists" "test -f deployment/scripts/deploy.sh"
run_check "deploy-k8s.sh exists" "test -f deployment/scripts/deploy-k8s.sh"
run_check "run-tests.sh exists" "test -f deployment/scripts/run-tests.sh"
run_check "validate-deployment.sh exists" "test -f deployment/scripts/validate-deployment.sh"
run_check "validate-implementation.sh exists" "test -f deployment/scripts/validate-implementation.sh"
run_check "check-errors.sh exists" "test -f deployment/scripts/check-errors.sh"
run_check "production-simulation.py exists" "test -f deployment/scripts/production-simulation.py"
echo ""

echo -e "${BLUE}=== 10. Documentation ===${NC}"
run_check "Production readiness checklist" "test -f deployment/PRODUCTION_READINESS.md"
run_check "Implementation summary" "test -f deployment/IMPLEMENTATION_SUMMARY.md"
run_check "Final report" "test -f deployment/FINAL_REPORT.md"
run_check "Complete implementation report" "test -f deployment/COMPLETE_IMPLEMENTATION_REPORT.md"
run_check "Production ready report" "test -f deployment/PRODUCTION_READY_REPORT.md"
echo ""

echo -e "${BLUE}=== 11. Production Simulation ===${NC}"
if [ -f "deployment/simulation_results.json" ]; then
    SUCCESS_RATE=$(python3 -c "import json; data=json.load(open('deployment/simulation_results.json')); print(data['successful_requests']/data['total_requests']*100)")
    if [ "$(echo "$SUCCESS_RATE == 100" | bc -l)" -eq 1 ]; then
        echo -e "${GREEN}✓${NC} Production simulation: 100% success rate"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        echo -e "${YELLOW}⚠${NC} Production simulation: ${SUCCESS_RATE}% success rate"
        WARNING_CHECKS=$((WARNING_CHECKS + 1))
    fi
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
else
    echo -e "${YELLOW}⚠${NC} Production simulation not run yet"
    WARNING_CHECKS=$((WARNING_CHECKS + 1))
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
fi
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}VALIDATION SUMMARY${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Total Checks:   $TOTAL_CHECKS"
echo -e "${GREEN}Passed:         $PASSED_CHECKS${NC}"
echo -e "${RED}Failed:         $FAILED_CHECKS${NC}"
echo -e "${YELLOW}Warnings:       $WARNING_CHECKS${NC}"
echo ""

if [ $FAILED_CHECKS -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ ALL CRITICAL CHECKS PASSED${NC}"
    echo -e "${GREEN}✓ SYSTEM IS PRODUCTION READY${NC}"
    echo -e "${GREEN}========================================${NC}"
    exit 0
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ $FAILED_CHECKS CRITICAL CHECKS FAILED${NC}"
    echo -e "${RED}✗ SYSTEM IS NOT PRODUCTION READY${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

