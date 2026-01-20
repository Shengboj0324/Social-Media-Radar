#!/bin/bash
# Comprehensive Deployment and Training Readiness Check

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASSED=0
FAILED=0
TOTAL=0

check() {
    local name="$1"
    local result="$2"
    TOTAL=$((TOTAL + 1))
    
    if [ "$result" = "PASS" ]; then
        echo -e "${GREEN}✓${NC} $name"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗${NC} $name"
        FAILED=$((FAILED + 1))
    fi
}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}DEPLOYMENT & TRAINING READINESS${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

echo -e "${BLUE}=== 1. Security Validation ===${NC}"
./deployment/scripts/security-audit-simple.sh > /dev/null 2>&1 && check "Security audit passed" "PASS" || check "Security audit passed" "FAIL"
echo ""

echo -e "${BLUE}=== 2. Code Quality ===${NC}"
./deployment/scripts/final-validation.sh > /dev/null 2>&1 && check "Final validation passed" "PASS" || check "Final validation passed" "FAIL"

find app/llm -name "*.py" -exec python3 -m py_compile {} + 2>/dev/null && check "LLM code compiles" "PASS" || check "LLM code compiles" "FAIL"

[[ -z $(grep -rn "TODO\|FIXME\|XXX\|HACK" app/llm --include="*.py") ]] && check "No TODOs in LLM code" "PASS" || check "No TODOs in LLM code" "FAIL"
echo ""

echo -e "${BLUE}=== 3. LLM Infrastructure ===${NC}"
python3 -c "import sys; sys.path.insert(0, '.'); from app.llm.router import LLMRouter" 2>/dev/null && check "LLM router imports" "PASS" || check "LLM router imports" "FAIL"

python3 -c "import sys; sys.path.insert(0, '.'); from app.llm.cache import get_llm_cache_manager" 2>/dev/null && check "LLM cache imports" "PASS" || check "LLM cache imports" "FAIL"

python3 -c "import sys; sys.path.insert(0, '.'); from app.llm.token_counter import get_token_counter" 2>/dev/null && check "Token counter imports" "PASS" || check "Token counter imports" "FAIL"

python3 -c "import sys; sys.path.insert(0, '.'); from app.llm.monitoring import get_metrics_collector" 2>/dev/null && check "LLM monitoring imports" "PASS" || check "LLM monitoring imports" "FAIL"
echo ""

echo -e "${BLUE}=== 4. Training Infrastructure ===${NC}"
python3 -c "import sys; sys.path.insert(0, '.'); from app.llm.training.data_pipeline import TrainingDataPipeline" 2>/dev/null && check "Data pipeline imports" "PASS" || check "Data pipeline imports" "FAIL"

python3 -c "import sys; sys.path.insert(0, '.'); from app.llm.training.lora_trainer import LoRATrainer" 2>/dev/null && check "LoRA trainer imports" "PASS" || check "LoRA trainer imports" "FAIL"

python3 -c "import sys; sys.path.insert(0, '.'); from app.llm.training.evaluator import ModelEvaluator" 2>/dev/null && check "Model evaluator imports" "PASS" || check "Model evaluator imports" "FAIL"

test -d app/llm/training && check "Training directory exists" "PASS" || check "Training directory exists" "FAIL"
echo ""

echo -e "${BLUE}=== 5. Deployment Files ===${NC}"
test -f deployment/docker/Dockerfile.llm && check "Dockerfile exists" "PASS" || check "Dockerfile exists" "FAIL"

test -f deployment/docker/docker-compose.llm.yml && check "Docker Compose config exists" "PASS" || check "Docker Compose config exists" "FAIL"

test -f deployment/prometheus/prometheus.yml && check "Prometheus config exists" "PASS" || check "Prometheus config exists" "FAIL"

test -f deployment/prometheus/alertmanager.yml && check "Alertmanager config exists" "PASS" || check "Alertmanager config exists" "FAIL"

test -f deployment/scripts/deploy.sh && check "Deployment script exists" "PASS" || check "Deployment script exists" "FAIL"
echo ""

echo -e "${BLUE}=== 6. Documentation ===${NC}"
test -f deployment/PRODUCTION_READY_REPORT.md && check "Production ready report" "PASS" || check "Production ready report" "FAIL"

test -f deployment/QUICK_START.md && check "Quick start guide" "PASS" || check "Quick start guide" "FAIL"

test -f .env.example && check "Environment template" "PASS" || check "Environment template" "FAIL"
echo ""

echo -e "${BLUE}=== 7. Production Simulation ===${NC}"
test -f deployment/scripts/production-simulation.py && check "Production simulation script" "PASS" || check "Production simulation script" "FAIL"

test -f deployment/simulation_results.json && check "Simulation results exist" "PASS" || check "Simulation results exist" "FAIL"

if [ -f deployment/simulation_results.json ]; then
    SUCCESS_RATE=$(python3 -c "import json; data=json.load(open('deployment/simulation_results.json')); print(data['successful_requests']/data['total_requests']*100)" 2>/dev/null || echo "0")
    [ "$(echo "$SUCCESS_RATE == 100" | bc -l 2>/dev/null || echo 0)" -eq 1 ] && check "100% simulation success rate" "PASS" || check "100% simulation success rate" "FAIL"
fi
echo ""

echo -e "${BLUE}=== 8. Dependencies ===${NC}"
test -f requirements.txt && check "Requirements file exists" "PASS" || check "Requirements file exists" "FAIL"

python3 -c "import tiktoken" 2>/dev/null && check "tiktoken installed" "PASS" || check "tiktoken installed" "FAIL"

python3 -c "import redis" 2>/dev/null && check "redis installed" "PASS" || check "redis installed" "FAIL"

python3 -c "import fastapi" 2>/dev/null && check "FastAPI installed" "PASS" || check "FastAPI installed" "FAIL"
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}READINESS SUMMARY${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Total Checks:   $TOTAL"
echo -e "${GREEN}Passed:         $PASSED${NC}"
echo -e "${RED}Failed:         $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ DEPLOYMENT READY${NC}"
    echo -e "${GREEN}✓ TRAINING READY${NC}"
    echo -e "${GREEN}✓ PRODUCTION READY${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo "1. Configure API keys in deployment/.env"
    echo "2. Run: ./deployment/scripts/deploy.sh production"
    echo "3. For training: python3 -m app.llm.training.lora_trainer"
    echo "4. Monitor: http://localhost:9090 (Prometheus)"
    echo ""
    exit 0
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ $FAILED CHECKS FAILED${NC}"
    echo -e "${RED}✗ NOT READY FOR DEPLOYMENT${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

