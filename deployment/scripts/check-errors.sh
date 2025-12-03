#!/bin/bash
# Comprehensive error checking script
# Checks for syntax errors, import errors, and code quality issues

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
echo "Comprehensive Error Checking"
echo "=========================================="
echo ""

cd "$PROJECT_ROOT"

ERRORS=0
WARNINGS=0

# Function to check syntax
check_syntax() {
    echo -e "${BLUE}=== Checking Python Syntax ===${NC}"
    
    local failed=0
    while IFS= read -r file; do
        if ! python3 -m py_compile "$file" 2>/dev/null; then
            echo -e "${RED}✗${NC} Syntax error in: $file"
            python3 -m py_compile "$file" 2>&1 | head -5
            ((failed++))
        fi
    done < <(find app -name "*.py" -type f)
    
    if [ "$failed" -eq 0 ]; then
        echo -e "${GREEN}✓${NC} All Python files have valid syntax"
    else
        echo -e "${RED}✗${NC} Found $failed files with syntax errors"
        ((ERRORS+=failed))
    fi
}

# Function to check imports
check_imports() {
    echo ""
    echo -e "${BLUE}=== Checking Critical Imports ===${NC}"
    
    python3 << 'EOF'
import sys
sys.path.insert(0, '.')

errors = []

# Test LLM core imports
try:
    from app.llm.cache import get_llm_cache_manager
    from app.llm.token_counter import get_token_counter
    from app.llm.base_client import EnhancedBaseLLMClient
    from app.llm.router import LLMRouter
    from app.llm.config import DEFAULT_LLM_CONFIG
    from app.llm.models import LLMMessage, LLMResponse
    from app.llm.exceptions import LLMError
    from app.llm.monitoring import get_metrics_collector
    print('✓ LLM core imports successful')
except Exception as e:
    errors.append(f'LLM core imports failed: {e}')
    print(f'✗ LLM core imports failed: {e}')

# Test provider imports
try:
    from app.llm.providers import OpenAILLMClient, VLLMClient
    print('✓ Provider imports successful')
except Exception as e:
    errors.append(f'Provider imports failed: {e}')
    print(f'✗ Provider imports failed: {e}')

# Test training imports
try:
    from app.llm.training.data_pipeline import TrainingDataPipeline
    from app.llm.training.lora_trainer import LoRATrainer
    from app.llm.training.evaluator import ModelEvaluator
    print('✓ Training imports successful')
except Exception as e:
    errors.append(f'Training imports failed: {e}')
    print(f'✗ Training imports failed: {e}')

# Test API imports
try:
    from app.api.routes.llm import router
    print('✓ API imports successful')
except Exception as e:
    errors.append(f'API imports failed: {e}')
    print(f'✗ API imports failed: {e}')

if errors:
    sys.exit(1)
EOF
    
    if [ $? -ne 0 ]; then
        ((ERRORS++))
    fi
}

# Function to check for TODOs
check_todos() {
    echo ""
    echo -e "${BLUE}=== Checking for TODOs ===${NC}"
    
    local llm_todos=$(grep -rn "TODO\|FIXME\|XXX\|HACK" app/llm --include="*.py" 2>/dev/null | wc -l)
    local other_todos=$(grep -rn "TODO\|FIXME\|XXX\|HACK" app --include="*.py" --exclude-dir=llm 2>/dev/null | wc -l)
    
    if [ "$llm_todos" -eq 0 ]; then
        echo -e "${GREEN}✓${NC} LLM infrastructure: No TODOs"
    else
        echo -e "${YELLOW}⚠${NC} LLM infrastructure: Found $llm_todos TODOs"
        ((WARNINGS++))
    fi
    
    if [ "$other_todos" -gt 0 ]; then
        echo -e "${YELLOW}⚠${NC} Other modules: Found $other_todos TODOs"
        ((WARNINGS++))
    fi
}

# Function to check for print statements
check_prints() {
    echo ""
    echo -e "${BLUE}=== Checking for Print Statements ===${NC}"
    
    local llm_prints=$(grep -rn "print(" app/llm --include="*.py" | grep -v "# print(" | grep -v "pprint" | wc -l)
    local other_prints=$(grep -rn "print(" app --include="*.py" --exclude-dir=llm | grep -v "# print(" | grep -v "pprint" | wc -l)
    
    if [ "$llm_prints" -eq 0 ]; then
        echo -e "${GREEN}✓${NC} LLM infrastructure: No print statements"
    else
        echo -e "${RED}✗${NC} LLM infrastructure: Found $llm_prints print statements"
        ((ERRORS++))
    fi
    
    if [ "$other_prints" -gt 0 ]; then
        echo -e "${YELLOW}⚠${NC} Other modules: Found $other_prints print statements"
        ((WARNINGS++))
    fi
}

# Function to check for bare except
check_bare_except() {
    echo ""
    echo -e "${BLUE}=== Checking for Bare Except Clauses ===${NC}"
    
    local bare_except=$(grep -rn "except:" app --include="*.py" | grep -v "# except:" | grep -v "except Exception" | grep -v "except (" | wc -l)
    
    if [ "$bare_except" -eq 0 ]; then
        echo -e "${GREEN}✓${NC} No bare except clauses found"
    else
        echo -e "${RED}✗${NC} Found $bare_except bare except clauses"
        ((ERRORS++))
    fi
}

# Run all checks
check_syntax
check_imports
check_todos
check_prints
check_bare_except

echo ""
echo "=========================================="
if [ "$ERRORS" -eq 0 ]; then
    echo -e "${GREEN}✓ All error checks passed!${NC}"
    echo -e "Warnings: $WARNINGS"
    exit 0
else
    echo -e "${RED}✗ Error checking failed with $ERRORS errors${NC}"
    echo -e "Warnings: $WARNINGS"
    exit 1
fi

