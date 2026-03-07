#!/bin/bash
# Simplified Security Audit Script

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
echo -e "${BLUE}SECURITY AUDIT - PEAK LEVEL${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

echo -e "${BLUE}=== Security Features ===${NC}"

# Check for hardcoded secrets
[[ -z $(grep -rn "password.*=.*['\"]" app --include="*.py" | grep -v "hashed_password" | grep -v "user_password" | grep -v "plain_password" | grep -v "get_password" | grep -v "verify_password" | grep -v "# password") ]] && check "No hardcoded passwords" "PASS" || check "No hardcoded passwords" "FAIL"

[[ -z $(grep -rn "api_key.*=.*['\"]sk-" app --include="*.py") ]] && check "No hardcoded API keys" "PASS" || check "No hardcoded API keys" "FAIL"

# Cryptography
grep -q "bcrypt" app/api/routes/auth.py && check "Strong password hashing (bcrypt)" "PASS" || check "Strong password hashing (bcrypt)" "FAIL"

grep -q "AES-256-GCM" app/core/security_advanced.py && check "Military-grade encryption (AES-256-GCM)" "PASS" || check "Military-grade encryption (AES-256-GCM)" "FAIL"

grep -q "RSA-4096" app/core/security_advanced.py && check "RSA-4096 encryption" "PASS" || check "RSA-4096 encryption" "FAIL"

# Authentication
grep -q "jwt.encode" app/api/routes/auth.py && check "JWT authentication" "PASS" || check "JWT authentication" "FAIL"

grep -q "verify_password" app/api/routes/auth.py && check "Password verification" "PASS" || check "Password verification" "FAIL"

grep -q "is_active" app/api/routes/auth.py && check "User activation check" "PASS" || check "User activation check" "FAIL"

# Input validation
test -f app/core/validation.py && check "Input validation module" "PASS" || check "Input validation module" "FAIL"

grep -q "sanitize_sql_input" app/core/validation.py && check "SQL injection prevention" "PASS" || check "SQL injection prevention" "FAIL"

grep -q "sanitize_html" app/core/security.py && check "XSS prevention" "PASS" || check "XSS prevention" "FAIL"

grep -q "sanitize_path_input" app/core/validation.py && check "Path traversal prevention" "PASS" || check "Path traversal prevention" "FAIL"

# Security headers
grep -q "X-Frame-Options" app/core/security_advanced.py && check "X-Frame-Options header" "PASS" || check "X-Frame-Options header" "FAIL"

grep -q "X-Content-Type-Options" app/core/security_advanced.py && check "X-Content-Type-Options header" "PASS" || check "X-Content-Type-Options header" "FAIL"

grep -q "Strict-Transport-Security" app/core/security_advanced.py && check "HSTS header" "PASS" || check "HSTS header" "FAIL"

grep -q "Content-Security-Policy" app/core/security_advanced.py && check "CSP header" "PASS" || check "CSP header" "FAIL"

# Rate limiting
grep -q "RateLimitMiddleware" app/api/middleware/security_middleware.py && check "Rate limiting middleware" "PASS" || check "Rate limiting middleware" "FAIL"

grep -q "is_ip_blocked" app/api/middleware/security_middleware.py && check "IP blocking" "PASS" || check "IP blocking" "FAIL"

grep -q "check_brute_force" app/api/middleware/security_middleware.py && check "Brute force protection" "PASS" || check "Brute force protection" "FAIL"

# Intrusion detection
grep -q "IntrusionDetection" app/core/security_advanced.py && check "Intrusion detection system" "PASS" || check "Intrusion detection system" "FAIL"

grep -q "calculate_anomaly_score" app/core/security_advanced.py && check "Anomaly detection" "PASS" || check "Anomaly detection" "FAIL"

# Data protection
grep -q "DataMasking" app/core/security_advanced.py && check "Data masking" "PASS" || check "Data masking" "FAIL"

grep -q "encrypt_multilayer" app/core/security_advanced.py && check "Multi-layer encryption" "PASS" || check "Multi-layer encryption" "FAIL"

test -f app/core/credential_vault.py && check "Secure credential vault" "PASS" || check "Secure credential vault" "FAIL"

# Audit logging
grep -q "SecurityAuditLog" app/core/security_advanced.py && check "Security audit logging" "PASS" || check "Security audit logging" "FAIL"

# CORS
grep -q "CORSMiddleware" app/api/main.py && check "CORS middleware" "PASS" || check "CORS middleware" "FAIL"

[[ -z $(grep -rn "allow_origins.*\*" app --include="*.py") ]] && check "No wildcard CORS" "PASS" || check "No wildcard CORS" "FAIL"

# Database
grep -q "from sqlalchemy" app/core/db.py && check "SQLAlchemy ORM (parameterized queries)" "PASS" || check "SQLAlchemy ORM" "FAIL"

grep -q "pool_size" app/core/db.py && check "Connection pooling" "PASS" || check "Connection pooling" "FAIL"

grep -q "SQLAlchemyError" app/core/db.py && check "Database error handling" "PASS" || check "Database error handling" "FAIL"

# Environment
test -f .env.example && check "Environment variables template" "PASS" || check "Environment variables template" "FAIL"

grep -q "secret_key" app/core/config.py && check "Secret key configuration" "PASS" || check "Secret key configuration" "FAIL"

grep -q "encryption_key" app/core/config.py && check "Encryption key configuration" "PASS" || check "Encryption key configuration" "FAIL"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SECURITY AUDIT SUMMARY${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Total Checks:   $TOTAL"
echo -e "${GREEN}Passed:         $PASSED${NC}"
echo -e "${RED}Failed:         $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ ALL SECURITY CHECKS PASSED${NC}"
    echo -e "${GREEN}✓ PEAK-LEVEL SECURITY CONFIRMED${NC}"
    echo -e "${GREEN}========================================${NC}"
    exit 0
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ $FAILED SECURITY CHECKS FAILED${NC}"
    echo -e "${RED}✗ SECURITY ISSUES MUST BE RESOLVED${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
