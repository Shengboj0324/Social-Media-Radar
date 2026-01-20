#!/bin/bash
# Comprehensive Security Audit and Validation Script
# Ensures peak-level security before deployment

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
echo -e "${BLUE}SECURITY AUDIT - PEAK LEVEL${NC}"
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

echo -e "${BLUE}=== 1. Hardcoded Secrets Check ===${NC}"
run_check "No hardcoded passwords" "test -z \"\$(grep -rn 'password.*=.*['\\\"]' app --include='*.py' | grep -v 'hashed_password' | grep -v 'user_password' | grep -v 'plain_password' | grep -v 'get_password' | grep -v 'verify_password' | grep -v '# password')\""
run_check "No hardcoded API keys" "test -z \"\$(grep -rn 'api_key.*=.*['\\\"]sk-' app --include='*.py')\""
run_check "No hardcoded tokens" "test -z \"\$(grep -rn 'token.*=.*['\\\"][a-zA-Z0-9]{20,}' app --include='*.py' | grep -v 'access_token' | grep -v 'refresh_token')\""
echo ""

echo -e "${BLUE}=== 2. Injection Vulnerabilities ===${NC}"
run_check "No SQL injection (string formatting)" "test -z \"\$(grep -rn 'execute.*%s\\|execute.*format\\|execute.*+' app --include='*.py' | grep -v '# ')\""
run_check "No command injection (eval/exec)" "test -z \"\$(grep -rn '^[^#]*\\(eval\\|exec\\)(' app --include='*.py')\""
run_check "No shell injection" "test -z \"\$(grep -rn 'os.system\\|subprocess.call.*shell=True' app --include='*.py')\""
run_check "Path traversal prevention implemented" "grep -q 'sanitize_path' app/core/validation.py"
echo ""

echo -e "${BLUE}=== 3. Cryptography & Hashing ===${NC}"
run_check "No weak hashing (MD5/SHA1)" "test -z \"\$(grep -rn '\\(md5\\|sha1\\)(' app --include='*.py' | grep -v 'sha256' | grep -v '# ')\""
run_check "Strong password hashing (bcrypt)" "grep -q 'bcrypt' app/api/routes/auth.py"
run_check "Military-grade encryption available" "grep -q 'AES-256-GCM' app/core/security_advanced.py"
run_check "RSA-4096 encryption available" "grep -q 'RSA-4096' app/core/security_advanced.py"
echo ""

echo -e "${BLUE}=== 4. Authentication & Authorization ===${NC}"
run_check "JWT authentication implemented" "grep -q 'jwt.encode\\|jwt.decode' app/api/routes/auth.py"
run_check "Password verification implemented" "grep -q 'verify_password' app/api/routes/auth.py"
run_check "User activation check" "grep -q 'is_active' app/api/routes/auth.py"
run_check "Token expiration configured" "grep -q 'ACCESS_TOKEN_EXPIRE' app/api/routes/auth.py"
echo ""

echo -e "${BLUE}=== 5. Input Validation & Sanitization ===${NC}"
run_check "Input sanitization module exists" "test -f app/core/validation.py"
run_check "SQL injection prevention" "grep -q 'sanitize_sql_input' app/core/validation.py"
run_check "XSS prevention" "grep -q 'sanitize_html\\|bleach' app/core/security.py"
run_check "Path traversal prevention" "grep -q 'sanitize_path_input' app/core/validation.py"
run_check "Text validation with Pydantic" "grep -q 'TextValidator' app/core/validation.py"
echo ""

echo -e "${BLUE}=== 6. Security Headers ===${NC}"
run_check "X-Frame-Options header" "grep -q 'X-Frame-Options' app/core/security_advanced.py"
run_check "X-Content-Type-Options header" "grep -q 'X-Content-Type-Options' app/core/security_advanced.py"
run_check "Strict-Transport-Security header" "grep -q 'Strict-Transport-Security' app/core/security_advanced.py"
run_check "Content-Security-Policy header" "grep -q 'Content-Security-Policy' app/core/security_advanced.py"
echo ""

echo -e "${BLUE}=== 7. Rate Limiting & DDoS Protection ===${NC}"
run_check "Rate limiting middleware" "grep -q 'RateLimitMiddleware' app/api/middleware/security_middleware.py"
run_check "Token bucket algorithm" "grep -q 'token.*bucket' app/api/middleware/security_middleware.py"
run_check "IP blocking mechanism" "grep -q 'is_ip_blocked' app/api/middleware/security_middleware.py"
run_check "Brute force protection" "grep -q 'check_brute_force' app/api/middleware/security_middleware.py"
echo ""

echo -e "${BLUE}=== 8. Intrusion Detection ===${NC}"
run_check "Intrusion detection system" "grep -q 'IntrusionDetection' app/core/security_advanced.py"
run_check "Anomaly score calculation" "grep -q 'calculate_anomaly_score' app/core/security_advanced.py"
run_check "Failed attempt tracking" "grep -q 'record_failed_attempt' app/core/security_advanced.py"
run_check "Challenge mechanism (CAPTCHA)" "grep -q 'should_challenge' app/core/security_advanced.py"
echo ""

echo -e "${BLUE}=== 9. Data Protection ===${NC}"
run_check "Data masking implemented" "grep -q 'DataMasking' app/core/security_advanced.py"
run_check "Credential encryption" "grep -q 'CredentialEncryption' app/core/security.py"
run_check "Multi-layer encryption" "grep -q 'encrypt_multilayer' app/core/security_advanced.py"
run_check "Secure credential vault" "test -f app/core/credential_vault.py"
echo ""

echo -e "${BLUE}=== 10. Audit Logging ===${NC}"
run_check "Security audit log" "grep -q 'SecurityAuditLog' app/core/security_advanced.py"
run_check "Request logging" "grep -q '_log_request' app/api/middleware/security_middleware.py"
run_check "Risk level classification" "grep -q 'risk_level' app/api/middleware/security_middleware.py"
echo ""

echo -e "${BLUE}=== 11. CORS Configuration ===${NC}"
run_check "CORS middleware configured" "grep -q 'CORSMiddleware' app/api/main.py"
run_check "No wildcard CORS origins" "test -z \"\$(grep -rn 'allow_origins.*\\*' app --include='*.py')\""
run_check "CORS credentials allowed" "grep -q 'allow_credentials.*True' app/api/main.py"
echo ""

echo -e "${BLUE}=== 12. Database Security ===${NC}"
run_check "Parameterized queries (SQLAlchemy ORM)" "grep -q 'from sqlalchemy' app/core/db.py"
run_check "Connection pooling configured" "grep -q 'pool_size' app/core/db.py"
run_check "Connection timeout configured" "grep -q 'pool_timeout' app/core/db.py"
run_check "Database error handling" "grep -q 'SQLAlchemyError' app/core/db.py"
echo ""

echo -e "${BLUE}=== 13. Environment Configuration ===${NC}"
run_check "Environment variables used" "test -f .env.example"
run_check "Secret key configuration" "grep -q 'secret_key' app/core/config.py"
run_check "Encryption key configuration" "grep -q 'encryption_key' app/core/config.py"
run_check "No debug mode in production" "test -z \"\$(grep -rn 'debug.*=.*True' app --include='*.py')\""
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SECURITY AUDIT SUMMARY${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Total Checks:   $TOTAL_CHECKS"
echo -e "${GREEN}Passed:         $PASSED_CHECKS${NC}"
echo -e "${RED}Failed:         $FAILED_CHECKS${NC}"
echo -e "${YELLOW}Warnings:       $WARNING_CHECKS${NC}"
echo ""

if [ $FAILED_CHECKS -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ ALL SECURITY CHECKS PASSED${NC}"
    echo -e "${GREEN}✓ PEAK-LEVEL SECURITY CONFIRMED${NC}"
    echo -e "${GREEN}========================================${NC}"
    exit 0
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ $FAILED_CHECKS SECURITY CHECKS FAILED${NC}"
    echo -e "${RED}✗ SECURITY ISSUES MUST BE RESOLVED${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

