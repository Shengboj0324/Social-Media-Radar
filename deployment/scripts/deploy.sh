#!/bin/bash
# Deployment script for LLM infrastructure
# Usage: ./deploy.sh [environment]

set -e

ENVIRONMENT=${1:-production}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "LLM Infrastructure Deployment"
echo "Environment: $ENVIRONMENT"
echo "=========================================="

# Load environment variables
if [ -f "$PROJECT_ROOT/deployment/.env.$ENVIRONMENT" ]; then
    echo "Loading environment from .env.$ENVIRONMENT"
    export $(cat "$PROJECT_ROOT/deployment/.env.$ENVIRONMENT" | grep -v '^#' | xargs)
elif [ -f "$PROJECT_ROOT/deployment/.env" ]; then
    echo "Loading environment from .env"
    export $(cat "$PROJECT_ROOT/deployment/.env" | grep -v '^#' | xargs)
else
    echo "ERROR: No .env file found"
    echo "Please copy .env.template to .env and configure it"
    exit 1
fi

# Validate required environment variables
required_vars=("OPENAI_API_KEY" "ANTHROPIC_API_KEY" "DATABASE_URL")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "ERROR: Required environment variable $var is not set"
        exit 1
    fi
done

echo "✓ Environment variables validated"

# Build Docker image
echo ""
echo "Building Docker image..."
cd "$PROJECT_ROOT"
docker build -f deployment/docker/Dockerfile.llm -t social-media-radar-llm:latest .
echo "✓ Docker image built"

# Deploy with Docker Compose
echo ""
echo "Deploying with Docker Compose..."
cd "$PROJECT_ROOT/deployment/docker"
docker-compose -f docker-compose.llm.yml up -d

echo ""
echo "Waiting for services to be healthy..."
sleep 10

# Check service health
echo ""
echo "Checking service health..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ LLM service is healthy"
        break
    fi
    
    attempt=$((attempt + 1))
    echo "Waiting for service... ($attempt/$max_attempts)"
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "ERROR: Service failed to become healthy"
    docker-compose -f docker-compose.llm.yml logs llm-app
    exit 1
fi

# Run database migrations (if needed)
echo ""
echo "Running database migrations..."
docker-compose -f docker-compose.llm.yml exec -T llm-app alembic upgrade head || true

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Services:"
echo "  - LLM API: http://localhost:8000"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana: http://localhost:3000 (admin/admin)"
echo ""
echo "Useful commands:"
echo "  - View logs: docker-compose -f deployment/docker/docker-compose.llm.yml logs -f"
echo "  - Stop services: docker-compose -f deployment/docker/docker-compose.llm.yml down"
echo "  - Restart services: docker-compose -f deployment/docker/docker-compose.llm.yml restart"
echo ""

