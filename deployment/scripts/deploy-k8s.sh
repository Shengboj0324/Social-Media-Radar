#!/bin/bash
# Kubernetes deployment script for LLM infrastructure
# Usage: ./deploy-k8s.sh [namespace]

set -e

NAMESPACE=${1:-default}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "LLM Infrastructure Kubernetes Deployment"
echo "Namespace: $NAMESPACE"
echo "=========================================="

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "ERROR: kubectl is not installed"
    exit 1
fi

# Check if cluster is accessible
if ! kubectl cluster-info &> /dev/null; then
    echo "ERROR: Cannot connect to Kubernetes cluster"
    exit 1
fi

echo "✓ Kubernetes cluster is accessible"

# Create namespace if it doesn't exist
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    echo "Creating namespace: $NAMESPACE"
    kubectl create namespace "$NAMESPACE"
fi

# Build and push Docker image
echo ""
echo "Building Docker image..."
cd "$PROJECT_ROOT"
docker build -f deployment/docker/Dockerfile.llm -t social-media-radar-llm:latest .

# Tag and push to registry (modify for your registry)
REGISTRY=${DOCKER_REGISTRY:-localhost:5000}
docker tag social-media-radar-llm:latest "$REGISTRY/social-media-radar-llm:latest"

echo "Pushing image to registry: $REGISTRY"
docker push "$REGISTRY/social-media-radar-llm:latest" || echo "WARNING: Failed to push image. Using local image."

echo "✓ Docker image ready"

# Create secrets
echo ""
echo "Creating secrets..."
echo "IMPORTANT: Update deployment/kubernetes/llm-secrets.yaml with actual secrets"
echo "Press Enter to continue or Ctrl+C to abort..."
read

kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/llm-secrets.yaml" -n "$NAMESPACE"
echo "✓ Secrets created"

# Deploy application
echo ""
echo "Deploying LLM application..."
kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/llm-deployment.yaml" -n "$NAMESPACE"
echo "✓ Deployment created"

# Deploy HPA
echo ""
echo "Deploying Horizontal Pod Autoscaler..."
kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/hpa.yaml" -n "$NAMESPACE"
echo "✓ HPA created"

# Wait for deployment to be ready
echo ""
echo "Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s \
    deployment/social-media-radar-llm -n "$NAMESPACE"

echo "✓ Deployment is ready"

# Get service information
echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Service Information:"
kubectl get svc llm-service -n "$NAMESPACE"

echo ""
echo "Pod Status:"
kubectl get pods -l component=llm -n "$NAMESPACE"

echo ""
echo "Useful commands:"
echo "  - View logs: kubectl logs -f -l component=llm -n $NAMESPACE"
echo "  - Get pods: kubectl get pods -l component=llm -n $NAMESPACE"
echo "  - Describe deployment: kubectl describe deployment social-media-radar-llm -n $NAMESPACE"
echo "  - Scale deployment: kubectl scale deployment social-media-radar-llm --replicas=5 -n $NAMESPACE"
echo "  - Port forward: kubectl port-forward svc/llm-service 8000:8000 -n $NAMESPACE"
echo ""

