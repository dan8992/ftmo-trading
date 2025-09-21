# CI/CD Pipeline for FTMO Trading System

## Overview

This document describes the complete CI/CD pipeline for the FTMO trading system using GitHub Actions, GitHub Container Registry (GHCR), and ArgoCD for GitOps deployment to k3s.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Code Push     │───▶│  GitHub Actions │───▶│      GHCR       │
│   (main/develop)│    │   CI/CD Build   │    │ Container Images│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        │
                       ┌─────────────────┐               │
                       │   GitOps        │               │
                       │   Manifest      │               │
                       │   Update        │               │
                       └─────────────────┘               │
                                │                        │
                                ▼                        │
                       ┌─────────────────┐               │
                       │     ArgoCD      │◀──────────────┘
                       │   Auto Sync     │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   k3s Cluster   │
                       │  dax-trading    │
                       │   namespace     │
                       └─────────────────┘
```

## Pipeline Stages

### 1. Code Quality & Testing
- **Linting**: flake8, black, isort
- **Testing**: pytest with coverage
- **Security**: Trivy vulnerability scanning
- **Triggers**: All pushes and PRs

### 2. Container Image Build
- **Registry**: GitHub Container Registry (ghcr.io)
- **Services Built**:
  - postgres-data-collector
  - signal-generation-engine
  - technical-pattern-service
  - finbert-sentiment-service
  - news-ingestion-service
  - signal-monitoring
  - dax-llm-service
  - finbert-trading-llm
- **Tagging Strategy**:
  - `latest` for main branch
  - `branch-SHA` for all builds
  - `pr-NUMBER` for pull requests

### 3. GitOps Manifest Update
- **Automatic**: Updates image tags in Kubernetes manifests
- **Branch**: Creates/updates `manifests` branch with current image tags
- **Validation**: Kubeval for manifest validation

### 4. ArgoCD Deployment
- **Auto-sync**: Monitors `manifests` branch
- **Self-healing**: Automatically corrects configuration drift
- **Rollback**: Maintains revision history for easy rollbacks

## Services Configuration

### Container Resources

| Service | CPU Request | Memory Request | CPU Limit | Memory Limit |
|---------|-------------|---------------|-----------|--------------|
| postgres-data-collector | 250m | 256Mi | 500m | 512Mi |
| signal-generation-engine | 500m | 1Gi | 1000m | 2Gi |
| technical-pattern-service | 250m | 512Mi | 500m | 1Gi |
| finbert-sentiment-service | 1000m | 2Gi | 2000m | 4Gi |
| news-ingestion-service | 250m | 256Mi | 500m | 512Mi |
| signal-monitoring | 250m | 256Mi | 500m | 512Mi |

### Environment Variables

All services use the following common environment variables:
- `POSTGRES_HOST`: postgres-service
- `POSTGRES_PORT`: 5432
- `POSTGRES_DB`: finrl_dax
- `POSTGRES_USER`: finrl_user
- `POSTGRES_PASSWORD`: (from postgres-secret)

Additional API keys:
- `ALPHA_VANTAGE_KEY`: (from api-keys-secret)
- `FMP_KEY`: (from api-keys-secret)

## Setup Instructions

### 1. Install ArgoCD on k3s

```bash
# Run the setup script
./scripts/setup-argocd.sh
```

### 2. Configure Secrets

```bash
# Create PostgreSQL secret
kubectl create secret generic postgres-secret \
  --from-literal=POSTGRES_PASSWORD=your-strong-password \
  -n dax-trading

# Create API keys secret (optional)
kubectl create secret generic api-keys-secret \
  --from-literal=ALPHA_VANTAGE_KEY=your-api-key \
  --from-literal=FMP_KEY=your-fmp-key \
  -n dax-trading
```

### 3. Deploy Applications

```bash
# Apply ArgoCD applications
kubectl apply -f infra/argocd/
```

## Workflow Triggers

### Continuous Integration (CI)
- **Trigger**: Push to any branch, Pull Requests to main
- **Actions**: Test, Lint, Build, Security Scan

### Continuous Deployment (CD)
- **Trigger**: Push to main branch
- **Actions**: Build images, Update manifests, Deploy via ArgoCD

## Monitoring & Observability

### ArgoCD Dashboard
- **URL**: http://YOUR_K3S_IP:30080
- **Username**: admin
- **Password**: Retrieved during setup

### Application Health
- **Health Checks**: HTTP endpoints for web services
- **Liveness Probes**: Restart unhealthy containers
- **Readiness Probes**: Traffic routing control

## Security

### Container Security
- **Base Images**: Official Python slim images
- **Vulnerability Scanning**: Trivy in CI pipeline
- **Multi-platform**: linux/amd64, linux/arm64

### Secrets Management
- **K8s Secrets**: Database passwords, API keys
- **No Hardcoded Values**: All sensitive data in environment variables
- **Secret Rotation**: Manual process via kubectl

## Rollback Procedures

### Quick Rollback via ArgoCD
1. Access ArgoCD UI
2. Select ftmo-trading application
3. Navigate to History and Rollback
4. Select previous working revision

### Manual Rollback
```bash
# Get deployment history
kubectl rollout history deployment/signal-generation-engine -n dax-trading

# Rollback to previous version
kubectl rollout undo deployment/signal-generation-engine -n dax-trading
```

## Troubleshooting

### Common Issues

1. **Image Pull Errors**
   - Verify GHCR credentials
   - Check image tags in manifests

2. **ArgoCD Sync Issues**
   - Check application status in ArgoCD UI
   - Verify manifest syntax with kubeval

3. **Service Startup Failures**
   - Check pod logs: `kubectl logs -f deployment/SERVICE_NAME -n dax-trading`
   - Verify environment variables and secrets

### Debugging Commands

```bash
# Check ArgoCD applications
kubectl get applications -n argocd

# Monitor deployment status
kubectl get deployments -n dax-trading

# View service logs
kubectl logs -l app=signal-generation-engine -n dax-trading --tail=100

# Check service health
kubectl get pods -n dax-trading
```

## Performance Optimization

### Resource Tuning
- Monitor CPU/Memory usage with Grafana
- Adjust resource requests/limits based on actual usage
- Scale replicas for high-traffic services

### Image Optimization
- Multi-stage builds to reduce image size
- BuildKit cache for faster builds
- Layer optimization for better caching

## Future Enhancements

1. **Automated Testing**: Integration tests in CI pipeline
2. **Blue-Green Deployment**: Zero-downtime deployments
3. **Canary Releases**: Gradual rollout of new versions
4. **Multi-environment**: Staging and production environments
5. **Disaster Recovery**: Backup and restore procedures