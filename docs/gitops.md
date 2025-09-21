# GitOps Setup Guide

This guide walks you through setting up GitOps for the FTMO Trading System using ArgoCD or FluxCD.

## Prerequisites

- Kubernetes cluster (local or cloud)
- kubectl configured and connected to your cluster
- GitHub repository with appropriate permissions
- SSH access to your cluster

## Option A: ArgoCD Setup (Recommended)

### 1. Install ArgoCD

```bash
# Create ArgoCD namespace
kubectl create namespace argocd

# Install ArgoCD
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Wait for ArgoCD to be ready
kubectl wait --for=condition=available --timeout=300s deployment/argocd-server -n argocd
```

### 2. Access ArgoCD UI

```bash
# Port forward to access the UI
kubectl port-forward svc/argocd-server -n argocd 8080:443

# Get the initial admin password
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
```

Access ArgoCD at `https://localhost:8080` with username `admin` and the password from above.

### 3. Configure Repository

In the ArgoCD UI:
1. Go to Settings → Repositories
2. Click "Connect Repo using SSH"
3. Add your repository: `git@github.com:dan8992/ftmo-trading-monorepo.git`
4. Upload your SSH private key or use SSH agent

### 4. Create ArgoCD Application

```yaml
# Save as argocd-application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ftmo-trading-system
  namespace: argocd
spec:
  project: default
  source:
    repoURL: git@github.com:dan8992/ftmo-trading-monorepo.git
    targetRevision: manifests
    path: infra/k8s-base
  destination:
    server: https://kubernetes.default.svc
    namespace: dax-trading
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
    - CreateNamespace=true
```

Apply the application:
```bash
kubectl apply -f argocd-application.yaml
```

## Option B: FluxCD Setup

### 1. Install Flux CLI

```bash
curl -s https://fluxcd.io/install.sh | sudo bash
```

### 2. Bootstrap Flux

```bash
export GITHUB_TOKEN=<your-github-token>
export GITHUB_USER=dan8992
export GITHUB_REPO=ftmo-trading-monorepo

flux bootstrap github \
  --owner=$GITHUB_USER \
  --repository=$GITHUB_REPO \
  --branch=main \
  --path=./infra/gitops/flux \
  --personal
```

### 3. Create GitRepository Source

```yaml
# infra/gitops/flux/git-source.yaml
apiVersion: source.toolkit.fluxcd.io/v1beta2
kind: GitRepository
metadata:
  name: ftmo-trading-monorepo
  namespace: flux-system
spec:
  interval: 1m
  ref:
    branch: manifests
  url: https://github.com/dan8992/ftmo-trading-monorepo
```

### 4. Create Kustomization

```yaml
# infra/gitops/flux/kustomization.yaml
apiVersion: kustomize.toolkit.fluxcd.io/v1beta2
kind: Kustomization
metadata:
  name: ftmo-trading-system
  namespace: flux-system
spec:
  interval: 10m
  path: "./infra/k8s-base"
  prune: true
  sourceRef:
    kind: GitRepository
    name: ftmo-trading-monorepo
  targetNamespace: dax-trading
```

## GitHub Actions Integration

### Required Secrets

Set these secrets in your GitHub repository (Settings → Secrets and variables → Actions):

```bash
# For image registry
GITHUB_TOKEN=<automatically-provided>

# For notifications (optional)
SLACK_WEBHOOK_URL=<your-slack-webhook>

# For ArgoCD webhook (optional)
ARGOCD_WEBHOOK_URL=<your-argocd-webhook>
```

### Workflow Overview

1. **CI Pipeline** (`.github/workflows/ci.yml`):
   - Runs tests and linting on every push/PR
   - Builds Docker images and pushes to GHCR
   - Performs security scanning

2. **GitOps Pipeline** (`.github/workflows/gitops-deploy.yml`):
   - Updates image tags in Kubernetes manifests
   - Commits changes to `manifests` branch
   - Triggers ArgoCD/Flux sync

## Setting Up Secrets

### Kubernetes Secrets

Create the required secrets in your cluster:

```bash
# Create namespace
kubectl create namespace dax-trading

# Database secrets
kubectl create secret generic postgres-secret \
  --from-literal=POSTGRES_USER=finrl_user \
  --from-literal=POSTGRES_PASSWORD=your-strong-password \
  -n dax-trading

# API secrets (replace with actual values)
kubectl create secret generic api-secrets \
  --from-literal=FINANCIAL_API_KEY=your-api-key \
  --from-literal=NEWS_API_KEY=your-news-api-key \
  -n dax-trading
```

### Sealed Secrets (Recommended for Production)

Install Sealed Secrets controller:

```bash
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.18.0/controller.yaml
```

Create sealed secrets:

```bash
# Install kubeseal
wget https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.18.0/kubeseal-linux-amd64 -O kubeseal
sudo install -m 755 kubeseal /usr/local/bin/kubeseal

# Create sealed secret
echo -n your-password | kubectl create secret generic postgres-secret \
  --dry-run=client --from-file=password=/dev/stdin -o yaml | \
  kubeseal -o yaml > postgres-sealed-secret.yaml
```

## Monitoring GitOps

### ArgoCD Monitoring

- **Application Health**: Check in ArgoCD UI
- **Sync Status**: Green = synced, Yellow = out of sync
- **Events**: View detailed deployment events

### Flux Monitoring

```bash
# Check GitRepository status
flux get sources git

# Check Kustomization status
flux get kustomizations

# View logs
flux logs
```

## Troubleshooting

### Common Issues

1. **Image Pull Errors**:
   ```bash
   kubectl create secret docker-registry ghcr-secret \
     --docker-server=ghcr.io \
     --docker-username=dan8992 \
     --docker-password=$GITHUB_TOKEN \
     -n dax-trading
   ```

2. **Permission Errors**:
   - Ensure GitHub token has package:read permissions
   - Check RBAC settings for ArgoCD/Flux

3. **Sync Failures**:
   - Validate YAML manifests: `kubectl apply --dry-run=client -f manifest.yaml`
   - Check resource limits and quotas

### Debug Commands

```bash
# Check pod status
kubectl get pods -n dax-trading

# View logs
kubectl logs -f deployment/signal-generation-engine -n dax-trading

# Describe resources
kubectl describe deployment signal-generation-engine -n dax-trading

# Check events
kubectl get events -n dax-trading --sort-by='.lastTimestamp'
```

## Best Practices

1. **Branching Strategy**:
   - `main`: Production-ready code
   - `develop`: Integration branch
   - `manifests`: Auto-generated deployment manifests

2. **Environment Promotion**:
   - Dev → Staging → Production
   - Use separate namespaces or clusters

3. **Security**:
   - Use sealed secrets for sensitive data
   - Enable RBAC and network policies
   - Regularly update base images

4. **Monitoring**:
   - Set up alerts for deployment failures
   - Monitor application health metrics
   - Track deployment frequency and lead time

## Next Steps

1. Set up monitoring with Prometheus and Grafana
2. Configure alerting rules
3. Implement backup strategies
4. Set up disaster recovery procedures
5. Add environment-specific configurations