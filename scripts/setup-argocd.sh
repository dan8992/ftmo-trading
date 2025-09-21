#!/bin/bash
set -e

echo "🚀 Setting up ArgoCD on k3s cluster..."

# Create argocd namespace
kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -

# Install ArgoCD
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Wait for ArgoCD to be ready
echo "⏳ Waiting for ArgoCD to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/argocd-server -n argocd

# Patch ArgoCD server service to use NodePort for easier access
kubectl patch svc argocd-server -n argocd -p '{"spec":{"type":"NodePort","ports":[{"port":80,"targetPort":8080,"nodePort":30080}]}}'

# Get initial admin password
echo "🔑 Getting ArgoCD admin password..."
ARGOCD_PASSWORD=$(kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d)

# Create GitHub repository secret for private access (optional)
echo "📝 Creating GitHub repository secret..."
kubectl create secret generic github-repo-secret \
  -n argocd \
  --from-literal=type=git \
  --from-literal=url=https://github.com/dan8992/ftmo-trading.git \
  --dry-run=client -o yaml | kubectl apply -f -

# Apply ArgoCD applications
echo "📦 Deploying ArgoCD applications..."
kubectl apply -f infra/argocd/

echo "✅ ArgoCD setup complete!"
echo ""
echo "🌐 Access ArgoCD at: http://YOUR_K3S_NODE_IP:30080"
echo "👤 Username: admin"
echo "🔐 Password: $ARGOCD_PASSWORD"
echo ""
echo "📋 Next steps:"
echo "1. Access ArgoCD UI and verify applications are syncing"
echo "2. Push code changes to trigger CI/CD pipeline"
echo "3. Monitor deployment status in ArgoCD dashboard"