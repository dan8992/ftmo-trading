# 🔐 Security Guide

## Environment Setup

### 1. Create Environment File
```bash
cp .env.template .env
# Edit .env with your secure credentials
```

### 2. Required Environment Variables

**Database:**
```bash
POSTGRES_PASSWORD=your_secure_32_char_password_here
```

**API Keys:**
```bash
NEWS_API_KEY=your_news_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
FINANCIAL_API_KEY=your_financial_api_key
```

**Security:**
```bash
JWT_SECRET_KEY=your_32_character_random_secret_key
ENCRYPTION_KEY=your_32_character_encryption_key
GRAFANA_ADMIN_PASSWORD=your_secure_grafana_password
```

### 3. Kubernetes Secrets

Create PostgreSQL secret:
```bash
kubectl create secret generic postgres-secret \
  --from-literal=POSTGRES_USER=finrl_user \
  --from-literal=POSTGRES_PASSWORD=your_secure_password \
  -n dax-trading
```

Create API keys secret:
```bash
kubectl create secret generic api-keys-secret \
  --from-literal=NEWS_API_KEY=your_news_api_key \
  --from-literal=ALPHA_VANTAGE_KEY=your_alpha_vantage_key \
  --from-literal=FINANCIAL_API_KEY=your_financial_api_key \
  -n dax-trading
```

### 4. GitHub Secrets (for CI/CD)

Add these secrets to your GitHub repository:
- `POSTGRES_PASSWORD`
- `NEWS_API_KEY` 
- `ALPHA_VANTAGE_KEY`
- `GRAFANA_ADMIN_PASSWORD`
- `SLACK_WEBHOOK_URL` (optional)

## Security Best Practices

### 🔑 Password Requirements
- Minimum 16 characters
- Include uppercase, lowercase, numbers, symbols
- Use password manager to generate

### 🛡️ API Key Security
- Never commit API keys to git
- Rotate keys regularly
- Use different keys for dev/staging/prod
- Monitor API key usage

### 🚫 What NOT to Commit
- `.env` files
- `secrets.yaml` files
- Database backups
- Private keys (.key, .pem)
- Credentials in any form

### ✅ Safe to Commit
- `.env.template` (with placeholder values)
- Kubernetes manifests using `secretKeyRef`
- Docker compose files using `${VARIABLE}` syntax

## Emergency Response

If credentials are accidentally committed:
1. **Immediately** rotate all exposed credentials
2. Remove from git history: `git filter-branch` or BFG Repo-Cleaner
3. Force push cleaned history
4. Update all deployment environments
5. Monitor for unauthorized access

## Compliance

This system is designed for FTMO compliance with:
- No credential exposure in public repositories
- Secure environment variable handling
- Kubernetes-native secret management
- Audit trail for all credential access
