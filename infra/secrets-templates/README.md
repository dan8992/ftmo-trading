# Kubernetes Secrets Templates

## Usage

1. **Copy templates to working directory:**
   ```bash
   cp postgres-secret.yaml postgres-secret-production.yaml
   cp api-keys-secret.yaml api-keys-secret-production.yaml
   ```

2. **Encode your actual credentials:**
   ```bash
   echo -n "your_actual_password" | base64
   echo -n "your_actual_api_key" | base64
   ```

3. **Replace placeholder values in YAML files**

4. **Apply to cluster:**
   ```bash
   kubectl apply -f postgres-secret-production.yaml
   kubectl apply -f api-keys-secret-production.yaml
   ```

5. **Verify secrets:**
   ```bash
   kubectl get secrets -n dax-trading
   kubectl describe secret postgres-secret -n dax-trading
   ```

## Security Notes

- ⚠️  **NEVER commit the actual secret files to git**
- ⚠️  **Always use production-specific filenames** 
- ⚠️  **Delete working files after applying**
- ✅  **Only commit these template files with placeholder values**

## Quick Setup Script

```bash
#!/bin/bash
# Create secrets from environment variables
kubectl create secret generic postgres-secret \
  --from-literal=POSTGRES_USER=$POSTGRES_USER \
  --from-literal=POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
  -n dax-trading

kubectl create secret generic api-keys-secret \
  --from-literal=NEWS_API_KEY=$NEWS_API_KEY \
  --from-literal=ALPHA_VANTAGE_KEY=$ALPHA_VANTAGE_KEY \
  --from-literal=FINANCIAL_API_KEY=$FINANCIAL_API_KEY \
  -n dax-trading
```
