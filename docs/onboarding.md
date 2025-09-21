# Developer Onboarding Guide

Welcome to the FTMO Trading System! This guide will help you get up and running quickly.

## 🚀 Quick Start (5 minutes)

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Git
- kubectl (for Kubernetes deployment)

### 1. Clone and Setup

```bash
# Clone the repository
git clone git@github.com:dan8992/ftmo-trading-monorepo.git
cd ftmo-trading-monorepo

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
vim .env
```

### 2. Local Development with Docker

```bash
# Start all services
make dev-up

# View logs
make dev-logs

# Stop services
make dev-down
```

### 3. Verify Setup

```bash
# Check service health
curl http://localhost:8080/health  # Signal Generation Engine
curl http://localhost:8001/health  # FinBERT Trading LLM
curl http://localhost:3000         # Grafana Dashboard
```

## 📁 Repository Structure

```
ftmo-trading-monorepo/
├── services/              # Microservices
│   ├── postgres/          # Database with schema
│   ├── signal-generation-engine/  # Core trading logic
│   ├── finbert-trading-llm/       # AI sentiment analysis
│   └── ...
├── common-lib/           # Shared FTMO compliance libraries
├── infra/               # Kubernetes manifests
├── charts/              # Helm charts
├── ci/                  # CI/CD utilities
└── docs/                # Documentation
```

## 🛠️ Development Workflow

### Adding a New Service

1. **Create service directory**:
   ```bash
   mkdir -p services/my-new-service/{app,tests}
   cd services/my-new-service
   ```

2. **Create basic files**:
   ```bash
   # Dockerfile
   cp ../signal-generation-engine/Dockerfile .
   
   # Requirements
   touch requirements.txt
   
   # Main application
   touch app/main.py
   
   # Tests
   touch tests/test_main.py
   
   # Kubernetes manifests
   touch k8s-deployment.yaml
   
   # README
   touch README.md
   ```

3. **Implement service**:
   - Add your Python code in `app/main.py`
   - Include health endpoint: `GET /health`
   - Add metrics endpoint: `GET /metrics`
   - Write tests in `tests/`

4. **Update CI/CD**:
   - Add service to `.github/workflows/ci.yml` matrix
   - Update build targets in `Makefile`

### Code Standards

- **Python**: Follow PEP 8, use Black for formatting
- **Imports**: Use isort for import organization
- **Testing**: pytest for unit tests, aim for >80% coverage
- **Documentation**: Docstrings for all functions and classes

### Testing

```bash
# Run all tests
make test-all

# Run specific service tests
make test-service SERVICE=signal-generation-engine

# Run with coverage
pytest services/signal-generation-engine/tests/ --cov=services/signal-generation-engine

# Lint code
make lint

# Format code
make format
```

## 🐳 Docker Development

### Building Images

```bash
# Build all services
make build-all

# Build specific service
make build-service SERVICE=signal-generation-engine

# Build with custom tag
docker build -t my-service:dev services/signal-generation-engine/
```

### Running Individual Services

```bash
# Run signal generation engine
docker run -d \
  --name signal-engine \
  -p 8080:8080 \
  -e POSTGRES_HOST=localhost \
  -e POSTGRES_PASSWORD=password \
  ftmo-trading/signal-generation-engine:latest

# Check logs
docker logs -f signal-engine
```

## 🎯 FTMO Integration

### Understanding FTMO Rules

The system enforces these FTMO compliance rules:

1. **Daily Loss Limit**: Maximum 5% loss per day
2. **Total Drawdown**: Maximum 10% total loss
3. **Position Sizing**: Maximum 2% risk per trade
4. **Market Conditions**: No trading during weekends/holidays
5. **News Events**: Avoid high-impact news periods

### Using FTMO Libraries

```python
# Import FTMO compliance modules
from common_lib.ftmo_realtime_compliance import FTMOComplianceMonitor
from common_lib.ftmo_position_sizer import FTMOPositionSizer

# Initialize compliance monitor
compliance = FTMOComplianceMonitor(account_balance=100000)

# Check if trade is allowed
if compliance.is_trading_allowed():
    # Calculate position size
    sizer = FTMOPositionSizer(account_balance=100000)
    position_size = sizer.calculate_position_size(
        entry_price=1.1000,
        stop_loss=1.0950,
        risk_percentage=0.02
    )
```

### Testing FTMO Compliance

```bash
# Run FTMO validation tests
python common-lib/ftmo_system_validation.py

# Run backtest with FTMO rules
python common-lib/ftmo_realistic_backtester.py
```

## 🤖 AI/ML Development

### FinBERT Sentiment Analysis

```python
# Example usage of FinBERT service
import requests

response = requests.post(
    "http://localhost:8001/analyze",
    json={"text": "The market is showing strong bullish momentum"}
)

result = response.json()
# Returns: {"sentiment": 0.85, "confidence": 0.92}
```

### Adding New Models

1. **Create model directory**: `services/my-model-service/`
2. **Implement inference endpoint**: Return predictions with confidence scores
3. **Add to signal generation**: Update ensemble logic
4. **Test integration**: Ensure predictions flow to trading signals

### Model Performance Monitoring

- All predictions stored in `ai_predictions` table
- Confidence scores tracked for calibration
- Actual vs predicted values for accuracy metrics

## 📊 Database Development

### Schema Changes

1. **Create migration**:
   ```sql
   -- services/postgres/migrations/001_add_new_table.sql
   CREATE TABLE new_feature (
       id SERIAL PRIMARY KEY,
       created_at TIMESTAMPTZ DEFAULT NOW()
   );
   ```

2. **Test migration**:
   ```bash
   psql -U finrl_user -d dax_trading -f services/postgres/migrations/001_add_new_table.sql
   ```

3. **Update schema**: Add to `services/postgres/init-sql/schema.sql`

### Database Access

```python
import psycopg2
import os

# Connection configuration
db_config = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
    'database': os.getenv('POSTGRES_DB', 'dax_trading'),
    'user': os.getenv('POSTGRES_USER', 'finrl_user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'password')
}

# Connect and query
conn = psycopg2.connect(**db_config)
with conn.cursor() as cur:
    cur.execute("SELECT * FROM trading_signals LIMIT 10")
    signals = cur.fetchall()
```

## 🚢 Kubernetes Development

### Local Cluster Setup

```bash
# Using kind (Kubernetes in Docker)
kind create cluster --name ftmo-trading

# Deploy to local cluster
kubectl apply -f infra/k8s-base/
kubectl apply -f services/*/k8s-*.yaml
```

### Debugging in Kubernetes

```bash
# Check pod status
kubectl get pods -n dax-trading

# View logs
kubectl logs -f deployment/signal-generation-engine -n dax-trading

# Port forward for testing
kubectl port-forward svc/signal-generation-service 8080:8080 -n dax-trading

# Execute into pod
kubectl exec -it deployment/signal-generation-engine -n dax-trading -- bash
```

### Resource Management

- **CPU/Memory**: Set appropriate requests and limits
- **Storage**: Use persistent volumes for stateful services
- **Networking**: Use services for inter-pod communication

## 🔧 Troubleshooting

### Common Issues

1. **Port conflicts**: Check `docker ps` and `lsof -i :8080`
2. **Database connection**: Verify `POSTGRES_*` environment variables
3. **Model loading**: Check model cache directory permissions
4. **Import errors**: Ensure all requirements are installed

### Debug Commands

```bash
# Check service health
make health-check

# View all logs
make dev-logs

# Reset local environment
make dev-reset

# Database shell
make db-shell
```

### Getting Help

1. **Check documentation**: `/docs` directory
2. **Search issues**: GitHub issues for common problems
3. **Ask team**: Create GitHub issue with detailed description
4. **Debug mode**: Set `DEBUG=True` in environment

## 📈 Performance Guidelines

### Code Performance

- **Database queries**: Use indexes, avoid N+1 queries
- **Model inference**: Batch requests when possible
- **Memory usage**: Monitor with `memory_profiler`
- **Response times**: Target <500ms for API endpoints

### Resource Limits

```yaml
# Example resource specification
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

## 🔐 Security Best Practices

1. **Secrets**: Never commit secrets to Git
2. **Environment variables**: Use for configuration
3. **Network policies**: Limit inter-service communication
4. **Image scanning**: Use Trivy for vulnerability detection
5. **Dependencies**: Keep packages updated

## 📚 Learning Resources

- **Kubernetes**: [Official Documentation](https://kubernetes.io/docs/)
- **FinBERT**: [Hugging Face Model](https://huggingface.co/ProsusAI/finbert)
- **FTMO Rules**: [Official FTMO Website](https://ftmo.com/en/trading-rules/)
- **Technical Analysis**: [TA-Lib Documentation](https://ta-lib.org/)

## 🎯 Next Steps

1. **Complete setup**: Verify all services are running
2. **Run tests**: Ensure everything works correctly
3. **Explore codebase**: Understand the architecture
4. **Make changes**: Start with small improvements
5. **Deploy**: Test on Kubernetes cluster

Welcome to the team! 🚀