# FTMO Trading System Makefile

.PHONY: help build-all test-all dev-up dev-down deploy clean

# Default target
help:
	@echo "FTMO Trading System - Available targets:"
	@echo ""
	@echo "Development:"
	@echo "  dev-up          - Start all services with docker-compose"
	@echo "  dev-down        - Stop all services"
	@echo "  dev-logs        - Show logs from all services"
	@echo "  dev-reset       - Reset development environment"
	@echo ""
	@echo "Building:"
	@echo "  build-all       - Build all Docker images"
	@echo "  build-service   - Build specific service (SERVICE=name)"
	@echo ""
	@echo "Testing:"
	@echo "  test-all        - Run all tests"
	@echo "  test-service    - Test specific service (SERVICE=name)"
	@echo "  lint            - Run linting on all code"
	@echo "  format          - Format all Python code"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy          - Deploy to Kubernetes"
	@echo "  deploy-staging  - Deploy to staging environment"
	@echo "  undeploy        - Remove from Kubernetes"
	@echo ""
	@echo "Utilities:"
	@echo "  db-shell        - Open PostgreSQL shell"
	@echo "  health-check    - Check all service health endpoints"
	@echo "  clean           - Clean up build artifacts"

# Development commands
dev-up:
	@echo "Starting FTMO Trading System..."
	docker-compose up -d
	@echo "Services started. Check status with: make dev-logs"

dev-down:
	@echo "Stopping FTMO Trading System..."
	docker-compose down

dev-logs:
	docker-compose logs -f

dev-reset:
	docker-compose down -v
	docker-compose build --no-cache
	docker-compose up -d

# Build commands
build-all:
	@echo "Building all Docker images..."
	@for service in postgres-data-collector dax-llm-service finbert-trading-llm finbert-sentiment-service signal-generation-engine technical-pattern-service news-ingestion-service signal-monitoring; do \
		echo "Building $$service..."; \
		docker build -t ftmo-trading/$$service:latest services/$$service/; \
	done
	@echo "All images built successfully!"

build-service:
	@if [ -z "$(SERVICE)" ]; then \
		echo "Usage: make build-service SERVICE=<service-name>"; \
		exit 1; \
	fi
	@echo "Building $(SERVICE)..."
	docker build -t ftmo-trading/$(SERVICE):latest services/$(SERVICE)/

# Testing commands
test-all:
	@echo "Running all tests..."
	python -m pytest services/*/tests/ common-lib/tests/ -v --cov=services --cov=common-lib

test-service:
	@if [ -z "$(SERVICE)" ]; then \
		echo "Usage: make test-service SERVICE=<service-name>"; \
		exit 1; \
	fi
	@echo "Testing $(SERVICE)..."
	python -m pytest services/$(SERVICE)/tests/ -v

lint:
	@echo "Running linting..."
	flake8 services common-lib --max-line-length=127
	black --check services common-lib
	isort --check-only services common-lib

format:
	@echo "Formatting code..."
	black services common-lib
	isort services common-lib

# Deployment commands
deploy:
	@echo "Deploying to Kubernetes..."
	kubectl create namespace dax-trading --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -f infra/k8s-base/
	@for service in services/*/k8s-*.yaml; do \
		kubectl apply -f $$service; \
	done
	@echo "Deployment complete!"

deploy-staging:
	@echo "Deploying to staging..."
	kubectl apply -f infra/k8s-base/ --namespace=dax-trading-staging

undeploy:
	@echo "Removing from Kubernetes..."
	kubectl delete namespace dax-trading --ignore-not-found=true

# Utility commands
db-shell:
	@echo "Opening database shell..."
	docker-compose exec postgres psql -U finrl_user -d dax_trading

health-check:
	@echo "Checking service health..."
	@curl -s http://localhost:8080/health || echo "Signal Generation Engine: DOWN"
	@curl -s http://localhost:8001/health || echo "FinBERT Trading LLM: DOWN"
	@curl -s http://localhost:8002/health || echo "Sentiment Service: DOWN"
	@curl -s http://localhost:3000 > /dev/null && echo "Grafana: UP" || echo "Grafana: DOWN"

clean:
	@echo "Cleaning up..."
	docker system prune -f
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

# Install development dependencies
install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

# Generate requirements from all services
requirements:
	@echo "Generating consolidated requirements..."
	@find services -name requirements.txt -exec cat {} \; | sort -u > requirements.txt